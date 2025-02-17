
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from models import Transformer, iTransformer  # Only these two models are allowed
from layers.SelfAttention_Family import TemporalAttention
import os


class ADA_PPO(nn.Module):
    """
    ADA_PPO implements a reinforcement learning (PPO-based) framework with a Transformer backbone.

    The model supports only 'Transformer' and 'itransformer'. The backbone is used to extract features,
    which are then processed by separate heads for horizon selection and portfolio allocation.
    """

    def __init__(self, model_name, configs, setting, deterministic=False):
        super(ADA_PPO, self).__init__()
        self.model_name = configs.model
        # Restrict the available models to Transformer and iTransformer only.
        self.model_dict = {
            'Transformer': Transformer,
            'iTransformer': iTransformer
        }
        self.model = self.model_dict[model_name].Model(configs)

        # Transfer learning settings: load pre-trained weights and freeze parameters if specified.
        if configs.transfer:
            checkpoint_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path))
            if configs.freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.periods = configs.horizons
        self.num_periods = len(self.periods)
        self.deterministic = deterministic
        self.Temporal = TemporalAttention(configs.d_model)

        self.shared_rep = nn.Sequential(
            nn.Linear(configs.d_model, 128),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )

        self.horizon_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(128, self.num_periods)
        )
        self.portfolio_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(128, self.pred_len)
        )

        # Actor-Critic layers for each trading horizon for portfolio allocation
        self.layer_mu = nn.Linear(self.pred_len, 1)
        self.layer_std = nn.Linear(self.pred_len, 1)
        self.layer_value = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pred_len, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in self.periods
        ])
        # Additional head for portfolio return prediction (if needed)
        self.layer_pred = nn.Linear(self.pred_len, 1)

    def pi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for the policy network.

        Returns:
            action: Final action (portfolio allocation) after tanh activation.
            total_log_prob: Combined log probability from horizon selection and action sampling.
            total_entropy: Mean entropy of the policy.
            final_idx: The most frequent selected horizon index.
        """
        # Obtain prediction scores from the backbone model
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # shape: [N, pred_len]
        pred_scores = self.Temporal(pred_scores)
        shared_features = self.shared_rep(pred_scores)

        # Horizon selection head
        horizon_logits = self.horizon_head(shared_features)  # shape: [N, num_periods]
        epsilon = 1e-8
        selection_probs = F.softmax(horizon_logits, dim=1) + epsilon
        selection_probs = selection_probs / selection_probs.sum(dim=1, keepdim=True)
        selection_dist = Categorical(selection_probs)
        horizon_entropy = selection_dist.entropy()

        if (not self.training) or self.deterministic:
            selected_horizon = torch.argmax(selection_probs, dim=1)  # [N]
        else:
            selected_horizon = selection_dist.sample()
        unique_vals, counts = selected_horizon.unique(return_counts=True)
        final_idx = unique_vals[counts.argmax()]

        # Portfolio allocation head
        portfolio_repr = self.portfolio_head(shared_features)  # shape: [N, pred_len]
        mu = self.layer_mu(portfolio_repr).squeeze(-1)
        std = torch.clamp(F.softplus(self.layer_std(portfolio_repr)), min=1e-2).squeeze(-1)

        if (not self.training) or self.deterministic:
            action = torch.tanh(mu)
            return action, None, horizon_entropy.mean(), final_idx

        action_dist = Normal(mu, std)
        raw_action = action_dist.rsample()  # [N]
        action = torch.tanh(raw_action)

        log_prob = action_dist.log_prob(raw_action)
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-5))
        log_prob = log_prob.squeeze(-1)

        horizon_log_prob = selection_dist.log_prob(selected_horizon)  # [N]
        total_log_prob = horizon_log_prob + log_prob
        action_entropy = action_dist.entropy()  # [N]
        total_entropy = (horizon_entropy + action_entropy).mean()

        return action, total_log_prob.sum(), total_entropy, final_idx

    def value(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Estimate the value function by computing a weighted sum over horizons.

        Returns:
            A scalar value representing the mean value estimation.
        """
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred_scores = self.Temporal(pred_scores)
        shared_features = self.shared_rep(pred_scores)
        portfolio_repr = self.portfolio_head(shared_features)
        horizon_logits = self.horizon_head(shared_features)
        selection_probs = F.softmax(horizon_logits, dim=1)  # [N, num_periods]

        value_list = []
        for i, period in enumerate(self.periods):
            value_i = self.layer_value[i](portfolio_repr).squeeze(-1)  # [N]
            value_list.append(value_i)
        values_all = torch.stack(value_list, dim=1)  # [N, num_periods]
        value_portfolio = (selection_probs * values_all).sum(dim=1)  # [N]
        return value_portfolio.mean(dim=0).squeeze()

    def pred(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Predict portfolio return using the portfolio prediction head.

        Returns:
            pred_return: Predicted return as a scalar.
        """
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred_scores = self.Temporal(pred_scores)
        shared_features = self.shared_rep(pred_scores)
        portfolio_repr = self.portfolio_head(shared_features)
        pred_return = self.layer_pred(portfolio_repr).squeeze(-1)
        return pred_return
