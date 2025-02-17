import copy
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, moe, ppo, ada_ppo, tradingenv, Informer, Reformer, Autoformer, Fedformer, Flowformer, Flashformer, itransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, port_visual
from utils.metrics import metric
from utils.backtest import *
from itertools import tee
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def get_sample(dataset, index, device):
    """
    Retrieve a sample from the dataset and convert it to torch tensors.
    """
    sample = dataset[index]  # Calls __getitem__
    (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) = sample
    return (
        torch.tensor(batch_x, dtype=torch.float).to(device),
        torch.tensor(batch_y, dtype=torch.float).to(device),
        torch.tensor(batch_x_mark, dtype=torch.float).to(device),
        torch.tensor(batch_y_mark, dtype=torch.float).to(device),
        torch.tensor(ground_true, dtype=torch.float).to(device)
    )


class Exp_DeepAries(Exp_Basic):
    """
    Exp_DeepAries implements the DeepAries experiment: a reinforcement learning
    framework for dynamic portfolio management using a Transformer-based backbone and PPO updates.
    """
    def __init__(self, args, setting):
        super(Exp_DeepAries, self).__init__(args, setting)
        self.setting = setting
        # Load a pretrained ADA_PPO model; its parameters are frozen.
        self.old_model = ada_ppo.ADA_PPO(args.model, args, setting, deterministic=False).to(self.device)
        for param in self.old_model.parameters():
            param.requires_grad = False

        self.horizons = args.horizons
        self.temperature = args.temperature
        self.env = tradingenv.TradingEnvironment(self.args)
        self.buffer_size = 1  # Rollout buffer size
        self.ent_coef = 0.01
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.1
        self.beta = 0.1
        self.data = []
        self.max_clip = 5  # Clipping range for log probability ratio
        self.min_clip = -5

    def _build_model(self):
        """
        Build the ADA_PPO model. If using multiple GPUs, wrap the model in DataParallel.
        """
        model = ada_ppo.ADA_PPO(self.args.model, self.args, self.setting, deterministic=False)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def put_data(self, item):
        """
        Append rollout data to the buffer.
        """
        self.data.append(item)

    def _get_data(self, flag):
        """
        Retrieve the dataset and dataloader for the specified flag.
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Select the Adam optimizer for trainable parameters.
        """
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        return optim.Adam(trainable_params, lr=self.args.learning_rate)

    def _select_criterion(self):
        """
        Select MSELoss as the training criterion.
        """
        return nn.MSELoss()

    def train(self, setting):
        """
        Train the DeepAries model using PPO-based reinforcement learning.
        """
        train_dataset, _ = self._get_data(flag='train')
        n_data = len(train_dataset)
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.logger.info(f"[Train] Epoch {epoch + 1}/{self.args.train_epochs}")
            self.model.train()
            self.env.reset()

            i = 0
            epoch_loss = []
            epoch_time = time.time()

            while i < n_data:
                batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = get_sample(train_dataset, i, self.device)
                # Prepare decoder input with zero padding for prediction part
                dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
                scores, log_prob, entropy, selected_period_indices = self.old_model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                # Select top stocks based on scores
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                final_weights = torch.zeros_like(scores)
                final_weights[top_indices] = topk_weights
                returns = ground_true[:, selected_period_indices]

                # Calculate simulated returns for each horizon
                simulated_returns = torch.tensor(
                    [(final_weights * ground_true[:, i]).sum() for i in range(len(self.horizons))],
                    device=ground_true.device
                )
                optimal_horizon_index = torch.argmax(simulated_returns)
                bonus_ratio = 0.2
                reward = self.env.step(final_weights, returns)
                if optimal_horizon_index.item() == selected_period_indices.item():
                    reward *= (1 + bonus_ratio)
                else:
                    reward *= (1 - bonus_ratio)

                chosen_horizon = self.horizons[selected_period_indices]
                next_i = i + chosen_horizon
                done = (next_i >= n_data - 1)
                if not done:
                    next_sample = get_sample(train_dataset, next_i, self.device)
                    next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, _ = next_sample
                else:
                    next_batch_x = torch.zeros_like(batch_x)
                    next_batch_y = torch.zeros_like(batch_y)
                    next_batch_x_mark = torch.zeros_like(batch_x_mark)
                    next_batch_y_mark = torch.zeros_like(batch_y_mark)

                transition = (
                    batch_x, batch_y, batch_x_mark, batch_y_mark,  # current state
                    scores.detach(), reward,
                    next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,  # next state
                    log_prob.detach(),
                    done,
                    returns,
                )
                self.env.rollout.append(transition)

                i = next_i
                if len(self.env.rollout) >= self.env.rollout_len:
                    self.put_data(self.env.rollout)
                    self.env.rollout = []
                    loss_val = self.train_net(K_epoch=1, model_optim=model_optim)
                    # Update the old model with the current model weights
                    self.old_model.load_state_dict(self.model.state_dict())
                    if loss_val is not None:
                        epoch_loss.append(loss_val)
                    torch.cuda.empty_cache()
            vali_loss, vali_reward, test_loss, test_reward = self.vali()
            avg_train_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            torch.cuda.empty_cache()
            self.logger.info(
                f"[Epoch {epoch + 1}] TrainLoss={avg_train_loss:.4f}, "
                f"ValiLoss={vali_loss:.4f}, TestLoss={test_loss:.4f}"
            )
            self.wandb.log({
                "Train Loss": avg_train_loss,
                "Validation Loss": vali_loss,
                "Validation Reward": vali_reward,
                "Test Loss": test_loss,
                "Test Reward": test_reward,
                "Epoch": epoch
            })

            # Early stopping based on validation loss
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopped!")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Load the best model checkpoint after training
        best_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_path))
        self.logger.info("Training completed, best model loaded.")

    def vali(self):
        """
        Evaluate the model on the validation and test datasets.
        """
        vali_dataset, _ = self._get_data('val')
        v_loss, v_reward = self._evaluate_dataset(vali_dataset)

        test_dataset, _ = self._get_data('test')
        t_loss, t_reward = self._evaluate_dataset(test_dataset)

        return v_loss, v_reward, t_loss, t_reward

    def _evaluate_dataset(self, dataset):
        """
        Compute evaluation metrics (loss and total reward) on the given dataset.
        """
        self.model.eval()
        env_temp = tradingenv.TradingEnvironment(self.args)
        env_temp.reset()
        total_loss, total_value_loss, total_pred_loss = 0.0, 0.0, 0.0
        n_data = len(dataset)
        i = 0
        total_reward = 0.0

        while i < n_data:
            batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = get_sample(dataset, i, self.device)
            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)

            with torch.no_grad():
                scores, log_prob, entropy, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                final_weights = torch.zeros_like(scores)
                final_weights[top_indices] = topk_weights
                returns = ground_true[:, selected_period_indices]
                reward = env_temp.step(final_weights, returns)
                chosen_horizon = self.horizons[selected_period_indices]
                i += chosen_horizon
                done = (i >= n_data - 1)
                total_reward += reward
                if not done:
                    next_sample = get_sample(dataset, i, self.device)
                    next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, _ = next_sample
                else:
                    next_batch_x = torch.zeros_like(batch_x)
                    next_batch_y = torch.zeros_like(batch_y)
                    next_batch_x_mark = torch.zeros_like(batch_x_mark)
                    next_batch_y_mark = torch.zeros_like(batch_y_mark)
                value = self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark)
                next_value = self.model.value(next_batch_x, next_batch_x_mark, next_batch_y, next_batch_y_mark)
                td_target = reward + self.gamma * next_value * (0 if done else 1)
                value_loss = F.smooth_l1_loss(value, td_target)
                pred_loss = F.smooth_l1_loss(scores, returns)
                loss = value_loss + self.beta * pred_loss

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_pred_loss += pred_loss.item()
        avg_loss = total_loss
        self.logger.info(
            f"[Validation] Total Reward={total_reward:.4f}, Total Loss={avg_loss:.4f}, "
            f"Value Loss={total_value_loss:.4f}, Pred Loss={total_pred_loss:.4f}"
        )
        self.model.train()  # Reset model to training mode
        return avg_loss, total_reward

    def backtest(self, setting, load=False):
        """
        Backtest the trained DeepAries model using reinforcement learning principles.
        """
        if load:
            best_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_path))
            self.logger.info(f"Loaded best model from {best_path}")

        backtest_dataset, _ = self._get_data('backtest')
        n_data = len(backtest_dataset)
        self.model.eval()

        env_test = tradingenv.TradingEnvironment(self.args)
        env_test.reset()

        portfolio_values = [1.0]
        portfolio_dates = [backtest_dataset.unique_dates[0]]
        i = 0
        while i < n_data:
            batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = get_sample(backtest_dataset, i, self.device)
            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
            with torch.no_grad():
                scores, log_prob, entropy, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
            current_date = backtest_dataset.unique_dates[i + self.args.seq_len]
            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
            selected_scores = scores[top_indices]
            topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
            final_weights = torch.zeros_like(scores)
            final_weights[top_indices] = topk_weights
            returns = ground_true[:, selected_period_indices]
            chosen_horizon = self.horizons[selected_period_indices]
            reward = env_test.step(final_weights, returns)
            portfolio_values.append(env_test.portfolio_value)
            portfolio_dates.append(current_date)
            i += chosen_horizon

        final_pf = env_test.portfolio_value
        self.logger.info(f"[BackTest] Final Portfolio Value = {final_pf:.4f}")
        self.wandb.log({"BackTest Final Portfolio Value": final_pf})
        start_date = backtest_dataset.unique_dates[0]
        end_date = backtest_dataset.unique_dates[-1]

        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        raw_data = pd.read_csv(csv_path)
        raw_data['date'] = pd.to_datetime(raw_data['date']).dt.tz_localize(None)
        index_name = self._get_market_index_name()
        index_data = fetch_index_data(index_name, start_date, end_date)
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        metrics = run_backtest(
            data=raw_data,
            index_data=index_data,
            start_date=start_date,
            end_date=end_date,
            fee_rate=self.args.fee_rate,
            external_portfolio=np.array(portfolio_values),
            external_dates=pd.to_datetime(portfolio_dates),
            pred_len=self.args.pred_len,
            total_periods=len(backtest_dataset.unique_dates),
            folder_path=folder_path
        )
        strategy_columns = ["External Portfolio"]
        log_data = {}
        for i, metric_name in enumerate(metrics["Metric"]):
            for strategy in strategy_columns:
                value = metrics[strategy][i]
                log_data[f"test/{strategy}/{metric_name}"] = value

        self.wandb.log(log_data)
        return final_pf

    def make_batch(self):
        """
        Convert rollout data into mini-batches.
        """
        batch_data = {
            "batch_x": [],
            "batch_y": [],
            "batch_x_mark": [],
            "batch_y_mark": [],
            "scores": [],
            "reward": [],
            "next_batch_x": [],
            "next_batch_y": [],
            "next_batch_x_mark": [],
            "next_batch_y_mark": [],
            "log_prob": [],
            "done": [],
            "return_data": []
        }

        for _ in range(self.buffer_size):
            rollout = self.data.pop(0)
            for transition in rollout:
                (batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward,
                 next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, log_prob, done, return_data) = transition
                batch_data["batch_x"].append(batch_x)
                batch_data["batch_y"].append(batch_y)
                batch_data["batch_x_mark"].append(batch_x_mark)
                batch_data["batch_y_mark"].append(batch_y_mark)
                batch_data["scores"].append(scores)
                batch_data["reward"].append([reward])
                batch_data["next_batch_x"].append(next_batch_x)
                batch_data["next_batch_y"].append(next_batch_y)
                batch_data["next_batch_x_mark"].append(next_batch_x_mark)
                batch_data["next_batch_y_mark"].append(next_batch_y_mark)
                batch_data["log_prob"].append(log_prob)
                done_mask = 0 if done else 1
                batch_data["done"].append([done_mask])
                batch_data["return_data"].append(return_data)

        for key in batch_data:
            try:
                if key in ["reward", "done"]:
                    batch_data[key] = torch.tensor(batch_data[key], dtype=torch.float).squeeze().to(self.device)
                else:
                    batch_data[key] = torch.stack(batch_data[key], dim=0).to(self.device)
            except Exception as e:
                print(f"Error processing key {key}: {e}")

        mini_batches = []
        for i in range(len(batch_data["batch_x"])):
            mini_batches.append((
                batch_data["batch_x"][i],
                batch_data["batch_y"][i],
                batch_data["batch_x_mark"][i],
                batch_data["batch_y_mark"][i],
                batch_data["scores"][i],
                batch_data["reward"][i],
                batch_data["next_batch_x"][i],
                batch_data["next_batch_y"][i],
                batch_data["next_batch_x_mark"][i],
                batch_data["next_batch_y_mark"][i],
                batch_data["done"][i],
                batch_data["log_prob"][i],
                batch_data["return_data"][i]
            ))
        return mini_batches

    def calc_advantage(self, data):
        """
        Calculate advantage values and TD targets for PPO.
        """
        data_with_adv = []
        td_target_lst = []
        delta_lst = []
        for i, transition in enumerate(data):
            batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward, next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, done_mask, old_log_prob, return_data = transition

            with torch.no_grad():
                v_s_next = self.model.value(next_batch_x, next_batch_x_mark, next_batch_y, next_batch_y_mark)
                v_s = self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark)
                td_target = reward + self.gamma * v_s_next * done_mask
                delta = td_target - v_s
            td_target_lst.append(td_target.item())
            delta_lst.append(delta.item())
        advantage_lst = []
        running_adv = 0.0
        for delta_t in reversed(delta_lst):
            running_adv = self.gamma * self.lmbda * running_adv + delta_t
            advantage_lst.append(running_adv)
        advantage_lst.reverse()

        advantage_tensor = torch.tensor(advantage_lst, dtype=torch.float, device=self.device)
        advantage_mean = advantage_tensor.mean()
        advantage_std = advantage_tensor.std() + 1e-8
        advantage_tensor = (advantage_tensor - advantage_mean) / advantage_std

        for i, transition in enumerate(data):
            (batch_x, batch_y, batch_x_mark, batch_y_mark,
             scores, reward,
             next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
             done_mask, old_log_prob, return_data) = transition

            td_target_tensor = torch.tensor(td_target_lst[i], dtype=torch.float, device=self.device)
            adv_val = advantage_tensor[i]

            data_with_adv.append((
                batch_x, batch_y, batch_x_mark, batch_y_mark,
                scores, reward,
                next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                done_mask, old_log_prob,
                td_target_tensor, adv_val,
                return_data
            ))

        return data_with_adv

    def train_net(self, K_epoch=10, model_optim=None):
        """
        Perform PPO network updates over multiple epochs.
        """
        if len(self.data) == self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            losses = 0.0

            for _ in range(K_epoch):
                for mini_batch in data:
                    (batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward,
                     next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                     done_mask, old_log_prob, td_target, advantage, return_data) = mini_batch

                    scores, log_prob, total_entropy, _ = self.model.pi(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=self.min_clip, max=self.max_clip))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                    entropy_mean = total_entropy
                    pred_loss = F.smooth_l1_loss(self.model.pred(batch_x, batch_x_mark, batch_y, batch_y_mark), return_data)
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.smooth_l1_loss(self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark), td_target)
                    loss = self.beta * pred_loss + policy_loss + value_loss - self.ent_coef * entropy_mean
                    losses += loss.mean().item()
                    model_optim.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()
            return losses

    def _get_market_index_name(self):
        """
        Return the market index name based on the selected market.
        """
        market_indices = {
            'kospi': '^KS11',
            'dj30': '^DJI',
            'sp500': '^GSPC',
            'nasdaq': '^IXIC',
            'csi300': '000300.SS',
            'ftse': '^FTSE',
        }
        index_name = market_indices.get(self.args.market)
        if not index_name:
            raise ValueError(f"Unsupported market: {self.args.market}")
        return index_name

    def simulate_episode(self):
        """
        Simulate a single episode in the reinforcement learning environment.

        Returns:
            total_reward (float): Total accumulated reward over the episode.
        """
        self.env.reset()
        done = False
        total_reward = 0.0
        dataset, _ = self._get_data('train')
        i = 0
        n_data = len(dataset)

        while not done and i < n_data:
            batch = get_sample(dataset, i, self.device)
            batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = batch
            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
            with torch.no_grad():
                scores, _, _, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
            selected_scores = scores[top_indices]
            topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
            final_weights = torch.zeros_like(scores)
            final_weights[top_indices] = topk_weights
            returns = ground_true[:, selected_period_indices]
            reward = self.env.step(final_weights, returns)
            total_reward += reward
            chosen_horizon = self.horizons[selected_period_indices]
            i += chosen_horizon
            if i >= n_data - 1:
                done = True

        return total_reward
