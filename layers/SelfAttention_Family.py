import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class TemporalAttention(nn.Module):
    """
    TemporalAttention applies a simple linear transformation to the input sequence
    and then uses the last time step as a query to compute a weighted sum of the entire sequence.

    Input:
        z (Tensor): Input tensor of shape [N, T, D] where N is the batch size,
                    T is the sequence length, and D is the feature dimension.
    Output:
        Tensor: Aggregated output of shape [N, D].
    """

    def __init__(self, d_model):
        super(TemporalAttention, self).__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # Apply linear transformation: [N, T, D]
        h = self.trans(z)
        # Use the last time step as query: shape [N, D] -> [N, D, 1]
        query = h[:, -1, :].unsqueeze(-1)
        # Compute attention weights: [N, T, D] x [N, D, 1] -> [N, T]
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # Shape: [N, 1, T]
        # Weighted sum over the time dimension: [N, 1, T] x [N, T, D] -> [N, 1, D]
        output = torch.matmul(lam, z).squeeze(1)  # Shape: [N, D]
        return output

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


