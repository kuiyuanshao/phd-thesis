import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BayesianMHA(d_model, nhead, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)

        self.dropout_ffn1 = nn.Dropout(dropout)
        self.dropout_ffn2 = nn.Dropout(dropout)
        self.dropout_res = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # Attention sub-layer with residual connection
        res = x
        x = self.norm1(x)
        x = res + self.dropout_res(self.attn(x, x, x))

        # Feed-forward sub-layer with residual connection
        res = x
        x = self.norm2(x)
        x = self.ffn1(self.dropout_ffn1(x))
        x = self.activation(x)
        x = self.ffn2(self.dropout_ffn2(x))

        return res + self.dropout_res(x)


class BayesianMHA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_qkv = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        B, N, C = x.shape

        # Apply Bayesian dropout before projections
        x_drop = self.dropout_qkv(x)
        y_drop = self.dropout_qkv(y)
        z_drop = self.dropout_qkv(z)

        q = self.q_proj(x_drop).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(y_drop).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(z_drop).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = self.attn_dropout(F.softmax(attn_weights, dim=-1))
        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)

        return self.out_proj(self.dropout_out(out))