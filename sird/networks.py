import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Callable
from torch_frame import stype
from torch_frame.nn.encoder import LinearEncoder, EmbeddingEncoder
from bayesian_transformer import BayesianTransformerBlock
ModuleType = Union[str, Callable[..., nn.Module]]

class FeatureTokenizer(nn.Module):
    def __init__(self, config, dim_info):
        super().__init__()
        self.d_model = config["model"]["channels"]
        stats = dim_info['pt_frame_stats']

        self.num_x_encoder = LinearEncoder(
            out_channels=self.d_model,
            stats_list=stats['x_num'],
            stype=stype.numerical,
            post_module=None
        )
        self.has_cond_num = len(stats['cond_num']) > 0
        if self.has_cond_num:
            self.num_cond_encoder = LinearEncoder(
                out_channels=self.d_model,
                stats_list=stats['cond_num'],
                stype=stype.numerical,
                post_module=None
            )

        self.cat_x_encoder = EmbeddingEncoder(
            out_channels=self.d_model,
            stats_list=stats['x_cat'],
            stype=stype.categorical,
            post_module=None
        )

        self.has_cond_cat = len(stats['cond_cat']) > 0
        if self.has_cond_cat:
            self.cat_cond_encoder = EmbeddingEncoder(
                out_channels=self.d_model,
                stats_list=stats['cond_cat'],
                stype=stype.categorical,
                post_module=None
            )

        self.x_num_count = len(stats['x_num'])
        self.cond_num_count = len(stats['cond_num'])
        self.x_cat_sizes = dim_info['x_cat_sizes']
        self.cond_cat_sizes = dim_info['cond_cat_sizes']

    def forward(self, x_curr, cond):
        x_list = []
        if self.x_num_count > 0:
            x_list.append(self.num_x_encoder(x_curr[:, :self.x_num_count]))
        if self.x_cat_sizes:
            x_indices = self._recover_indices(x_curr[:, self.x_num_count:], self.x_cat_sizes)
            x_list.append(self.cat_x_encoder(x_indices))
        x_tokens = torch.cat(x_list, dim=1) if x_list else None

        # --- Process Cond (Control) ---
        c_list = []
        if self.cond_num_count > 0:
            c_list.append(self.num_cond_encoder(cond[:, :self.cond_num_count]))
        if self.cond_cat_sizes:
            c_indices = self._recover_indices(cond[:, self.cond_num_count:], self.cond_cat_sizes)
            c_list.append(self.cat_cond_encoder(c_indices))
        cond_tokens = torch.cat(c_list, dim=1) if c_list else None

        return x_tokens, cond_tokens

    def _recover_indices(self, flat_one_hot, sizes):
        if not sizes:
            return None
        indices_list = []
        curr = 0
        for k in sizes:
            chunk = flat_one_hot[:, curr: curr + k]
            indices_list.append(chunk.argmax(dim=1).unsqueeze(1))
            curr += k
        return torch.cat(indices_list, dim=1)

"""

Adapted from https://github.com/ermongroup/CSDI/blob/main/diff_models.py

"""
class GatedResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_emb_dim, nhead, dropout, is_bayesian=False):
        super().__init__()
        self.channels = channels
        self.diffusion_projection = nn.Linear(diffusion_emb_dim, channels)
        if is_bayesian:
            self.feature_layer = BayesianTransformerBlock(
                d_model=channels, nhead=nhead, d_ff=channels * 2, dropout=dropout
            )
        else:
            self.feature_layer = nn.TransformerEncoderLayer(
                d_model=channels, nhead=nhead, dim_feedforward=channels * 2,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True
            )
        self.mid_projection = nn.Linear(channels, 2 * channels)
        self.cond_projection = nn.Linear(channels, 2 * channels)
        self.output_projection = nn.Linear(channels, 2 * channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, cond_emb, diffusion_emb):
        B, K, C = x.shape
        residual_base = x
        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(1)
        y = x + diff_proj
        y = self.feature_layer(y)
        y = self.mid_projection(self.dropout1(y))
        cond_proj = self.cond_projection(cond_emb)
        y = y + cond_proj
        gate, filter = torch.chunk(y, 2, dim=-1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(self.dropout2(y))
        residual, skip = torch.chunk(y, 2, dim=-1)
        return (residual_base + residual) / math.sqrt(2.0), skip


class AttnBackbone(nn.Module):
    def __init__(self, config, device, dim_info):
        super().__init__()
        self.device = device
        self.config = config
        self.task = config["diffusion"]["task"]
        self.d_main = config["model"]["channels"]
        self.d_emb = config["diffusion"]["diffusion_embedding_dim"]
        self.num_layers = config["model"]["layers"]
        self.nhead = config["model"]["nheads"]
        self.dropout_rate = config["model"]["dropout"]

        self.out_num_dim = dim_info['out_num_dim']
        self.out_cat_dim = dim_info['out_cat_dim']
        self.dropout_input = nn.Dropout(self.dropout_rate / 2)
        self.tokenizer = FeatureTokenizer(config, dim_info)
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_emb, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )

        self.layers = nn.ModuleList([
            GatedResidualBlock(
                channels=self.d_main,
                diffusion_emb_dim=self.d_main,
                nhead=self.nhead,
                dropout=self.dropout_rate,
                is_bayesian=self.config["sample"].get('mi_approx') == "DROPOUT"
            )
            for _ in range(self.num_layers)
        ])
        self.dropout_last = nn.Dropout(self.dropout_rate)
        if self.out_cat_dim > 0:
            self.head_cat = nn.Linear(self.d_main, self.out_cat_dim)

        if self.out_num_dim > 0:
            if self.task == "Res-N":
                self.head_num_res = nn.Linear(self.d_main, self.out_num_dim)
                self.head_num_eps = nn.Linear(self.d_main, self.out_num_dim)
            else:
                self.head_num = nn.Linear(self.d_main, self.out_num_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_main))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_curr, cond, t):
        t_emb_raw = timestep_embedding(t, self.d_emb).to(self.device)
        t_emb = self.time_embed(t_emb_raw)  # [B, d_main]
        x_tokens, cond_tokens = self.tokenizer(x_curr, cond)
        cond_context = cond_tokens.mean(dim=1, keepdim=True)
        cls_tokens = self.cls_token.expand(x_tokens.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x_tokens], dim=1)
        if self.config["sample"].get('mi_approx') == "DROPOUT":
            x = self.dropout_input(x)
            cond_context = self.dropout_input(cond_context)
        skips = []
        for layer in self.layers:
            x, skip_connection = layer(x, cond_context, t_emb)
            skips.append(skip_connection)

        x_final = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.layers))
        x_latent = self.dropout_last(x_final[:, 0])
        out_cat = self.head_cat(x_latent) if self.out_cat_dim > 0 else None
        out_num = None
        if self.out_num_dim > 0:
            if self.task == "Res-N":
                out_num = torch.cat([self.head_num_res(x_latent), self.head_num_eps(x_latent)], dim=1)
            else:
                out_num = self.head_num(x_latent)

        return out_num, out_cat

"""

Adapted from https://github.com/Yura52/rtdl

"""

class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResNetBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout, d_cond, d_emb):
        super().__init__()
        self.normalization = nn.BatchNorm1d(d_main)
        self.linear1 = nn.Linear(d_main, d_hidden)
        self.activation1 = ReGLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden // 2, d_main)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_proj = nn.Linear(d_emb, d_main)
        self.cond_proj = nn.Linear(d_cond, d_hidden // 2)

    def forward(self, x, cond, emb):
        identity = x
        x_in = self.normalization(x)
        x_in = x_in + self.emb_proj(emb)
        x_in = self.dropout1(self.activation1(self.linear1(x_in)))
        x_in = x_in + self.cond_proj(cond)
        x_in = self.dropout2(self.linear2(x_in))
        return x_in + identity

class MlpBackbone(nn.Module):
    def __init__(self, config, device, dim_info):
        super().__init__()
        self.device = device
        self.config = config
        self.task = config["diffusion"]["task"]
        self.d_main = config["model"]["channels"]
        self.d_emb = config["diffusion"].get("diffusion_embedding_dim", 128)

        self.input_dim = dim_info['input_dim']
        self.cond_dim = dim_info['cond_dim']
        self.out_num_dim = dim_info['out_num_dim']
        self.out_cat_dim = dim_info['out_cat_dim']

        self.dropout_input = nn.Dropout(config["model"]["dropout"] / 2)
        self.dropout_cond = nn.Dropout(config["model"]["dropout"])
        self.time_embed = nn.Sequential(nn.Linear(self.d_emb, self.d_emb),
                                        nn.SiLU(),
                                        nn.Linear(self.d_emb, self.d_emb))
        self.first_layer = nn.Linear(self.input_dim, self.d_main)

        self.blocks = nn.ModuleList([
            ResNetBlock(self.d_main, self.d_main * 2,
                        config["model"]["dropout"], self.cond_dim, self.d_emb)
            for _ in range(config["model"]["layers"])
        ])
        self.final_norm = nn.BatchNorm1d(self.d_main)
        self.dropout_last = nn.Dropout(config["model"]["dropout"])
        if self.out_cat_dim > 0:
            self.head_cat = nn.Linear(self.d_main, self.out_cat_dim)

        if self.out_num_dim > 0:
            if self.task == "Res-N":
                self.head_num_res = nn.Linear(self.d_main, self.out_num_dim)
                self.head_num_eps = nn.Linear(self.d_main, self.out_num_dim)
            else:
                self.head_num = nn.Linear(self.d_main, self.out_num_dim)

    def forward(self, x_curr, cond, t):
        emb = self.time_embed(timestep_embedding(t, self.d_emb))
        if self.config["sample"]['mi_approx'] == "DROPOUT":
            x_curr = self.dropout_input(x_curr)

        x = self.first_layer(x_curr)
        if self.config["sample"]['mi_approx'] == "DROPOUT":
            x = self.dropout_input(x)
        for block in self.blocks:
            cond_in = self.dropout_cond(cond)
            x = block(x, cond_in, emb)
        x = self.final_norm(x)

        out_cat = self.head_cat(x) if self.out_cat_dim > 0 else None
        out_num = None
        if self.out_num_dim > 0:
            if self.task == "Res-N":
                out_num = torch.cat([self.head_num_res(x), self.head_num_eps(x)], dim=1)
            else:
                out_num = self.head_num(x)
        return out_num, out_cat


class SIRD_NET(nn.Module):
    def __init__(self, config, device, dim_info):
        super().__init__()
        if config['model']['net'] == "ResNet":
            self.model = MlpBackbone(config, device, dim_info)
        else:
            self.model = AttnBackbone(config, device, dim_info)

    def forward(self, x_curr, cond, t):
        return self.model(x_curr, cond, t)
