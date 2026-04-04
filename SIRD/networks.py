import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class CFGConditionMask(nn.Module):
    def __init__(self, d_cond, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.null_condition = nn.Parameter(torch.zeros(d_cond))

    def forward(self, cond, force_drop=False):
        if self.drop_prob == 0 and not force_drop:
            return cond
        if force_drop:
            return self.null_condition.unsqueeze(0).expand(cond.size(0), -1)
        mask = torch.rand(cond.size(0), device=cond.device) < self.drop_prob
        mask = mask.view(-1, 1).expand_as(cond)
        cond_out = torch.where(mask, self.null_condition.unsqueeze(0).expand_as(cond), cond)
        return cond_out

class DenseMLPBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_main, d_hidden)
        self.norm1 = nn.BatchNorm1d(d_hidden)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.norm2 = nn.BatchNorm1d(d_hidden)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        input = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        return torch.cat([input, x], dim=1)

class ResNetBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout):
        super().__init__()
        self.normalization = nn.BatchNorm1d(d_main)
        self.linear1 = nn.Linear(d_main, d_hidden)
        self.activation1 = ReGLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden // 2, d_main)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, t_emb, c_emb):
        x_in = self.normalization(x)
        x_in += t_emb + c_emb
        x_in = self.dropout1(self.activation1(self.linear1(x_in)))
        x_in = self.dropout2(self.linear2(x_in))
        return x + x_in

class MainModel(nn.Module):
    def __init__(self, config, device, dim_info):
        super().__init__()
        self.device = device
        self.config = config

        diffusion_cfg = config.get("diffusion", {})
        model_cfg = config.get("model", {})

        self.task = diffusion_cfg.get("task", "Res-N")
        self.d_main = model_cfg.get("channels", 256)
        self.d_emb = diffusion_cfg.get("diffusion_embedding_dim", 128)

        self.input_dim = dim_info['input_dim']
        self.cond_dim = dim_info['cond_dim']
        self.out_num_dim = dim_info['out_num_dim']
        self.out_cat_dim = dim_info['out_cat_dim']
        self.has_cond = self.cond_dim > 0

        self.net = model_cfg.get("net", "ResNet")
        dropout_rate = model_cfg.get("dropout", 0.5)
        num_layers = model_cfg.get("layers", 5)

        self.first_layer = nn.Linear(self.input_dim, self.d_main)
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_emb, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        if self.has_cond:
            self.dropout_cond = nn.Dropout(dropout_rate)
            self.cond_embed = nn.Sequential(
                nn.Linear(self.cond_dim, self.d_main),
                nn.SiLU()
            )
        self.blocks = nn.ModuleList()
        current_dim = self.d_main

        for _ in range(num_layers):
            if self.net == "ResNet":
                self.blocks.append(ResNetBlock(self.d_main, self.d_main * 2, dropout_rate))
            else:
                self.blocks.append(DenseMLPBlock(current_dim, self.d_main, dropout_rate))
                current_dim += self.d_main

        self.head = nn.Sequential(
            nn.BatchNorm1d(current_dim),
            nn.ReLU()
        )

        if self.out_cat_dim > 0:
            self.head_cat = nn.Linear(current_dim, self.out_cat_dim)
        if self.out_num_dim > 0:
            if self.task == "Res-N":
                self.head_num_res = nn.Linear(current_dim, self.out_num_dim)
                self.head_num_eps = nn.Linear(current_dim, self.out_num_dim)
            else:
                self.head_num = nn.Linear(current_dim, self.out_num_dim)

    def forward(self, x_curr, cond, t):
        t_emb = self.time_embed(timestep_embedding(t, self.d_emb))
        if self.has_cond:
            c_emb = self.dropout_cond(self.cond_embed(self.dropout_cond(cond)))
        else:
            c_emb = 0
        x = self.first_layer(x_curr)

        if self.net == "ResNet":
            x_in = x #  + t_emb + c_emb
            for block in self.blocks:
                x_in = block(x_in, t_emb, c_emb)
        else:
            x_in = x + t_emb + c_emb
            for block in self.blocks:
                x_in = block(x_in)

        x = self.head(x_in)
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
        self.cond_drop_prob = config.get("train", {}).get("cond_drop_prob", 0.0)
        self.cfg_scale_num = config.get("sample", {}).get("cfg_scale_num", 1.0)
        self.cfg_scale_cat = config.get("sample", {}).get("cfg_scale_cat", 1.0)

        self.cfg_mask = CFGConditionMask(dim_info['cond_dim'], self.cond_drop_prob)

        self.model = MainModel(config, device, dim_info)

    def forward(self, x_curr, cond, t):
        if self.training:
            cond = self.cfg_mask(cond)
            return self.model(x_curr, cond, t)
        else:
            use_cfg = (self.cond_drop_prob > 0.0) and (self.cfg_scale_num != 1.0 or self.cfg_scale_cat != 1.0)
            if not use_cfg:
                cond = self.cfg_mask(cond)
                return self.model(x_curr, cond, t)

            x_double = torch.cat([x_curr, x_curr], dim=0)
            t_double = torch.cat([t, t], dim=0)
            cond_cond = cond
            cond_null = self.cfg_mask(cond, force_drop=True)
            cond_double = torch.cat([cond_cond, cond_null], dim=0)
            out_num_double, out_cat_double = self.model(x_double, cond_double, t_double)
            out_num = None
            if out_num_double is not None:
                out_num_cond, out_num_uncond = out_num_double.chunk(2, dim=0)
                out_num = out_num_uncond + self.cfg_scale_num * (out_num_cond - out_num_uncond)
            out_cat = None
            if out_cat_double is not None:
                out_cat_cond, out_cat_uncond = out_cat_double.chunk(2, dim=0)
                out_cat = out_cat_uncond + self.cfg_scale_cat * (out_cat_cond - out_cat_uncond)

            return out_num, out_cat