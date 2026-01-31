import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, dropout_rate=0):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim // 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)
        self.input_dropout = nn.Dropout(dropout_rate / 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        return F.silu(self.projection2(self.dropout(F.silu(self.projection1(self.input_dropout(x))))))

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class Block(nn.Module):
    def __init__(self, net, input_dim, output_dim, dropout_rate):
        super().__init__()
        self.net = net
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        if self.net == "Dense":
            return torch.cat([self.block(x), x], dim = 1)
        else:
            return x + self.block(x)

class RDDM_NET(nn.Module):
    def __init__(self, config, device, variable_schema):
        super().__init__()
        self.device = device
        self.task = config["else"]["task"]
        self.num_steps = config["diffusion"]["num_steps"]
        self.dropout_rate = config["model"]["dropout"]
        self.hidden_dim = config["model"]["channels"] * 4
        self.layers = config["model"]["layers"]
        self.net = config["model"]["net"]
        self.target_schema = [v for v in variable_schema if 'aux' not in v['type']]
        self.aux_schema = [v for v in variable_schema if 'aux' in v['type']]
        self.num_vars = [v['name'] for v in self.target_schema if v['type'] == 'numeric']
        self.cat_vars = [v for v in self.target_schema if v['type'] == 'categorical']

        self.input_dim = 0
        self.out_dim_map = {}
        for var in self.target_schema:
            dim = var['num_classes'] if 'categorical' in var['type'] else 1
            self.input_dim += (dim * 2)
            self.out_dim_map[var['name']] = dim
        for var in self.aux_schema:
            dim = var['num_classes'] if 'categorical' in var['type'] else 1
            self.input_dim += dim
        self.time_emb_dim = config["diffusion"]["diffusion_embedding_dim"]
        self.diffusion_embedding = DiffusionEmbedding(self.num_steps, self.time_emb_dim, self.dropout_rate)

        self.project_in = nn.Linear(self.input_dim + self.time_emb_dim, self.hidden_dim)
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.blocks = nn.ModuleList()
        self.current_dim = self.hidden_dim
        for _ in range(self.layers):
            if self.net == "Dense":
                self.blocks.append(Block(
                    net=self.net,
                    input_dim=self.current_dim,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate
                ))
                self.current_dim += self.hidden_dim
            else:
                self.blocks.append(Block(
                    net=self.net,
                    input_dim=self.current_dim,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate
                ))
        self.final_norm = nn.BatchNorm1d(self.current_dim)
        self.final_activation = nn.ReLU()
        self.cat_heads = nn.ModuleDict()
        for var in self.target_schema:
            if var['type'] == 'categorical':
                name = var['name']
                out_d = self.out_dim_map[name]
                self.cat_heads[name] = nn.Linear(self.current_dim, out_d)
        n_num = len(self.num_vars)
        if n_num > 0:
            if self.task == "Res-N":
                self.numeric_res_head = nn.Linear(self.current_dim, n_num)
                self.numeric_eps_head = nn.Linear(self.current_dim, n_num)
            else:
                self.numeric_head = nn.Linear(self.current_dim, n_num)

    def forward(self, x_t_dict, t, p1_dict, aux_dict):
        flat_list = []
        for var in self.target_schema:
            flat_list.append(x_t_dict[var['name']])
            flat_list.append(p1_dict[var['name']])
        for var in self.aux_schema:
            flat_list.append(aux_dict[var['name']])
        t_emb = self.diffusion_embedding(t)  # (Batch, time_emb_dim)
        x = torch.cat(flat_list + [t_emb], dim=1)
        x = self.input_dropout(self.project_in(self.input_dropout(x)))
        for block in self.blocks:
            x = block(x)
        x = self.final_activation(self.final_norm(x))
        x = self.dropout(x)
        output_dict = {}
        for name in self.cat_heads:
            output_dict[name] = self.cat_heads[name](x)
        if len(self.num_vars) > 0:
            if self.task == "Res-N":
                all_res = self.numeric_res_head(x)
                all_eps = self.numeric_eps_head(x)
                for i, name in enumerate(self.num_vars):
                    r = all_res[:, i:i + 1]
                    e = all_eps[:, i:i + 1]
                    output_dict[name] = torch.cat([r, e], dim=1)
            else:
                all_pred = self.numeric_head(x)
                for i, name in enumerate(self.num_vars):
                    output_dict[name] = all_pred[:, i:i + 1]
        return output_dict