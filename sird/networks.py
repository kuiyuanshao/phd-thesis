import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_torch_trans(heads=8, layers=1, channels=64, dropout_rate=0):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", dropout=dropout_rate
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class AttnDiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, dropout_rate=0):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim // 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        return F.silu(self.projection2(self.dropout(F.silu(self.projection1(self.dropout(x))))))

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class FeatureEmbedder(nn.Module):
    def __init__(self, schema, channels, dropout_rate=0):
        super().__init__()
        self.schema = schema
        self.channels = channels
        self.var_names = [v['name'] for v in schema]
        self.projectors = nn.ModuleDict()

        for var in schema:
            name = var['name']
            dim = var['num_classes'] if 'categorical' in var['type'] else 1

            self.projectors[name] = nn.Sequential(
                nn.Linear(dim, channels),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(channels, channels)
            )

        self.feature_embeddings = nn.Parameter(torch.randn(len(schema), channels))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_dict):
        embeddings = []
        for i, name in enumerate(self.var_names):
            val = input_dict[name]
            token = self.projectors[name](self.dropout(val))
            token = token + self.feature_embeddings[i].unsqueeze(0)
            embeddings.append(token)
        sequence = torch.stack(embeddings, dim=1)
        return sequence.permute(0, 2, 1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, dropout_rate=0):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.self_attn = get_torch_trans(heads=nheads, layers=1, channels=channels, dropout_rate=dropout_rate)
        self.cross_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=nheads, dropout=dropout_rate)
        self.norm_cross = nn.LayerNorm(channels)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, aux, diffusion_emb):
        x_in = x
        time_emb = self.diffusion_projection(self.dropout(diffusion_emb)).unsqueeze(-1)
        y = x + time_emb

        y_seq = y.permute(2, 0, 1)
        y_seq = self.self_attn(self.dropout(y_seq))

        if aux is not None:
            context = aux.permute(2, 0, 1)
            attn_out, _ = self.cross_attn(query=y_seq, key=context, value=context)
            y_seq = self.norm_cross(y_seq + self.dropout(attn_out))

        y = y_seq.permute(1, 2, 0)
        y = self.mid_projection(self.dropout(y))
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(self.dropout(y))
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x_in + residual) / math.sqrt(2.0), skip


class AttnBackbone(nn.Module):
    def __init__(self, config, device, variable_schema):
        super().__init__()
        self.device = device
        self.channels = config["model"]["channels"]
        self.num_steps = config["diffusion"]["num_steps"]
        self.nheads = config["model"]["nheads"]
        self.layers = config["model"]["layers"]
        self.dropout_rate = config["model"]["dropout"]
        self.task = config["else"]["task"]

        self.target_schema = [v for v in variable_schema if 'aux' not in v['type']]
        self.aux_schema = [v for v in variable_schema if 'aux' in v['type']]

        self.target_embedder = FeatureEmbedder(self.target_schema, self.channels, self.dropout_rate)
        self.combined_context_schema = self.aux_schema + self.target_schema
        self.aux_embedder = FeatureEmbedder(self.combined_context_schema, self.channels, self.dropout_rate)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.diffusion_embedding = AttnDiffusionEmbedding(self.num_steps, self.channels, self.dropout_rate)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.channels, self.channels, self.nheads, self.dropout_rate) for _ in range(self.layers)
        ])

        self.output_heads = nn.ModuleDict()
        for var in self.target_schema:
            name = var['name']
            if var['type'] == 'categorical':
                self.output_heads[name] = nn.Linear(self.channels, var['num_classes'])
            else:
                if self.task == "Res-N":
                    self.output_heads[name] = nn.ModuleDict({
                        'res': nn.Sequential(nn.Linear(self.channels, self.channels), nn.SiLU(),
                                             nn.Dropout(self.dropout_rate), nn.Linear(self.channels, 1)),
                        'eps': nn.Sequential(nn.Linear(self.channels, self.channels), nn.SiLU(),
                                             nn.Dropout(self.dropout_rate), nn.Linear(self.channels, 1))
                    })
                else:
                    self.output_heads[name] = nn.Sequential(
                        nn.Linear(self.channels, self.channels), nn.SiLU(), nn.Dropout(self.dropout_rate),
                        nn.Linear(self.channels, 1))

    def forward(self, x_t_dict, t, p1_dict, aux_dict):
        x_t_emb = self.target_embedder(x_t_dict)
        context_dict = {**aux_dict, **p1_dict}
        aux_emb = self.aux_embedder(context_dict) if len(self.combined_context_schema) > 0 else None
        dif_emb = self.diffusion_embedding(t)

        sequence = x_t_emb
        skip_accum = 0
        for block in self.res_blocks:
            sequence, skip = block(sequence, aux_emb, dif_emb)
            skip_accum += skip

        target_features = (skip_accum / math.sqrt(self.layers)).permute(0, 2, 1)
        output_dict = {}
        for i, var in enumerate(self.target_schema):
            name = var['name']
            feature = target_features[:, i, :]
            if var['type'] == 'categorical':
                output_dict[name] = self.output_heads[name](self.dropout(feature))
            else:
                if self.task == "Res-N":
                    res_pred = self.output_heads[name]['res'](self.dropout(feature))
                    eps_pred = self.output_heads[name]['eps'](self.dropout(feature))
                    output_dict[name] = torch.cat([res_pred, eps_pred], dim=1)
                else:
                    output_dict[name] = self.output_heads[name](self.dropout(feature))
        return output_dict


class MlpDiffusionEmbedding(nn.Module):
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


class ResBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)


class MlpBackbone(nn.Module):
    def __init__(self, config, device, variable_schema):
        super().__init__()
        self.device = device
        self.task = config["else"]["task"]
        self.num_steps = config["diffusion"]["num_steps"]
        self.dropout_rate = config["model"]["dropout"]
        self.hidden_dim = config["model"]["channels"]
        self.layers = config["model"]["layers"]
        self.target_schema = [v for v in variable_schema if 'aux' not in v['type']]
        self.aux_schema = [v for v in variable_schema if 'aux' in v['type']]
        self.num_vars = [v['name'] for v in self.target_schema if v['type'] == 'numeric']

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
        self.diffusion_embedding = MlpDiffusionEmbedding(self.num_steps, self.time_emb_dim, self.dropout_rate)

        self.project_in = nn.Linear(self.input_dim + self.time_emb_dim, self.hidden_dim)
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.blocks = nn.ModuleList([ResBlock(self.hidden_dim, self.dropout_rate) for _ in range(self.layers)])

        self.final_norm = nn.BatchNorm1d(self.hidden_dim)
        self.final_activation = nn.ReLU()

        self.cat_heads = nn.ModuleDict()
        for var in self.target_schema:
            if var['type'] == 'categorical':
                name = var['name']
                self.cat_heads[name] = nn.Linear(self.hidden_dim, self.out_dim_map[name])

        n_num = len(self.num_vars)
        if n_num > 0:
            if self.task == "Res-N":
                self.numeric_res_head = nn.Linear(self.hidden_dim, n_num)
                self.numeric_eps_head = nn.Linear(self.hidden_dim, n_num)
            else:
                self.numeric_head = nn.Linear(self.hidden_dim, n_num)

    def forward(self, x_t_dict, t, p1_dict, aux_dict):
        flat_list = []
        for var in self.target_schema:
            flat_list.append(x_t_dict[var['name']])
            flat_list.append(p1_dict[var['name']])
        for var in self.aux_schema:
            flat_list.append(aux_dict[var['name']])

        t_emb = self.diffusion_embedding(t)
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
                all_res, all_eps = self.numeric_res_head(x), self.numeric_eps_head(x)
                for i, name in enumerate(self.num_vars):
                    output_dict[name] = torch.cat([all_res[:, i:i + 1], all_eps[:, i:i + 1]], dim=1)
            else:
                all_pred = self.numeric_head(x)
                for i, name in enumerate(self.num_vars):
                    output_dict[name] = all_pred[:, i:i + 1]
        return output_dict


class SIRD_NET(nn.Module):
    def __init__(self, config, device, variable_schema):
        super().__init__()
        self.net_type = config["model"]["net"]

        if self.net_type == "AttnNet":
            print("[SIRD_NET] Initializing Attention-based Backbone")
            self.model = AttnBackbone(config, device, variable_schema)
        elif self.net_type == "ResNet":
            print("[SIRD_NET] Initializing ResNet Backbone")
            self.model = MlpBackbone(config, device, variable_schema)
        else:
            raise ValueError(f"Unknown architecture: {self.net_type}. Options: 'AttnNet', 'ResNet'")

    def forward(self, x_t_dict, t, p1_dict, aux_dict):
        return self.model(x_t_dict, t, p1_dict, aux_dict)