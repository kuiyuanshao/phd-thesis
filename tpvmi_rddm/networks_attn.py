# networks_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
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
    def __init__(self, schema, channels, dropout_rate=0, is_target=False, expansion_factor=1):
        super().__init__()
        self.schema = schema
        self.channels = channels
        self.is_target = is_target
        self.var_names = [v['name'] for v in schema]
        self.projectors = nn.ModuleDict()
        for var in schema:
            name = var['name']
            dim = var['num_classes'] if 'categorical' in var['type'] else 1
            input_dim = dim * 2 if is_target else dim
            self.projectors[name] = nn.Sequential(
                nn.Linear(input_dim, channels),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(channels, channels)
            )
        self.feature_embeddings = nn.Parameter(torch.randn(len(schema), channels))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, primary_dict, secondary_dict=None):
        embeddings = []
        for i, name in enumerate(self.var_names):
            val = primary_dict[name]
            if self.is_target and secondary_dict is not None:
                val = torch.cat([val, secondary_dict[name]], dim=-1)

            token = self.projectors[name](self.dropout(val))
            token = token + self.feature_embeddings[i].unsqueeze(0)
            embeddings.append(token)
        # Shape: (B, K, C)
        sequence = torch.stack(embeddings, dim=1)
        # Permute to (Batch, Channels, Num_Vars) / (B, C, K)
        return sequence.permute(0, 2, 1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, dropout_rate=0):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # 1. Self-Attention (Target-Target)
        # Expects input (Sequence, Batch, Channels). Here Sequence = Num_Features.
        self.self_attn = get_torch_trans(heads=nheads, layers=1, channels=channels)
        # 2. Cross-Attention (Target-Aux)
        self.cross_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=nheads, dropout=dropout_rate)
        self.norm_cross = nn.LayerNorm(channels)
        # 3. Feed Forward components
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, aux, diffusion_emb):
        # x shape: (Batch, Channels, K)
        x_in = x
        # time_emb: (Batch, Channels, 1) to broadcast over K
        time_emb = self.diffusion_projection(self.dropout(diffusion_emb)).unsqueeze(-1)
        y = x + time_emb
        # ==========================================
        # 1. Feature Attention (Self-Attention)
        # ==========================================
        # Permute to (K, Batch, Channels) for Transformer
        y_seq = y.permute(2, 0, 1)
        y_seq = self.self_attn(self.dropout(y_seq))
        # ==========================================
        # 2. Auxiliary Cross-Attention
        # ==========================================
        if aux is not None:
            # Permute to (M, Batch, Channels) for Attention Key/Value
            context = aux.permute(2, 0, 1)
            # Output: (K, B, C) - Attention weights are (B, K, M) implicitly
            attn_out, _ = self.cross_attn(query=y_seq, key=context, value=context)
            y_seq = self.norm_cross(y_seq + self.dropout(attn_out))
        # ==========================================
        # 3. Feed Forward (Conv1d)
        # ==========================================
        # Permute back to (Batch, Channels, K) for Conv1d
        y = y_seq.permute(1, 2, 0)
        y = self.mid_projection(self.dropout(y))
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(self.dropout(y))
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x_in + residual) / math.sqrt(2.0), skip


class RDDM_NET(nn.Module):
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
        # 1. Target Embedder
        self.target_embedder = FeatureEmbedder(
            self.target_schema,
            self.channels,
            self.dropout_rate,
            is_target=True,
            expansion_factor = 2
        )
        # 2. Aux Embedder
        self.aux_embedder = FeatureEmbedder(
            self.aux_schema,
            self.channels,
            self.dropout_rate,
            is_target=False
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.diffusion_embedding = DiffusionEmbedding(self.num_steps, self.channels, self.dropout_rate)
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
                        'res': nn.Sequential(
                            nn.Linear(self.channels, self.channels),
                            nn.SiLU(),
                            nn.Dropout(self.dropout_rate),
                            nn.Linear(self.channels, 1)
                        ),
                        'eps': nn.Sequential(
                            nn.Linear(self.channels, self.channels),
                            nn.SiLU(),
                            nn.Dropout(self.dropout_rate),
                            nn.Linear(self.channels, 1)
                        )
                    })
                else:
                    self.output_heads[name] = nn.Sequential(
                            nn.Linear(self.channels, self.channels),
                            nn.SiLU(),
                            nn.Dropout(self.dropout_rate),
                            nn.Linear(self.channels, 1))


    def forward(self, x_t_dict, t, p1_dict, aux_dict):
        x_t_emb = self.target_embedder(x_t_dict, p1_dict)
        dif_emb = self.diffusion_embedding(t)
        aux_emb = None
        if len(self.aux_schema) > 0:
            aux_emb = self.aux_embedder(aux_dict)
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