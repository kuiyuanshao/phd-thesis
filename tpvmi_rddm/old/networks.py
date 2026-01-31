# networks.py: Hybrid D3PM/RDDM Backbone
# Implements Schema-Aware Embeddings and Multi-Head Outputs.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim // 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        return self.projection2(F.silu(self.projection1(x)))

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # Side info projection (P1 + Aux) is handled in the main embedder now,
        # but we keep this for global time/feature mixing if needed.
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward(self, x, side_info, diffusion_emb):
        B, C, L = x.shape
        base_shape = x.shape
        x_in = x

        # 1. Inject Time Embedding
        time_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B, C, 1)
        y = x + time_emb

        # 2. Inject Side Info (Conditioning)
        # side_info is (B, Side_Dim, L)
        y = y + self.cond_projection(side_info)[:, :C, :]  # Simple add for now

        # 3. Bi-Directional Transformer Processing
        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Gate Mechanism
        y = self.mid_projection(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        # 5. Output Project & Residual
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        return (x_in + residual) / math.sqrt(2.0), skip


class HybridFeatureEmbedder(nn.Module):
    """
    Encodes heterogeneous inputs (Mixed Continuous/Discrete) into a unified channel space.
    Used for x_t, P1, and Aux.
    """

    def __init__(self, schema, channels):
        super().__init__()
        self.schema = schema
        self.channels = channels

        # We hold specific embedding layers for categorical variables
        # and simple linear projections for numeric variables.
        self.cat_embeddings = nn.ModuleDict()
        self.num_projections = nn.ModuleDict()

        for var in schema:
            name = var['name']
            if 'categorical' in var['type']:
                # D3PM: Discrete Embedding
                self.cat_embeddings[name] = nn.Embedding(
                    num_embeddings=var['num_classes'],
                    embedding_dim=channels
                )
            else:
                # RDDM: Numeric Projection
                self.num_projections[name] = nn.Linear(1, channels)

    def forward(self, x_tensor):
        """
        x_tensor: (Batch, Num_Vars_In_Schema)
        Output: (Batch, Channels, Num_Vars_In_Schema)
        """
        B, N = x_tensor.shape
        embeddings_list = []

        # Iterate over columns corresponding to the schema order
        # We assume x_tensor columns align perfectly with schema list order
        for i, var in enumerate(self.schema):
            name = var['name']
            col_data = x_tensor[:, i]  # (B,)

            if 'categorical' in var['type']:
                # Input must be Long Indices
                indices = col_data.long()
                emb = self.cat_embeddings[name](indices)  # (B, Channels)
            else:
                # Input is Float
                val = col_data.unsqueeze(1)  # (B, 1)
                emb = self.num_projections[name](val)  # (B, Channels)

            embeddings_list.append(emb)

        # Stack to create sequence: (B, N, Channels) -> Permute to (B, Channels, N)
        return torch.stack(embeddings_list, dim=1).permute(0, 2, 1)


class CSDI_Hybrid(nn.Module):
    def __init__(self, config, device, variable_schema, aux_dim=0):
        super().__init__()
        self.device = device
        self.channels = config["model"]["channels"]
        self.num_steps = config["diffusion"]["num_steps"]
        self.nheads = config["model"]["nheads"]
        self.layers = config["model"]["layers"]

        # --- 1. Schema Splitting ---
        # We split the unified schema into Target (P2) and Aux parts
        # to build separate embedders.
        self.target_schema = [v for v in variable_schema if 'aux' not in v['type']]
        self.aux_schema = [v for v in variable_schema if 'aux' in v['type']]

        # --- 2. Input Embedders ---
        # Embedder for x_t (Noisy State) and P1 (Proxy State) - They share the same schema/domain
        self.target_embedder = HybridFeatureEmbedder(self.target_schema, self.channels)

        # Embedder for Aux conditioning
        self.aux_embedder = HybridFeatureEmbedder(self.aux_schema, self.channels)

        # Projection to mix x_t and P1/Conditioning
        # Input to backbone will be projection(cat(x_t_emb, p1_emb))
        self.input_mixer = Conv1d_with_init(2 * self.channels, self.channels, 1)

        # --- 3. Backbone ---
        self.diffusion_embedding = DiffusionEmbedding(self.num_steps, self.channels)

        self.res_blocks = nn.ModuleList()
        for i in range(self.layers):
            self.res_blocks.append(
                ResidualBlock(
                    side_dim=self.channels,  # Aux is appended as tokens, so side_dim is just channel dim
                    channels=self.channels,
                    diffusion_embedding_dim=self.channels,
                    nheads=self.nheads
                )
            )

        # --- 4. Multi-Head Outputs ---
        # We need specific heads for each variable type
        self.output_heads = nn.ModuleDict()

        for var in self.target_schema:
            name = var['name']
            if var['type'] == 'categorical':
                # D3PM Head: Output Logits (Channels -> Num_Classes)
                self.output_heads[name] = nn.Sequential(
                    nn.Linear(self.channels, self.channels),
                    nn.GELU(),
                    nn.Linear(self.channels, var['num_classes'])
                )
            else:
                # RDDM Head: Output Scalar (Channels -> 1)
                self.output_heads[name] = nn.Sequential(
                    nn.Linear(self.channels, self.channels),
                    nn.GELU(),
                    nn.Linear(self.channels, 1)
                )

    def forward(self, x_t, t, p1, aux):
        """
        x_t: (B, N_Target) - Mixed Floats/Indices
        t: (B,) - Timesteps
        p1: (B, N_Target) - Proxy Values (Condition)
        aux: (B, N_Aux) - Auxiliary Values (Condition)
        """

        # 1. Embed Inputs
        # x_t_emb: (B, C, N)
        x_t_emb = self.target_embedder(x_t)
        # p1_emb: (B, C, N)
        p1_emb = self.target_embedder(p1)
        # aux_emb: (B, C, N_Aux)
        aux_emb = self.aux_embedder(aux)

        # 2. Time Embedding
        dif_emb = self.diffusion_embedding(t)  # (B, C)

        # 3. Construct Input Sequence
        # We combine Noisy State (x_t) and Proxy (p1) via concatenation -> mixing
        # This tells the model: "Here is the noisy version, here is the proxy version."
        combined_input = torch.cat([x_t_emb, p1_emb], dim=1)  # (B, 2C, N)
        mixed_input = self.input_mixer(combined_input)  # (B, C, N)

        # 4. Append Aux as Extra Context Tokens
        # Full Sequence: [Targets, Aux] -> Length = N + N_Aux
        # This allows attention to attend to Aux variables for every Target variable
        sequence = torch.cat([mixed_input, aux_emb], dim=2)  # (B, C, N + N_Aux)

        # 5. Residual Backbone
        skip_accum = 0
        for block in self.res_blocks:
            # We pass 'sequence' as both x and side_info (self-conditioning structure)
            # or just rely on the attention mechanism to mix them.
            # In this implementation, we pass sequence as x.
            # The 'side_info' arg in ResidualBlock is used for global conditioning if needed.
            # Here, we treat 'sequence' as the main data.
            # We can pass dummy side info or use the time embedding twice.

            # To strictly follow CSDI logic, we'll just pass sequence through.
            # Note: ResidualBlock expects (x, side_info, diff_emb).
            # We will use the same sequence for side_info to allow self-attention mixing.

            sequence, skip = block(sequence, sequence, dif_emb)
            skip_accum += skip

        skip_accum = skip_accum / math.sqrt(self.layers)

        # 6. Multi-Head Output Decoding
        # We only care about the first N tokens (The Targets), ignore Aux tokens now.
        target_features = skip_accum[:, :, :x_t.shape[1]]  # (B, C, N)
        target_features = target_features.permute(0, 2, 1)  # (B, N, C) for Linear Layers

        output_dict = {}

        # Iterate over target schema to apply correct head
        for i, var in enumerate(self.target_schema):
            name = var['name']
            feature_vector = target_features[:, i, :]  # (B, C)

            # Apply specific head
            prediction = self.output_heads[name](feature_vector)

            # Squeeze numeric outputs to (B, 1) -> (B) if needed, but usually (B, 1) is fine
            # We keep standard shape (B, OutDim)
            output_dict[name] = prediction

        return output_dict