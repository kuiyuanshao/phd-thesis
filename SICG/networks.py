import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
        super(Generator, self).__init__()
        self.config = config
        self.noise_dim = config['model']['generator']['noise_dim']
        self.backbone_input_dim = self.noise_dim + p1_dim + cond_cols_dim

        hidden_dim = config['model']['generator']['hidden_dim']
        num_layers = config['model']['generator']['layers']
        dropout_p = config['model']['generator']['dropout']

        self.dropout = nn.Dropout(dropout_p / 2)
        self.blocks = nn.ModuleList()

        current_input_dim = self.backbone_input_dim
        for _ in range(num_layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(current_input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ELU(),
                    nn.Dropout(dropout_p)
                )
            )
            current_input_dim += hidden_dim
        self.output_layer = nn.Linear(current_input_dim, p2_dim)

    def forward(self, z, p1, cond):
        conds = torch.cat([p1, cond], dim=1)
        if self.config.get('sample', {}).get('mi_approx') == "DROPOUT":
            conds = self.dropout(conds)
        x = torch.cat([z, conds], dim=1)
        for block in self.blocks:
            out = block(x)
            x = torch.cat([x, out], dim=1)
        return self.output_layer(x)

class Discriminator(nn.Module):
    def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
        super(Discriminator, self).__init__()
        self.config = config
        self.input_dim = p1_dim + p2_dim + cond_cols_dim
        self.pack = config['model']['discriminator']['pack']
        hidden_dim = config['model']['discriminator']['hidden_dim']
        num_layers = config['model']['discriminator']['layers']
        layers = []
        self.pack_dim = self.pack * self.input_dim
        current_dim = self.pack_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.config['train']['loss']['loss_info'] > 0:
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.LeakyReLU(0.2))
            if self.pack > 1:
                layers.append(nn.Dropout(0.5))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        if self.config['train']['loss']['loss_info'] > 0:
            layers.append(nn.ReLU(True))
            self.info_head = nn.Sequential(*layers[:len(layers) - 2])
        self.seq_head = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.pack_dim)
        out = self.seq_head(input)
        if self.config['train']['loss']['loss_info'] > 0:
            info = self.info_head(input)
            return out, info
        return out, 0


# class ReGLU(nn.Module):
#     def forward(self, x):
#         a, b = x.chunk(2, dim=-1)
#         return F.relu(a) * b
#
# class G_ResNetBlock(nn.Module):
#     def __init__(self, hidden_dim, dropout_p):
#         super(G_ResNetBlock, self).__init__()
#         self.bn = nn.BatchNorm1d(hidden_dim)
#         self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
#         self.reglu = ReGLU()
#         self.dropout1 = nn.Dropout(dropout_p)
#
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.dropout2 = nn.Dropout(dropout_p)
#
#     def forward(self, x):
#         out = self.bn(x)
#         out = self.linear1(out)
#         out = self.reglu(out)
#         out = self.dropout1(out)
#         out = self.linear2(out)
#         out = self.dropout2(out)
#         return out + x
#
#
# class D_ResNetBlock(nn.Module):
#     def __init__(self, hidden_dim, dropout_p):
#         super(D_ResNetBlock, self).__init__()
#         self.ln = nn.LayerNorm(hidden_dim)
#         self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
#         self.reglu = ReGLU()
#         self.dropout1 = nn.Dropout(dropout_p)
#
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.dropout2 = nn.Dropout(dropout_p)
#
#     def forward(self, x):
#         out = self.ln(x)
#         out = self.linear1(out)
#         out = self.reglu(out)
#         out = self.dropout1(out)
#         out = self.linear2(out)
#         out = self.dropout2(out)
#         return out + x

# class Generator(nn.Module):
#     def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
#         super(Generator, self).__init__()
#         self.config = config
#         self.noise_dim = config['model']['generator']['noise_dim']
#         self.backbone_input_dim = self.noise_dim * 2 #+ p1_dim + cond_cols_dim
#
#         hidden_dim = config['model']['generator']['hidden_dim']
#         num_layers = config['model']['generator']['layers']
#         dropout_p = config['model']['generator']['dropout']
#
#         self.dropout = nn.Dropout(dropout_p / 2)
#         self.cond_encoder = nn.Sequential(
#             nn.Linear(p1_dim + cond_cols_dim, self.noise_dim),
#             nn.Dropout(dropout_p / 2)
#         )
#         self.input_projection = nn.Sequential(
#             nn.Linear(self.backbone_input_dim, hidden_dim),
#             nn.Dropout(dropout_p)
#         )
#         self.blocks = nn.ModuleList()
#         for _ in range(num_layers):
#             self.blocks.append(
#                 G_ResNetBlock(hidden_dim, dropout_p)
#             )
#         self.output_layer = nn.Linear(hidden_dim, p2_dim)
#
#     def forward(self, z, p1, cond):
#         conds = torch.cat([p1, cond], dim=1)
#         if self.config.get('sample', {}).get('mi_approx') == "DROPOUT":
#             conds = self.dropout(conds)
#         conds = self.cond_encoder(conds)
#         x = torch.cat([z, conds], dim=1)
#         x = self.input_projection(x)
#         for block in self.blocks:
#             out = block(x)
#         return self.output_layer(out)
#
# class Discriminator(nn.Module):
#     def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
#         super(Discriminator, self).__init__()
#         self.config = config
#         self.input_dim = p1_dim + p2_dim + cond_cols_dim
#         self.pack = config['model']['discriminator']['pack']
#         hidden_dim = config['model']['discriminator']['hidden_dim']
#         num_layers = config['model']['discriminator']['layers']
#         layers = []
#         self.pack_dim = self.pack * self.input_dim
#         current_dim = self.pack_dim
#
#         layers.append(nn.Linear(current_dim, hidden_dim))
#         for _ in range(num_layers):
#             layers.append(D_ResNetBlock(hidden_dim, dropout_p=0.5))
#
#         layers.append(nn.LayerNorm(hidden_dim))
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
#         layers.append(nn.Linear(hidden_dim, 1))
#
#         if self.config['train']['loss']['loss_info'] > 0:
#             layers.append(nn.ReLU(True))
#             self.info_head = nn.Sequential(*layers[:len(layers) - 4])
#
#         self.seq_head = nn.Sequential(*layers)
#
#     def forward(self, input):
#         input = input.view(-1, self.pack_dim)
#         out = self.seq_head(input)
#         if self.config['train']['loss']['loss_info'] > 0:
#             info = self.info_head(input)
#             return out, info
#         return out, 0

