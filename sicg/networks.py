import torch
import torch.nn as nn
import torch.nn.functional as F

class ReGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.relu(x2)


class ResNetBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_p):
        super(ResNetBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.reglu = ReGLU()
        self.dropout1 = nn.Dropout(dropout_p)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_p)

        if in_dim != hidden_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_p)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(self.bn(x))
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.reglu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        return out + residual


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
                    nn.Linear(current_input_dim, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    ReGLU(),
                    nn.Dropout(dropout_p)
                )
            )
            current_input_dim += hidden_dim

        self.output_layer = nn.Linear(current_input_dim, p2_dim)

    def forward(self, z, p1, cond):
        x = torch.cat([z, p1, cond], dim=1)
        if self.config['sample']['mi_approx'] == "DROPOUT":
            x = self.dropout(x)
        for block in self.blocks:
            out = block(x)
            x = torch.cat([out, x], dim=1)
        return self.output_layer(x)


class Discriminator(nn.Module):
    def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
        super(Discriminator, self).__init__()
        self.config = config
        self.input_dim = p1_dim + p2_dim + cond_cols_dim
        self.pack = config['model']['discriminator'].get('pack', 1)
        hidden_dim = config['model']['discriminator']['hidden_dim']
        num_layers = config['model']['discriminator']['layers']
        layers = []
        self.pack_dim = self.pack * self.input_dim

        current_dim = self.pack_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.5))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.seq_head = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.pack_dim)
        out = self.seq_head(input)
        return out

