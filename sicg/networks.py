import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
        super(Generator, self).__init__()
        self.config = config

        self.noise_dim = config['model']['generator']['noise_dim']
        input_dim = self.noise_dim + p1_dim + cond_cols_dim
        output_dim = p2_dim

        hidden_dim = config['model']['generator']['hidden_dim']
        num_layers = config['model']['generator']['layers']
        dropout_p = config['model']['generator']['dropout']

        self.dropout = nn.Dropout(dropout_p / 2)
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p)
                )
            )

        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, p1, cond):
        x = torch.cat([z, p1, cond], dim=1)

        out = self.dropout(self.first_layer(x))
        for block in self.blocks:
            out = block(out) + out

        return self.last_layer(out)


class Discriminator(nn.Module):
    def __init__(self, config, input_dim):
        super(Discriminator, self).__init__()
        self.config = config

        self.pack = config['model']['discriminator'].get('pack', 1)

        hidden_dim = config['model']['discriminator']['hidden_dim']
        num_layers = config['model']['discriminator']['layers']

        layers = []
        current_dim = input_dim * self.pack

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.5))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        if self.pack > 1:
            if x.size(0) % self.pack != 0:
                truncate_len = x.size(0) - (x.size(0) % self.pack)
                x = x[:truncate_len]
            x = x.view(-1, x.size(1) * self.pack)

        return self.seq(x)