import torch
import torch.nn as nn

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
                    nn.ReLU(),
                    nn.Dropout(dropout_p)
                )
            )
            current_input_dim += hidden_dim
        self.output_layer = nn.Linear(current_input_dim, p2_dim)


    def forward(self, z, p1, cond):
        backbone_in = torch.cat([z, p1, cond], dim=1)
        x = self.dropout(backbone_in)
        for block in self.blocks:
            out = block(x)
            x = torch.cat([x, out], dim=1)
        return self.output_layer(x)


class Discriminator(nn.Module):
    def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
        super(Discriminator, self).__init__()
        self.config = config
        self.input_dim = p1_dim + p2_dim + cond_cols_dim
        self.pack = config['model']['discriminator'].get('pack', 1)
        hidden_dim = config['model']['discriminator']['hidden_dim']
        num_layers = config['model']['discriminator']['layers']
        dropout_p = config['model']['generator']['dropout']
        layers = []
        self.pack_dim = self.pack * self.input_dim

        current_dim = self.pack_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.seq_head = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.pack_dim)
        out = self.seq_head(input)
        return out


class BiasHunter(nn.Module):
    def __init__(self, config, p1_dim, p2_dim, cond_cols_dim):
        super(BiasHunter, self).__init__()
        self.config = config
        self.backbone_input_dim = p2_dim

        hidden_dim = config['model']['biashunter']['hidden_dim']
        num_layers = config['model']['biashunter']['layers']
        dropout_p = config['model']['biashunter']['dropout']

        self.blocks = nn.ModuleList()
        current_input_dim = self.backbone_input_dim

        for _ in range(num_layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(current_input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p)
                )
            )
            current_input_dim += hidden_dim
        self.output_layer = nn.Linear(current_input_dim, p1_dim + cond_cols_dim)

    def forward(self, p2):
        x = p2
        for block in self.blocks:
            out = block(x)
            x = torch.cat([x, out], dim=1)
        return self.output_layer(x)