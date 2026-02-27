import torch
import torch.nn as nn
# Code adapted from https://github.com/jsyoon0823/GAIN/tree/master
# @inproceedings{yoon2018gain,
#   title={Gain: Missing data imputation using generative adversarial nets},
#   author={Yoon, Jinsung and Jordon, James and Schaar, Mihaela},
#   booktitle={International conference on machine learning},
#   pages={5689--5698},
#   year={2018},
#   organization={PMLR}
# }
class Generator(nn.Module):
    def __init__(self, dim, h_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        return self.net(inputs)

class Discriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        return self.net(inputs)