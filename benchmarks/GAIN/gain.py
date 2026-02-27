import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
import sys
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but not supported on MPS.*")


def setup_project_paths():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == 'GAIN':
            project_root = parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            return
    sys.path.insert(0, str(current_path.parent))
setup_project_paths()

from data_transformer import DataTransformer
from networks import Generator, Discriminator


class GAINDataset(Dataset):
    def __init__(self, data_df):
        self.X_raw = data_df.to_numpy(dtype=np.float32)
        self.M = 1.0 - np.isnan(self.X_raw)
        self.X = np.nan_to_num(self.X_raw, nan=0.0)

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        return {
            'X': torch.tensor(self.X[idx], dtype=torch.float32),
            'M': torch.tensor(self.M[idx], dtype=torch.float32)
        }


class GAIN:
    def __init__(self, config, data_info, device=None):
        self.config = config
        self.data_info = data_info

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Active Device: {self.device}")
        self.generators = []

    def fit(self, file_path=None, provided_data=None):
        if provided_data is None:
            print(f"Loading data from {file_path}...")
            self.df_raw = pd.read_csv(file_path).reset_index(drop=True)
            self.df_raw = self.df_raw.loc[:, ~self.df_raw.columns.str.contains('^Unnamed')]
        else:
            self.df_raw = provided_data

        self.df_raw.columns = self.df_raw.columns.str.strip()
        for col in self.data_info.get('num_vars', []):
            if col in self.df_raw.columns:
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')

        self.transformer = DataTransformer(self.data_info)
        self.transformer.fit(self.df_raw)
        self.df_processed = self.transformer.transform()
        self.dim = self.df_processed.shape[1]

        self.num_idx = sum(1 for c in self.transformer.generated_columns
                           if c in self.data_info.get('num_vars', []))
        self.dataset = GAINDataset(self.df_processed)

        self._train_loop()
        return self

    def _train_loop(self):
        epochs = self.config['train']['epochs']
        loss_cfg = self.config['train']['loss']
        hint_rate = self.config['model'].get('hint_rate', 0.9)

        dataloader = self._get_dataloader()
        data_iter = iter(dataloader)

        G, D = self._init_model_pair()
        opt_G, opt_D = self._init_optimizers(G, D)

        G.train()
        D.train()

        pbar = tqdm(range(1, epochs + 1), desc="Training GAIN", colour='black')
        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            X_mb = batch['X'].to(self.device, non_blocking=True)
            M_mb = batch['M'].to(self.device, non_blocking=True)
            curr_batch_size = X_mb.size(0)

            Z_mb = torch.rand((curr_batch_size, self.dim), device=self.device) * 0.01
            H_mb_temp = (torch.rand((curr_batch_size, self.dim), device=self.device) < hint_rate).float()
            H_mb = M_mb * H_mb_temp
            X_mb_combined = M_mb * X_mb + (1 - M_mb) * Z_mb
            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            G_sample = G(X_mb_combined, M_mb)
            Hat_X = X_mb_combined * M_mb + G_sample * (1 - M_mb)

            D_prob = D(Hat_X.detach(), H_mb)

            d_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) +
                                 (1 - M_mb) * torch.log(1. - D_prob + 1e-8))
            d_loss.backward()
            opt_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_G.zero_grad()
            G_sample = G(X_mb_combined, M_mb)
            Hat_X = X_mb_combined * M_mb + G_sample * (1 - M_mb)

            D_prob = D(Hat_X, H_mb)

            g_loss_temp = -torch.mean((1 - M_mb) * torch.log(D_prob + 1e-8))

            if self.num_idx > 0:
                M_mb_num = M_mb[:, :self.num_idx]
                X_mb_num = X_mb[:, :self.num_idx]
                G_sample_num = G_sample[:, :self.num_idx]
                mse_loss = torch.sum((M_mb_num * X_mb_num - M_mb_num * G_sample_num) ** 2) / (
                            torch.sum(M_mb_num) + 1e-8)
            else:
                mse_loss = torch.tensor(0.0, device=self.device)
            if self.num_idx < self.dim:
                M_mb_cat = M_mb[:, self.num_idx:]
                X_mb_cat = X_mb[:, self.num_idx:]
                G_sample_cat = G_sample[:, self.num_idx:]
                bce_matrix = -(X_mb_cat * torch.log(G_sample_cat + 1e-8) +
                               (1 - X_mb_cat) * torch.log(1 - G_sample_cat + 1e-8))
                cat_loss = torch.sum(M_mb_cat * bce_matrix) / (torch.sum(M_mb_cat) + 1e-8)
            else:
                cat_loss = torch.tensor(0.0, device=self.device)

            g_loss = g_loss_temp + loss_cfg['alpha'] * mse_loss + loss_cfg['beta'] * cat_loss
            g_loss.backward()
            opt_G.step()

            logs = {
                'D_Loss': f"{d_loss.item():.4f}",
                'G_Loss': f"{g_loss_temp.item():.4f}",
                'MSE': f"{mse_loss.item():.4f}",
                'BCE': f"{cat_loss.item():.4f}"
            }
            pbar.set_postfix(logs)
        self.generators.append(G)

    def impute(self, save_path=None):
        print("Generating Imputed Dataset...")
        full_loader = DataLoader(self.dataset, batch_size=self.config['train']['batch_size'],
                                 shuffle=False, num_workers=0, pin_memory=True)

        curr_G = self.generators[0]
        curr_G.eval()

        imputed_rows = []
        with torch.no_grad():
            for batch in full_loader:
                X_mb = batch['X'].to(self.device, non_blocking=True)
                M_mb = batch['M'].to(self.device, non_blocking=True)
                curr_batch_size = X_mb.size(0)

                Z_mb = torch.rand((curr_batch_size, self.dim), device=self.device) * 0.01
                X_input = M_mb * X_mb + (1 - M_mb) * Z_mb

                imputed_array = curr_G(X_input, M_mb).cpu().numpy()
                final_batch = batch['M'].numpy() * batch['X'].numpy() + (1 - batch['M'].numpy()) * imputed_array
                imputed_rows.append(final_batch)

        full_gen = np.concatenate(imputed_rows, axis=0)
        df_gen = pd.DataFrame(full_gen, columns=self.df_processed.columns, index=self.df_processed.index)
        df_denorm = self.transformer.inverse_transform(df_gen)
        df_f = self.df_raw.copy()
        for col in df_denorm.columns:
            if col in df_f.columns:
                df_denorm[col] = df_denorm[col].astype(df_f[col].dtype)

        target_vars = self.data_info.get('num_vars', []) + self.data_info.get('cat_vars', [])
        for c in target_vars:
            df_f[c] = df_f[c].fillna(df_denorm[c])

        if save_path:
            df_f.to_parquet(save_path, index=False)
            print(f"Saved imputed data to: {save_path}")
        return df_f

    def _get_dataloader(self):
        batch_size = self.config['train']['batch_size']
        iterations = self.config['train']['epochs']
        total_samples_needed = iterations * batch_size
        sampler = RandomSampler(self.dataset, replacement=False, num_samples=total_samples_needed)

        return DataLoader(self.dataset, batch_size=batch_size,
                          sampler=sampler, drop_last=True, num_workers=0, pin_memory=True)

    def _init_model_pair(self):
        G = Generator(self.dim, self.dim).to(self.device)
        D = Discriminator(self.dim, self.dim).to(self.device)
        return G, D

    def _init_optimizers(self, G, D):
        tc = self.config['train']['Adam']
        opt_G = optim.Adam(G.parameters(), lr=tc['lr_g'], betas=tuple(tc.get('betas_g', (0.9, 0.999))),
                           weight_decay=tc.get('weight_decay', 0))
        opt_D = optim.Adam(D.parameters(), lr=tc['lr_d'], betas=tuple(tc.get('betas_d', (0.9, 0.999))),
                           weight_decay=tc.get('weight_decay', 0))
        return opt_G, opt_D