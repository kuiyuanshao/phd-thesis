import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, RandomSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but not supported on MPS.*")
from .data_transformer import DataTransformer, ImputationDataset
from .networks import Generator, Discriminator
from .utils import gumbel_activation, gradient_penalty, recon_loss, project_categorical
from .swag import SWAG


class SICG:
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
        self.swag_model = None
        self.confusion_matrices = {}
        self.proj_groups = []

    def fit(self, file_path):
        print(f"Loading data from {file_path}...")
        self.df_raw = pd.read_csv(file_path).reset_index(drop=True)
        self.df_raw = self.df_raw.loc[:, ~self.df_raw.columns.str.contains('^Unnamed')]
        self.df_raw.columns = self.df_raw.columns.str.strip()

        # 1. Identify Phase 2 Training Rows
        p2_check_col = self.data_info['phase2_vars'][0]
        self.p1_indices = self.df_raw.index[self.df_raw[p2_check_col].isna()].tolist()
        self.p2_indices = self.df_raw.index[self.df_raw[p2_check_col].notna()].tolist()
        print(f"Phase 2 Training Samples: {len(self.p2_indices)}")

        # 2. Fast Numeric Coercion
        for col in self.data_info.get('num_vars', []):
            if col in self.df_raw.columns:
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')

        # 3. Residual Transformation

        # 4. Transform
        print("Preprocessing data...")
        self.transformer = DataTransformer(self.data_info, self.config)
        self.transformer.fit(self.df_raw)

        self._calculate_confusion_matrices(self.df_raw)
        self.df_processed = self.transformer.transform()

        p1_raw, p2_raw, cond_raw = self.transformer.get_dims()

        # 5. Optimize Layout
        self.p1_cols = p1_raw
        self.cond_cols = cond_raw
        self.p2_cols, self.layout, self.layout_map = self._optimize_p2_layout(p2_raw)

        valid_cols = [c for c in (self.p1_cols + self.p2_cols + self.cond_cols) if c in self.df_processed.columns]
        self.df_processed = self.df_processed[valid_cols]

        self.p1_cols = [c for c in self.p1_cols if c in valid_cols]
        self.p2_cols = [c for c in self.p2_cols if c in valid_cols]
        self.cond_cols = [c for c in self.cond_cols if c in valid_cols]

        self._build_projection_groups()

        # 6. Training Dataset
        df_p1 = self.df_processed.iloc[self.p1_indices].reset_index(drop=True)
        self.p1_dataset = ImputationDataset(df_p1, self.p1_cols, self.p2_cols, self.cond_cols)
        df_p2 = self.df_processed.iloc[self.p2_indices].reset_index(drop=True)
        self.p2_dataset = ImputationDataset(df_p2, self.p1_cols, self.p2_cols, self.cond_cols)

        # 7. Weights
        weight_var = self.data_info.get('weight_var')
        self.use_weighted_sampler = False
        if weight_var and weight_var in self.df_raw.columns:
            w_vals = self.df_raw.loc[self.p2_indices, weight_var].fillna(0).values.astype(np.float32)
            if np.std(w_vals) > 1e-6:
                self.use_weighted_sampler = True
                self.global_weights = torch.from_numpy(w_vals)
            else:
                self.global_weights = None
        else:
            self.global_weights = None

        self._train_loop()
        return self

    def _calculate_confusion_matrices(self, df):
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        cat_vars = set(self.data_info['cat_vars'])

        print("Calculating Confusion Matrices...")
        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in cat_vars and p2 in cat_vars:
                if p1 not in self.transformer.cat_encoders: continue
                enc = self.transformer.cat_encoders[p1]

                mask = df[p1].notna() & df[p2].notna()
                if mask.sum() == 0: continue

                v1 = df.loc[mask, p1].astype(str).to_numpy().reshape(-1, 1)
                v2 = df.loc[mask, p2].astype(str).to_numpy().reshape(-1, 1)

                try:
                    idx1 = enc.transform(v1).argmax(axis=1)
                    idx2 = enc.transform(v2).argmax(axis=1)
                except:
                    continue

                n_cat = len(enc.categories_[0])
                mat = torch.zeros(n_cat, n_cat, device=self.device)

                indices = np.stack([idx2, idx1], axis=1)
                for i, j in indices:
                    mat[i, j] += 1

                row_sums = mat.sum(dim=1, keepdim=True)
                mat = mat / (row_sums + 1e-8)

                self.confusion_matrices[p2] = mat

    def _optimize_p2_layout(self, p2_cols):
        var_map = {};
        processed = set()

        for base in self.data_info['cat_vars']:
            prefix = f"{base}_"
            cols = [c for c in p2_cols if c.startswith(prefix)]
            if cols:
                var_map[base] = sorted(cols)
                processed.update(cols)

        for base in self.data_info['num_vars']:
            prefix = f"{base}_mode_"
            cols = [c for c in p2_cols if c.startswith(prefix)]
            if cols:
                cols = sorted(cols, key=lambda x: int(x.split('_')[-1]))
                var_map[f"{base}_mode"] = cols;
                processed.update(cols)

        numericals = [c for c in p2_cols if c not in processed]
        groups = {};
        if numericals: groups[0] = [numericals]

        for base, cols in var_map.items():
            card = len(cols)
            if card not in groups: groups[card] = []
            groups[card].append((base, cols))

        sorted_cols = [];
        layout = [];
        layout_map = []
        curr = 0

        if 0 in groups:
            for cols in groups[0]:
                sorted_cols.extend(cols)
                w = len(cols)
                layout.append((curr, curr + w, 0))
                curr += w

        for card in sorted([k for k in groups.keys() if k != 0]):
            for item in groups[card]:
                base, cols = item
                sorted_cols.extend(cols)
                w = len(cols)
                layout.append((curr, curr + w, card))
                layout_map.append({
                    'base_name': base,
                    'p2_range': (curr, curr + w),
                    'card': card,
                    'cols': cols
                })
                curr += w

        return sorted_cols, layout, layout_map

    def _build_projection_groups(self):
        self.proj_groups = []
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']

        p1_col_map = {c: i for i, c in enumerate(self.p1_cols)}

        for info in self.layout_map:
            base_p2 = info['base_name']
            if base_p2 not in p2_vars: continue

            try:
                idx = p2_vars.index(base_p2)
                base_p1 = p1_vars[idx]
            except ValueError:
                continue

            if base_p2 not in self.confusion_matrices: continue

            if base_p1 not in self.transformer.cat_encoders: continue
            enc = self.transformer.cat_encoders[base_p1]
            p1_sub_cols = [f"{base_p1}_{c}" for c in enc.categories_[0]]

            indices = []
            for c in p1_sub_cols:
                if c in p1_col_map:
                    indices.append(p1_col_map[c])

            if not indices: continue

            p1_start = min(indices)
            p1_end = max(indices) + 1

            group = {
                'base_name': base_p2,
                'p2_range': info['p2_range'],
                'p1_range': (p1_start, p1_end),
                'card': info['card'],
                'matrix': self.confusion_matrices[base_p2]
            }
            self.proj_groups.append(group)

    def _train_loop(self):
        cfg = self.config['train']
        total_steps = cfg['epochs']
        mi_approx = cfg['mi_approx']
        num_models = cfg.get('m', 5) if mi_approx == 'bootstrap' else 1
        loss_cfg = self.config['train']['loss']
        noise_dim = self.config['model']['generator']['noise_dim']

        for k in range(num_models):
            print(f"\nTraining Model {k + 1}/{num_models} ({mi_approx})")

            dataloader_p1 = self._get_dataloader("p1", is_bootstrap=(mi_approx == 'bootstrap'))
            data_iter_p1 = iter(dataloader_p1)
            dataloader_p2 = self._get_dataloader("p2", is_bootstrap=(mi_approx == 'bootstrap'))
            data_iter_p2 = iter(dataloader_p2)

            G, D = self._init_model_pair()
            opt_G, opt_D = self._init_optimizers(G, D)

            current_swag = None
            if mi_approx == 'SWAG':
                current_swag = SWAG(G, max_num_models=20, var_clamp=1e-30).to(self.device)
                swag_start = int(total_steps * 0.8)
                scheduler_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=swag_start,
                                                                   eta_min=cfg['optimizer']['lr_g'] * 0.5)
                scheduler_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=swag_start,
                                                                   eta_min=cfg['optimizer']['lr_d'] * 0.5)
            G.train()
            D.train()

            pbar = tqdm(range(1, total_steps + 1), desc=f"Model {k + 1}", colour='black')
            for step in pbar:
                # --- Discriminator Step ---
                try:
                    batch = next(data_iter_p2)
                except StopIteration:
                    data_iter_p2 = iter(dataloader_p2)
                    batch = next(data_iter_p2)

                A = batch['A'].to(self.device, non_blocking=True)
                X = batch['X'].to(self.device, non_blocking=True)
                C = batch['C'].to(self.device, non_blocking=True)
                bs = A.size(0)

                for _ in range(loss_cfg['discriminator_steps']):
                    opt_D.zero_grad()
                    z = torch.randn(bs, noise_dim, device=self.device)
                    fake_raw = G(z, A, C)
                    fake_act = gumbel_activation(fake_raw, self.layout, loss_cfg['tau'], loss_cfg['hard_gumbel'])
                    fake_cat = torch.cat([fake_act, A, C], dim=1)
                    real_cat = torch.cat([X, A, C], dim=1)
                    d_fake = D(fake_cat.detach())
                    d_real = D(real_cat)
                    d_loss = d_fake.mean() - d_real.mean()
                    if loss_cfg['lambda_gp'] > 0:
                        d_loss += gradient_penalty(D, real_cat, fake_cat, loss_cfg['lambda_gp'])
                    d_loss.backward()
                    opt_D.step()

                # --- Generator Step ---
                opt_G.zero_grad()

                # Fetch P2 Batch
                try:
                    batch_p2 = next(data_iter_p2)
                except StopIteration:
                    data_iter_p2 = iter(dataloader_p2)
                    batch_p2 = next(data_iter_p2)

                A_p2 = batch_p2['A'].to(self.device, non_blocking=True)
                X_p2 = batch_p2['X'].to(self.device, non_blocking=True)
                C_p2 = batch_p2['C'].to(self.device, non_blocking=True)

                z_p2 = torch.randn(A_p2.size(0), noise_dim, device=self.device)
                fake_p2 = G(z_p2, A_p2, C_p2)

                loss_p2 = recon_loss(fake_p2, X_p2, mode='p2', layout=self.layout,
                                     alpha=loss_cfg['alpha'], beta=loss_cfg['beta'])

                fake_p2_act = gumbel_activation(fake_p2, self.layout, loss_cfg['tau'], loss_cfg['hard_gumbel'])
                fake_cat_p2 = torch.cat([fake_p2_act, A_p2, C_p2], dim=1)

                g_adv2 = -D(fake_cat_p2).mean()

                g_loss = g_adv2 + loss_p2
                g_loss.backward()

                if mi_approx == 'SWAG':
                    torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()

                if self.config['model']['semi_supervised']:
                    for p in D.parameters():
                        p.requires_grad = False
                    opt_G.zero_grad()
                    try:
                        batch_p1 = next(data_iter_p1)
                    except StopIteration:
                        data_iter_p1 = iter(dataloader_p1)
                        batch_p1 = next(data_iter_p1)

                    A_p1 = batch_p1['A'].to(self.device, non_blocking=True)
                    C_p1 = batch_p1['C'].to(self.device, non_blocking=True)

                    z_p1 = torch.randn(A_p1.size(0), noise_dim, device=self.device)
                    fake_p1 = G(z_p1, A_p1, C_p1)

                    loss_p1 = recon_loss(fake_p1, A_p1, mode='p1', proj_groups=self.proj_groups)
                    fake_p1_act = gumbel_activation(fake_p1, self.layout, loss_cfg['tau'], loss_cfg['hard_gumbel'])

                    fake_cat_p1 = torch.cat([fake_p1_act, A_p1, C_p1], dim=1)

                    g_adv1 = -D(fake_cat_p1).mean()
                    g_loss = g_adv1 + loss_p1
                    g_loss.backward()

                    if mi_approx == 'SWAG':
                        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                    opt_G.step()

                    for p in D.parameters():
                        p.requires_grad = True

                pbar.set_postfix({
                    'D_Loss': f"{d_loss.item():.4f}",
                    'G_Adv': f"{g_adv1.item() + g_adv2.item():.4f}",
                    'Recon': f"{loss_p1.item() + loss_p2.item():.4f}"
                })

                if mi_approx == 'SWAG':
                    if step < swag_start:
                        scheduler_G.step();
                        scheduler_D.step()
                    elif (step - swag_start) % 50 == 0:
                        current_swag.collect_model(G)

            if mi_approx == 'SWAG': self.swag_model = current_swag
            self.generators.append(G)

    def impute(self, m=None, save_path=None):
        m = m if m is not None else self.config['train'].get('m', 5)
        print(f"Generating {m} Imputations...")
        all_imputed_dfs = []
        mi_approx = self.config['train']['mi_approx']

        # Inference on Full Data
        full_dataset = ImputationDataset(self.df_processed, self.p1_cols, self.p2_cols, self.cond_cols)
        full_loader = DataLoader(full_dataset, batch_size=self.config['train']['batch_size'], shuffle=False,
                                 num_workers=0, pin_memory=True)
        pbar = tqdm(total=m, desc="Imputation Rounds", colour='black')
        for i in range(m):
            if mi_approx == 'BOOTSTRAP':
                curr = self.generators[i % len(self.generators)]
            elif mi_approx == 'SWAG' and self.swag_model:
                curr = self.swag_model.sample(scale=1.0, cov=True)
            elif mi_approx == 'DROPOUT':
                curr = self.generators[0];
                curr.eval()
                for mod in curr.modules():
                    if isinstance(mod, torch.nn.Dropout): mod.train()
            else:
                curr = self.generators[0]

            curr.eval()
            imputed_rows = []
            with torch.no_grad():
                for batch in full_loader:
                    A = batch['A'].to(self.device, non_blocking=True)
                    C = batch['C'].to(self.device, non_blocking=True)
                    z = torch.randn(A.size(0), self.config['model']['generator']['noise_dim'], device=self.device)
                    fake_raw = curr(z, A, C)
                    fake_act = gumbel_activation(fake_raw, self.layout, 1, True)
                    imputed_rows.append(fake_act.cpu().numpy())

            full_gen = np.concatenate(imputed_rows, axis=0)
            df_gen = pd.DataFrame(full_gen, columns=self.p2_cols, index=self.df_processed.index)
            temp = self.df_processed.copy();
            temp[self.p2_cols] = df_gen[self.p2_cols]
            denorm = self.transformer.inverse_transform(temp)
            final = self.df_raw.copy()
            final = final.fillna(denorm)
            final['imp_id'] = i + 1
            all_imputed_dfs.append(final)
            pbar.update(1)
        pbar.close()
        if save_path:
            final_df = pd.concat(all_imputed_dfs, ignore_index=True)
            for c in final_df.columns:
                if final_df[c].dtype == 'object':
                    try:
                        final_df[c] = pd.to_numeric(final_df[c])
                    except:
                        final_df[c] = final_df[c].astype(str)

            final_df.to_parquet(save_path, index=False)
            print(f"Saved stacked imputations to: {save_path}")
            return all_imputed_dfs
        else:
            return all_imputed_dfs

    def _get_dataloader(self, Set, is_bootstrap):
        if Set == "p2":
            n = len(self.p2_dataset)
            if is_bootstrap:
                boot_idx = np.random.choice(np.arange(n), size=n, replace=True)
                p2_subset = Subset(self.p2_dataset, boot_idx)
                current_weights = np.array(self.global_weights)[boot_idx]
                sampler = WeightedRandomSampler(weights=current_weights,
                                                num_samples=n,
                                                replacement=True)
                return DataLoader(p2_subset, batch_size=self.config['train']['batch_size'], sampler=sampler,
                                  shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
            else:
                sampler = WeightedRandomSampler(self.global_weights, n,
                                                replacement=True) if self.use_weighted_sampler else None
                return DataLoader(self.p2_dataset, batch_size=self.config['train']['batch_size'],
                                  sampler=sampler, shuffle=(sampler is None), drop_last=True,
                                  num_workers=0, pin_memory=True)
        if Set == "p1":
            n = len(self.p1_dataset)
            if is_bootstrap:
                boot_idx = np.random.choice(np.arange(n), size=n, replace=True)
                return DataLoader(Subset(self.p1_dataset, boot_idx), batch_size=self.config['train']['batch_size'],
                                  shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
            else:
                sampler = RandomSampler(self.p1_dataset, replacement=True, num_samples=n)
                return DataLoader(self.p1_dataset, batch_size=self.config['train']['batch_size'],
                                  sampler=sampler, drop_last=True,
                                  num_workers=0, pin_memory=True)

    def _init_model_pair(self):
        G = Generator(self.config, p1_dim=len(self.p1_cols), p2_dim=len(self.p2_cols),
                      cond_cols_dim=len(self.cond_cols)).to(self.device)
        d_input = len(self.p2_cols) + len(self.p1_cols) + len(self.cond_cols)
        D = Discriminator(self.config, input_dim=d_input).to(self.device)
        return G, D

    def _init_optimizers(self, G, D):
        tc = self.config['train']['optimizer']
        if self.config['train']['mi_approx'] == 'SWAG':
            opt_G = optim.SGD(G.parameters(), lr=tc['lr_g'], momentum=0.9, weight_decay=tc['weight_decay'])
            opt_D = optim.SGD(D.parameters(), lr=tc['lr_d'], momentum=0.9, weight_decay=tc['weight_decay'])
        else:
            opt_G = optim.Adam(G.parameters(), lr=tc['lr_g'], betas=tuple(tc['betas_g']),
                               weight_decay=tc['weight_decay'])
            opt_D = optim.Adam(D.parameters(), lr=tc['lr_d'], betas=tuple(tc['betas_d']),
                               weight_decay=tc['weight_decay'])
        return opt_G, opt_D