import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but not supported on MPS.*")
from .data_transformer import DataTransformer, ImputationDataset
from .networks import Generator, Discriminator, MINE
from .utils import gumbel_activation, gradient_penalty, recon_loss, moment_matching_loss, PCA, calc_HSIC_loss, \
    calculate_mine_loss
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
        self._pca = PCA(n_components=None, device=self.device)

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

        # 4. Transform
        self.transformer = DataTransformer(self.data_info, self.config)

        # NOTE: Transformer fit now normalizes categories internally
        self.transformer.fit(self.df_raw)

        # 5. Calculate Matrix (Using Robust Normalization)
        self._calculate_confusion_matrices(self.df_raw)

        self.df_processed = self.transformer.transform()

        p1_raw, p2_raw, cond_raw = self.transformer.get_dims()

        # A. Analyze Layout
        self.p2_cols, self.layout, self.layout_map = self._analyze_layout(p2_raw, self.data_info)
        self.p1_cols, layout_p1, _ = self._analyze_layout(p1_raw, self.data_info)
        self.cond_cols, layout_cond, _ = self._analyze_layout(cond_raw, self.data_info)

        # Build Map for P1 Columns to help Projection Grouping
        self.p1_col_map = {c: i for i, c in enumerate(self.p1_cols)}

        # B. Merge Layouts
        p1_width = len(self.p1_cols)
        layout_cond_shifted = [
            (start + p1_width, end + p1_width, card)
            for start, end, card in layout_cond
        ]
        self.layout_attrs = layout_p1 + layout_cond_shifted

        # C. Reorder Columns
        all_final_cols = self.p1_cols + self.p2_cols + self.cond_cols
        missing = [c for c in all_final_cols if c not in self.df_processed.columns]
        if missing:
            print(f"Warning: Columns missing after transform: {missing[:5]}...")
            for c in missing: self.df_processed[c] = 0.0

        self.df_processed = self.df_processed[all_final_cols]

        # D. Dimensions
        self.p1_dim = len(self.p1_cols)
        self.p2_dim = len(self.p2_cols)
        self.cond_dim = len(self.cond_cols)

        self._build_projection_groups()

        # 6. Training Dataset
        df_p1 = self.df_processed.iloc[self.p1_indices].reset_index(drop=True)
        self.p1_dataset = ImputationDataset(df_p1, self.p1_cols, self.p2_cols, self.cond_cols)
        df_p2 = self.df_processed.iloc[self.p2_indices].reset_index(drop=True)
        self.p2_dataset = ImputationDataset(df_p2, self.p1_cols, self.p2_cols, self.cond_cols)

        pca_cols = self.p2_cols + self.p1_cols + self.cond_cols
        pca_cols = [c for c in pca_cols if c in df_p2.columns]
        self._pca.fit(torch.from_numpy(df_p2[pca_cols].values.astype("float32")))

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

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in cat_vars and p2 in cat_vars:
                # 1. Get Independent Encoders
                enc_p1 = self.transformer.cat_encoders.get(p1)
                enc_p2 = self.transformer.cat_encoders.get(p2)
                if not enc_p1 or not enc_p2: continue

                # 2. Prepare Valid Data (Intersection)
                mask = df[p1].notna() & df[p2].notna()
                if mask.sum() == 0: continue

                # 3. Robust Normalization (Using same logic as Transformer)
                v1_raw = df.loc[mask, p1].apply(DataTransformer._robust_string)
                v2_raw = df.loc[mask, p2].apply(DataTransformer._robust_string)

                # 4. Map Categories to Indices
                cats_p1 = enc_p1.categories_[0]
                cats_p2 = enc_p2.categories_[0]

                c1_map = {c: i for i, c in enumerate(cats_p1)}
                c2_map = {c: i for i, c in enumerate(cats_p2)}

                # 5. Build Matrix: (Rows=P2, Cols=P1)
                mat = torch.zeros(len(cats_p2), len(cats_p1), device=self.device)

                # Iterate and count
                # Note: Iterating pandas series is slow but safe for pre-processing step.
                for val1, val2 in zip(v1_raw, v2_raw):
                    if val1 in c1_map and val2 in c2_map:
                        r = c2_map[val2]  # P2 (Source)
                        c = c1_map[val1]  # P1 (Target)
                        mat[r, c] += 1

                # 6. Normalize by Row (P(P1 | P2))
                row_sums = mat.sum(dim=1, keepdim=True)
                mat = mat / (row_sums + 1e-8)

                self.confusion_matrices[p2] = mat

    def _build_projection_groups(self):
        self.proj_groups = []
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']

        # Iterate via Layout Map (which is based on P2 columns)
        for info in self.layout_map:
            base_p2 = info['base_name']

            if base_p2 not in p2_vars: continue
            try:
                idx = p2_vars.index(base_p2)
                base_p1 = p1_vars[idx]
            except ValueError:
                continue

            # Get P1 Encoder to find corresponding P1 columns
            if base_p1 not in self.transformer.cat_encoders: continue
            enc_p1 = self.transformer.cat_encoders[base_p1]

            # Construct expected P1 column names based on encoder categories
            p1_sub_cols = [f"{base_p1}_{c}" for c in enc_p1.categories_[0]]

            # Find these columns in the P1 section of the model input
            indices = [self.p1_col_map[c] for c in p1_sub_cols if c in self.p1_col_map]

            if not indices: continue

            # Assume P1 columns are contiguous (standard behavior of _analyze_layout)
            p1_start = min(indices)
            p1_end = max(indices) + 1

            group = {
                'base_name': base_p2,
                'p2_range': info['range'],
                'p1_range': (p1_start, p1_end),
                'card': info['card'],  # P2 Cardinality
                'matrix': self.confusion_matrices.get(base_p2)
            }
            if group['matrix'] is not None:
                self.proj_groups.append(group)

    def _train_loop(self):
        # ... (Identical to previous implementation) ...
        cfg = self.config['train']
        total_steps = cfg['epochs']
        mi_approx = cfg['mi_approx']
        num_models = cfg.get('m', 5) if mi_approx == 'BOOTSTRAP' else 1
        loss_cfg = self.config['train']['loss']
        noise_dim = self.config['model']['generator']['noise_dim']

        for k in range(num_models):
            print(f"\nTraining Model {k + 1}/{num_models} ({mi_approx})")
            dataloader_p1 = self._get_dataloader("p1", is_bootstrap=(mi_approx == 'BOOTSTRAP'))
            data_iter_p1 = iter(dataloader_p1)
            dataloader_p2 = self._get_dataloader("p2", is_bootstrap=(mi_approx == 'BOOTSTRAP'))
            data_iter_p2 = iter(dataloader_p2)

            if self.config['model']['mine']:
                G, D, B = self._init_model_pair()
                opt_G, opt_D, opt_B = self._init_optimizers(G, D, B)
            else:
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
                for _ in range(loss_cfg['discriminator_steps']):
                    try:
                        batch = next(data_iter_p2)
                    except StopIteration:
                        data_iter_p2 = iter(dataloader_p2)
                        batch = next(data_iter_p2)

                    A = batch['A'].to(self.device, non_blocking=True)
                    X = batch['X'].to(self.device, non_blocking=True)
                    C = batch['C'].to(self.device, non_blocking=True)
                    opt_D.zero_grad()
                    z = torch.randn(A.size(0), noise_dim, device=self.device)
                    fake_raw = G(z, A, C)
                    fake_act = gumbel_activation(fake_raw, self.layout, loss_cfg['tau'], loss_cfg['hard_gumbel'])
                    fake_d = torch.cat([fake_act, A, C], dim=1)
                    true_d = torch.cat([X, A, C], dim=1)
                    d_fake = D(fake_d.detach())
                    d_real = D(true_d)
                    d_loss = d_fake.mean() - d_real.mean()
                    if loss_cfg['lambda_gp'] > 0:
                        d_loss += gradient_penalty(D, true_d, fake_d, loss_cfg['lambda_gp'])
                    d_loss.backward()
                    opt_D.step()

                if self.config['model']['mine']:
                    for _ in range(5):
                        try:
                            batch = next(data_iter_p2)
                        except StopIteration:
                            data_iter_p2 = iter(dataloader_p2)
                            batch = next(data_iter_p2)

                        A = batch['A'].to(self.device, non_blocking=True)
                        X = batch['X'].to(self.device, non_blocking=True)
                        C = batch['C'].to(self.device, non_blocking=True)
                        opt_B.zero_grad()
                        z = torch.randn(A.size(0), noise_dim, device=self.device)
                        fake_raw = G(z, A, C)
                        fake_act = gumbel_activation(fake_raw, self.layout, loss_cfg['tau'], loss_cfg['hard_gumbel'])
                        residual = fake_act - X
                        loss_B, loss_m = calculate_mine_loss(residual, A, C, B, self.layout)
                        loss_m.backward()
                        opt_B.step()
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
                                     alpha=loss_cfg['loss_mse'], beta=loss_cfg['loss_ce'])

                fake_p2_act = gumbel_activation(fake_p2, self.layout, loss_cfg['tau'], loss_cfg['hard_gumbel'])
                fake_d = torch.cat([fake_p2_act, A_p2, C_p2], dim=1)

                g_adv2 = D(fake_d)
                g_adv2 = -g_adv2.mean()
                g_loss = g_adv2 + loss_p2

                if self.config['model']['mml']:
                    loss_marg = moment_matching_loss(self._pca, torch.cat([X_p2, A_p2, C_p2], dim=1), fake_d)
                    g_loss += loss_marg
                if self.config['model']['hsic']:
                    loss_hsic = calc_HSIC_loss(fake_p2, X_p2,
                                               torch.cat([A_p2, C_p2], dim=1),
                                               self.layout,
                                               self.layout_attrs,
                                               loss_cfg['loss_hsic'])
                    g_loss += loss_hsic
                if self.config['model']['mine']:
                    loss_B_adv, _ = calculate_mine_loss(fake_p2_act - X_p2, A_p2, C_p2, B,
                                                        self.layout)
                    g_loss += loss_B_adv
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
                    loss_p1 = recon_loss(fake_p1, A_p1, mode='p1', proj_groups=self.proj_groups,
                                         alpha=loss_cfg['alpha'], beta=loss_cfg['beta'])
                    fake_p1_act = gumbel_activation(fake_p1, self.layout, loss_cfg['tau'], loss_cfg['hard_gumbel'])
                    fake_d = torch.cat([fake_p1_act, A_p1, C_p1], dim=1)
                    g_adv1 = D(fake_d)
                    g_adv1 = -g_adv1.mean()
                    g_loss = g_adv1 + loss_p1
                    g_loss.backward()

                    if mi_approx == 'SWAG':
                        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                    opt_G.step()

                    for p in D.parameters():
                        p.requires_grad = True
                logs = {
                    'D_Loss': f"{d_loss.item():.4f}",
                    'G_Adv': f"{g_adv2.item():.4f}",
                    'Recon': f"{loss_p2.item():.4f}"
                }
                if self.config['model']['mine']:
                    logs['Mine'] = f"{loss_m.item():.4f}"
                    logs['M_Adv'] = f"{loss_B_adv.item():.4f}"

                if self.config['model']['mml']:
                    logs['MM'] = f"{loss_marg.item():.4f}"

                if self.config['model']['hsic']:
                    logs['HSIC'] = f"{loss_hsic.item():.4f}"

                pbar.set_postfix(logs)

                if mi_approx == 'SWAG':
                    if step < swag_start:
                        scheduler_G.step();
                        scheduler_D.step()
                    elif (step - swag_start) % 50 == 0:
                        current_swag.collect_model(G)

            if mi_approx == 'SWAG': self.swag_model = current_swag
            self.generators.append(G)

    # ... (Rest of the class) ...
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
                curr = self.generators[i]
                curr.eval()
            elif mi_approx == 'SWAG' and self.swag_model:
                curr = self.swag_model.sample(scale=1.0, cov=True)
                curr.eval()
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
                    fake_act = gumbel_activation(fake_raw, self.layout, 0.2, True)
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
                current_weights = np.array(self.global_weights)[boot_idx] if self.use_weighted_sampler else None
                sampler = WeightedRandomSampler(weights=current_weights,
                                                num_samples=n,
                                                replacement=True) if self.use_weighted_sampler else None
                return DataLoader(p2_subset, batch_size=self.config['train']['batch_size'], sampler=sampler,
                                  shuffle=(sampler is None), drop_last=True, num_workers=0, pin_memory=True)
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
                return DataLoader(self.p1_dataset, batch_size=self.config['train']['batch_size'], drop_last=True,
                                  shuffle=True, num_workers=0, pin_memory=True)

    def _init_model_pair(self):
        G = Generator(self.config, len(self.p1_cols), len(self.p2_cols), len(self.cond_cols)).to(self.device)
        D = Discriminator(self.config, len(self.p1_cols), len(self.p2_cols), len(self.cond_cols)).to(self.device)
        if self.config['model']['mine']:
            B = MINE(self.config, len(self.p1_cols) + len(self.p2_cols) + len(self.cond_cols)).to(self.device)
            return G, D, B
        return G, D

    def _init_optimizers(self, G, D, B=None):
        tc = self.config['train']['optimizer']
        if self.config['train']['mi_approx'] == 'SWAG':
            opt_G = optim.SGD(G.parameters(), lr=tc['lr_g'], momentum=0.9, weight_decay=tc['weight_decay'])
            opt_D = optim.SGD(D.parameters(), lr=tc['lr_d'], momentum=0.9, weight_decay=tc['weight_decay'])
            if B is not None:
                opt_B = optim.SGD(B.parameters(), lr=tc['lr_d'], momentum=0.9, weight_decay=tc['weight_decay'])
                return opt_G, opt_D, opt_B
            return opt_G, opt_D
        else:
            opt_G = optim.Adam(G.parameters(), lr=tc['lr_g'], betas=tuple(tc['betas_g']),
                               weight_decay=tc['weight_decay'])
            opt_D = optim.Adam(D.parameters(), lr=tc['lr_d'], betas=tuple(tc['betas_d']),
                               weight_decay=tc['weight_decay'])
            if B is not None:
                opt_B = optim.Adam(B.parameters(), lr=tc['lr_d'], betas=tuple(tc['betas_d']),
                                   weight_decay=tc['weight_decay'])
                return opt_G, opt_D, opt_B
            return opt_G, opt_D

    def _analyze_layout(self, cols_to_process, data_info):
        """
        严谨的布局分析器：确保每一列有且只有一个归属，列数绝对守恒。
        """
        var_groups = {}  # 存储 base_name -> [columns]
        remaining = []

        # 1. 遍历输入列，进行“单次归属”检查
        for col in cols_to_process:
            matched = False

            # A. 检查是否属于 GMM 模式列 (如 income_mode_0)
            if "_mode_" in col:
                base_name = col.split("_mode_")[0]
                if base_name in data_info['num_vars']:
                    group_key = f"{base_name}_mode"
                    if group_key not in var_groups: var_groups[group_key] = []
                    var_groups[group_key].append(col)
                    matched = True

            # B. 检查是否属于类别变量 (如 sex_Male)
            if not matched and "_" in col:
                # 寻找所有可能匹配的 base_name
                potential_bases = [b for b in data_info['cat_vars'] if col.startswith(f"{b}_")]
                if potential_bases:
                    # 【核心修复】：取长度最长的 base_name，确保匹配最精确，防止 sex 抢走 sex_oriented 的列
                    best_base = max(potential_bases, key=len)
                    if best_base not in var_groups: var_groups[best_base] = []
                    var_groups[best_base].append(col)
                    matched = True

            # C. 兜底：所有不匹配的列归为连续/通用块 (比如原本就是数值型的列)
            if not matched:
                remaining.append(col)

        # 2. 组装结果
        sorted_cols = []
        hsic_layout = []
        meta_map = []
        curr = 0

        # 第一块：连续变量块 (card=0)
        if remaining:
            sorted_cols.extend(remaining)
            w = len(remaining)
            hsic_layout.append((curr, curr + w, 0))
            curr += w

        # 后续块：离散变量块
        for base_key in sorted(var_groups.keys()):
            cols = var_groups[base_key]
            # 内部排序
            if "_mode" in base_key:
                cols = sorted(cols, key=lambda x: int(x.split('_')[-1]))
            else:
                cols = sorted(cols)

            w = len(cols)
            sorted_cols.extend(cols)
            hsic_layout.append((curr, curr + w, w))

            # 只有原始类别变量需要 meta_map
            if base_key in data_info['cat_vars']:
                meta_map.append({
                    'base_name': base_key,
                    'range': (curr, curr + w),
                    'card': w,
                    'cols': cols
                })
            curr += w

        # 终极校验：输入 141，输出必须是 141
        if len(sorted_cols) != len(cols_to_process):
            raise ValueError(f"维度不匹配！输入 {len(cols_to_process)}, 得到 {len(sorted_cols)}")

        return sorted_cols, hsic_layout, meta_map