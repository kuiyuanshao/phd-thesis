import torch
import torch.nn.functional as F
import os, sys, numpy as np, pandas as pd
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from scipy.linalg import block_diag
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import SIRD_NET
from swag import SWAG
from data_transformer import DataTransformer

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but not supported on MPS.*")


class SIRD:
    def __init__(self, config, data_info, device=None):
        self.config = config or {}
        self.data_info = data_info
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f"Active Device: {self.device}")

        diffusion_cfg = self.config.get("diffusion", {})
        sample_cfg = self.config.get("sample", {})
        train_cfg = self.config.get("train", {})

        self.num_steps = diffusion_cfg.get("num_steps", 30)
        self.sum_scale = torch.tensor(diffusion_cfg.get("sum_scale", 0.01)).to(self.device)
        self.task = diffusion_cfg.get("task", "Res-N")
        self.discrete = diffusion_cfg.get("discrete", "Multinomial")

        self.mi_approx = sample_cfg.get("mi_approx", None)
        self.m = sample_cfg.get("m", 5)
        self.num_train_mods = self.m if self.mi_approx == "BOOTSTRAP" else 1
        self.pmm = sample_cfg.get("pmm", False)
        self.donors = sample_cfg.get("donors", 5)
        self.eval_bs = sample_cfg.get("eval_batch_size", 1024)

        self.epochs = train_cfg.get("epochs", 5000)
        self.bs = train_cfg.get("batch_size", 128)
        self.valid_ratio = train_cfg.get("valid", 0.0)
        self.loss_num = train_cfg.get("loss_num", 1.0)
        self.loss_cat = train_cfg.get("loss_cat", 1.0)

        b = torch.linspace(0, 1, self.num_steps).to(self.device) ** 3
        betas = b / b.sum() * self.sum_scale
        betas_cumsum = betas.cumsum(dim=0).clip(0, 1)

        a = torch.flip(torch.linspace(0, 1, self.num_steps).to(self.device), dims=[0])
        alphas = a / a.sum()
        self.alpha_bars = alphas.cumsum(dim=0).clip(0, 1)

        alphas_cumsum_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=self.alpha_bars[1])
        betas_cumsum_prev = F.pad(betas_cumsum[:-1], (1, 0), value=betas_cumsum[1])
        self.beta_bars = torch.sqrt(betas_cumsum)

        self.post_w_p = F.pad(self.alpha_bars[:-1], (1, 0), value=0.0).to(self.device)
        self.post_w_s = 1.0 - self.post_w_p
        self.post_beta_t = (1.0 - (1.0 - self.alpha_bars) / (1.0 - self.post_w_p + 1e-30)).clamp(1e-5, 1.0 - 1e-5)
        self.log_post_w_p = torch.log(self.post_w_p + 1e-30)
        self.log_post_w_s = torch.log(self.post_w_s + 1e-30)

        posterior_variance = betas * betas_cumsum_prev / betas_cumsum
        posterior_variance[0] = 0
        self.posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = betas_cumsum_prev / betas_cumsum
        self.posterior_mean_coef2 = (betas * alphas_cumsum_prev - betas_cumsum_prev * alphas) / betas_cumsum
        self.posterior_mean_coef3 = betas / betas_cumsum
        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1

        self.model = None
        self.model_list = []
        self.swag_model = None
        self.transformer = None

        self.X_num = None
        self.X_cat = None
        self.A_num = None
        self.A_cat = None
        self.Covariates = None
        self.Q_block = None
        self.dim_info = {}
        self.dim_dict = {}
        self.cat_sizes = []
        self.has_partner_mask = []
        self.num_names = data_info["num_vars"]
        self.p2_vars = data_info["phase2_vars"]

    def fit(self, file_path=None, provided_data=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if provided_data is None:
            df_raw = pd.read_csv(file_path).loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]
        else:
            df_raw = provided_data
        for c in self.data_info.get('num_vars', []):
            if c in df_raw.columns:
                try:
                    df_raw[c] = pd.to_numeric(df_raw[c], errors='raise')
                except (ValueError, TypeError):
                    df_raw[c] = pd.factorize(df_raw[c])[0]

        self.transformer = DataTransformer(self.data_info, self.config).fit(df_raw)
        df_proc = self.transformer.transform()

        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        pair_map = {p2: p1 for p1, p2 in zip(p1_vars, p2_vars)}
        for p2 in p2_vars:
            if p2 in self.data_info['num_vars']:
                p1 = pair_map.get(p2)
                if not p1: continue
                n_p1 = len([c for c in df_proc.columns if c.startswith(f"{p1}_mode_")])
                n_p2 = len([c for c in df_proc.columns if c.startswith(f"{p2}_mode_")])
                if n_p1 != n_p2: print(f"Mode Mismatch Identified: {p1} ({n_p1} modes) vs {p2} ({n_p2} modes)")

        self.raw_df = df_raw
        schema = self.transformer.get_sird_schema()
        self._tensorize_data(df_proc, schema)

        w_col = self.data_info.get('weight_var')
        self._global_weights = torch.from_numpy(
            df_raw[w_col].fillna(0).values if w_col in df_raw else np.ones(len(df_raw))).float().to(self.device)
        p2_check = next((v['name'] for v in schema if 'p2' in v['type'] and '_mode' not in v['name']), None)
        if p2_check and p2_check in df_raw:
            self.valid_rows = np.where(df_raw[p2_check].notna().values)[0]
        else:
            self.valid_rows = np.arange(len(df_raw))

        print(f"Global Valid Rows: {len(self.valid_rows)}")
        do_validation = self.valid_ratio > 0.0
        if do_validation:
            num_val = int(len(self.valid_rows) * self.valid_ratio)
            shuffled_rows = np.random.permutation(self.valid_rows)
            val_rows = shuffled_rows[:num_val]
            train_rows = shuffled_rows[num_val:]
            print(f"Validation enabled: {len(train_rows)} train rows, {len(val_rows)} val rows.")
        else:
            train_rows = self.valid_rows
            val_rows = []

        for k in range(self.num_train_mods):
            print(f"\nTraining Model {k + 1}/{self.num_train_mods}...")

            idx = np.random.choice(train_rows, len(train_rows),
                                   replace=True) if self.mi_approx == "BOOTSTRAP" else train_rows
            t_idx = torch.from_numpy(idx).long().to(self.device)

            loader = DataLoader(TensorDataset(t_idx), batch_size=self.bs,
                                sampler=WeightedRandomSampler(self._global_weights[t_idx], len(t_idx) * 4, True),
                                drop_last=False)

            if do_validation:
                v_idx = torch.from_numpy(val_rows).long().to(self.device)
                val_loader = DataLoader(TensorDataset(v_idx), batch_size=self.bs, shuffle=False)

            self.model = SIRD_NET(self.config, self.device, self.dim_info).to(self.device)
            self.model.train()

            if self.mi_approx == "SWAG":
                swa_start_iter = int(self.epochs * 0.50)
                sgd_cfg = self.config.get('train', {}).get('SGD', {})
                sgd_lr = sgd_cfg.get('lr', 1.0e-2)
                optim = SGD(
                    self.model.parameters(),
                    lr=sgd_lr,
                    momentum=sgd_cfg.get('momentum', 0.9),
                    weight_decay=sgd_cfg.get('weight_decay', 1.0e-6)
                )
                scheduler = CosineAnnealingLR(optim, T_max=swa_start_iter, eta_min=sgd_lr * 0.25)
                self.swag_model = SWAG(self.model, max_num_models=50).to(self.device)
            else:
                adam_cfg = self.config.get('train', {}).get('Adam', {})
                optim = AdamW(
                    self.model.parameters(),
                    lr=adam_cfg.get('lr', 2.0e-4),
                    weight_decay=adam_cfg.get('weight_decay', 1.0e-6)
                )

            pbar = tqdm(range(self.epochs), desc=f"Training M{k + 1}", leave=False)
            iter_loader = iter(loader)

            best_loss = float('inf')
            steps_no_improve = 0
            patience = self.config.get('train', {}).get('patience', 2000)
            running_train_loss = 0.0

            for step in pbar:
                try:
                    batch_idx = next(iter_loader)[0]
                except StopIteration:
                    iter_loader = iter(loader)
                    batch_idx = next(iter_loader)[0]

                optim.zero_grad()
                loss, n_loss, c_loss = self.calc_loss(batch_idx)
                loss.backward()

                running_train_loss += loss.item()
                if self.mi_approx == "SWAG":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()

                if (step + 1) % 50 == 0:
                    avg_train_loss = running_train_loss / 50
                    running_train_loss = 0.0

                    postfix_dict = {
                        "continuous_loss": f"{n_loss.item():.4f}",
                        "discrete_loss": f"{c_loss.item():.4f}"
                    }
                    if do_validation:
                        self.model.eval()
                        mse_loss = 0.0
                        cat_loss = 0.0
                        with torch.no_grad():
                            for v_batch in val_loader:
                                v_batch_idx = v_batch[0]
                                _, mse_loss_val, cat_loss_val = self.calc_loss(v_batch_idx)
                                mse_loss += mse_loss_val.item()
                                cat_loss += cat_loss_val.item()

                        avg_mse = mse_loss / len(val_loader)
                        avg_cat = cat_loss / len(val_loader)
                        current_metric = avg_mse + avg_cat
                        postfix_dict["val_cont_loss"] = f"{avg_mse:.4f}"
                        postfix_dict["val_disc_loss"] = f"{avg_cat:.4f}"
                        self.model.train()
                    else:
                        current_metric = avg_train_loss

                    if current_metric < best_loss:
                        best_loss = current_metric
                        steps_no_improve = 0
                    else:
                        steps_no_improve += 50
                    pbar.set_postfix(**postfix_dict)

                    if steps_no_improve >= patience:
                        target_name = "Validation" if do_validation else "Training"
                        print(f"\nEarly stopping triggered. {target_name} loss hasn't improved for {patience} steps.")
                        break

                if self.mi_approx == "SWAG":
                    if step < swa_start_iter:
                        scheduler.step()
                    else:
                        if step == swa_start_iter:
                            for param_group in optim.param_groups:
                                param_group['lr'] = sgd_lr * 0.25
                        if (step - swa_start_iter) % 50 == 0:
                            self.swag_model.collect_model(self.model)
            self.model_list.append(self.model)
        return self

    def _tensorize_data(self, df, schema):
        is_bits = self.discrete == 'AnalogBits'

        p2_nums = [v for v in schema if 'numeric_p2' in v['type']]
        p2_cats = [v for v in schema if 'categorical_p2' in v['type']]

        x_num_list, x_cat_list = [], []
        p1_num_list, p1_cat_list = [], []
        self.cat_sizes = []
        self.has_partner_mask = []
        q_blocks = []

        for v in p2_nums:
            x_num_list.append(df[v['name']].values.reshape(-1, 1))
            partner = v['pair_partner']
            if partner:
                p1_num_list.append(df[partner].values.reshape(-1, 1))
            else:
                p1_num_list.append(np.zeros((len(df), 1)))

        for v in p2_cats:
            k = v['num_classes']
            if is_bits:
                x_num_list.append(df[df.columns[v['start_idx']:v['end_idx']]].values)
                partner = v['pair_partner']
                if partner:
                    p1_cols = [c for c in df.columns if c.startswith(partner + "_")]
                    p1_num_list.append(df[p1_cols].values)
                else:
                    p1_num_list.append(np.zeros((len(df), k)))
            else:
                self.cat_sizes.append(k)
                x_cat_list.append(df[df.columns[v['start_idx']:v['end_idx']]].values)
                partner = v['pair_partner']
                if partner:
                    p1_cols = [c for c in df.columns if c.startswith(partner + "_")]
                    p1_cat_list.append(df[p1_cols].values)
                    self.has_partner_mask.append(True)
                    if v['name'] in self.transformer.Q_matrices:
                        q_blocks.append(self.transformer.Q_matrices[v['name']].numpy())
                    else:
                        q_blocks.append(np.eye(k))
                else:
                    p1_cat_list.append(np.zeros((len(df), k)))
                    self.has_partner_mask.append(False)
                    q_blocks.append(np.eye(k))

        np_x_num = np.hstack(x_num_list) if x_num_list else np.empty((len(df), 0))
        np_x_cat = np.hstack(x_cat_list) if x_cat_list else np.empty((len(df), 0))
        np_p1_num = np.hstack(p1_num_list) if p1_num_list else np.empty((len(df), 0))
        np_p1_cat = np.hstack(p1_cat_list) if p1_cat_list else np.empty((len(df), 0))

        aux_num_list, aux_cat_list = [], []
        for v in schema:
            if 'aux' in v['type']:
                if 'numeric' in v['type']:
                    aux_num_list.append(df[v['name']].values.reshape(-1, 1))
                else:
                    if is_bits:
                        aux_num_list.append(df[df.columns[v['start_idx']:v['end_idx']]].values)
                    else:
                        aux_cat_list.append(df[df.columns[v['start_idx']:v['end_idx']]].values)

        np_aux_num = np.hstack(aux_num_list) if aux_num_list else np.empty((len(df), 0))
        np_aux_cat = np.hstack(aux_cat_list) if aux_cat_list else np.empty((len(df), 0))

        cov_list = []
        if np_aux_num.shape[1] > 0: cov_list.append(np_aux_num)
        if np_aux_cat.shape[1] > 0: cov_list.append(np_aux_cat)

        np_cov = np.hstack(cov_list) if cov_list else np.empty((len(df), 0))

        self.X_num = torch.tensor(np_x_num, dtype=torch.float32).to(self.device)
        self.X_cat = torch.tensor(np_x_cat, dtype=torch.float32).to(self.device)
        self.A_num = torch.tensor(np_p1_num, dtype=torch.float32).to(self.device)
        self.A_cat = torch.tensor(np_p1_cat, dtype=torch.float32).to(self.device)
        self.Covariates = torch.tensor(np_cov, dtype=torch.float32).to(self.device)

        self.dim_dict = {
            'p1_n': np_p1_num.shape[1],
            'aux_n': np_aux_num.shape[1],
            'p1_c': np_p1_cat.shape[1],
        }

        self.dim_info = {
            'input_dim': self.X_num.shape[1] + self.X_cat.shape[1],
            'cond_dim': self.Covariates.shape[1],
            'out_num_dim': self.X_num.shape[1],
            'out_cat_dim': self.X_cat.shape[1],
        }

        self.Q_block = torch.tensor(block_diag(*q_blocks), dtype=torch.float32).to(
            self.device) if q_blocks else torch.empty(0).to(self.device)

    def calc_loss(self, batch_idx):
        B = len(batch_idx)
        b_x_num = self.X_num[batch_idx]
        b_x_cat = self.X_cat[batch_idx]
        b_a_num = self.A_num[batch_idx]
        b_a_cat = self.A_cat[batch_idx]
        b_cond = self.Covariates[batch_idx]

        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        a_bar = self.alpha_bars[t].view(B, 1)
        b_bar = self.beta_bars[t].view(B, 1)

        eps_num_p2 = torch.randn_like(b_x_num)
        true_res_p2 = b_a_num - b_x_num
        x_t_num = b_x_num + a_bar * true_res_p2 + b_bar * eps_num_p2

        if self.Q_block.numel() > 0:
            pi_prior_raw = b_a_cat @ self.Q_block
        else:
            pi_prior_raw = torch.ones_like(b_x_cat)

        pi_prior_list = []
        for i, p in enumerate(torch.split(pi_prior_raw, self.cat_sizes, dim=1)):
            if self.has_partner_mask[i]:
                pi_prior_list.append(p)
            else:
                pi_prior_list.append(torch.ones_like(p) / p.shape[1])
        pi_prior = torch.cat(pi_prior_list, dim=1) if pi_prior_list else torch.empty_like(pi_prior_raw)

        log_probs = torch.logaddexp(
            torch.log((1 - a_bar) + 1e-30) + torch.log(b_x_cat + 1e-30),
            torch.log(a_bar + 1e-30) + torch.log(pi_prior + 1e-30),
        )
        gumbel = -torch.log(-torch.log(torch.rand_like(log_probs) + 1e-30) + 1e-30)
        noisy_indices = (log_probs + gumbel)
        x_t_cat_list = []
        curr_idx = 0
        for k in self.cat_sizes:
            chunk = noisy_indices[:, curr_idx: curr_idx + k]
            x_t_cat_list.append(F.one_hot(chunk.argmax(dim=1), k).float())
            curr_idx += k

        x_t_cat = torch.cat(x_t_cat_list, dim=1) if x_t_cat_list else torch.empty((B, 0), device=self.device)

        x_curr = torch.cat([x_t_num, x_t_cat], dim=1)
        out_num, out_cat = self.model(x_curr, b_cond, t)

        loss_n = torch.tensor(0.0, device=self.device)
        loss_c = torch.tensor(0.0, device=self.device)

        if out_num is not None:
            dim_total = out_num.shape[1] // 2 if self.task == "Res-N" else out_num.shape[1]
            p2_n_dim = self.X_num.shape[1]

            if self.task == "Res-N":
                p_res_p2 = out_num[:, :p2_n_dim]
                p_eps_p2 = out_num[:, dim_total: dim_total + p2_n_dim]
                loss_n += self.loss_num * F.mse_loss(p_res_p2, true_res_p2) + F.mse_loss(p_eps_p2, eps_num_p2)
            elif self.task == "Res":
                loss_n += self.loss_num * F.mse_loss(out_num[:, :p2_n_dim], true_res_p2)
            else:
                loss_n += self.loss_num * F.mse_loss(out_num[:, :p2_n_dim], eps_num_p2)

        if out_cat is not None and self.discrete == "Multinomial":
            for logits, xt, x0, p in zip(
                torch.split(out_cat, self.cat_sizes, dim=1),
                torch.split(x_t_cat, self.cat_sizes, dim=1),
                torch.split(b_x_cat, self.cat_sizes, dim=1),
                torch.split(pi_prior, self.cat_sizes, dim=1),
            ):
                pred_probs = F.softmax(logits, dim=-1)
                log_true_post = self._compute_posterior_vec(xt, x0, p, t)
                log_model_post = self._compute_posterior_vec(xt, pred_probs, p, t)
                loss_c += self.loss_cat * F.kl_div(log_model_post, log_true_post, reduction='batchmean',
                                                   log_target=True)

        total_loss = loss_n + loss_c
        return total_loss, loss_n, loss_c

    def _reverse_diffusion(self, model_to_use, batch_size=2048):
        N = self.X_num.shape[0]
        collected_outputs = {'num': [], 'cat': []}
        p2_n_dim = self.X_num.shape[1]

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                B = end - start

                b_cond = self.Covariates[start:end]
                b_a_num = self.A_num[start:end]
                b_a_cat_chunk = self.A_cat[start:end]

                x_t_num = b_a_num + torch.sqrt(self.sum_scale) * torch.randn_like(b_a_num)

                if self.Q_block.numel() > 0:
                    pi_prior_p2_raw = b_a_cat_chunk @ self.Q_block
                else:
                    pi_prior_p2_raw = torch.ones((B, sum(self.cat_sizes)), device=self.device)

                pi_prior_list = []
                for idx, p in enumerate(torch.split(pi_prior_p2_raw, self.cat_sizes, dim=1)):
                    if self.has_partner_mask[idx]:
                        pi_prior_list.append(p)
                    else:
                        pi_prior_list.append(torch.ones_like(p) / p.shape[1])
                pi_prior = torch.cat(pi_prior_list, dim=1) if pi_prior_list else torch.empty_like(pi_prior_p2_raw)

                x_t_cat_list = []
                prior_split = torch.split(pi_prior, self.cat_sizes, dim=1)
                for j, (k, p) in enumerate(zip(self.cat_sizes, prior_split)):
                    if self.has_partner_mask[j]:
                        x_t_cat_list.append(F.one_hot(p.argmax(dim=1), k).float())
                    else:
                        log_p = torch.log(p + 1e-30)
                        g = -torch.log(-torch.log(torch.rand_like(log_p) + 1e-30) + 1e-30)
                        x_t_cat_list.append(F.one_hot((log_p + g).argmax(dim=1), k).float())
                x_t_cat = torch.cat(x_t_cat_list, dim=1) if x_t_cat_list else torch.empty((B, 0), device=self.device)

                for i in reversed(range(self.num_steps)):
                    tt = torch.full((B,), i, device=self.device).long()
                    x_curr = torch.cat([x_t_num, x_t_cat], dim=1)
                    out_num, out_cat = model_to_use(x_curr, b_cond, tt)

                    if out_num is not None:
                        if self.task == "Res-N":
                            dim_total = out_num.shape[1] // 2
                            res, eps = out_num[:, :dim_total], out_num[:, dim_total:]
                        elif self.task == "Res":
                            res, eps = out_num, torch.zeros_like(out_num)
                        else:
                            res, eps = torch.zeros_like(out_num), out_num

                        res_p2, eps_p2 = res[:, :p2_n_dim], eps[:, :p2_n_dim]
                        x_t_p2 = x_t_num[:, :p2_n_dim]

                        if self.task == "Res-N":
                            x_start_p2 = x_t_p2 - self.alpha_bars[i] * res_p2 - self.beta_bars[i] * eps_p2
                            res_use_p2 = res_p2
                        elif self.task == "Res":
                            eps_p2 = (x_t_p2 - b_a_num - (self.alpha_bars[i] - 1.0) * res_p2) / (
                                        self.beta_bars[i] + 1e-8)
                            x_start_p2 = x_t_p2 - self.alpha_bars[i] * res_p2 - self.beta_bars[i] * eps_p2
                            res_use_p2 = res_p2
                        else:
                            x_start_p2 = (x_t_p2 - self.alpha_bars[i] * b_a_num - self.beta_bars[i] * eps_p2) / (
                                        1.0 - self.alpha_bars[i] + 1e-8)
                            x_start_p2 = x_start_p2.clamp(-5, 5)
                            res_use_p2 = b_a_num - x_start_p2

                        x_start_p2 = x_start_p2.clamp(-5, 5)
                        post_mean_p2 = (self.posterior_mean_coef1[i] * x_t_p2 +
                                        self.posterior_mean_coef2[i] * res_use_p2 +
                                        self.posterior_mean_coef3[i] * x_start_p2)
                        noise_p2 = torch.randn_like(x_t_p2) if i > 0 else 0.0
                        x_t_num = post_mean_p2 + (0.5 * self.posterior_log_variance[i]).exp() * noise_p2

                    if out_cat is not None:
                        logits_list = torch.split(out_cat, self.cat_sizes, dim=1)
                        xt_list = torch.split(x_t_cat, self.cat_sizes, dim=1)
                        new_cat_list = []
                        for j in range(len(self.cat_sizes)):
                            logits, xt, p = logits_list[j], xt_list[j], prior_split[j]
                            pred_probs = F.softmax(logits, dim=-1)
                            log_post = self._compute_posterior_vec(xt, pred_probs, p, tt)
                            if i > 0:
                                g = -torch.log(-torch.log(torch.rand_like(log_post) + 1e-30) + 1e-30)
                                sample = log_post + g
                            else:
                                sample = log_post
                            new_cat_list.append(F.one_hot(sample.argmax(dim=1), self.cat_sizes[j]).float())
                        x_t_cat = torch.cat(new_cat_list, dim=1)

                collected_outputs['num'].append(x_t_num[:, :p2_n_dim].cpu())

                curr_c = 0
                p2_cat_chunks = []
                for k in self.cat_sizes:
                    p2_cat_chunks.append(x_t_cat[:, curr_c: curr_c + k])
                    curr_c += k
                if p2_cat_chunks:
                    collected_outputs['cat'].append(torch.cat(p2_cat_chunks, dim=1).cpu())

        final_dict = {}
        full_num = torch.cat(collected_outputs['num'], dim=0).to(self.device) if collected_outputs['num'] else None
        full_cat = torch.cat(collected_outputs['cat'], dim=0).to(self.device) if collected_outputs['cat'] else None

        schema = self.transformer.get_sird_schema()
        curr_n, curr_c = 0, 0
        p2_nums = [v for v in schema if 'numeric_p2' in v['type']]
        p2_cats = [v for v in schema if 'categorical_p2' in v['type']]
        is_bits = self.discrete == 'AnalogBits'

        if full_num is not None:
            for v in p2_nums:
                final_dict[v['name']] = full_num[:, curr_n:curr_n + 1]
                curr_n += 1
            if is_bits:
                for v in p2_cats:
                    k = v['num_classes']
                    final_dict[v['name']] = full_num[:, curr_n:curr_n + k]
                    curr_n += k

        if full_cat is not None and not is_bits:
            for v in p2_cats:
                k = v['num_classes']
                final_dict[v['name']] = full_cat[:, curr_c:curr_c + k]
                curr_c += k

        return final_dict

    def impute(self, m=None, save_path=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        pd.set_option('future.no_silent_downcasting', True)
        m_s = m if m else self.m
        eval_bs = self.eval_bs

        if self.mi_approx == "SWAG":
            if self.swag_model is None or self.swag_model.n_models.item() == 0:
                print("Warning: No SWAG collected. Using base model.")
            else:
                print(f"Using SWAG sampling ({self.swag_model.n_models.item()} models)")

        all_imputed_dfs = []
        pbar = tqdm(total=m_s, desc="Imputation Rounds", file=sys.stdout)

        for samp_i in range(1, m_s + 1):
            if self.mi_approx == "SWAG" and self.swag_model.n_models.item() > 1:
                model_to_use = self.swag_model.sample(scale=1, cov=True)
                model_to_use.eval()
            elif self.mi_approx == "DROPOUT":
                model_to_use = self.model_list[0]
                model_to_use.eval()
                for modu in model_to_use.modules():
                    if modu.__class__.__name__.startswith('Dropout'): modu.train()
            elif self.mi_approx == "BOOTSTRAP":
                model_to_use = self.model_list[samp_i - 1]
                model_to_use.eval()
            else:
                model_to_use = self.model_list[0]
                model_to_use.eval()

            x_0_dict = self._reverse_diffusion(model_to_use, eval_bs)
            gen_df_dict = {}
            for v in self.transformer.get_sird_schema():
                if 'p2' in v['type']:
                    val = x_0_dict[v['name']].cpu().numpy()
                    if 'numeric' in v['type']:
                        gen_df_dict[v['name']] = val.flatten()
                    elif 'categorical' in v['type']:
                        start, num_cls = v['start_idx'], v['num_classes']
                        cols = self.transformer.generated_columns[start: start + num_cls]
                        for k, col_name in enumerate(cols): gen_df_dict[col_name] = val[:, k]

            df_gen_p2 = pd.DataFrame(gen_df_dict)
            df_denorm = self.transformer.inverse_transform(df_gen_p2)
            df_f = self.raw_df.copy()
            df_denorm.index = df_f.index

            for col in df_denorm.columns:
                if col in df_f.columns:
                    df_denorm[col] = df_denorm[col].astype(df_f[col].dtype)

            for c in self.p2_vars:
                if self.pmm:
                    obs_idx = df_f[df_f[c].notna()].index
                    miss_idx = df_f[df_f[c].isna()].index
                    if c in self.num_names:
                        y_obs = df_f.loc[obs_idx, c].values
                        yhat_obs = df_denorm.loc[obs_idx, c].values
                        yhat_miss = df_denorm.loc[miss_idx, c].values
                        imputed_series = pd.Series(
                            self.pmm(yhat_obs, yhat_miss, y_obs, k=self.donors).flatten(),
                            index=miss_idx)
                        df_f.loc[miss_idx, c] = imputed_series
                    else:
                        df_f[c] = df_f[c].fillna(df_denorm[c])
                else:
                    df_f[c] = df_f[c].fillna(df_denorm[c])

            df_f.insert(0, "imp_id", samp_i)
            all_imputed_dfs.append(df_f)
            pbar.update(1)

        pbar.close()
        if save_path is not None:
            pd.concat(all_imputed_dfs, ignore_index=True).to_parquet(save_path, index=False)
            print(f"Saved stacked imputations to: {save_path}")
        return all_imputed_dfs

    def _compute_posterior_vec(self, x_t, x_start, x_proxy, t):
        log_w_s = self.log_post_w_s[t].unsqueeze(1)
        log_w_p = self.log_post_w_p[t].unsqueeze(1)
        beta_t = self.post_beta_t[t].unsqueeze(1)
        log_prior = torch.logaddexp(
            log_w_s + torch.log(x_start + 1e-30),
            log_w_p + torch.log(x_proxy + 1e-30)
        )
        p_proxy_at_xt = (x_t * x_proxy).sum(dim=1, keepdim=True)

        log_lik_noise = torch.log(beta_t * p_proxy_at_xt + 1e-30)
        log_lik_stay = torch.log((1.0 - beta_t) + beta_t * p_proxy_at_xt + 1e-30)
        log_likelihood = x_t * log_lik_stay + (1.0 - x_t) * log_lik_noise

        log_unnorm = log_prior + log_likelihood
        return log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)

    def pmm(self, yhat_obs, yhat_miss, y_obs, k=5):
        d = np.array(yhat_obs).reshape(-1, 1)
        t = np.array(yhat_miss).reshape(-1, 1)
        v = np.array(y_obs).reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(d)
        _, neighbor_indices = nbrs.kneighbors(t)
        rand_cols = np.random.randint(0, k, size=neighbor_indices.shape[0])
        final_indices = neighbor_indices[np.arange(neighbor_indices.shape[0]), rand_cols]
        return v[final_indices]
