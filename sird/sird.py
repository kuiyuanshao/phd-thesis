import torch
import torch.nn.functional as F
import os, sys, numpy as np, pandas as pd
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from scipy.linalg import block_diag
from sklearn.neighbors import NearestNeighbors
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import SIRD_NET
from swag import SWAG
from data_transformer import DataTransformer
import warnings
from sklearn.exceptions import ConvergenceWarning
from utils import calc_HSIC_loss

warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but not supported on MPS.*")


class SIRD:
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

        self.num_steps = config["diffusion"]["num_steps"]
        self.sum_scale = torch.tensor(config["diffusion"]["sum_scale"]).to(self.device)
        self.task = config["diffusion"]["task"]
        self.discrete = config["diffusion"]["discrete"]

        if self.task == "Res-F":
            self.betas = torch.linspace(1e-4, 0.02, self.num_steps).to(self.device)
            self.alphas = 1.0 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
            self.alpha_bars_t_minus_1 = torch.roll(self.alpha_bars, shifts=1, dims=0)
            self.alpha_bars_t_minus_1[0] = self.alpha_bars_t_minus_1[1]
            self.betas_bars = (1 - self.alpha_bars_t_minus_1) / (1 - self.alpha_bars) * self.betas
            abs_dist = torch.abs(torch.sqrt(self.alpha_bars) - 0.5)
            self.T_acc = abs_dist.argmin().item() + 1

        else:
            b = torch.linspace(0, 1, self.num_steps).to(self.device) ** 3
            betas = b / b.sum() * self.sum_scale
            betas_cumsum = betas.cumsum(dim=0).clip(0, 1)

            alpha_schedule = self.config['diffusion'].get('alpha_schedule', 'exp')
            a = torch.flip(torch.linspace(0, 1, self.num_steps).to(self.device), dims=[0])
            alphas = a / a.sum()
            self.alpha_bars = alphas.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(self.alpha_bars[:-1], (1, 0),
                                       value=self.alpha_bars[1])  # Padding logic from original

            betas_cumsum_prev = F.pad(betas_cumsum[:-1], (1, 0), value=betas_cumsum[1])
            self.beta_bars = torch.sqrt(betas_cumsum)

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

        self.X_num = None;
        self.X_cat = None
        self.Cond = None;
        self.Q_block = None
        self.dim_info = {};
        self.cat_sizes = [];
        self.has_partner_mask = []
        self.num_names = data_info["num_vars"]
        self.p2_vars = data_info["phase2_vars"]
        self.Q_t = None

    def fit(self, file_path=None, provided_data=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if provided_data is None:
            df_raw = pd.read_csv(file_path).loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]
        else:
            df_raw = provided_data
        for c in self.data_info.get('num_vars', []):
            if c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

        self.transformer = DataTransformer(self.data_info, self.config).fit(df_raw)
        df_proc = self.transformer.transform(df_raw)

        print("\n[DEBUG] Checking Mode Count Consistency...")
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        pair_map = {p2: p1 for p1, p2 in zip(p1_vars, p2_vars)}
        for p2 in p2_vars:
            if p2 in self.data_info['num_vars']:
                p1 = pair_map.get(p2)
                if not p1: continue
                n_p1 = len([c for c in df_proc.columns if c.startswith(f"{p1}_mode_")])
                n_p2 = len([c for c in df_proc.columns if c.startswith(f"{p2}_mode_")])
                if n_p1 != n_p2: print(f"!!! MODE MISMATCH: {p1} ({n_p1} modes) vs {p2} ({n_p2} modes)")
        print("[DEBUG] Complete.\n")

        self.raw_df = df_raw
        schema = self.transformer.get_sird_schema()
        self._tensorize_data(df_proc, schema)

        w_col = self.data_info.get('weight_var')
        self._global_weights = torch.from_numpy(
            df_raw[w_col].fillna(0).values if w_col in df_raw else np.ones(len(df_raw))).double().to(self.device)
        p2_check = next((v['name'] for v in schema if 'p2' in v['type'] and '_mode' not in v['name']), None)
        if p2_check and p2_check in df_raw:
            self.valid_rows = np.where(df_raw[p2_check].notna().values)[0]
        else:
            self.valid_rows = np.arange(len(df_raw))

        print(f"[Fit] Global Valid Rows (Observed Phase 2): {len(self.valid_rows)}")
        valid_ratio = self.config.get("train", {}).get("valid", 0.0)
        do_validation = valid_ratio > 0.0
        if do_validation:
            num_val = int(len(self.valid_rows) * valid_ratio)
            shuffled_rows = np.random.permutation(self.valid_rows)
            val_rows = shuffled_rows[:num_val]
            train_rows = shuffled_rows[num_val:]
            print(f"[Fit] Validation enabled: {len(train_rows)} train rows, {len(val_rows)} val rows.")
        else:
            train_rows = self.valid_rows
            val_rows = []
        epochs = self.config["train"]["epochs"]
        bs = self.config["train"]["batch_size"]
        mi_approx = self.config["sample"]["mi_approx"]
        num_train_mods = self.config["sample"]["m"] if mi_approx == "BOOTSTRAP" else 1

        for k in range(num_train_mods):
            print(f"\n[SIRD] Training Model {k + 1}/{num_train_mods}...")

            idx = np.random.choice(train_rows, len(train_rows),
                                   replace=True) if mi_approx == "BOOTSTRAP" else train_rows
            t_idx = torch.from_numpy(idx).long().to(self.device)

            loader = DataLoader(TensorDataset(t_idx), batch_size=bs,
                                sampler=WeightedRandomSampler(self._global_weights[t_idx], len(t_idx) * 4, True),
                                drop_last=False)
            if do_validation:
                v_idx = torch.from_numpy(val_rows).long().to(self.device)
                val_loader = DataLoader(TensorDataset(v_idx), batch_size=bs, shuffle=False)

            self.model = SIRD_NET(self.config, self.device, self.dim_info).to(self.device)
            self.model.train()

            if mi_approx == "SWAG":
                swa_start_iter = int(epochs * 0.50)
                optim = SGD(self.model.parameters(), lr=self.config['train']['SGD']['lr'], momentum=self.config['train']['SGD']['momentum'])
                scheduler = CosineAnnealingLR(optim, T_max=swa_start_iter, eta_min=self.config['train']['SGD']['lr'] * 0.25)
                self.swag_model = SWAG(self.model, max_num_models=50).to(self.device)
            else:
                optim = Adam(self.model.parameters(), lr=self.config['train']['Adam']['lr'],
                             weight_decay=self.config['train']['Adam']["weight_decay"])

            pbar = tqdm(range(epochs), desc=f"Training M{k + 1}", leave=False)
            iter_loader = iter(loader)

            best_loss = float('inf')
            steps_no_improve = 0
            patience = self.config['train']['patience']
            running_train_loss = 0.0

            for step in pbar:
                try:
                    batch_idx = next(iter_loader)[0]
                except:
                    iter_loader = iter(loader); batch_idx = next(iter_loader)[0]

                optim.zero_grad()
                loss, n_loss, c_loss, hsic_loss = self.calc_loss(batch_idx)
                loss.backward()

                running_train_loss += loss.item()
                if mi_approx == "SWAG":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()

                if step % 50 == 0:
                    avg_train_loss = running_train_loss / (50 if step > 0 else 1)
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
                                _, mse_loss, cat_loss, v_hsic = self.calc_loss(v_batch_idx)
                                mse_loss += mse_loss.item()
                                cat_loss += cat_loss.item()

                        avg_mse = mse_loss / len(val_loader)
                        avg_cat = cat_loss / len(val_loader)
                        avg_hsic = v_hsic / len(val_loader)
                        current_metric = avg_mse + avg_cat + avg_hsic
                        postfix_dict["val_cont_loss"] = f"{avg_mse:.4f}"
                        postfix_dict["val_disc_loss"] = f"{avg_cat:.4f}"
                        self.model.train()
                    else:
                        current_metric = avg_train_loss
                    if self.config['model']['loss_hsic'] > 0:
                        postfix_dict["val_disc_loss"] = f"{hsic_loss:.4f}"

                    if current_metric < best_loss:
                        best_loss = current_metric
                        steps_no_improve = 0
                    else:
                        steps_no_improve += 50
                    pbar.set_postfix(**postfix_dict)

                    if steps_no_improve >= patience:
                        target_name = "Validation" if do_validation else "Training"
                        print(f"\nEarly stopping triggered! {target_name} loss hasn't improved for {patience} steps.")
                        break

                if mi_approx == "SWAG":
                    if step < swa_start_iter:
                        scheduler.step()
                    else:
                        for param_group in optim.param_groups: param_group['lr'] = self.config['train']['SGD']['lr'] * 0.25
                        if (step - swa_start_iter) % 50 == 0:
                            self.swag_model.collect_model(self.model)
            self.model_list.append(self.model)
        return self

    def _tensorize_data(self, df, schema):
        from torch_frame.data.stats import StatType
        x_num_list, x_cat_list = [], []
        p1_num_list, aux_num_list = [], []
        p1_cat_list, aux_cat_list = [], []
        self.cat_sizes = []  # X (Target) categorical sizes
        self.cond_cat_sizes = []  # Cond categorical sizes
        self.has_partner_mask = []
        q_blocks = []

        p2_nums = [v for v in schema if 'numeric_p2' in v['type']]
        for v in p2_nums:
            x_num_list.append(df[v['name']].values.reshape(-1, 1))
            partner = v['pair_partner']
            if partner:
                p1_num_list.append(df[partner].values.reshape(-1, 1))
            else:
                p1_num_list.append(np.zeros((len(df), 1)))

        p2_cats = [v for v in schema if 'categorical_p2' in v['type']]
        p1_cat_widths = []  # Track width of each one-hot P1 var for slicing later

        for v in p2_cats:
            k = v['num_classes']
            self.cat_sizes.append(k)
            x_cat_list.append(df[df.columns[v['start_idx']:v['end_idx']]].values)

            # Partner P1
            partner = v['pair_partner']
            if partner:
                p1_cols = [c for c in df.columns if c.startswith(partner + "_")]
                val = df[p1_cols].values
                p1_cat_list.append(val)
                self.cond_cat_sizes.append(k)
                p1_cat_widths.append(val.shape[1])  # Should be equal to k
                self.has_partner_mask.append(True)

                # Q Matrix
                if v['name'] in self.transformer.Q_matrices:
                    q_blocks.append(self.transformer.Q_matrices[v['name']].numpy())
                else:
                    q_blocks.append(np.eye(k))
            else:
                # No partner
                val = np.zeros((len(df), k))
                p1_cat_list.append(val)
                self.cond_cat_sizes.append(k)
                p1_cat_widths.append(k)
                self.has_partner_mask.append(False)
                q_blocks.append(np.eye(k))

        for v in schema:
            if 'aux' in v['type']:
                if 'numeric' in v['type']:
                    aux_num_list.append(df[v['name']].values.reshape(-1, 1))
                else:
                    k = v['num_classes']
                    aux_cat_list.append(df[df.columns[v['start_idx']:v['end_idx']]].values)
                    self.cond_cat_sizes.append(k)

        self.X_num = torch.tensor(np.hstack(x_num_list) if x_num_list else np.empty((len(df), 0)),
                                  dtype=torch.float32).to(self.device)
        self.X_cat = torch.tensor(np.hstack(x_cat_list) if x_cat_list else np.empty((len(df), 0)),
                                  dtype=torch.float32).to(self.device)

        np_p1_num = np.hstack(p1_num_list) if p1_num_list else np.empty((len(df), 0))
        np_aux_num = np.hstack(aux_num_list) if aux_num_list else np.empty((len(df), 0))
        np_p1_cat = np.hstack(p1_cat_list) if p1_cat_list else np.empty((len(df), 0))
        np_aux_cat = np.hstack(aux_cat_list) if aux_cat_list else np.empty((len(df), 0))

        self.Cond_Num = np.hstack([np_p1_num, np_aux_num]) if (np_p1_num.size or np_aux_num.size) else np.empty(
            (len(df), 0))
        self.Cond_Cat = np.hstack([np_p1_cat, np_aux_cat]) if (np_p1_cat.size or np_aux_cat.size) else np.empty(
            (len(df), 0))

        self.Cond = torch.tensor(np.hstack([self.Cond_Num, self.Cond_Cat]), dtype=torch.float32).to(self.device)

        p1_num_dim = np_p1_num.shape[1]
        p1_cat_start_idx = self.Cond_Num.shape[1]
        p1_cat_dim = np_p1_cat.shape[1]

        self.dim_info = {
            'input_dim': self.X_num.shape[1] + self.X_cat.shape[1],
            'cond_dim': self.Cond.shape[1],
            'out_num_dim': self.X_num.shape[1],
            'out_cat_dim': self.X_cat.shape[1],
            # Tokenizer Slicing Info
            'x_num_count': self.X_num.shape[1],
            'cond_num_count': self.Cond_Num.shape[1],
            # Residual Slicing Info (For Reverse Diffusion)
            'p1_num_idx': (0, p1_num_dim),
            'p1_cat_idx': (p1_cat_start_idx, p1_cat_start_idx + p1_cat_dim)
        }
        x_num_stats = [
            {StatType.MEAN: m.item(), StatType.STD: s.item()}
            for m, s in zip(self.X_num.mean(0), self.X_num.std(0))
        ]
        x_cat_stats = [
            {StatType.COUNT: [[None] * int(k)]}
            for k in self.cat_sizes
        ]
        cond_num_t = torch.tensor(self.Cond_Num, dtype=torch.float32)
        cond_num_stats = [
            {StatType.MEAN: m.item(), StatType.STD: s.item()}
            for m, s in zip(cond_num_t.mean(0), cond_num_t.std(0))
        ] if cond_num_t.shape[1] > 0 else []
        cond_cat_stats = [
            {StatType.COUNT: [[None] * int(k)]}
            for k in self.cond_cat_sizes
        ]
        self.dim_info['pt_frame_stats'] = {
            'x_num': x_num_stats,
            'x_cat': x_cat_stats,
            'cond_num': cond_num_stats,
            'cond_cat': cond_cat_stats
        }
        self.dim_info['x_cat_sizes'] = self.cat_sizes
        self.dim_info['cond_cat_sizes'] = self.cond_cat_sizes

        self.P1_num = torch.tensor(np_p1_num, dtype=torch.float32).to(self.device)
        self.P1_cat_full = torch.tensor(np_p1_cat, dtype=torch.float32).to(self.device)
        self.Q_block = torch.tensor(block_diag(*q_blocks), dtype=torch.float32).to(
            self.device) if q_blocks else torch.empty(0).to(self.device)

        # ---------------------------------------------------------
        # Build Layouts for HSIC Loss
        # ---------------------------------------------------------
        self.layout_res = []
        curr_idx = 0
        num_dim = self.X_num.shape[1]
        if num_dim > 0:
            self.layout_res.append((curr_idx, curr_idx + num_dim, 0))
            curr_idx += num_dim
            if self.task == "Res-N":
                self.layout_res.append((curr_idx, curr_idx + num_dim, 0))
                curr_idx += num_dim
        for k in self.cat_sizes:
            self.layout_res.append((curr_idx, curr_idx + int(k), int(k)))
            curr_idx += int(k)
        self.layout_cond = []
        curr_idx = 0

        cond_num_dim = self.Cond_Num.shape[1]
        if cond_num_dim > 0:
            self.layout_cond.append((curr_idx, curr_idx + cond_num_dim, 0))
            curr_idx += cond_num_dim

        for k in self.cond_cat_sizes:
            self.layout_cond.append((curr_idx, curr_idx + int(k), int(k)))
            curr_idx += int(k)

    def calc_loss(self, batch_idx):
        b_x_num = self.X_num[batch_idx]
        b_x_cat = self.X_cat[batch_idx]
        b_cond = self.Cond[batch_idx]
        b_p1_num = self.P1_num[batch_idx]
        b_p1_cat = self.P1_cat_full[batch_idx]
        B = len(batch_idx)

        cond_drop_prob = self.config.get("train", {}).get("cond_drop_prob", 0.2)
        if cond_drop_prob > 0.0 and self.model.training:
            drop_mask = (torch.rand(B, 1, device=self.device) > cond_drop_prob).float()
            b_cond = b_cond * drop_mask

        if self.task == "Res-F":
            t = torch.randint(0, self.T_acc, (B,), device=self.device).long()
            a_bar = self.alpha_bars[t].view(B, 1)
            true_res = b_p1_num - b_x_num
            eps_num = torch.randn_like(b_x_num)
            x_t_num = (torch.sqrt(a_bar) * b_x_num + (1 - torch.sqrt(a_bar)) * true_res +
                       torch.sqrt(1 - a_bar) * eps_num)
        else:
            t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
            a_bar = self.alpha_bars[t].view(B, 1)
            b_bar = self.beta_bars[t].view(B, 1)
            eps_num = torch.randn_like(b_x_num)
            true_res = b_p1_num - b_x_num
            x_t_num = b_x_num + a_bar * true_res + b_bar * eps_num

        if self.Q_block.numel() > 0:
            pi_prior = b_p1_cat @ self.Q_block
        else:
            pi_prior = torch.ones_like(b_x_cat)
        log_probs = torch.logaddexp(
            torch.log((1 - a_bar) + 1e-30) + torch.log(b_x_cat + 1e-30),
            torch.log(a_bar + 1e-30) + torch.log(pi_prior + 1e-30)
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
            gamma = self.config["model"]["loss_num"]
            if self.task == "Res-N":
                dim = out_num.shape[1] // 2
                p_res, p_eps = out_num[:, :dim], out_num[:, dim:]
                loss_n = gamma * F.mse_loss(p_res, true_res) + F.mse_loss(p_eps, eps_num)
            elif self.task == "Res":
                loss_n = gamma * F.mse_loss(out_num, true_res)
            elif self.task == "Res-F":
                # alpha_t = self.alphas[t].view(B, 1)
                # beta_t = self.betas[t].view(B, 1)
                # resnoise = eps_num + (1 - torch.sqrt(alpha_t)) * torch.sqrt(1 - a_bar) / beta_t * true_res
                # loss_n = gamma * F.mse_loss(out_num, resnoise)
                loss_n = gamma * F.mse_loss(out_num, b_x_num)
            else:
                loss_n = F.mse_loss(out_num, eps_num)

        if out_cat is not None:
            logits_list = torch.split(out_cat, self.cat_sizes, dim=1)
            xt_list = torch.split(x_t_cat, self.cat_sizes, dim=1)
            x0_list = torch.split(b_x_cat, self.cat_sizes, dim=1)
            prior_list = torch.split(pi_prior, self.cat_sizes, dim=1)
            if self.discrete == "KL":
                for i, (logits, xt, x0, prior) in enumerate(zip(logits_list, xt_list, x0_list, prior_list)):
                    p = prior if self.has_partner_mask[i] else torch.ones_like(prior) / prior.shape[1]
                    #logits = torch.log(p + 1.0e-30) + logits
                    pred_probs = F.softmax(logits, dim=-1)
                    log_true_post = self._compute_posterior_vec(xt, x0, p, t)
                    log_model_post = self._compute_posterior_vec(xt, pred_probs, p, t)
                    loss_c += (self.config["model"]["loss_cat"] *
                               F.kl_div(log_model_post, log_true_post, reduction='batchmean', log_target=True))
            else:
                for i, (logits, x0, prior) in enumerate(zip(logits_list, x0_list, prior_list)):
                    p = prior if self.has_partner_mask[i] else torch.ones_like(prior) / prior.shape[1]
                    #logits = torch.log(p + 1.0e-30) + logits
                    loss_c += (self.config["model"]["loss_cat"] *
                               F.cross_entropy(logits, x0, reduction='mean'))
        # ---------------------------------------------------------
        # Assemble Tensors for One-Shot HSIC Calculation
        # ---------------------------------------------------------
        lambda_hsic = self.config["model"].get("loss_hsic", 0)
        loss_hsic = torch.tensor(0.0, device=self.device)

        if lambda_hsic > 0.0:
            if self.task == "Res-N":
                pred_n = out_num
                true_n = torch.cat([true_res, eps_num], dim=1)
            elif self.task == "Res":
                pred_n = out_num
                true_n = true_res
            else:
                pred_n = out_num
                true_n = eps_num

            pred_c_list = []
            if out_cat is not None:
                for i, (logits, prior) in enumerate(zip(logits_list, prior_list)):
                    p = prior if self.has_partner_mask[i] else torch.ones_like(prior) / prior.shape[1]
                    #logits = torch.log(p + 1.0e-30) + logits
                    pred_c_list.append(F.softmax(logits, dim=-1))

            pred_c = torch.cat(pred_c_list, dim=1) if pred_c_list else torch.empty((B, 0), device=self.device)
            true_c = b_x_cat

            x_fake = torch.cat([pred_n, pred_c], dim=1)
            x_real = torch.cat([true_n, true_c], dim=1)

            loss_hsic = calc_HSIC_loss(
                x_fake=x_fake,
                x_real=x_real,
                attrs=b_cond,
                layout_res=self.layout_res,
                layout_attr=self.layout_cond,
                lambda_hsic=lambda_hsic,
            )

        # Combine everything
        total_loss = loss_n + loss_c + loss_hsic
        return total_loss, loss_n, loss_c, loss_hsic

    def _compute_posterior_vec(self, x_t, x_start, x_proxy, t):
        B = x_t.shape[0]
        curr_alpha = self.alpha_bars[t].view(B, 1)
        prev_alpha = self.alpha_bars[(t - 1).clamp(0)].view(B, 1)
        prev_alpha[t == 0] = 0.0
        beta_t = (1.0 - (1.0 - curr_alpha) / (1.0 - prev_alpha + 1e-30)).clamp(1e-5, 1.0 - 1e-5)
        w_s_prev = 1.0 - prev_alpha
        w_p_prev = prev_alpha
        log_prior = torch.logaddexp(
            torch.log(w_s_prev + 1e-30) + torch.log(x_start + 1e-30),
            torch.log(w_p_prev + 1e-30) + torch.log(x_proxy + 1e-30)
        )
        p_proxy_at_xt = (x_t * x_proxy).sum(dim=1, keepdim=True)
        log_lik_noise = torch.log(beta_t * p_proxy_at_xt + 1e-30)
        log_lik_stay = torch.log((1.0 - beta_t) + beta_t * p_proxy_at_xt + 1e-30)
        log_likelihood = x_t * log_lik_stay + (1.0 - x_t) * log_lik_noise
        log_unnorm = log_prior + log_likelihood
        return log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)

    def _reverse_diffusion(self, full_data, model_to_use, batch_size=2048):
        N = full_data.shape[0]
        collected_outputs = {'num': [], 'cat': []}
        p1_n_start, p1_n_end = self.dim_info['p1_num_idx']
        p1_c_start, p1_c_end = self.dim_info['p1_cat_idx']
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                B = end - start
                full_cond = full_data[start:end]
                p1_num = full_cond[:, p1_n_start:p1_n_end]

                if self.task == "Res-F":
                    alpha_hat_T = self.alpha_bars[self.T_acc - 1]
                    x_t_num = torch.sqrt(alpha_hat_T) * p1_num + torch.sqrt(1 - alpha_hat_T) * torch.randn_like(p1_num)
                    loop_range = reversed(range(self.T_acc))
                else:
                    x_t_num = p1_num + torch.sqrt(self.sum_scale) * torch.randn_like(p1_num)
                    loop_range = reversed(range(self.num_steps))

                p1_cat_chunk = full_cond[:, p1_c_start:p1_c_end]

                if self.Q_block.numel() > 0:
                    pi_prior = p1_cat_chunk @ self.Q_block
                else:
                    pi_prior = torch.ones((B, self.X_cat.shape[1]), device=self.device)

                x_t_cat_list = []
                prior_list = torch.split(pi_prior, self.cat_sizes, dim=1)
                for k, p in zip(self.cat_sizes, prior_list):
                    log_p = torch.log(p + 1e-30)
                    g = -torch.log(-torch.log(torch.rand_like(log_p) + 1e-30) + 1e-30)
                    x_t_cat_list.append(F.one_hot((log_p + g).argmax(dim=1), k).float())
                x_t_cat = torch.cat(x_t_cat_list, dim=1) if x_t_cat_list else torch.empty((B, 0), device=self.device)

                cfg_scale = self.config.get("sample", {}).get("cfg_scale", 1.0)
                null_cond = torch.zeros_like(full_cond) if cfg_scale > 1.0 else None
                for i in loop_range:
                    t = torch.full((B,), i, device=self.device).long()
                    x_curr = torch.cat([x_t_num, x_t_cat], dim=1)

                    out_num, out_cat = model_to_use(x_curr, full_cond, t)
                    if cfg_scale > 1.0:
                        out_num_cond, out_cat_cond = model_to_use(x_curr, full_cond, t)
                        out_num_uncond, out_cat_uncond = model_to_use(x_curr, null_cond, t)
                        out_num = None
                        if out_num_cond is not None:
                            out_num = out_num_uncond + cfg_scale * (out_num_cond - out_num_uncond)
                        out_cat = None
                        if out_cat_cond is not None:
                            out_cat = out_cat_uncond + cfg_scale * (out_cat_cond - out_cat_uncond)
                    else:
                        out_num, out_cat = model_to_use(x_curr, full_cond, t)

                    if out_num is not None:
                        if self.task == "Res-F":
                            # alpha_t = self.alphas[i]
                            # a_bar = self.alpha_bars[i]
                            # beta_t = self.betas[i]
                            # beta_bar = self.betas_bars[i]
                            # pred_resnoise = out_num
                            # sigma = torch.sqrt(beta_bar)
                            # z = torch.randn_like(x_t_num) if i > 0 else 0
                            # x_t_num = 1 / torch.sqrt(alpha_t) * (
                            #         x_t_num - (beta_t / torch.sqrt(1 - a_bar)) * pred_resnoise
                            # ) + sigma * z
                            alpha_t = self.alphas[i]
                            a_bar = self.alpha_bars[i]
                            a_bar_prev = self.alpha_bars_t_minus_1[i]
                            beta_hat = self.betas_bars[i]

                            # 1. Network predicts x_0 directly
                            pred_x_0 = out_num

                            # 2. Implied residual: R = condition - x_0
                            pred_residual = p1_num - pred_x_0

                            sigma = torch.sqrt(beta_hat)
                            z = torch.randn_like(x_t_num) if i > 0 else 0

                            if i == 0:
                                # Final step: just return the predicted clean data
                                x_t_num = pred_x_0
                            else:
                                # ResFusion Formula 44 (Sample Mode)
                                term1 = torch.sqrt(alpha_t) * (1 - a_bar_prev) * (x_t_num - pred_residual)
                                term2 = torch.sqrt(a_bar_prev) * (1 - alpha_t) * (pred_x_0 - pred_residual)

                                x_t_num = ((term1 + term2) / (1 - a_bar)) + pred_residual + sigma * z
                        else:
                            if self.task == "Res-N":
                                dim = out_num.shape[1] // 2
                                res, eps = out_num[:, :dim], out_num[:, dim:]
                                x_start = x_t_num - self.alpha_bars[i] * res - self.beta_bars[i] * eps
                            elif self.task == "Res":
                                res = out_num
                                eps = (x_t_num - p1_num - (self.alpha_bars[i] - 1) * res) / (self.beta_bars[i] + 1e-8)
                                x_start = x_t_num - self.alpha_bars[i] * res - self.beta_bars[i] * eps
                            else:
                                eps = out_num
                                x_start = (x_t_num - self.alpha_bars[i] * p1_num - self.beta_bars[i] * eps) / (
                                            1 - self.alpha_bars[i] + 1e-8)
                                x_start = x_start.clamp(-4, 4)
                                res = p1_num - x_start

                            x_start = x_start.clamp(-4, 4)
                            post_mean = self.posterior_mean_coef1[i] * x_t_num + \
                                        self.posterior_mean_coef2[i] * res + \
                                        self.posterior_mean_coef3[i] * x_start
                            noise = torch.randn_like(x_t_num) if i > 0 else 0
                            x_t_num = (post_mean + (0.5 * self.posterior_log_variance[i]).exp() * noise).clamp(-4, 4)

                    if out_cat is not None:
                        logits_list = torch.split(out_cat, self.cat_sizes, dim=1)
                        xt_list = torch.split(x_t_cat, self.cat_sizes, dim=1)
                        new_cat_list = []
                        for j, (logits, xt, prior) in enumerate(zip(logits_list, xt_list, prior_list)):
                            p = prior if self.has_partner_mask[j] else torch.ones_like(prior) / prior.shape[1]
                            # logits = torch.log(p + 1.0e-30) + logits
                            pred_probs = F.softmax(logits, dim=-1)
                            log_post = self._compute_posterior_vec(xt, pred_probs, p, t)
                            g = -torch.log(-torch.log(torch.rand_like(log_post) + 1e-30) + 1e-30)
                            sample = (log_post + g)
                            new_cat_list.append(F.one_hot(sample.argmax(dim=1), self.cat_sizes[j]).float())
                        x_t_cat = torch.cat(new_cat_list, dim=1)

                collected_outputs['num'].append(x_t_num.cpu())
                collected_outputs['cat'].append(x_t_cat.cpu())

        final_dict = {}
        full_num = torch.cat(collected_outputs['num'], dim=0).to(self.device) if collected_outputs['num'] else None
        full_cat = torch.cat(collected_outputs['cat'], dim=0).to(self.device) if collected_outputs['cat'] else None

        schema = self.transformer.get_sird_schema()
        curr_n, curr_c = 0, 0
        p2_nums = [v for v in schema if 'numeric_p2' in v['type']]
        p2_cats = [v for v in schema if 'categorical_p2' in v['type']]

        if full_num is not None:
            for v in p2_nums:
                final_dict[v['name']] = full_num[:, curr_n:curr_n + 1]
                curr_n += 1

        if full_cat is not None:
            for v in p2_cats:
                k = v['num_classes']
                final_dict[v['name']] = full_cat[:, curr_c:curr_c + k]
                curr_c += k

        return final_dict

    def impute(self, m=None, save_path=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        pd.set_option('future.no_silent_downcasting', True)
        m_s = m if m else self.config["sample"]["m"]
        eval_bs = self.config["train"].get("eval_batch_size", 2048)

        if self.config["sample"]["mi_approx"] == "SWAG":
            if self.swag_model is None or self.swag_model.n_models.item() == 0:
                print("Warning: No SWAG collected. Using base model.")
            else:
                print(f"Using SWAG sampling ({self.swag_model.n_models.item()} models)")

        all_imputed_dfs = []
        pbar = tqdm(total=m_s, desc="Imputation Rounds", file=sys.stdout)

        for samp_i in range(1, m_s + 1):
            if self.config["sample"]["mi_approx"] == "SWAG" and self.swag_model.n_models.item() > 1:
                model_to_use = self.swag_model.sample(scale=1, cov=True)
                model_to_use.eval()
            elif self.config["sample"]["mi_approx"] == "DROPOUT":
                model_to_use = self.model_list[0];
                model_to_use.eval()
                for modu in model_to_use.modules():
                    if modu.__class__.__name__.startswith('Dropout'): modu.train()
            elif self.config["sample"]["mi_approx"] == "BOOTSTRAP":
                model_to_use = self.model_list[samp_i - 1];
                model_to_use.eval()
            else:
                model_to_use = self.model_list[0]; model_to_use.eval()

            # Pass full Cond tensor
            x_0_dict = self._reverse_diffusion(self.Cond, model_to_use, eval_bs)

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
                if self.config["sample"]["pmm"]:
                    obs_idx = df_f[df_f[c].notna()].index;
                    miss_idx = df_f[df_f[c].isna()].index
                    if c in self.num_names:
                        y_obs = df_f.loc[obs_idx, c].values
                        yhat_obs = df_denorm.loc[obs_idx, c].values
                        yhat_miss = df_denorm.loc[miss_idx, c].values
                        imputed_series = pd.Series(
                            self.pmm(yhat_obs, yhat_miss, y_obs, k=self.config["sample"]["donors"]).flatten(),
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

    def pmm(self, yhat_obs, yhat_miss, y_obs, k=5):
        d = np.array(yhat_obs).reshape(-1, 1)
        t = np.array(yhat_miss).reshape(-1, 1)
        v = np.array(y_obs).reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(d)
        _, neighbor_indices = nbrs.kneighbors(t)
        rand_cols = np.random.randint(0, k, size=neighbor_indices.shape[0])
        final_indices = neighbor_indices[np.arange(neighbor_indices.shape[0]), rand_cols]
        return v[final_indices]