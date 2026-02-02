import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import numpy as np
import pandas as pd
import copy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks_attn import RDDM_NET
from utils import inverse_transform_data, process_data


# ==========================================
# 1. Full SWAG Implementation (Low-Rank + Diag)
# ==========================================
class FullSWAG(nn.Module):
    def __init__(self, base_model, max_num_models=20, var_clamp=1e-30):
        super(FullSWAG, self).__init__()
        self.base = copy.deepcopy(base_model)
        self.base.train()
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        self.n_models = torch.zeros([1], dtype=torch.long)
        self.params = list()

        for name, param in self.base.named_parameters():
            safe_name = name.replace(".", "_")
            self.register_buffer(f"{safe_name}_mean", torch.zeros_like(param.data))
            self.register_buffer(f"{safe_name}_sq_mean", torch.zeros_like(param.data))
            self.register_buffer(f"{safe_name}_cov_mat_sqrt", torch.empty(0, param.numel()))
            self.params.append((name, safe_name, param))

    def collect_model(self, base_model):
        curr_params = dict(base_model.named_parameters())
        n = self.n_models.item()
        for name, safe_name, _ in self.params:
            if name not in curr_params: continue
            param = curr_params[name]
            mean = getattr(self, f"{safe_name}_mean")
            sq_mean = getattr(self, f"{safe_name}_sq_mean")
            cov_mat_sqrt = getattr(self, f"{safe_name}_cov_mat_sqrt")

            mean = mean * n / (n + 1.0) + param.data.to(mean.device) / (n + 1.0)
            sq_mean = sq_mean * n / (n + 1.0) + (param.data.to(sq_mean.device) ** 2) / (n + 1.0)
            dev = (param.data.to(mean.device) - mean).view(-1, 1)
            cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.t()), dim=0)

            if (cov_mat_sqrt.size(0)) > self.max_num_models:
                cov_mat_sqrt = cov_mat_sqrt[1:, :]

            setattr(self, f"{safe_name}_mean", mean)
            setattr(self, f"{safe_name}_sq_mean", sq_mean)
            setattr(self, f"{safe_name}_cov_mat_sqrt", cov_mat_sqrt)
        self.n_models.add_(1)

    def sample(self, scale=1, cov=True):
        scale_sqrt = scale ** 0.5
        for name, safe_name, base_param in self.params:
            mean = getattr(self, f"{safe_name}_mean")
            sq_mean = getattr(self, f"{safe_name}_sq_mean")
            cov_mat_sqrt = getattr(self, f"{safe_name}_cov_mat_sqrt")

            var = torch.clamp(sq_mean - mean ** 2, min=self.var_clamp)
            var_sample = var.sqrt() * torch.randn_like(var)

            if cov and cov_mat_sqrt.size(0) > 0:
                K = cov_mat_sqrt.size(0)
                z2 = torch.randn(K, 1, device=mean.device)
                cov_sample = cov_mat_sqrt.t().matmul(z2).view_as(mean)
                cov_sample /= (self.max_num_models - 1) ** 0.5
                rand_sample = var_sample + cov_sample
            else:
                rand_sample = var_sample

            sample = mean + scale_sqrt * rand_sample
            base_param.data.copy_(sample)
        return self.base


class TPVMI_RDDM:
    def __init__(self, config, data_info, device=None):
        self.config = config
        self.data_info = data_info
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = config["diffusion"]["num_steps"]
        self.sum_scale = torch.tensor(config["diffusion"]["sum_scale"])
        self.task = config["else"]["task"]
        # Schedules
        b = torch.linspace(0, 1, self.num_steps).to(self.device) ** 3
        a = torch.flip(torch.linspace(0, 1, self.num_steps).to(self.device), dims=[0])
        alphas = a / a.sum()
        betas = b / b.sum() * self.sum_scale
        betas_cumsum = betas.cumsum(dim=0).clip(0, 1)

        self.alpha_bars = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=self.alpha_bars[1])

        betas_cumsum_prev = F.pad(betas_cumsum[:-1], (1, 0), value=betas_cumsum[1])
        self.beta_bars = torch.sqrt(betas_cumsum)

        posterior_variance = betas * betas_cumsum_prev / betas_cumsum
        posterior_variance[0] = 0
        self.posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = betas_cumsum_prev / betas_cumsum
        self.posterior_mean_coef2 = (betas * alphas_cumsum_prev-betas_cumsum_prev*alphas) / betas_cumsum
        self.posterior_mean_coef3 = betas / betas_cumsum
        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1

        self.model = None
        self.model_list = []
        self.swag_model = None
        self._global_p1 = None
        self._global_p2 = None
        self._global_aux = None
        self.num_idxs = None
        self.cat_idxs = None
        self.num_vars = []
        self.cat_vars = []

    def _get_cosine_schedule(self, num_steps, s=0.008):
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]

    def _map_schema_indices(self):
        self.num_vars = [v['name'] for v in self.variable_schema if v['type'] == 'numeric']
        self.cat_vars = [v for v in self.variable_schema if v['type'] == 'categorical']
        num_indices_list, cat_indices_list = [], []
        curr_ptr = 0
        for var in self.variable_schema:
            if 'aux' in var['type']: continue
            if var['type'] == 'numeric':
                num_indices_list.append(curr_ptr)
            elif var['type'] == 'categorical':
                cat_indices_list.append(curr_ptr)
            curr_ptr += 1
        self.num_idxs = torch.tensor(num_indices_list, device=self.device).long()
        self.cat_idxs = torch.tensor(cat_indices_list, device=self.device).long()

    def fit(self, file_path, train_indices = None, val_indices=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        lr = self.config["train"]["lr"]
        epochs = self.config["train"]["epochs"]
        batch_size = self.config["train"]["batch_size"]
        mi_approx = self.config["else"]["mi_approx"]

        (proc_data, proc_mask, p1_idx, p2_idx, weight_idx, self.variable_schema, self.norm_stats,
         self.raw_df) = process_data(file_path, self.data_info)

        p1_idx, p2_idx = p1_idx.astype(int), p2_idx.astype(int)
        self._map_schema_indices()

        p2_mask = proc_mask[:, p2_idx]
        valid_rows = np.where(p2_mask.mean(axis=1) > 0.5)[0]

        print(f"[Fit] Global Valid Rows: {len(valid_rows)}")
        if train_indices is not None:
            print(f"[Fit] OVERRIDE: Using provided training subset (N={len(train_indices)})")

        self._global_p1 = torch.from_numpy(proc_data[:, p1_idx]).float().to(self.device)
        self._global_p2 = torch.from_numpy(proc_data[:, p2_idx]).float().to(self.device)

        all_idx = set(range(proc_data.shape[1]))
        reserved = set(p1_idx) | set(p2_idx)
        aux_idx = np.array(sorted(list(all_idx - reserved)), dtype=int)
        if len(aux_idx) > 0:
            self._global_aux = torch.from_numpy(proc_data[:, aux_idx]).float().to(self.device)
        else:
            self._global_aux = torch.empty((proc_data.shape[0], 0)).float().to(self.device)

        self.Q_dict = {}
        if len(self.cat_idxs) > 0:
            rel_cat_idxs = self.cat_idxs.cpu().numpy()
            valid_p1_entries = proc_data[valid_rows][:, p1_idx[rel_cat_idxs]].astype(int)
            valid_p2_entries = proc_data[valid_rows][:, p2_idx[rel_cat_idxs]].astype(int)
            for i, var in enumerate(self.cat_vars):
                name = var['name']
                K = var['num_classes']
                cm = confusion_matrix(valid_p2_entries[:, i], valid_p1_entries[:, i], labels=range(K))
                cm_smooth = cm + 1
                Q = cm_smooth / cm_smooth.sum(axis=1, keepdims=True)
                self.Q_dict[name] = torch.tensor(Q, device=self.device).float()

        num_train_mods = self.config["else"]["m"] if mi_approx == "bootstrap" else 1

        for k in range(num_train_mods):
            print(f"\n[TPVMI-RDDM] Training Model {k + 1}/{num_train_mods}...")
            if train_indices is not None:
                train_rows = train_indices
            else:
                rng = np.random.default_rng()
                if mi_approx == "bootstrap":
                    current_rows_indices = rng.choice(np.arange(len(valid_rows)), size=len(valid_rows), replace=True)
                    train_rows = valid_rows[current_rows_indices]
                else:
                    train_rows = valid_rows.copy()

            if val_indices is not None:
                self.best_epoch_ = 0
                val_p1 = self._global_p1[val_indices]
                val_p2 = self._global_p2[val_indices]
                val_aux = self._global_aux[val_indices]

            if self.config["else"]["samp"] == "BLS":
                cat_class_indices = {}
                cat_vars_names = []
                if len(self.cat_vars) > 0:
                    all_cat_vars_map = []
                    n_targets = len(p2_idx)
                    for i, var in enumerate(self.variable_schema):
                        if 'categorical' in var['type']:
                            if 'aux' in var['type']:
                                if len(aux_idx) > 0:
                                    rel_idx = i - n_targets
                                    if 0 <= rel_idx < len(aux_idx):
                                        all_cat_vars_map.append({'name': var['name'], 'global_col': aux_idx[rel_idx]})
                            else:
                                all_cat_vars_map.append({'name': var['name'], 'global_col': p2_idx[i]})

                    train_data_subset = proc_data[train_rows]
                    severity_scores = []
                    for var_info in all_cat_vars_map:
                        col_idx = var_info['global_col']
                        name = var_info['name']
                        col_vals = train_data_subset[:, col_idx].astype(int)
                        unique_labels, counts = np.unique(col_vals, return_counts=True)
                        indices_map = {}
                        for label in unique_labels:
                            indices_map[label] = np.where(col_vals == label)[0]
                        cat_class_indices[name] = indices_map

                        N = len(col_vals)
                        probs = counts / N
                        entropy = -np.sum(probs * np.log(probs + 1e-9))
                        K = len(unique_labels)
                        if K > 1:
                            max_entropy = np.log(K)
                            severity = max_entropy - entropy
                        else:
                            severity = 0.0
                        severity_scores.append(severity)
                        cat_vars_names.append(name)
                    severity_scores = np.array(severity_scores)
                    severity_scores = np.power(severity_scores, 0.25)
                    cat_sampling_probs = severity_scores / severity_scores.sum()
                    if k == 0:
                        print(f"[Sampler] Computed Imbalance Severity (KL-Div). Top imbalance:")
                        sort_idx = np.argsort(cat_sampling_probs)[::-1]
                        for i in range(min(3, len(sort_idx))):
                            idx = sort_idx[i]
                            print(f"  - {cat_vars_names[idx]}: P={cat_sampling_probs[idx]:.4f}")

            self.model = RDDM_NET(self.config, self.device, self.variable_schema).to(self.device)
            self.model.train()
            if mi_approx == "SWAG":
                swa_start_iter = int(epochs * 0.80)
                optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
                scheduler = CosineAnnealingLR(optimizer, T_max=swa_start_iter, eta_min=lr * 0.5)
                self.swag_model = FullSWAG(self.model, max_num_models=int(self.config["else"]["m"] * 5)).to(self.device)
            else:
                optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=self.config["train"]["weight_decay"])

            # --- Early Stopping Variables ---
            best_val_loss = float('inf')
            patience_counter = 0
            patience_limit = 20  # Stop after 10 checks (20 * 50 = 1000 epochs) without improvement
            best_model_state = None

            pbar = tqdm(range(epochs), desc=f"Training M{k+1}", file=sys.stdout)
            for step in pbar:
                if self.config["else"]["samp"] == "BLS":
                    current_balance_map = None
                    current_var_name = ""
                    if len(cat_class_indices) > 0:
                        target_var_name = np.random.choice(cat_vars_names, p=cat_sampling_probs)
                        current_balance_map = cat_class_indices[target_var_name]
                        current_var_name = target_var_name
                    if current_balance_map is not None:
                        classes = list(current_balance_map.keys())
                        batch_classes = np.random.choice(classes, batch_size, replace=True)
                        batch_indices_local = []
                        for c in batch_classes:
                            idx = np.random.choice(current_balance_map[c])
                            batch_indices_local.append(idx)
                        b_idx = train_rows[np.array(batch_indices_local)]
                        np.random.shuffle(b_idx)
                else:
                    idx_sample = np.random.randint(0, len(train_rows), batch_size)
                    b_idx = train_rows[idx_sample]

                b_p1 = self._global_p1[b_idx]
                b_p2 = self._global_p2[b_idx]
                b_aux = self._global_aux[b_idx]

                optimizer.zero_grad()
                loss = self.calc_unified_loss(b_p1, b_p2, b_aux)
                loss.backward()

                if mi_approx == "SWAG":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if step % 50 == 0:
                    status_str = f"L:{loss.item():.4f}"
                    if val_indices is not None:
                        current_val_loss = self._validate(val_p1, val_p2, val_aux, bs=1024)
                        status_str += f" | Val:{current_val_loss:.4f}"
                        if current_val_loss < best_val_loss:
                            best_val_loss = current_val_loss
                            current_model_best_epoch = step
                            best_model_state = copy.deepcopy(self.model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        if patience_counter >= patience_limit:
                            print(f"\n[Early Stopping] No improvement for {patience_limit * 50} epochs. Stopping.")
                            if best_model_state is not None:
                                self.model.load_state_dict(best_model_state)
                            break
                    else:
                        if self.config["else"]["samp"] == "BLS":
                            var_tag = f"Bal:{current_var_name}" if current_var_name else "Std"
                            pbar.set_postfix(loss=f"{loss.item():.4f}", v=var_tag)

                    pbar.set_postfix_str(status_str)

                if mi_approx == "SWAG":
                    if step < swa_start_iter:
                        scheduler.step()
                    else:
                        for param_group in optimizer.param_groups: param_group['lr'] = lr * 0.5
                        if (step - swa_start_iter) % 50 == 0:
                            self.swag_model.collect_model(self.model)
            if val_indices is not None:
                self.best_epoch_ = current_model_best_epoch
            self.model_list.append(self.model)
        return self

    def _validate(self, p1, p2, aux, bs):
        self.model.eval()
        tl, count = 0, 0
        with torch.no_grad():
            for i in range(0, p1.shape[0], bs):
                end = min(i + bs, p1.shape[0])
                b_p1 = p1[i:end]
                b_p2 = p2[i:end]
                b_aux = aux[i:end]
                loss = self.calc_unified_loss(b_p1, b_p2, b_aux)
                tl += loss.item() * (end - i)
                count += (end - i)
        self.model.train()
        return tl / count

    def calc_unified_loss(self, p1, p2, aux):
        B = p1.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        a_bar = self.alpha_bars[t].view(B, 1)
        b_bar = self.beta_bars[t].view(B, 1)

        x_t_dict, p1_dict, aux_dict = {}, {}, {}

        # --- NUMERIC ---
        if len(self.num_idxs) > 0:
            batch_p2_num = p2[:, self.num_idxs]
            batch_p1_num = p1[:, self.num_idxs]
            true_residual = batch_p1_num - batch_p2_num
            eps = torch.randn_like(batch_p2_num)

            x_t_num = batch_p2_num + a_bar * true_residual + b_bar * eps
            for i, name in enumerate(self.num_vars):
                x_t_dict[name] = x_t_num[:, i:i + 1]
                p1_dict[name] = batch_p1_num[:, i:i + 1]

        # --- CATEGORICAL ---
        target_cat_list = []
        if len(self.cat_idxs) > 0:
            batch_p1_cat = p1[:, self.cat_idxs].long()
            batch_p2_cat = p2[:, self.cat_idxs].long()

            for i, var in enumerate(self.cat_vars):
                name = var['name']
                K = var['num_classes']
                oh_p1 = F.one_hot(batch_p1_cat[:, i], K).float()
                oh_p2 = F.one_hot(batch_p2_cat[:, i], K).float()
                p1_indices = batch_p1_cat[:, i]  # B
                noise_dist = self.Q_dict[name][p1_indices]
                proxy_mix = (1 - b_bar) * oh_p1 + b_bar * noise_dist
                pi_t = oh_p2 + a_bar * (proxy_mix - oh_p2)
                x_t_indices = torch.multinomial(pi_t, 1).squeeze(-1)
                x_t_dict[name] = F.one_hot(x_t_indices, K).float()
                p1_dict[name] = oh_p1
                target_cat_list.append(batch_p2_cat[:, i])

        if aux.shape[1] > 0:
            aux_c = 0
            for var in self.variable_schema:
                if 'aux' in var['type']:
                    curr = aux[:, aux_c:aux_c + 1]
                    if var['type'] == 'categorical_aux':
                        aux_dict[var['name']] = F.one_hot(curr.long().squeeze(), var['num_classes']).float()
                    else:
                        aux_dict[var['name']] = curr
                    aux_c += 1

        model_out = self.model(x_t_dict, t, p1_dict, aux_dict)
        loss_total = 0.0

        if len(self.num_vars) > 0:
            pred_stack = torch.stack([model_out[name] for name in self.num_vars], dim=1)
            if self.task == "Res-N":
                loss_total += (F.mse_loss(pred_stack[:, :, 0], true_residual) +
                               F.mse_loss(pred_stack[:, :, 1], eps))
            elif self.task == "Res":
                loss_total += F.mse_loss(pred_stack[:, :, 0], true_residual)
            elif self.task == "N":
                loss_total += F.mse_loss(pred_stack[:, :, 0], eps)

        if len(self.cat_vars) > 0:
            for i, var in enumerate(self.cat_vars):
                ce_loss = F.cross_entropy(model_out[var['name']], target_cat_list[i], reduction='mean', label_smoothing=0)
                loss_total += ce_loss

        return loss_total

    def impute(self, m=None, save_path="imputed_results.parquet", batch_size=None, eta=0.0):
        if not save_path.endswith('.parquet'):
            base = os.path.splitext(save_path)[0]
            save_path = f"{base}.parquet"
            print(f"Note: File extension changed to .parquet: {save_path}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        pd.set_option('future.no_silent_downcasting', True)

        m_s = m if m else self.config["else"]["m"]
        eval_bs = batch_size if batch_size else self.config["train"].get("eval_batch_size", 64)
        N = self._global_p1.shape[0]

        if self.config["else"]["mi_approx"] == "SWAG":
            if self.swag_model is None or self.swag_model.n_models.item() == 0:
                print("Warning: No SWAG models collected. Using base model point estimate.")
            else:
                print(f"Using Full SWAG for sampling (Collected {self.swag_model.n_models.item()} models)")

        all_imputed_dfs = []

        # Calculate total steps for the single progress bar
        steps_per_imp = (N + eval_bs - 1) // eval_bs
        total_steps = m_s * steps_per_imp

        # Initialize single progress bar
        pbar = tqdm(total=total_steps, desc="Imputation", file=sys.stdout)

        for samp_i in range(1, m_s + 1):
            # Update description to show current imputation set
            pbar.set_description(f"Imputing Set {samp_i}/{m_s}")

            if self.config["else"]["mi_approx"] == "SWAG":
                if self.swag_model is not None and self.swag_model.n_models.item() > 1:
                    model_to_use = self.swag_model.sample(scale=1, cov=True)
                    model_to_use.eval()
            elif self.config["else"]["mi_approx"] == "dropout":
                model_to_use = self.model_list[0]
                model_to_use.eval()
                for modu in model_to_use.modules():
                    if modu.__class__.__name__.startswith('Dropout'):
                        modu.train()
            elif self.config["else"]["mi_approx"] == "bootstrap":
                model_to_use = self.model_list[samp_i - 1]
                model_to_use.eval()
            else:
                model_to_use = self.model_list[0]
                model_to_use.eval()
            all_p2 = []

            with torch.no_grad():
                # Standard range loop instead of tqdm
                for start in range(0, N, eval_bs):
                    end = min(start + eval_bs, N)
                    B = end - start
                    b_p1 = self._global_p1[start:end]
                    b_aux = self._global_aux[start:end]

                    x_t_dict, p1_dict, aux_dict = {}, {}, {}
                    if len(self.num_idxs) > 0:
                        curr_p1_num = b_p1[:, self.num_idxs]
                        init_eps = torch.randn_like(curr_p1_num)
                        x_T_num = curr_p1_num + torch.sqrt(self.sum_scale) * init_eps
                        for k, name in enumerate(self.num_vars):
                            cp1 = curr_p1_num[:, k:k + 1]
                            x_t_dict[name] = x_T_num[:, k:k + 1]
                            p1_dict[name] = cp1

                    if len(self.cat_idxs) > 0:
                        curr_p1_cat = b_p1[:, self.cat_idxs].long()
                        for k, var in enumerate(self.cat_vars):
                            name = var['name']
                            K = var['num_classes']
                            oh_p1 = F.one_hot(curr_p1_cat[:, k], K).float()

                            p1_indices = curr_p1_cat[:, k]
                            noise_dist = self.Q_dict[name][p1_indices]  # B x K

                            pi_T = (1 - torch.sqrt(self.sum_scale)) * oh_p1 + torch.sqrt(self.sum_scale) * noise_dist
                            x_T_indices = torch.multinomial(pi_T, 1).squeeze(-1)
                            x_t_dict[name] = F.one_hot(x_T_indices, K).float()
                            p1_dict[name] = oh_p1

                    if b_aux.shape[1] > 0:
                        aux_c = 0
                        for var in self.variable_schema:
                            if 'aux' in var['type']:
                                curr = b_aux[:, aux_c:aux_c + 1]
                                if var['type'] == 'categorical_aux':
                                    aux_dict[var['name']] = F.one_hot(curr.long().squeeze(), var['num_classes']).float()
                                else:
                                    aux_dict[var['name']] = curr
                                aux_c += 1

                    for t in reversed(range(self.num_steps)):
                        t_b = torch.full((B,), t, device=self.device).long()
                        out = model_to_use(x_t_dict, t_b, p1_dict, aux_dict)

                        # Numeric Update
                        for name in self.num_vars:
                            x_t = x_t_dict[name]
                            x_input = p1_dict[name]
                            if self.task == "Res-N":
                                pred_res = out[name][:, 0:1]
                                pred_eps = out[name][:, 1:2]
                                x_start_pred = x_t - self.alpha_bars[t] * pred_res - self.beta_bars[t] * pred_eps
                                x_start_pred = x_start_pred.clamp(-5.0, 5.0)
                            elif self.task == "Res":
                                pred_res = out[name]
                                pred_eps = (x_t - x_input - (self.alpha_bars[t] - 1) * pred_res) / self.beta_bars[t]
                                x_start_pred = x_t - self.alpha_bars[t] * pred_res - self.beta_bars[t] * pred_eps
                                x_start_pred = x_start_pred.clamp(-5.0, 5.0)
                            elif self.task == "N":
                                pred_eps = out[name]
                                x_start_pred = (x_t - self.alpha_bars[t] * x_input - self.beta_bars[t] * pred_eps) / (
                                            1 - self.alpha_bars[t]).clamp(min=1e-5)
                                x_start_pred = x_start_pred.clamp(-5.0, 5.0)
                                pred_res = x_input - x_start_pred

                            posterior_mean = self.posterior_mean_coef1[t] * x_t + self.posterior_mean_coef2[
                                t] * pred_res + self.posterior_mean_coef3[t] * x_start_pred
                            noise = torch.randn_like(x_t) if t > 0 else 0
                            pred_x_t = posterior_mean + (0.5 * self.posterior_log_variance[t]).exp() * noise
                            x_t_dict[name] = pred_x_t

                        # Categorical Update
                        for k, var in enumerate(self.cat_vars):
                            name = var['name']
                            K = var['num_classes']
                            logits = out[name]
                            curr_oh_p1 = p1_dict[name]
                            pred_x0_probs = F.softmax(logits, dim=-1)  # \hat{x_0} probs
                            curr_x_t = x_t_dict[name]  # One-hot (B x K)

                            a_t = self.alpha_bars[t]
                            a_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0).to(self.device)
                            b_prev = self.beta_bars[t - 1] if t > 0 else torch.tensor(0.0).to(self.device)

                            alpha_t = (1 - a_t) / (1 - a_prev + 1e-6)
                            alpha_t = torch.clamp(alpha_t, 0.0, 1.0)
                            noise_dist_weighted = torch.einsum('bk,kn->bn', pred_x0_probs, self.Q_dict[name])
                            proxy_mix_prev = (1 - b_prev) * curr_oh_p1 + b_prev * noise_dist_weighted
                            q_xprev_given_x0 = a_prev * pred_x0_probs + (1 - a_prev) * proxy_mix_prev
                            x_t_idx = torch.argmax(curr_x_t, dim=-1)
                            prob_noise_to_xt = proxy_mix_prev.gather(1, x_t_idx.unsqueeze(1))  # (B, 1)
                            lik_vec = (1 - alpha_t) * prob_noise_to_xt + alpha_t * curr_x_t
                            numerator = lik_vec * q_xprev_given_x0
                            posterior_probs = numerator / (numerator.sum(dim=-1, keepdim=True) + 1e-10)

                            x_next_indices = torch.multinomial(posterior_probs, 1).squeeze(-1)
                            x_t_dict[name] = F.one_hot(x_next_indices, K).float()

                    batch_res = []
                    for var in self.variable_schema:
                        if 'aux' in var['type']: continue
                        if var['type'] == 'categorical':
                            final = torch.argmax(x_t_dict[var['name']], dim=-1, keepdim=True).float()
                            batch_res.append(final)
                        else:
                            batch_res.append(x_t_dict[var['name']])
                    all_p2.append(torch.cat(batch_res, dim=1).cpu())

                    # Update global progress bar by 1 batch
                    pbar.update(1)

            df_p2 = inverse_transform_data(torch.cat(all_p2, dim=0).numpy(), self.norm_stats, self.data_info)
            df_f = self.raw_df.copy()

            for c in df_p2.columns:
                if c in df_f.columns:
                    df_f[c] = df_f[c].fillna(df_p2[c])

            df_f.insert(0, "imp_id", samp_i)
            all_imputed_dfs.append(df_f)

        pbar.close()

        final_df = pd.concat(all_imputed_dfs, ignore_index=True)
        final_df['imp_id'] = final_df['imp_id'].astype(int)
        final_df.to_parquet(save_path, index=False)
        print(f"Saved stacked imputations to: {save_path} (Shape: {final_df.shape})")