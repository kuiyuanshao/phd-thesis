import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import pandas as pd
import copy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import SIRD_NET
from utils import inverse_transform_data, process_data
from swag import SWAG


class SIRD:
    def __init__(self, config, data_info, device=None):
        """
        Initialize the SIRD model and pre-calculate diffusion schedules.
        """
        self.config = config
        self.data_info = data_info
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = config["diffusion"]["num_steps"]
        self.sum_scale = torch.tensor(config["diffusion"]["sum_scale"])
        self.task = config["else"]["task"]

        # Define diffusion schedules using cubic interpolation
        b = torch.linspace(0, 1, self.num_steps).to(self.device) ** 3
        a = torch.flip(torch.linspace(0, 1, self.num_steps).to(self.device), dims=[0])
        alphas = a / a.sum()
        betas = b / b.sum() * self.sum_scale
        betas_cumsum = betas.cumsum(dim=0).clip(0, 1)

        # Calculate alpha bars (cumulative product of alphas approx) and beta bars
        self.alpha_bars = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=self.alpha_bars[1])

        betas_cumsum_prev = F.pad(betas_cumsum[:-1], (1, 0), value=betas_cumsum[1])
        self.beta_bars = torch.sqrt(betas_cumsum)

        # Pre-calculate posterior variance for the reverse step
        posterior_variance = betas * betas_cumsum_prev / betas_cumsum
        posterior_variance[0] = 0
        self.posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))

        # Coefficients for posterior mean calculation (used in reverse diffusion)
        self.posterior_mean_coef1 = betas_cumsum_prev / betas_cumsum
        self.posterior_mean_coef2 = (betas * alphas_cumsum_prev - betas_cumsum_prev * alphas) / betas_cumsum
        self.posterior_mean_coef3 = betas / betas_cumsum
        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1

        self.model = None
        self.model_list = []
        self.swag_model = None

        self._global_data = None
        self.variable_schema = []

        self.num_vars = []
        self.cat_vars = []
        self.p1_slices = {}
        self.p2_slices = {}
        self.aux_slices = {}
        self.Q_dict = {}

    def _map_schema_indices(self):
        """
        Map variable names to their corresponding column indices in the data matrix.
        Separates variables into Numeric, Categorical, Phase 1, Phase 2, and Auxiliary groups.
        """
        self.num_vars = [v for v in self.variable_schema if 'numeric' in v['type']]
        self.cat_vars = [v for v in self.variable_schema if 'categorical' in v['type']]
        self.p1_slices = {v['name']: slice(v['start_idx'], v['end_idx']) for v in self.variable_schema if
                          'p1' in v['type']}
        self.p2_slices = {v['name']: slice(v['start_idx'], v['end_idx']) for v in self.variable_schema if
                          'p2' in v['type']}
        self.aux_slices = {v['name']: slice(v['start_idx'], v['end_idx']) for v in self.variable_schema if
                           'aux' in v['type']}

    def fit(self, file_path=None, provided_data=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        lr = self.config["train"]["lr"]
        epochs = self.config["train"]["epochs"]
        batch_size = self.config["train"]["batch_size"]
        mi_approx = self.config["else"]["mi_approx"]
        self.gamma = self.config["model"]["gamma"]
        self.zeta = self.config["model"]["zeta"]

        # Load data from file or use provided data dictionary
        if provided_data is not None:
            self._global_data = provided_data['data'].float().to(self.device)
            self._global_weights = torch.from_numpy(provided_data['weights']).double()
            self.variable_schema = provided_data['schema']
            self.norm_stats = provided_data['stats']
            valid_rows = np.arange(self._global_data.shape[0])
        elif file_path is not None:
            (proc_data, proc_mask, weights_raw, self.variable_schema, self.norm_stats,
             self.raw_df) = process_data(file_path, self.data_info)

            # Filter rows where Phase 2 outcomes are actually observed
            p2_check_idx = next((i for i, v in enumerate(self.variable_schema) if 'numeric_p2' in v['type']), None)
            valid_rows = np.where(proc_mask[:, p2_check_idx] == 1)[0]
            print(f"[Fit] Global Valid Rows (Observed Phase 2): {len(valid_rows)}")

            self._global_data = torch.from_numpy(proc_data).float().to(self.device)
            self._global_weights = torch.from_numpy(weights_raw).double().to(self.device)

        self._map_schema_indices()
        self.Q_dict = {}

        # Construct transition matrices Q for categorical variables
        # Q approximates the noise distribution between Phase 1 (proxy) and Phase 2 (true)
        if len(self.cat_vars) > 0:
            for v in self.cat_vars:
                if 'p2' not in v['type']: continue
                name = v['name']
                partner = v['pair_partner']

                sl_p1 = self.p1_slices[partner]
                sl_p2 = self.p2_slices[name]

                d_p1 = self._global_data[valid_rows, sl_p1].argmax(dim=1).cpu().numpy()
                d_p2 = self._global_data[valid_rows, sl_p2].argmax(dim=1).cpu().numpy()

                K = v['num_classes']
                cm = confusion_matrix(d_p2, d_p1, labels=range(K))
                cm_smooth = cm + 1
                Q = cm_smooth / cm_smooth.sum(axis=1, keepdims=True)
                self.Q_dict[name] = torch.tensor(Q, device=self.device).float()

        num_train_mods = self.config["else"]["m"] if mi_approx == "bootstrap" else 1

        # Main training loop (supports multiple models for bootstrap approximation)
        for k in range(num_train_mods):
            if provided_data is None:
                print(f"\n[SIRD] Training Model {k + 1}/{num_train_mods}...")

            rng = np.random.default_rng()
            if mi_approx == "bootstrap":
                current_rows_indices = rng.choice(np.arange(len(valid_rows)), size=len(valid_rows), replace=True)
                train_rows = valid_rows[current_rows_indices]
            else:
                train_rows = valid_rows.copy()

            train_rows = torch.from_numpy(train_rows).long().to(self.device)
            local_weights = self._global_weights[train_rows]

            # Setup Weighted Sampler to handle survey weights
            train_indices = torch.arange(len(train_rows))
            sampler = WeightedRandomSampler(
                weights=local_weights,
                num_samples=len(train_rows) * 4,
                replacement=True
            )
            loader = DataLoader(
                TensorDataset(train_indices),
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False
            )
            loader_iter = iter(loader)

            self.model = SIRD_NET(self.config, self.device, self.variable_schema).to(self.device)
            self.model.train()

            # Initialize optimizer or SWAG if selected
            if mi_approx == "SWAG":
                swa_start_iter = int(epochs * 0.80)
                optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
                scheduler = CosineAnnealingLR(optimizer, T_max=swa_start_iter, eta_min=lr * 0.5)
                self.swag_model = SWAG(self.model, max_num_models=int(self.config["else"]["m"] * 5)).to(self.device)
            else:
                optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=self.config["train"]["weight_decay"])

            pbar = tqdm(range(epochs), desc=f"Training M{k + 1}", file=sys.stdout, leave=False)

            for step in pbar:
                try:
                    batch_local_indices = next(loader_iter)[0]
                except StopIteration:
                    loader_iter = iter(loader)
                    batch_local_indices = next(loader_iter)[0]
                b_idx = train_rows[batch_local_indices.to(self.device)]

                b_data = self._global_data[b_idx]

                optimizer.zero_grad()
                loss = self.calc_loss(b_data)
                loss.backward()

                if mi_approx == "SWAG":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if step % 50 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                # Collect SWAG models after threshold
                if mi_approx == "SWAG":
                    if step < swa_start_iter:
                        scheduler.step()
                    else:
                        for param_group in optimizer.param_groups: param_group['lr'] = lr * 0.5
                        if (step - swa_start_iter) % 50 == 0:
                            self.swag_model.collect_model(self.model)
            self.model_list.append(self.model)
        return self

    def calc_loss(self, batch_data):
        """
        Calculate diffusion loss for a batch.
        Samples time step t, adds noise, and computes prediction error.
        """
        B = batch_data.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        a_bar = self.alpha_bars[t].view(B, 1)
        b_bar = self.beta_bars[t].view(B, 1)

        x_t_dict, p1_dict, aux_dict = {}, {}, {}

        # Handle Numeric Variables (Phase 2)
        numeric_p2 = [v for v in self.num_vars if 'p2' in v['type']]
        if len(numeric_p2) > 0:
            p1_num_cols = [self.p1_slices[v['pair_partner']].start for v in numeric_p2]
            p2_num_cols = [self.p2_slices[v['name']].start for v in numeric_p2]

            batch_p1_num = batch_data[:, p1_num_cols]
            batch_p2_num = batch_data[:, p2_num_cols]

            true_residual = batch_p1_num - batch_p2_num
            eps = torch.randn_like(batch_p2_num)

            # Forward diffusion process for numeric data
            x_t_num = batch_p2_num + a_bar * true_residual + b_bar * eps
            for i, v in enumerate(numeric_p2):
                name = v['name']
                x_t_dict[name] = x_t_num[:, i:i + 1]
                p1_dict[name] = batch_p1_num[:, i:i + 1]

        # Handle Categorical Variables (Phase 2)
        target_cat_list = []
        cat_p2_vars = [v for v in self.cat_vars if 'p2' in v['type']]
        if len(cat_p2_vars) > 0:
            for v in cat_p2_vars:
                name = v['name']
                partner = v['pair_partner']

                oh_p1 = batch_data[:, self.p1_slices[partner]]
                oh_p2 = batch_data[:, self.p2_slices[name]]

                # Mix true labels with proxy-derived noise
                noise_dist = oh_p1 @ self.Q_dict[name]
                proxy_mix = (1 - b_bar) * oh_p1 + b_bar * noise_dist
                pi_t = oh_p2 + a_bar * (proxy_mix - oh_p2)

                x_t_indices = torch.multinomial(pi_t, 1).squeeze(-1)
                x_t_dict[name] = F.one_hot(x_t_indices, v['num_classes']).float()
                p1_dict[name] = oh_p1
                target_cat_list.append(torch.argmax(oh_p2, dim=1))

        # Collect Auxiliary Variables
        for v in self.variable_schema:
            if 'aux' in v['type']:
                aux_dict[v['name']] = batch_data[:, self.aux_slices[v['name']]]

        # Model Prediction
        model_out = self.model(x_t_dict, t, p1_dict, aux_dict)
        loss_total = 0.0

        # Compute Numeric Loss based on task configuration (Residual, Noise, or Both)
        if len(numeric_p2) > 0:
            pred_stack = torch.stack([model_out[v['name']] for v in numeric_p2], dim=1)
            if self.task == "Res-N":
                loss_total += (self.gamma * F.mse_loss(pred_stack[:, :, 0], true_residual) +
                               F.mse_loss(pred_stack[:, :, 1], eps))
            elif self.task == "Res":
                loss_total += self.gamma * F.mse_loss(pred_stack[:, :, 0], true_residual)
            elif self.task == "N":
                loss_total += F.mse_loss(pred_stack[:, :, 0], eps)

        # Compute Categorical Cross Entropy Loss
        if len(cat_p2_vars) > 0:
            for i, v in enumerate(cat_p2_vars):
                ce_loss = F.cross_entropy(model_out[v['name']], target_cat_list[i], reduction='mean',
                                          label_smoothing=0)
                loss_total += self.zeta * ce_loss

        return loss_total

    def _reverse_diffusion(self, full_data, model_to_use, batch_size=2048):
        """
        Execute the reverse diffusion process (sampling) from T to 0.
        Generates Phase 2 variables conditioned on Phase 1 and Auxiliary variables.
        """
        N = full_data.shape[0]
        # FIX: Only initialize lists for P2 variables. P1 variables are inputs, not generated outputs.
        collected_outputs = {var['name']: [] for var in self.variable_schema if 'p2' in var['type']}

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                B = end - start
                b_data = full_data[start:end]

                x_t_dict, p1_dict, aux_dict = {}, {}, {}

                # Initialize Numeric variables with noise (Time T)
                numeric_p2 = [v for v in self.num_vars if 'p2' in v['type']]
                if len(numeric_p2) > 0:
                    p1_num_cols = [self.p1_slices[v['pair_partner']].start for v in numeric_p2]
                    curr_p1_num = b_data[:, p1_num_cols]
                    init_eps = torch.randn_like(curr_p1_num)
                    x_T_num = curr_p1_num + torch.sqrt(self.sum_scale) * init_eps
                    for k, v in enumerate(numeric_p2):
                        x_t_dict[v['name']] = x_T_num[:, k:k + 1]
                        p1_dict[v['name']] = curr_p1_num[:, k:k + 1]

                # Initialize Categorical variables (Time T)
                cat_p2_vars = [v for v in self.cat_vars if 'p2' in v['type']]
                if len(cat_p2_vars) > 0:
                    for v in cat_p2_vars:
                        name = v['name']
                        partner = v['pair_partner']
                        oh_p1 = b_data[:, self.p1_slices[partner]]
                        noise_dist = oh_p1 @ self.Q_dict[name]

                        pi_T = (1 - torch.sqrt(self.sum_scale)) * oh_p1 + torch.sqrt(self.sum_scale) * noise_dist
                        x_T_indices = torch.multinomial(pi_T, 1).squeeze(-1)
                        x_t_dict[name] = F.one_hot(x_T_indices, v['num_classes']).float()
                        p1_dict[name] = oh_p1

                for v in self.variable_schema:
                    if 'aux' in v['type']:
                        aux_dict[v['name']] = b_data[:, self.aux_slices[v['name']]]

                # Iterative Denoising Loop
                for t in reversed(range(self.num_steps)):
                    t_b = torch.full((B,), t, device=self.device).long()
                    out = model_to_use(x_t_dict, t_b, p1_dict, aux_dict)

                    # Update Numeric Variables
                    for v in numeric_p2:
                        name = v['name']
                        x_t = x_t_dict[name]
                        x_input = p1_dict[name]
                        # Calculate predicted x_start based on task type
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

                        # Apply posterior mean formula
                        posterior_mean = self.posterior_mean_coef1[t] * x_t + self.posterior_mean_coef2[
                            t] * pred_res + self.posterior_mean_coef3[t] * x_start_pred
                        noise = torch.randn_like(x_t) if t > 0 else 0
                        pred_x_t = posterior_mean + (0.5 * self.posterior_log_variance[t]).exp() * noise
                        x_t_dict[name] = pred_x_t

                    # Update Categorical Variables
                    for v in cat_p2_vars:
                        name = v['name']
                        logits = out[name]
                        curr_oh_p1 = p1_dict[name]
                        pred_x0_probs = F.softmax(logits, dim=-1)
                        curr_x_t = x_t_dict[name]

                        a_t = self.alpha_bars[t]
                        a_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0).to(self.device)
                        b_prev = self.beta_bars[t - 1] if t > 0 else torch.tensor(0.0).to(self.device)

                        alpha_t = (1 - a_t) / (1 - a_prev + 1e-6)
                        alpha_t = torch.clamp(alpha_t, 0.0, 1.0)

                        # Compute posterior probability for categorical transition
                        noise_dist_weighted = pred_x0_probs @ self.Q_dict[name]
                        proxy_mix_prev = (1 - b_prev) * curr_oh_p1 + b_prev * noise_dist_weighted
                        q_xprev_given_x0 = a_prev * pred_x0_probs + (1 - a_prev) * proxy_mix_prev
                        x_t_idx = torch.argmax(curr_x_t, dim=-1)
                        prob_noise_to_xt = proxy_mix_prev.gather(1, x_t_idx.unsqueeze(1))
                        lik_vec = (1 - alpha_t) * prob_noise_to_xt + alpha_t * curr_x_t
                        numerator = lik_vec * q_xprev_given_x0
                        posterior_probs = numerator / (numerator.sum(dim=-1, keepdim=True) + 1e-10)

                        x_next_indices = torch.multinomial(posterior_probs, 1).squeeze(-1)
                        x_t_dict[name] = F.one_hot(x_next_indices, v['num_classes']).float()

                for name, tensor in x_t_dict.items():
                    collected_outputs[name].append(tensor.cpu())

        final_dict = {k: torch.cat(v, dim=0).to(self.device) for k, v in collected_outputs.items()}
        return final_dict

    def impute(self, m=None, save_path=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        pd.set_option('future.no_silent_downcasting', True)

        m_s = m if m else self.config["else"]["m"]
        eval_bs = self.config["train"].get("eval_batch_size", 64)

        target_data = self._global_data

        if self.config["else"]["mi_approx"] == "SWAG":
            if self.swag_model is None or self.swag_model.n_models.item() == 0:
                print("Warning: No SWAG collected. Using base model.")
            else:
                print(f"Using SWAG sampling ({self.swag_model.n_models.item()} models)")

        all_imputed_dfs = []
        pbar = tqdm(total=m_s, desc="Imputation Rounds", file=sys.stdout)

        # Perform multiple imputations (m times)
        for samp_i in range(1, m_s + 1):
            # Select model based on approximation strategy (SWAG, Dropout, Bootstrap, or Standard)
            if self.config["else"]["mi_approx"] == "SWAG" and self.swag_model.n_models.item() > 1:
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

            x_0_dict = self._reverse_diffusion(target_data, model_to_use, eval_bs)

            batch_res = []
            p2_vars = [v['name'] for v in self.variable_schema if 'p2' in v['type']]

            for name in p2_vars:
                tensor = x_0_dict[name]
                batch_res.append(tensor)

            # Reconstruct original scale from normalized tensor
            combined_tensor = torch.cat(batch_res, dim=1).cpu().numpy()
            df_p2 = inverse_transform_data(combined_tensor, self.norm_stats, self.variable_schema, self.data_info)

            # Merge imputed values into original dataframe
            df_f = self.raw_df.copy()
            for c in df_p2.columns:
                if c in df_f.columns:
                    df_f[c] = df_f[c].fillna(df_p2[c])

            df_f.insert(0, "imp_id", samp_i)
            all_imputed_dfs.append(df_f)
            pbar.update(1)

        pbar.close()

        if save_path is not None:
            final_df = pd.concat(all_imputed_dfs, ignore_index=True)
            final_df['imp_id'] = final_df['imp_id'].astype(int)
            final_df.to_parquet(save_path, index=False)
            print(f"Saved stacked imputations to: {save_path} (Shape: {final_df.shape})")
            return all_imputed_dfs
        else:
            return all_imputed_dfs