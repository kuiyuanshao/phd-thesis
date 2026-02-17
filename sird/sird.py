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
from swag import SWAG
from data_transformer import DataTransformer


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
        self.task = config["else"]["task"]

        # Diffusion schedules (Original Logic)
        b = torch.linspace(0, 1, self.num_steps).to(self.device) ** 3
        a = torch.flip(torch.linspace(0, 1, self.num_steps).to(self.device), dims=[0])
        alphas = a / a.sum()
        betas = b / b.sum() * self.sum_scale
        betas_cumsum = betas.cumsum(dim=0).clip(0, 1)

        self.alpha_bars = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(self.alpha_bars[:-1], (1, 0),
                                   value=self.alpha_bars[1])  # Padding logic from original

        betas_cumsum_prev = F.pad(betas_cumsum[:-1], (1, 0), value=betas_cumsum[1])
        self.beta_bars = torch.sqrt(betas_cumsum)

        # Posterior params (Original Logic)
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

        self._global_data = None
        self.variable_schema = []
        self.transformer = None
        self.Q_dict = {}

        self.num_vars = []
        self.cat_vars = []
        self.p1_slices = {}
        self.p2_slices = {}
        self.aux_slices = {}

    def _map_schema_indices(self):
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

        # 1. Process Data
        if provided_data is None and file_path is not None:
            # Read Raw
            df_raw = pd.read_csv(file_path)
            df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]
            for col in self.data_info.get('num_vars', []):
                if col in df_raw.columns:
                    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

            weight_col = self.data_info.get('weight_var')
            if weight_col and weight_col in df_raw.columns:
                weights_raw = df_raw[weight_col].fillna(0).values
            else:
                weights_raw = np.ones(len(df_raw))

            # Initialize and Fit Transformer
            self.transformer = DataTransformer(self.data_info, self.config)
            self.transformer.fit(df_raw)

            # Transform
            df_proc = self.transformer.transform(df_raw)
            # --- DIAGNOSIS START ---
            # --- DIAGNOSIS START ---
            print("\n[DEBUG] Checking Mode Count Consistency...")
            p1_vars = self.data_info.get('phase1_vars', [])
            p2_vars = self.data_info.get('phase2_vars', [])
            pair_map = {p2: p1 for p1, p2 in zip(p1_vars, p2_vars)}

            for p2 in p2_vars:
                if p2 in self.data_info['num_vars']:
                    p1 = pair_map.get(p2)
                    if not p1: continue

                    # Count generated mode columns in the transformed dataframe
                    n_p1 = len([c for c in df_proc.columns if c.startswith(f"{p1}_mode_")])
                    n_p2 = len([c for c in df_proc.columns if c.startswith(f"{p2}_mode_")])

                    # Only print if there is a mismatch
                    if n_p1 != n_p2:
                        print(f"!!! MODE MISMATCH: {p1} ({n_p1} modes) vs {p2} ({n_p2} modes)")
            print("[DEBUG] Complete.\n")
            # --- DIAGNOSIS END ---
            # --- DIAGNOSIS END ---
            self.raw_df = df_raw

            # Schema & Q
            self.variable_schema = self.transformer.get_sird_schema()
            self._map_schema_indices()
            self.Q_dict = {k: v.to(self.device) for k, v in self.transformer.Q_matrices.items()}

            # Create Tensor
            self._global_data = torch.from_numpy(df_proc.values).float().to(self.device)
            self._global_weights = torch.from_numpy(weights_raw).double().to(self.device)

            # Define Valid Rows
            # Use presence of a primary Phase 2 variable
            p2_check = next((v['name'] for v in self.variable_schema if 'p2' in v['type'] and '_mode' not in v['name']),
                            None)
            if p2_check:
                if p2_check in df_raw.columns:
                    valid_mask = df_raw[p2_check].notna().values
                    valid_rows = np.where(valid_mask)[0]
                else:
                    valid_rows = np.arange(len(df_raw))  # Fallback
            else:
                valid_rows = np.arange(len(df_raw))

            print(f"[Fit] Global Valid Rows (Observed Phase 2): {len(valid_rows)}")

        elif provided_data is not None:
            self._global_data = provided_data['data'].float().to(self.device)
            self._global_weights = torch.from_numpy(provided_data['weights']).double()
            self.variable_schema = provided_data['schema']
            self.Q_dict = provided_data.get('Q_dict', {})
            valid_rows = np.arange(self._global_data.shape[0])
            self._map_schema_indices()

        # 3. Training Loop
        num_train_mods = self.config["else"]["m"] if mi_approx == "bootstrap" else 1

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
                loss, num, cat = self.calc_loss(b_data)
                loss.backward()

                if mi_approx == "SWAG":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if step % 50 == 0:
                    pbar.set_postfix(mse=f"{num.item():.4f}",
                                     kl_div=f"{cat.item():.4f}",)

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
        B = batch_data.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        a_bar = self.alpha_bars[t].view(B, 1)
        b_bar = self.beta_bars[t].view(B, 1)

        x_t_dict, p1_dict, aux_dict, eps_dict = {}, {}, {}, {}

        # 1. Numeric Variables (Phase 2)
        numeric_p2 = [v for v in self.num_vars if 'p2' in v['type']]
        if len(numeric_p2) > 0:
            for v in numeric_p2:
                name = v['name']
                partner = v['pair_partner']

                p2_val = batch_data[:, self.p2_slices[name]]
                p1_val = batch_data[:, self.p1_slices[partner]]

                true_residual = p1_val - p2_val
                eps = torch.randn_like(p2_val)

                # As a_bar grows (0->1), x_t moves from True(p2) to Proxy(p1)
                x_t_dict[name] = p2_val + a_bar * true_residual + b_bar * eps
                p1_dict[name] = p1_val
                eps_dict[name] = eps

        # 2. Categorical Variables (Phase 2 + Modes)

        cat_p2_vars = [v for v in self.cat_vars if 'p2' in v['type']]
        if len(cat_p2_vars) > 0:
            for v in cat_p2_vars:
                name = v['name']
                K = v['num_classes']
                partner = v['pair_partner']

                oh_p2 = batch_data[:, self.p2_slices[name]]

                # BRIDGE: Project Proxy to True Dimensionality via Q
                if partner is not None and name in self.Q_dict:
                    oh_p1 = batch_data[:, self.p1_slices[partner]]
                    Q = self.Q_dict[name]
                    pi_prior = oh_p1 @ Q  # (B, N) @ (N, M) -> (B, M)
                    pi_prior = pi_prior * 0.99 + 0.01 / K
                    p1_dict[name] = oh_p1  # Raw proxy for conditioning
                else:
                    # Fallback for modes (no proxy partner) or missing Q
                    pi_prior = torch.ones_like(oh_p2) / oh_p2.shape[1]
                    p1_dict[name] = torch.zeros(B, 1).to(self.device)  # Dummy? Or handle in network?
                eps_0 = 1e-30
                log_true = torch.log(oh_p2 + eps_0)
                log_proxy = torch.log(pi_prior + eps_0)
                term_true = torch.log((1.0 - a_bar) + eps_0) + log_true
                term_proxy = torch.log(a_bar + eps_0) + log_proxy
                log_probs_xt = torch.logaddexp(term_true, term_proxy)
                u_noise = torch.rand_like(log_probs_xt)
                gumbel = -torch.log(-torch.log(u_noise + eps_0) + eps_0)
                x_t_indices = (log_probs_xt + gumbel).argmax(dim=1)
                x_t_dict[name] = F.one_hot(x_t_indices, num_classes=K).float()

        # 3. Aux Variables
        for v in self.variable_schema:
            if 'aux' in v['type']:
                aux_dict[v['name']] = batch_data[:, self.aux_slices[v['name']]]

        # Model Forward
        model_out = self.model(x_t_dict, t, p1_dict, aux_dict)
        loss_num = torch.tensor(0.0, device=self.device)
        loss_cat = torch.tensor(0.0, device=self.device)
        # Numeric Loss
        if len(numeric_p2) > 0:
            for v in numeric_p2:
                name = v['name']
                true_residual = p1_dict[name] - batch_data[:, self.p2_slices[name]]
                pred_out = model_out[name]

                if self.task == "Res-N":
                    # Split channels
                    pred_res = pred_out[:, 0:1]
                    pred_eps = pred_out[:, 1:2]
                    # Original logic: MSE(res) + MSE(eps)
                    # Note: eps variable here must match the eps sampled above (which it does)
                    loss_num += (self.gamma * F.mse_loss(pred_res, true_residual) +
                                   F.mse_loss(pred_eps, eps_dict[name]))
                elif self.task == "Res":
                    loss_num += self.gamma * F.mse_loss(pred_out, true_residual)
                elif self.task == "N":
                    loss_num += F.mse_loss(pred_out, eps_dict[name])

        # Categorical Loss (KL Divergence)
        # ... Inside calc_loss categorical loop (Line 351) ...
        for v in cat_p2_vars:
            name = v['name']
            K = v['num_classes']
            x_t = x_t_dict[name]
            x_true = batch_data[:, self.p2_slices[name]]

            if v['pair_partner'] is not None and name in self.Q_dict:
                oh_p1 = batch_data[:, self.p1_slices[v['pair_partner']]]
                pi_prior = oh_p1 @ self.Q_dict[name]
            else:
                pi_prior = torch.ones_like(x_true) / x_true.shape[1]

            # FIX: UNINDENT THESE LINES. They must run for EVERY variable,
            # not just those without partners.
            pred_logits = model_out[name]
            pred_x0_probs = F.softmax(pred_logits, dim=-1)

            x_true_smooth = x_true * (1.0 - 1e-2) + (1e-2 / K)
            log_true_posterior = self._compute_posterior(x_t, x_true_smooth, pi_prior, t)
            log_model_posterior = self._compute_posterior(x_t, pred_x0_probs, pi_prior, t)

            kl = F.kl_div(log_model_posterior, log_true_posterior, reduction='batchmean',
                          log_target=True)
            loss_cat += kl

        return loss_num + self.zeta * loss_cat, loss_num, self.zeta * loss_cat

    def _compute_posterior(self, x_t, x_start, x_proxy, t):
        x_start = x_start.clamp(min=1e-6, max=1.0 - 1e-6)
        K = x_start.shape[1]
        eps = 1e-30

        # --- FIX START: Derive consistent Beta from Alpha ---
        # Get cumulative Alphas (Probability of being Proxy)
        curr_alpha_bar = self.alpha_bars.to(x_t.device)[t].view(-1, 1)

        # Handle t=0 edge case for previous alpha
        # If t=0, prev_alpha_bar is 0 (Probability of being Proxy is 0)
        idx_prev = (t - 1).clamp(min=0).view(-1, 1)
        prev_alpha_bar = self.alpha_bars.to(x_t.device)[idx_prev]
        prev_alpha_bar[t == 0] = 0.0

        # Calculate the STRICT mathematical beta derived from your forward schedule
        # Formula: (1 - curr) = (1 - prev) * (1 - beta)
        # Therefore: beta = 1 - (1 - curr) / (1 - prev)
        numerator = 1.0 - curr_alpha_bar
        denominator = 1.0 - prev_alpha_bar

        # Clamp denominator to avoid division by zero if alpha saturates
        fraction = numerator / (denominator + eps)

        # This is the TRUE one-step transition probability consistent with your forward pass
        beta_t = (1.0 - fraction).clamp(min=1e-5, max=1.0 - 1e-5)
        # --- FIX END ---

        # 1. Log-Prior at t-1: q(x_{t-1} | x_0, x_proxy)
        # This part was mostly correct, just ensures consistent weights
        w_s_prev = 1.0 - prev_alpha_bar
        w_p_prev = prev_alpha_bar

        log_true_val = torch.log(x_start + eps)
        log_proxy_val = torch.log(x_proxy + eps)

        log_prior = torch.logaddexp(
            torch.log(w_s_prev + eps) + log_true_val,
            torch.log(w_p_prev + eps) + log_proxy_val
        )

        # 2. Log-Likelihood: q(x_t | x_{t-1}, x_proxy)
        # Use the beta_t we just derived.
        # Logic: If I stay (1-beta), I keep state. If I transition (beta), I become Proxy.

        # Compute "Noise Likelihood" (Transition to Proxy)
        # This relies on the TARGET state x_t, not the source.
        p_proxy_at_xt = (x_t * x_proxy).sum(dim=1, keepdim=True)
        log_likelihood_noise = torch.log(beta_t * p_proxy_at_xt + eps)

        # Compute "Stay Likelihood" (Keep state + Chance of random match)
        # P(xt | xt-1=xt) = (1-beta) + beta * P(xt=xt|Proxy)
        log_likelihood_stay = torch.log((1.0 - beta_t) + beta_t * p_proxy_at_xt + eps)

        # Broadcast to all possible previous states x_{t-1}
        # If x_{t-1} matches x_t (x_t=1), we use Stay Likelihood.
        # If x_{t-1} differs (x_t=0), we use Noise Likelihood.
        log_likelihood = x_t * log_likelihood_stay + (1.0 - x_t) * log_likelihood_noise

        # 3. Bayes
        log_unnormalized = log_prior + log_likelihood
        log_z = torch.logsumexp(log_unnormalized, dim=1, keepdim=True)
        return log_unnormalized - log_z

    def _reverse_diffusion(self, full_data, model_to_use, batch_size=2048):
        N = full_data.shape[0]
        collected_outputs = {var['name']: [] for var in self.variable_schema if 'p2' in var['type']}

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                B = end - start
                b_data = full_data[start:end]

                x_t_dict, p1_dict, aux_dict = {}, {}, {}

                # 1. Initialize x_T

                # Numeric: x_T = Proxy + Noise
                numeric_p2 = [v for v in self.num_vars if 'p2' in v['type']]
                for v in numeric_p2:
                    name = v['name']
                    partner = v['pair_partner']

                    # Proxy (Anchored P1)
                    curr_p1 = b_data[:, self.p1_slices[partner]]
                    init_eps = torch.randn_like(curr_p1)
                    # Use Sum Scale for initial noise magnitude?
                    # Original sird.py: x_T_num = curr_p1_num + torch.sqrt(self.sum_scale) * init_eps
                    x_t_dict[name] = curr_p1 + torch.sqrt(self.sum_scale) * init_eps
                    p1_dict[name] = curr_p1

                # 1. Get Schedule weights at T (Final Step)
                # Normalization Z = a_bar_T + b_bar_T
                cat_p2_vars = [v for v in self.cat_vars if 'p2' in v['type']]
                for v in cat_p2_vars:
                    name = v['name']
                    partner = v['pair_partner']
                    K = v['num_classes']

                    if partner is not None and name in self.Q_dict:
                        oh_p1 = b_data[:, self.p1_slices[partner]]
                        pi_prior = oh_p1 @ self.Q_dict[name]
                        pi_prior = pi_prior * 0.99 + 0.01 / K
                        p1_dict[name] = oh_p1
                    else:
                        pi_prior = torch.ones((B, K), device=self.device) / K
                        p1_dict[name] = torch.zeros((B, K), device=self.device)

                    eps_0 = 1e-30
                    log_proxy = torch.log(pi_prior + eps_0)

                    # 2. Gumbel Noise
                    u_noise = torch.rand_like(log_proxy)
                    gumbel = -torch.log(-torch.log(u_noise + eps_0) + eps_0)

                    # 3. Argmax to get sample
                    x_T_idx = (log_proxy + gumbel).argmax(dim=1)
                    x_t_dict[name] = F.one_hot(x_T_idx, K).float()
                # Aux
                for v in self.variable_schema:
                    if 'aux' in v['type']:
                        aux_dict[v['name']] = b_data[:, self.aux_slices[v['name']]]

                # Loop
                for t in reversed(range(self.num_steps)):
                    t_b = torch.full((B,), t, device=self.device).long()
                    out = model_to_use(x_t_dict, t_b, p1_dict, aux_dict)

                    # Update Numeric
                    for v in numeric_p2:
                        name = v['name']
                        x_t = x_t_dict[name]
                        x_proxy = p1_dict[name]

                        # Res-N extraction
                        if self.task == "Res-N":
                            pred_res = out[name][:, 0:1]
                            pred_eps = out[name][:, 1:2]
                            # Original update formula
                            # x_start_pred calculation
                            x_start_pred = x_t - self.alpha_bars[t] * pred_res - self.beta_bars[t] * pred_eps
                            x_start_pred = x_start_pred.clamp(-5.0, 5.0)
                        elif self.task == "Res":
                            pred_res = out[name]
                            # Infer eps from residual
                            # This part is tricky without implicit epsilon.
                            # Using original sird formula:
                            pred_eps = (x_t - x_proxy - (self.alpha_bars[t] - 1) * pred_res) / self.beta_bars[t]
                            x_start_pred = x_t - self.alpha_bars[t] * pred_res - self.beta_bars[t] * pred_eps
                            x_start_pred = x_start_pred.clamp(-5.0, 5.0)
                        elif self.task == "N":
                            pred_eps = out[name]
                            x_start_pred = (x_t - self.alpha_bars[t] * x_proxy - self.beta_bars[t] * pred_eps) / \
                                           (1 - self.alpha_bars[t]).clamp(min=1e-5)
                            x_start_pred = x_start_pred.clamp(-5.0, 5.0)
                            pred_res = x_proxy - x_start_pred

                        # Posterior Mean Update
                        posterior_mean = self.posterior_mean_coef1[t] * x_t + \
                                         self.posterior_mean_coef2[t] * pred_res + \
                                         self.posterior_mean_coef3[t] * x_start_pred

                        noise = torch.randn_like(x_t) if t > 0 else 0
                        x_t_dict[name] = posterior_mean + (0.5 * self.posterior_log_variance[t]).exp() * noise

                    # Update Categorical
                    for v in cat_p2_vars:
                        name = v['name']
                        K = v['num_classes']
                        logits = out[name]
                        pred_x0_probs = F.softmax(logits, dim=-1)

                        # Recompute Prior for this batch
                        if v['pair_partner'] is not None and name in self.Q_dict:
                            oh_p1 = p1_dict[name]
                            pi_prior = oh_p1 @ self.Q_dict[name]
                            pi_prior = pi_prior * 0.99 + 0.01 / K
                        else:
                            pi_prior = torch.ones_like(pred_x0_probs) / pred_x0_probs.shape[1]

                        log_posterior = self._compute_posterior(x_t_dict[name],
                                                                pred_x0_probs, pi_prior, t_b)
                        u_noise = torch.rand_like(log_posterior)
                        gumbel = -torch.log(-torch.log(u_noise + eps_0) + eps_0)
                        x_t_noise = (log_posterior + gumbel)
                        if t == 0:
                            x_t_dict[name] = F.softmax(x_t_noise, dim=-1)
                        else:
                            x_t_dict[name] = F.one_hot(x_t_noise.argmax(dim=1), K).float()

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

        for samp_i in range(1, m_s + 1):
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

            # Reverse Diffusion
            x_0_dict = self._reverse_diffusion(target_data, model_to_use, eval_bs)

            gen_df_dict = {}
            for v in self.variable_schema:
                if 'p2' in v['type']:
                    # x_0_dict[name] is (Batch, 1) for numeric, (Batch, K) for categorical
                    val = x_0_dict[v['name']].cpu().numpy()

                    if 'numeric' in v['type']:
                        # FIX: Use the original name (e.g. INCOME).
                        # DataTransformer.inverse_transform looks for this exact key.
                        gen_df_dict[v['name']] = val.flatten()

                    elif 'categorical' in v['type']:
                        start = v['start_idx']
                        num_cls = v['num_classes']
                        cols = self.transformer.generated_columns[start: start + num_cls]
                        for k, col_name in enumerate(cols):
                            gen_df_dict[col_name] = val[:, k]

            # 2. Reconstruct into DataFrame
            df_gen_p2 = pd.DataFrame(gen_df_dict)

            # 3. Inverse Transform (Denormalize and map categories)
            df_denorm = self.transformer.inverse_transform(df_gen_p2)

            # 4. FIX: Align Index before Merge
            # fillna matches by index. If raw_df index is non-standard, merge will fail without this.
            df_f = self.raw_df.copy()
            df_denorm.index = df_f.index

            # 5. Merge Generated Data into original gaps
            for c in df_denorm.columns:
                if c in df_f.columns:
                    df_f[c] = df_f[c].fillna(df_denorm[c])

            df_f.insert(0, "imp_id", samp_i)
            all_imputed_dfs.append(df_f)
            pbar.update(1)

        pbar.close()

        if save_path is not None:
            final_df = pd.concat(all_imputed_dfs, ignore_index=True)
            # Final conversion for Parquet compatibility
            for col in final_df.columns:
                if final_df[col].dtype == 'object':
                    final_df[col] = final_df[col].astype('string')

            final_df.to_parquet(save_path, index=False)
            print(f"Saved stacked imputations to: {save_path}")
            return all_imputed_dfs

        return all_imputed_dfs