# mime_csdi.py: Unified Target-to-Proxy Diffusion Driver
# Implements Linear RDDM (Numeric) and Linear Probability D3PM (Discrete).

import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import pandas as pd
import math
from torch.optim import Adam
from tqdm import tqdm

# Ensure script directory is prioritized
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from networks import CSDI_Hybrid
from utils import inverse_transform_data, process_data


class MIME_CSDI:
    def __init__(self, config, data_info, device=None):
        self.config = config
        self.data_info = data_info
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.variable_schema = None
        self.norm_stats = None
        self.raw_df = None

        # Diffusion Schedules
        self.num_steps = config["diffusion"]["num_steps"]

        # 1. Path Schedule (Alpha): Controls the Mean Shift
        # Linearly interpolate from 1.0 (Target) -> 0.0 (Proxy)
        # We use a linear beta schedule to derive alphas, but you could just linspace alphas directly.
        # Keeping existing logic for compatibility, but interpreted linearly now.
        betas = np.linspace(1e-4, 0.02, self.num_steps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.tensor(np.cumprod(alphas)).float().to(self.device)

        # 2. Noise Config (Beta): Controls the Variance Width
        # RDDM Imputation usually wants low noise (e.g. 0.01 - 0.1)
        self.beta_end = config["diffusion"].get("beta_end", 0.1)

        # GPU Data Containers
        self._p1_data_on_device = None
        self._p2_data_on_device = None
        self._aux_data_on_device = None

    def fit(self, file_path):
        print(f"\n[MIME-CSDI] Initiating Unified Target-to-Proxy Diffusion...")
        (processed_data, processed_mask, p1_idx, p2_idx,
         weight_idx, self.variable_schema, self.norm_stats, self.raw_df) = process_data(file_path, self.data_info)

        # 1. Slice Full Data
        p1_data = processed_data[:, p1_idx]
        p2_data = processed_data[:, p2_idx]

        all_indices = set(range(processed_data.shape[1]))
        p1_p2_set = set(p1_idx) | set(p2_idx)
        if weight_idx is not None: p1_p2_set.add(weight_idx)
        aux_idx = np.array(sorted(list(all_indices - p1_p2_set)))
        aux_data = processed_data[:, aux_idx]

        # 2. Store FULL data (Needed for Imputation later)
        self._p1_data_on_device = torch.from_numpy(p1_data).float().to(self.device)
        self._p2_data_on_device = torch.from_numpy(p2_data).float().to(self.device)
        self._aux_data_on_device = torch.from_numpy(aux_data).float().to(self.device)

        # 3. IDENTIFY AUDIT SET (Rows where P2 is observed)
        p2_mask = processed_mask[:, p2_idx]
        is_audit = (p2_mask.mean(axis=1) > 0.5)
        all_audit_indices = np.where(is_audit)[0]

        # Validation Split Logic
        val_ratio = self.config["train"].get("val_ratio", 0.0)
        val_interval = self.config["train"].get("val_interval", 100)

        if val_ratio > 0:
            n_val = int(len(all_audit_indices) * val_ratio)
            perm = np.random.permutation(len(all_audit_indices))
            val_indices = all_audit_indices[perm[:n_val]]
            train_indices = all_audit_indices[perm[n_val:]]
        else:
            val_indices = []
            train_indices = all_audit_indices

        n_train = len(train_indices)
        n_val = len(val_indices)

        print(f"   [Audit Filter] Total Audit Rows: {len(all_audit_indices)}")
        print(f"   [Data Split] Training: {n_train} | Validation: {n_val}")

        # 4. Create Tensors for Train/Val
        train_p1 = self._p1_data_on_device[train_indices]
        train_p2 = self._p2_data_on_device[train_indices]
        train_aux = self._aux_data_on_device[train_indices]

        if n_val > 0:
            val_p1 = self._p1_data_on_device[val_indices]
            val_p2 = self._p2_data_on_device[val_indices]
            val_aux = self._aux_data_on_device[val_indices]

        # 5. Model Initialization
        self.model = CSDI_Hybrid(
            config=self.config,
            device=self.device,
            variable_schema=self.variable_schema,
            aux_dim=self._aux_data_on_device.shape[1]
        ).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.config["train"]["lr"])
        batch_size = self.config["train"]["batch_size"]

        # 6. Training Loop
        self.model.train()
        for epoch in range(self.config["train"]["epochs"]):
            perm = torch.randperm(n_train, device=self.device)
            it = tqdm(range(0, n_train, batch_size), desc=f"   Epoch {epoch + 1}", mininterval=1, file=sys.stdout)

            epoch_loss = []

            for i in it:
                idx = perm[i: i + batch_size]
                loss = self.calc_unified_loss(
                    p1=train_p1[idx],
                    p2=train_p2[idx],
                    aux=train_aux[idx]
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                it.set_postfix(loss=f"{loss.item():.4f}")

            # Validation Step
            if n_val > 0 and (epoch + 1) % val_interval == 0:
                val_loss = self._validate(val_p1, val_p2, val_aux, batch_size)
                print(
                    f"   [Validation] Epoch {epoch + 1} | Train Loss: {np.mean(epoch_loss):.4f} | Val Loss: {val_loss:.4f}")
                self.model.train()

        return self

    def _validate(self, p1, p2, aux, batch_size):
        self.model.eval()
        total_loss = 0
        count = 0
        N = p1.shape[0]

        with torch.no_grad():
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                batch_p1 = p1[i:end]
                batch_p2 = p2[i:end]
                batch_aux = aux[i:end]
                loss = self.calc_unified_loss(batch_p1, batch_p2, batch_aux)
                total_loss += loss.item() * (end - i)
                count += (end - i)

        return total_loss / count if count > 0 else 0.0

    def calc_unified_loss(self, p1, p2, aux):
        B = p1.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()

        # 1. ALPHA SCHEDULE (The Mean Path)
        # We use alphas_cumprod to control the interpolation.
        # Starts at ~1.0 (Target), ends at ~0.0 (Proxy).
        alpha_path = self.alphas_cumprod[t].unsqueeze(1)  # (B, 1)

        # 2. BETA SCHEDULE (The Noise Intensity)
        # Decoupled noise schedule.
        # We assume linear growth of noise variance from 0 -> beta_end
        current_time_ratio = t.unsqueeze(1).float() / self.num_steps
        max_noise_std = math.sqrt(self.beta_end)  # if beta_end is variance
        noise_scale = current_time_ratio * max_noise_std

        x_t_parts = []
        p1_cursor = 0
        p2_cursor = 0

        for var in self.variable_schema:
            if 'aux' in var['type']: continue

            curr_p1 = p1[:, p1_cursor:p1_cursor + 1]
            curr_p2 = p2[:, p2_cursor:p2_cursor + 1]

            if var['type'] == 'numeric':
                # --- LINEAR RDDM FORMULA ---
                # Mean = (1 - ratio) * Proxy + ratio * Target
                # Since alpha_path goes 1->0, it is the Target Ratio.
                signal = alpha_path * curr_p2 + (1 - alpha_path) * curr_p1

                # Independent Noise
                noise = torch.randn_like(curr_p2) * noise_scale

                x_t_parts.append(signal + noise)

            elif var['type'] == 'categorical':
                # --- LINEAR PROBABILITY DRIFT ---
                # Bernoulli mask using alpha_path as "Keep Probability"
                probs = alpha_path.expand(B, 1)
                mask = torch.bernoulli(probs)

                curr_x_t = mask * curr_p2 + (1 - mask) * curr_p1
                x_t_parts.append(curr_x_t)

            p1_cursor += 1
            p2_cursor += 1

        x_t = torch.cat(x_t_parts, dim=1)
        model_out = self.model(x_t, t, p1, aux)

        # Loss Calculation
        loss_accum = 0.0
        p2_target_cursor = 0

        for var in self.variable_schema:
            if 'aux' in var['type']: continue

            if var['type'] == 'numeric':
                pred = model_out[var['name']]
                target = p2[:, p2_target_cursor:p2_target_cursor + 1]
                loss_accum += F.mse_loss(pred, target)
                p2_target_cursor += 1
            elif var['type'] == 'categorical':
                pred_logits = model_out[var['name']]
                target = p2[:, p2_target_cursor].long()
                loss_accum += F.cross_entropy(pred_logits, target)
                p2_target_cursor += 1

        return loss_accum

    def impute(self, m=None, save_path="imputed_results.xlsx", batch_size=None):
        m_samples = m if m else self.config["else"]["m"]
        if m_samples <= 0: return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if batch_size is None:
            batch_size = self.config["train"].get("eval_batch_size", self.config["train"]["batch_size"])

        N_total = self._p1_data_on_device.shape[0]
        self.model.eval()

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for i in range(1, m_samples + 1):
                print(f"[Sample {i}/{m_samples}] Generating...", flush=True)
                all_p2_generated = []

                with torch.no_grad():
                    for start in tqdm(range(0, N_total, batch_size)):
                        end = min(start + batch_size, N_total)
                        batch_p1 = self._p1_data_on_device[start:end]
                        batch_aux = self._aux_data_on_device[start:end]
                        B = batch_p1.shape[0]

                        # Start at P1 (Proxy)
                        x_t = batch_p1.clone()

                        for t in reversed(range(self.num_steps)):
                            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
                            model_out = self.model(x_t, t_batch, batch_p1, batch_aux)

                            x_next_parts = []
                            p1_cursor = 0

                            # Get alpha for next step (t-1)
                            # alpha_prev represents the target ratio we want to achieve
                            alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0).to(self.device)

                            for var in self.variable_schema:
                                if 'aux' in var['type']: continue

                                pred = model_out[var['name']]  # Predicted P2 (Target)
                                curr_p1 = batch_p1[:, p1_cursor:p1_cursor + 1]

                                if var['type'] == 'numeric':
                                    # --- LINEAR REVERSE STEP ---
                                    # Move mean linearly: Target * alpha_prev + Proxy * (1 - alpha_prev)
                                    mean_next = alpha_prev * pred + (1 - alpha_prev) * curr_p1

                                    # Add scaled noise for diversity
                                    if t > 0:
                                        # Use same scaling logic as forward but reduced for sampling stability
                                        time_ratio = float(t - 1) / self.num_steps
                                        sigma = time_ratio * math.sqrt(self.beta_end)
                                        x_next = mean_next + torch.randn_like(mean_next) * sigma
                                    else:
                                        x_next = mean_next

                                    x_next_parts.append(x_next)

                                elif var['type'] == 'categorical':
                                    # --- PROBABILITY SWAP ---
                                    probs_x0 = F.softmax(pred, dim=-1)
                                    candidate_x0 = torch.multinomial(probs_x0, 1).float()

                                    # As t -> 0, flip_prob should increase (trust prediction more)
                                    # Simple heuristic: 1/(t+1) chance to update from current state
                                    flip_prob = 1.0 / (t + 1)
                                    flip_mask = torch.bernoulli(torch.full((B, 1), flip_prob, device=self.device))

                                    curr_slice = x_t[:, len(x_next_parts):len(x_next_parts) + 1]
                                    x_next = flip_mask * candidate_x0 + (1 - flip_mask) * curr_slice
                                    x_next_parts.append(x_next)

                                p1_cursor += 1

                            x_t = torch.cat(x_next_parts, dim=1)

                        all_p2_generated.append(x_t.cpu())

                full_p2 = torch.cat(all_p2_generated, dim=0).numpy()
                df_partial = inverse_transform_data(full_p2, self.norm_stats, self.data_info)

                df_final = self.raw_df.copy()
                for col in df_partial.columns:
                    if col in df_final.columns:
                        df_final[col] = df_final[col].fillna(df_partial[col])

                df_final.to_excel(writer, sheet_name=f"Imputation_{i}", index=False)

        print(f"\nSaved {m_samples} samples to {save_path}.")