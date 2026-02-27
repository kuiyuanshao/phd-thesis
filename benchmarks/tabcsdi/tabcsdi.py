import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import typing as ty
import math
import sys
from pathlib import Path
def setup_project_paths():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == 'gain':
            project_root = parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            return
    sys.path.insert(0, str(current_path.parent))
setup_project_paths()
from diff_models_table import diff_CSDI
from data_transformer import DataTransformer

class Tokenizer(nn.Module):
    def __init__(
            self,
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            d_token: int,
            bias: bool,
    ) -> None:
        super().__init__()

        d_bias = d_numerical + len(categories)
        category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.d_token = d_token
        self.register_buffer("category_offsets", category_offsets)
        self.category_embeddings = nn.Embedding(sum(categories) + 1, self.d_token)
        self.category_embeddings.weight.requires_grad = False
        nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(Tensor(d_numerical, self.d_token))
        self.weight.requires_grad = False

        self.bias = nn.Parameter(Tensor(d_bias, self.d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            self.bias.requires_grad = False

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        x_cat = x_cat.type(torch.int32) if x_cat is not None else None

        assert x_some is not None
        x = self.weight.T * x_num

        x = x[:, np.newaxis, :, :]
        x = x.permute(0, 1, 3, 2)
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=2,
            )
        if self.bias is not None:
            x = x + self.bias[None]
        return x

    def recover(self, Batch, d_numerical):
        B, L, K = Batch.shape
        L_new = int(L / self.d_token)
        Batch = Batch.reshape(B, L_new, self.d_token)
        Batch = Batch - self.bias

        Batch_numerical = Batch[:, :d_numerical, :]
        Batch_numerical = Batch_numerical / self.weight
        Batch_numerical = torch.mean(Batch_numerical, 2, keepdim=False)

        if Batch.shape[1] > d_numerical:
            Batch_cat = Batch[:, d_numerical:, :]
            new_Batch_cat = torch.zeros([Batch_cat.shape[0], Batch_cat.shape[1]])
            for i in range(Batch_cat.shape[1]):
                token_start = self.category_offsets[i] + 1
                if i == Batch_cat.shape[1] - 1:
                    token_end = self.category_embeddings.weight.shape[0] - 1
                else:
                    token_end = self.category_offsets[i + 1]
                emb_vec = self.category_embeddings.weight[token_start: token_end + 1, :]
                for j in range(Batch_cat.shape[0]):
                    distance = torch.norm(emb_vec - Batch_cat[j, i, :], dim=1)
                    nearest = torch.argmin(distance)
                    new_Batch_cat[j, i] = nearest + 1
            new_Batch_cat = new_Batch_cat.to(Batch_numerical.device)
            return torch.cat([Batch_numerical, new_Batch_cat], dim=1)
        return Batch_numerical


class TabCSDIDataset(Dataset):
    def __init__(self, observed_values, observed_masks):
        self.observed_values = observed_values
        self.observed_masks = observed_masks
        self.num_features = observed_values.shape[1]

    def __len__(self):
        return len(self.observed_values)

    def __getitem__(self, index):
        return {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.observed_masks[index],
            "timepoints": np.arange(self.num_features),
        }


class TabCSDI(nn.Module):
    def __init__(self, config, data_info, device=None):
        super().__init__()
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

        self.transformer = DataTransformer(self.config, self.data_info)
        self.df_raw = None

        # Model components (Initialized dynamically in fit)
        self.tokenizer = None
        self.embed_layer = None
        self.diffmodel = None

    def _build_networks(self, cont_list, num_cate_list):
        self.cont_list = cont_list
        self.num_cate_list = num_cate_list

        self.emb_time_dim = self.config["model"]["timeemb"]
        self.emb_feature_dim = self.config["model"]["featureemb"]
        self.is_unconditional = self.config["model"].get("is_unconditional", False)
        self.token_dim = self.config["model"].get("token_emb_dim", 1)

        self.tokenizer = Tokenizer(d_numerical=len(cont_list), categories=num_cate_list, d_token=self.token_dim,
                                   bias=True)

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + (0 if self.is_unconditional else 1)
        self.embed_layer = nn.Embedding(num_embeddings=1, embedding_dim=self.emb_feature_dim)

        config_diff = self.config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5,
                                     self.num_steps) ** 2)
        else:
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

        self.to(self.device)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def fit(self, file_path=None, provided_data=None):
        if provided_data is None:
            print(f"Loading data from {file_path}...")
            self.df_raw = pd.read_csv(file_path).reset_index(drop=True)
            self.df_raw = self.df_raw.loc[:, ~self.df_raw.columns.str.contains('^Unnamed')]
        else:
            self.df_raw = provided_data.copy()

        self.df_raw.columns = self.df_raw.columns.str.strip()

        for col in self.data_info.get('num_vars', []):
            if col in self.df_raw.columns:
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')

        self.transformer.fit(self.df_raw)
        transformed_values, observed_masks = self.transformer.transform(self.df_raw)

        cont_indices, num_cate_list = self.transformer.get_model_metadata()
        self._build_networks(cont_indices, num_cate_list)

        dataset = TabCSDIDataset(transformed_values, observed_masks)
        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'], shuffle=True)

        optimizer = Adam(self.parameters(), lr=self.config["train"]["lr"], weight_decay=1e-6)
        epochs = self.config["train"]["epochs"]
        p0, p1, p2, p3 = int(0.25 * epochs), int(0.5 * epochs), int(0.75 * epochs), int(0.9 * epochs)
        lr_scheduler = MultiStepLR(optimizer, milestones=[p0, p1, p2, p3], gamma=0.1)

        print(f"Starting TabCSDI training for {epochs} epochs...")
        self.train()
        for epoch in range(epochs):
            avg_loss = 0
            with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False) as it:
                for batch_no, batch in enumerate(it, start=1):
                    optimizer.zero_grad()
                    loss = self.forward(batch)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
                    it.set_postfix(ordered_dict={"avg_loss": avg_loss / batch_no})
            lr_scheduler.step()
        print("Training complete.")

    def impute(self, m=None, save_path=None):
        m = m if m is not None else self.config.get('sample', {}).get('m', 5)
        print(f"Generating {m} Imputations...")
        transformed_values, observed_masks = self.transformer.transform(self.df_raw)
        dataset = TabCSDIDataset(transformed_values, observed_masks)
        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'], shuffle=False)
        self.eval()
        all_samples = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Imputing Batches"):
                observed_data, observed_mask, observed_tp, gt_mask = self._process_batch(batch)
                side_info = self.get_side_info(observed_tp, gt_mask)
                batch_samples = self._reverse_diffusion(observed_data, gt_mask, side_info, m)
                all_samples.append(batch_samples)

        full_generated_data = torch.cat(all_samples, dim=0)
        all_imputed_dfs = []

        for i in range(m):
            samples_i = full_generated_data[:, i, :, :]
            samples_i = samples_i.permute(0, 2, 1)
            if self.config['model']['task'] == "ft":
                recovered = self.tokenizer.recover(samples_i, len(self.cont_list)).cpu().numpy()
            else:
                recovered = samples_i.squeeze(2).cpu().numpy()  # Just remove the K dimension
            df_denorm = self.transformer.inverse_transform(recovered)
            df_f = self.df_raw.copy()
            for col in df_denorm.columns:
                if col in df_f.columns:
                    try:
                        df_denorm[col] = df_denorm[col].astype(df_f[col].dtype)
                    except ValueError:
                        pass
                df_f[col] = df_f[col].fillna(df_denorm[col])

            df_f['imp_id'] = i + 1
            all_imputed_dfs.append(df_f)

        if save_path:
            final_df = pd.concat(all_imputed_dfs, ignore_index=True)
            final_df.to_parquet(save_path, index=False)
            print(f"Saved stacked imputations to: {save_path}")

        return all_imputed_dfs

    def _reverse_diffusion(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        cond_mask_rep = torch.repeat_interleave(cond_mask, self.token_dim, dim=2)
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask_rep * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask_rep) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)

                B_in, old_input_dim, K_in, L_in = diff_input.shape
                diff_input = diff_input.reshape(B_in, old_input_dim, K_in, int(L_in / self.token_dim),
                                                self.token_dim).permute(0, 1, 4, 2, 3)
                diff_input = diff_input.reshape(B_in, old_input_dim * self.token_dim, K_in, int(L_in / self.token_dim))

                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def _process_batch(self, batch):
        observed_data = batch["observed_data"][:, np.newaxis, :].to(self.device).float()

        x_num = observed_data[:, :, :len(self.cont_list)]
        x_cat = observed_data[:, :, len(self.cont_list):] if len(self.num_cate_list) > 0 else None

        if self.config['model']['task'] == "ft":
            x_num = observed_data[:, :, :len(self.cont_list)]
            x_cat = observed_data[:, :, len(self.cont_list):] if len(self.num_cate_list) > 0 else None

            observed_data = self.tokenizer(x_num, x_cat)
            B, K, L, C = observed_data.shape
            observed_data = observed_data.reshape(B, K, L * C)

        observed_mask = batch["observed_mask"][:, np.newaxis, :].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"][:, np.newaxis, :].to(self.device).float()
        return observed_data, observed_mask, observed_tp, gt_mask

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)

        # Hardcoded to 1 as Tabular data does not use the variable target_dim.
        # This prevents it from crashing when target_dim is undefined.
        feature_embed = self.embed_layer(torch.arange(1).to(self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info):
        B, K, L = observed_data.shape
        t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)

        # Perform forward step. Adding noise to all data.
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)

        target_mask = observed_mask - cond_mask
        target_mask = torch.repeat_interleave(target_mask, self.token_dim, dim=2)
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_mask = torch.repeat_interleave(cond_mask, self.token_dim, dim=2)

        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        B, old_input_dim, K, L = total_input.shape
        total_input = total_input.reshape(
            B, old_input_dim, K, int(L / self.token_dim), self.token_dim
        )
        total_input = total_input.permute(0, 1, 4, 2, 3)
        total_input = total_input.reshape(
            B, old_input_dim * self.token_dim, K, int(L / self.token_dim)
        )

        return total_input

    def forward(self, batch):
        observed_data, observed_mask, observed_tp, _ = self._process_batch(batch)
        cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)

        # Directly call calc_loss without checking for validation triggers
        return self.calc_loss(observed_data, cond_mask, observed_mask, side_info)