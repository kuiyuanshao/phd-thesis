from torch import pca_lowrank
import pandas as pd
import torch
import torch.nn.functional as F

import torch


def pairwise_distances_squared(x):
    # (Batch, Dim) -> (Batch, Batch)
    x_norm = (x ** 2).sum(1).view(-1, 1)
    dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
    return torch.clamp(dist, 0.0, float('inf'))


def standardize_batch(x):
    if x.shape[1] == 0: return x
    std = x.std(dim=0, keepdim=True)
    mask = std > 1e-6
    x_out = x.clone()
    if mask.any():
        mean = x.mean(dim=0, keepdim=True)
        x_out[:, mask.squeeze()] = (x[:, mask.squeeze()] - mean[:, mask.squeeze()]) / (std[:, mask.squeeze()] + 1e-8)
    return x_out


def prepare_and_concat(data, layout):
    processed_chunks = []
    for start, end, card in layout:
        chunk = data[:, start:end]
        if card == 0:
            chunk_norm = standardize_batch(chunk)
            processed_chunks.append(chunk_norm)
        else:
            processed_chunks.append(chunk)

    if not processed_chunks:
        return None
    return torch.cat(processed_chunks, dim=1)


def calc_HSIC_loss(x_fake, x_real, attrs, layout_res, layout_attr, lambda_hsic=1.0):
    diff = x_fake - x_real
    diff_combined = prepare_and_concat(diff, layout_res)
    attrs_combined = prepare_and_concat(attrs, layout_attr)

    dist_res = pairwise_distances_squared(diff_combined)
    dist_attr = pairwise_distances_squared(attrs_combined)

    # 3. 计算 Sigma (Global Median)
    sigma_res = torch.median(dist_res).detach()
    if sigma_res < 1e-6: sigma_res = torch.tensor(1.0, device=diff.device)

    sigma_attr = torch.median(dist_attr).detach()
    if sigma_attr < 1e-6: sigma_attr = torch.tensor(1.0, device=diff.device)

    # 4. Kernel & HSIC
    K_res = torch.exp(-dist_res / sigma_res)
    L_attr = torch.exp(-dist_attr / sigma_attr)

    # Loss A: Set Independence
    loss_set = hsic_normalized(K_res, L_attr)

    # Loss B: Pairwise Independence (One-vs-Rest)
    # 依然可以用减法，因为是在同一个距离矩阵上操作
    loss_pairwise = 0.0

    # 为了做 Pairwise，我们需要知道每个 block 在 combined 矩阵里的索引范围
    # 重新扫一遍 layout 算索引
    current_idx = 0
    n_blocks = 0

    # 预先计算 diff_combined 的总距离用于减法
    # dist_res 已经是总距离了

    for s, e, c in layout_res:
        width = e - s
        # 提取当前 block 的数据 (注意：是处理过的数据)
        # 这里的切片是在 combined 矩阵上切
        target_data = diff_combined[:, current_idx: current_idx + width]

        # 1. Target Kernel
        d_target = pairwise_distances_squared(target_data)
        K_target = torch.exp(-d_target / sigma_res)  # 共享 Sigma

        # 2. Rest Kernel (直接减法！)
        # Rest = Total - Target
        d_rest = torch.clamp(dist_res - d_target, min=0.0)
        K_rest = torch.exp(-d_rest / sigma_res)

        loss_pairwise += hsic_normalized(K_target, K_rest)

        current_idx += width
        n_blocks += 1

    if n_blocks > 1:
        loss_pairwise /= n_blocks

    return lambda_hsic * torch.relu((loss_set + loss_pairwise) - 0.0009)


# 辅助函数
def hsic_normalized(K, L):
    m = K.shape[0]
    H = torch.eye(m, device=K.device) - 1.0 / m * torch.ones((m, m), device=K.device)
    Kc = torch.mm(H, torch.mm(K, H))
    Lc = torch.mm(H, torch.mm(L, H))
    return torch.trace(torch.mm(Kc, Lc)) / (m * m)

def project_categorical(fake, proj_groups):
    """
    Vectorized projection of Phase-2 logits to Phase-1 space.
    Batches operations by variable cardinality to avoid Python loops.
    """
    projections = {}
    if not proj_groups:
        return projections

    grouped_by_card = {}
    for g in proj_groups:
        c = g['card']
        if c not in grouped_by_card:
            grouped_by_card[c] = []
        grouped_by_card[c].append(g)

    for card, groups in grouped_by_card.items():
        matrices = torch.stack([g['matrix'] for g in groups], dim=0)
        slices = [fake[:, g['p2_range'][0]:g['p2_range'][1]] for g in groups]
        fake_batch = torch.stack(slices, dim=1)

        probs = F.softmax(fake_batch, dim=2)
        proj_probs = torch.einsum('bnc, ncd -> bnd', probs, matrices)

        # E. Log & Clamp (Vectorized)
        proj_logits = torch.log(torch.clamp(proj_probs, min=1e-8, max=1.0 - 1e-8))

        # F. Unbind and map back to variable names
        # unbind(dim=1) splits the tensor into a tuple of (Batch, Card) tensors
        results = torch.unbind(proj_logits, dim=1)

        for g, res in zip(groups, results):
            projections[g['base_name']] = res

    return projections

def gumbel_activation(x, layout, tau, hard):
    """
    Vectorized Gumbel Softmax.
    """
    out_parts = []
    for start, end, card in layout:
        chunk = x[:, start:end]
        if card == 0:
            out_parts.append(chunk)
        else:
            b, w = chunk.shape
            n_vars = w // card
            chunk_view = chunk.view(b, n_vars, card)
            act = F.gumbel_softmax(chunk_view, tau=tau, hard=hard, dim=2)
            out_parts.append(act.view(b, w))

    return torch.cat(out_parts, dim=1)

def gradient_penalty(discriminator, real, fake, lambda_gp):
    device = real.device
    pack = getattr(discriminator, 'pack', 1)
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)

    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake_output = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    if pack > 1:
        gradients = gradients.reshape(-1, pack * interpolates.size(1))

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


def recon_loss(fake, real, mode='p2', layout=None, proj_groups=None, alpha=1.0, beta=1.0):
    """
    Calculates reconstruction loss without masks (assuming full observation).

    Args:
        fake: Generator output.
        real: Ground truth target.
              If mode='p2', this is the Phase 2 real data (X).
              If mode='p1', this is the Phase 1 real data (A).
        mode: 'p2' for standard reconstruction, 'p1' for projection consistency.
        layout: Required for 'p2' mode.
        proj_groups: Required for 'p1' mode.
    """
    loss = torch.tensor(0.0, device=fake.device)

    if alpha <= 0 and beta <= 0:
        return loss
    # --- Mode 1: Phase 2 Standard Reconstruction (MSE + CE) ---
    if mode == 'p2':
        assert layout is not None, "Layout required for P2 reconstruction"

        ce_loss_sum = torch.tensor(0.0, device=fake.device)
        n_cat_groups = 0
        n_num_elements = 0
        mse_sum = torch.tensor(0.0, device=fake.device)

        for start, end, card in layout:
            fake_chunk = fake[:, start:end]
            real_chunk = real[:, start:end]

            if card == 0:
                # Numerical MM
                if alpha > 0:
                    mse_sum += ((fake_chunk - real_chunk) ** 2).sum()
                    n_num_elements += fake_chunk.numel()
            else:
                # Categorical CE
                if beta > 0:
                    b, w = fake_chunk.shape
                    # Reshape to (Batch * N_vars, Classes)
                    fake_view = fake_chunk.reshape(-1, card)
                    real_view = real_chunk.reshape(-1, card)

                    # Target indices from One-Hot
                    r_idx = torch.argmax(real_view, dim=1)

                    ce = F.cross_entropy(fake_view, r_idx, reduction='sum')
                    ce_loss_sum += ce
                    n_cat_groups += (w // card) * b

        # Combine
        if alpha > 0 and n_num_elements > 0:
            loss += alpha * (mse_sum / n_num_elements)

        if beta > 0 and n_cat_groups > 0:
            loss += beta * (ce_loss_sum / n_cat_groups)

    # --- Mode 2: Phase 1 Projection Consistency (CE Only) ---
    elif mode == 'p1':
        if beta > 0:
            # 1. Project Phase 2 Output (fake) -> Phase 1 Logits
            projections = project_categorical(fake, proj_groups)

            proj_loss_sum = torch.tensor(0.0, device=fake.device)
            total_obs = 0

            for group in proj_groups:
                base_name = group['base_name']

                # If this group was successfully projected
                if base_name in projections:
                    proj_logits = projections[base_name]  # (Batch, W)

                    # Get Phase 1 Ground Truth
                    p1_start, p1_end = group['p1_range']
                    target_p1 = real[:, p1_start:p1_end]

                    card = group['card']

                    # Reshape
                    fake_view = proj_logits.reshape(-1, card)  # Log Probs
                    real_view = target_p1.reshape(-1, card)  # One-Hot

                    # Convert One-Hot target to Class Indices
                    r_idx = torch.argmax(real_view, dim=1)

                    # NLL Loss (since we already have Log Softmax from projection)
                    ce = F.nll_loss(fake_view, r_idx, reduction='sum')

                    proj_loss_sum += ce
                    total_obs += r_idx.size(0)

            if beta > 0 and total_obs > 0:
                loss += beta * (proj_loss_sum / total_obs)
    return loss


def moment_matching_loss(pca, real, fake):
    real = pca(real)
    fake = pca(fake)

    real_mean = real.mean(dim=0)
    fake_mean = fake.mean(dim=0)

    real_std = real.std(dim=0)
    fake_std = fake.std(dim=0)

    loss_mean = torch.norm(real_mean - fake_mean, p=2)
    loss_std = torch.norm(real_std - fake_std, p=2)

    loss = loss_mean + loss_std
    return loss


class PCA():

    def __init__(self, n_components=None, device="cpu"):
        self.n_components = n_components
        self._device = device

    def __call__(self, X):
        return self.transform(X)

    def fit(self, X):

        if type(X) != torch.Tensor:
            X = self.to_torch(X)

        X = X.to(self._device)

        n_samples, n_features = X.shape

        if self.n_components is None or self.n_components > n_samples:
            self.n_components = min(n_samples, n_features)

        self.U, self.S, self.V = pca_lowrank(X, q=self.n_components, center=False)

        explained_variance_ = (self.S ** 2) / (n_samples - 1)

        total_var = explained_variance_.sum()

        explained_variance_ratio_ = explained_variance_ / total_var

        self.explained_variance_ = explained_variance_[: self.n_components]

        self.explained_variance_ratio_ = explained_variance_ratio_[: self.n_components]

        self.n_samples = n_samples

        self.n_features = n_features

        return self

    def set_device(self, device):

        self._device = device

    def transform(self, X):

        if type(X) == pd.DataFrame:

            self.columns = X.columns

        else:

            self.columns = None

        if type(X) != torch.Tensor:
            X = self.to_torch(X)

        X = X.to(self._device)

        self.dtype = X.dtype

        X_bar = torch.matmul(X, self.V[:, : self.n_components])

        return X_bar

    def inverse_transform(self, X_bar):

        X_prime = torch.matmul(X_bar, self.V[:, : self.n_components].T)

        X_prime = X_prime.to(self._device)

        X_prime = X_prime.to(self.dtype).numpy()

        if self.columns is not None:
            X_prime = pd.DataFrame(X_prime, columns=self.columns)

        return X_prime

    def to_torch(self, X):

        if type(X) == pd.DataFrame:
            X = X.values

        X = torch.from_numpy(X)

        return X
