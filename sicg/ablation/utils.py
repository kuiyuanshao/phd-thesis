from torch import pca_lowrank
import pandas as pd
import torch
import torch.nn.functional as F
import torch

import torch


# --- 基础数学工具 ---
def pairwise_distances_squared(x):
    """(Batch, Dim) -> (Batch, Batch)"""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
    return torch.clamp(dist, 0.0, float('inf'))


def batch_pairwise_distances_squared_1d(x_stack):
    """
    针对连续变量的批处理计算
    Input: (N_features, Batch, 1)
    Output: (N_features, Batch, Batch)
    """
    # x: (N, B, 1)
    # dist = (x - x.T)^2
    # 利用广播: (N, B, 1) - (N, 1, B) -> (N, B, B)
    diff = x_stack - x_stack.transpose(1, 2)
    return diff ** 2


def kernel_from_distance(dist, sigma=None):
    if sigma is None:
        median_dist = torch.median(dist)
        sigma = median_dist.detach()
        if sigma <= 1e-8: sigma = torch.tensor(1.0, device=dist.device)
    return torch.exp(-dist / sigma)


def batch_kernel_from_distance(dist_stack, sigma_stack=None):
    """
    批处理 Kernel 计算
    dist_stack: (N, B, B)
    sigma_stack: (N, 1, 1) or Scalar
    """
    if sigma_stack is None:
        # 针对每个特征单独算 median (N, 1, 1)
        # flatten last two dims to sort: (N, B*B)
        flat = dist_stack.view(dist_stack.shape[0], -1)
        median_vals = torch.median(flat, dim=1, keepdim=True).values  # (N, 1)
        sigma_stack = median_vals.view(-1, 1, 1).detach()
        sigma_stack = torch.clamp(sigma_stack, min=1e-8)

    return torch.exp(-dist_stack / sigma_stack)


def batch_hsic(K_stack, L_stack):
    """
    批处理 HSIC 计算
    K_stack: (N, B, B)
    L_stack: (N, B, B)
    Output: (N,) -> scalar sum
    """
    N, m, _ = K_stack.shape

    # H = I - 1/m
    # 显式构造 H (Batch 较小时更快)
    H = torch.eye(m, device=K_stack.device, dtype=K_stack.dtype) - 1.0 / m * torch.ones((m, m), device=K_stack.device,
                                                                                        dtype=K_stack.dtype)
    # 广播 H: (1, B, B)
    H = H.unsqueeze(0)

    # Kc = H K H
    Kc = torch.matmul(H, torch.matmul(K_stack, H))
    Lc = torch.matmul(H, torch.matmul(L_stack, H))

    # HSIC = Sum(Kc * Lc) / (m-1)^2
    # 在最后两个维度 (B, B) 上求和
    hsic_vals = torch.sum(Kc * Lc, dim=(1, 2)) / ((m - 1) ** 2)
    return hsic_vals.sum()  # 返回所有特征 HSIC 的总和


# --- 核心 Loss ---

def calc_HSIC_loss(x_fake, x_real, attrs, layout, lambda_hsic=1000):
    diff = x_fake - x_real

    # 1. 预处理：分离连续变量和类别变量
    cont_cols = []  # 存储连续变量列索引
    cat_blocks = []  # 存储类别变量 block 范围 (start, end)

    current_idx = 0
    # 收集索引
    for start, end, card in layout:
        # layout 是基于原始 x_fake 维度的，我们需要映射到 diff 的维度
        # diff 和 x_fake 维度一致，直接用
        width = end - start
        if card == 0:  # 连续变量
            # 连续变量需要 Standardize
            # 这一步必须做，但我们可以后面批量做
            cont_cols.extend(range(start, end))
        else:  # 类别变量
            cat_blocks.append((start, end))

    # 2. 准备数据
    # A. 连续变量处理
    if cont_cols:
        diff_cont = diff[:, cont_cols]
        diff_cont = standardize_batch(diff_cont)  # (Batch, N_cont)
    else:
        diff_cont = None

    # B. 类别变量处理
    diff_cats = [diff[:, s:e] for s, e in cat_blocks]

    # C. 拼接 Total Diff (用于 Attr Loss 和 Rest 计算)
    to_concat = diff_cats + ([diff_cont] if diff_cont is not None else [])
    total_diff = torch.cat(to_concat, dim=1)

    # 3. 计算 Attr Loss (只算一次大矩阵)
    dist_total = pairwise_distances_squared(total_diff)

    # --- 优化点：计算全局 Sigma，用于复用 ---
    # Rest 矩阵通常占据 Total 矩阵的绝大部分，分布极其相似。
    # 直接复用 Total 的 Sigma 可以省去大量的 torch.median 计算。
    median_total = torch.median(dist_total).detach()
    sigma_total = torch.clamp(median_total, min=1e-8)

    # 计算 Attr HSIC
    attrs_scaled = standardize_batch(attrs)
    dist_attrs = pairwise_distances_squared(attrs_scaled)
    loss_disentangle_attr = hsic_from_kernels(
        kernel_from_distance(dist_total, sigma=sigma_total),
        kernel_from_distance(dist_attrs)
    )

    # 4. 计算 Pairwise Indep Loss (混合策略)
    loss_pairwise_indep = 0.0
    total_blocks_count = 0

    # --- 策略 A: 连续变量完全向量化 (极速) ---
    if diff_cont is not None:
        # (Batch, N_cont) -> (N_cont, Batch, 1)
        x_stack = diff_cont.t().unsqueeze(-1)

        # 随机采样优化 (如果连续特征太多，比如 > 50)
        n_cont = x_stack.shape[0]

        # 1. 批量计算 Target 距离: (N_cont, B, B)
        dist_target_stack = batch_pairwise_distances_squared_1d(x_stack)

        # 2. 批量计算 Rest 距离: (N_cont, B, B)
        # 利用广播: (B, B) - (N, B, B) -> (N, B, B)
        # 注意: clamp 必须加，防止 float 误差
        dist_rest_stack = torch.clamp(dist_total.unsqueeze(0) - dist_target_stack, min=0.0)

        # 3. 批量 Kernel
        # Target Sigma: 必须单独算，因为 1D 分布和 N-D 分布差异大
        K_stack = batch_kernel_from_distance(dist_target_stack, sigma_stack=None)

        # Rest Sigma: 直接复用 sigma_total (广播)
        # 这省去了 N 次 median 计算！
        L_stack = torch.exp(-dist_rest_stack / sigma_total)

        # 4. 批量 HSIC 求和
        sum_hsic_cont = batch_hsic(K_stack, L_stack)

        loss_pairwise_indep += sum_hsic_cont
        total_blocks_count += n_cont

    # --- 策略 B: 类别变量循环 (数量通常较少，保持 Loop) ---
    # 如果类别变量也很多，同样可以加随机采样
    for s, e in cat_blocks:
        target = diff[:, s:e]

        # 1. Target Dist
        dist_target = pairwise_distances_squared(target)

        # 2. Rest Dist (减法)
        dist_rest = torch.clamp(dist_total - dist_target, min=0.0)

        # 3. Kernels
        K_target = kernel_from_distance(dist_target)  # 自身算 median
        L_rest = torch.exp(-dist_rest / sigma_total)  # 复用 global sigma

        # 4. HSIC
        loss_pairwise_indep += hsic_from_kernels(K_target, L_rest)
        total_blocks_count += 1

    if total_blocks_count > 0:
        loss_pairwise_indep /= total_blocks_count

    return lambda_hsic * (loss_disentangle_attr + loss_pairwise_indep)


# 还需要 helper: standardize_batch, hsic_from_kernels (保持之前版本即可)
def standardize_batch(x, epsilon=1e-8):
    if x.std() < 1e-6: return x
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + epsilon)


def hsic_from_kernels(K, L):
    m = K.shape[0]
    H = torch.eye(m, device=K.device, dtype=K.dtype) - 1.0 / m * torch.ones((m, m), device=K.device, dtype=K.dtype)
    Kc = torch.mm(H, torch.mm(K, H))
    Lc = torch.mm(H, torch.mm(L, H))
    return torch.sum(Kc * Lc) / ((m - 1) ** 2)


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
