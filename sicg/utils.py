from torch import pca_lowrank
import pandas as pd
import torch
import torch.nn.functional as F

def pairwise_distances_squared(x):
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

    sigma_res = torch.median(dist_res).detach()
    if sigma_res < 1e-6: sigma_res = torch.tensor(1.0, device=diff.device)

    sigma_attr = torch.median(dist_attr).detach()
    if sigma_attr < 1e-6: sigma_attr = torch.tensor(1.0, device=diff.device)

    K_res = torch.exp(-dist_res / sigma_res)
    L_attr = torch.exp(-dist_attr / sigma_attr)

    loss_set = hsic_normalized(K_res, L_attr)
    loss_pairwise = 0.0
    current_idx = 0
    n_blocks = 0

    for s, e, c in layout_res:
        width = e - s
        target_data = diff_combined[:, current_idx: current_idx + width]
        d_target = pairwise_distances_squared(target_data)
        K_target = torch.exp(-d_target / sigma_res)
        d_rest = torch.clamp(dist_res - d_target, min=0.0)
        K_rest = torch.exp(-d_rest / sigma_res)
        loss_pairwise += hsic_normalized(K_target, K_rest)

        current_idx += width
        n_blocks += 1

    if n_blocks > 1:
        loss_pairwise /= n_blocks

    return lambda_hsic * torch.relu((loss_set + loss_pairwise))


def hsic_normalized(K, L):
    m = K.shape[0]
    H = torch.eye(m, device=K.device) - 1.0 / m * torch.ones((m, m), device=K.device)
    Kc = torch.mm(H, torch.mm(K, H))
    Lc = torch.mm(H, torch.mm(L, H))
    return torch.trace(torch.mm(Kc, Lc)) / (m * m)
# =========================================================================
# 1. 核心统计算子 (JIT 编译加速，O(N^2) 复杂度)
# =========================================================================
#
# @torch.jit.script
# def fast_hsic(K: torch.Tensor, L: torch.Tensor):
#     """
#     实现 Standard HSIC (Biased V-statistic)。
#     复杂度 O(N^2)，完美替代 ckatorch.hsic0。
#     """
#     n = K.size(0)
#     # 中心化矩阵 H K H 的广播实现
#     K_mean_row = K.mean(dim=0, keepdim=True)
#     K_mean_col = K.mean(dim=1, keepdim=True)
#     K_mean_all = K.mean()
#     Kc = K - K_mean_row - K_mean_col + K_mean_all
#
#     L_mean_row = L.mean(dim=0, keepdim=True)
#     L_mean_col = L.mean(dim=1, keepdim=True)
#     L_mean_all = L.mean()
#     Lc = L - L_mean_row - L_mean_col + L_mean_all
#
#     # Tr(Kc * Lc) = sum(Kc * Lc)
#     score = (Kc * Lc).sum()
#     return score / ((n - 1) ** 2)
#
#
# # =========================================================================
# # 2. 辅助计算工具
# # =========================================================================
#
# def pairwise_distances_squared(x):
#     """极速欧氏距离平方计算"""
#     x_norm = (x ** 2).sum(1).view(-1, 1)
#     dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
#     return torch.clamp(dist, 0.0)
#
#
# def get_bandwidths(dist, scales=[0.1, 0.5, 1.0, 2.0, 5.0]):
#     """基于中位数启发式生成多尺度带宽 Tensor"""
#     median = torch.median(dist).detach()
#     median = median if median > 1e-6 else torch.tensor(1.0, device=dist.device)
#     scales_t = torch.tensor(scales, device=dist.device, dtype=dist.dtype)
#     return 2 * (scales_t ** 2) * median
#
#
# def compute_kernel_vectorized(dist, bw_tensor):
#     """向量化计算 RBF 核，避免 Python 循环"""
#     # dist: (N, N) -> (N, N, 1)
#     # bw_tensor: (S,) -> (1, 1, S)
#     kernels = torch.exp(-dist.unsqueeze(-1) / bw_tensor.view(1, 1, -1))
#     return kernels.mean(dim=-1)
#
#
# # =========================================================================
# # 3. 数据预处理逻辑 (保护 One-hot)
# # =========================================================================
#
# def standardize_batch(x):
#     if x.shape[1] == 0: return x
#     std = x.std(dim=0, keepdim=True)
#     mask = std > 1e-6
#     x_out = x.clone()
#     if mask.any():
#         mean = x.mean(dim=0, keepdim=True)
#         x_out[:, mask.squeeze()] = (x[:, mask.squeeze()] - mean[:, mask.squeeze()]) / (std[:, mask.squeeze()] + 1e-8)
#     return x_out
#
#
# def prepare_mixed_data(data, layout):
#     """数值列做 Z-Score + 行 L2；类别列保持原样"""
#     all_chunks = []
#     for start, end, card in layout:
#         chunk = data[:, start:end]
#         if card == 0:  # 数值列
#             chunk = standardize_batch(chunk)
#         else:
#             pass
#         all_chunks.append(chunk)
#     return torch.cat(all_chunks, dim=1)
#
#
# # =========================================================================
# # 4. 主 Loss 函数
# # =========================================================================
#
# def calc_HSIC_loss(x_fake, x_real, attrs, layout_res, layout_attr, lambda_hsic=1.0):
#     """
#     整合版 HSIC Loss
#     已移除 loss_inter，专注处理 Marginal Independence 和 Pairwise Independence。
#     """
#     # 1. 预处理数据
#     diff = x_fake - x_real
#     e_n = prepare_mixed_data(diff, layout_res)
#     Z_n = prepare_mixed_data(attrs, layout_attr)
#
#     # 2. 全局距离与带宽
#     dist_res = pairwise_distances_squared(e_n)
#     dist_attr = pairwise_distances_squared(Z_n)
#
#     bw_res = get_bandwidths(dist_res)
#     bw_attr = get_bandwidths(dist_attr)
#
#     # 3. 全局核矩阵
#     K_res = compute_kernel_vectorized(dist_res, bw_res)
#     L_attr = compute_kernel_vectorized(dist_attr, bw_attr)
#
#     # --- Loss A: Base Independence (e vs Z) ---
#     loss_set = fast_hsic(K_res, L_attr)
#
#     # --- Loss C: Pairwise Independence (Internal Decoupling) ---
#     loss_pairwise = 0.0
#     current_idx = 0
#
#     for s, e, c in layout_res:
#         width = e - s
#         target_data = e_n[:, current_idx: current_idx + width]
#
#         # 局部距离与带宽 (因为维度变化大，必须重算)
#         d_target = pairwise_distances_squared(target_data)
#         bw_target = get_bandwidths(d_target)
#         K_target = compute_kernel_vectorized(d_target, bw_target)
#
#         # 减法技巧优化 O(N^2)
#         d_rest = torch.clamp(dist_res - d_target, min=0.0)
#         K_rest = compute_kernel_vectorized(d_rest, bw_res)
#
#         loss_pairwise += fast_hsic(K_target, K_rest)
#         current_idx += width
#
#     if len(layout_res) > 0:
#         loss_pairwise = loss_pairwise / len(layout_res)
#
#     total_loss = loss_set + loss_pairwise
#     return lambda_hsic * total_loss
# def calc_HSIC_loss(x_fake, x_real, attrs, layout_res, layout_attr, lambda_hsic=1.0):
#     diff = x_fake - x_real
#     diff_combined = prepare_and_concat(diff, layout_res)
#     attrs_combined = prepare_and_concat(attrs, layout_attr)
#
#     dist_res = pairwise_distances_squared(diff_combined)
#     dist_attr = pairwise_distances_squared(attrs_combined)
#
#     # 3. 计算 Sigma (Global Median)
#     sigma_res = torch.median(dist_res).detach()
#     if sigma_res < 1e-6: sigma_res = torch.tensor(1.0, device=diff.device)
#
#     sigma_attr = torch.median(dist_attr).detach()
#     if sigma_attr < 1e-6: sigma_attr = torch.tensor(1.0, device=diff.device)
#
#     # 4. Kernel & HSIC
#     K_res = torch.exp(-dist_res / sigma_res)
#     L_attr = torch.exp(-dist_attr / sigma_attr)
#
#     # Loss A: Set Independence
#     loss_set = hsic_normalized(K_res, L_attr)
#
#     # Loss B: Pairwise Independence (One-vs-Rest)
#     # 依然可以用减法，因为是在同一个距离矩阵上操作
#     loss_pairwise = 0.0
#
#     # 为了做 Pairwise，我们需要知道每个 block 在 combined 矩阵里的索引范围
#     # 重新扫一遍 layout 算索引
#     current_idx = 0
#     n_blocks = 0
#
#     # 预先计算 diff_combined 的总距离用于减法
#     # dist_res 已经是总距离了
#
#     for s, e, c in layout_res:
#         width = e - s
#         # 提取当前 block 的数据 (注意：是处理过的数据)
#         # 这里的切片是在 combined 矩阵上切
#         target_data = diff_combined[:, current_idx: current_idx + width]
#
#         # 1. Target Kernel
#         d_target = pairwise_distances_squared(target_data)
#         K_target = torch.exp(-d_target / sigma_res)  # 共享 Sigma
#
#         # 2. Rest Kernel (直接减法！)
#         # Rest = Total - Target
#         d_rest = torch.clamp(dist_res - d_target, min=0.0)
#         K_rest = torch.exp(-d_rest / sigma_res)
#
#         loss_pairwise += hsic_normalized(K_target, K_rest)
#
#         current_idx += width
#         n_blocks += 1
#
#     if n_blocks > 1:
#         loss_pairwise /= n_blocks
#
#     return lambda_hsic * torch.relu((loss_set + loss_pairwise) - 0.0009)

def project_categorical(fake, proj_groups):
    """
    Vectorized projection of Phase-2 logits to Phase-1 space.
    Batches operations by (P2_Card, P1_Card) matrix shape to support distinct
    cardinalities between phases and across variables.
    """
    projections = {}
    if not proj_groups:
        return projections

    grouped_by_shape = {}
    for g in proj_groups:
        shape = g['matrix'].shape
        if shape not in grouped_by_shape:
            grouped_by_shape[shape] = []
        grouped_by_shape[shape].append(g)

    for shape, groups in grouped_by_shape.items():
        matrices = torch.stack([g['matrix'] for g in groups], dim=0)
        slices = [fake[:, g['p2_range'][0]:g['p2_range'][1]] for g in groups]
        fake_batch = torch.stack(slices, dim=0)
        probs = F.softmax(fake_batch, dim=2)

        proj_probs = torch.einsum('nbc, ncd -> nbd', probs, matrices)

        # Log & Clamp
        proj_logits = torch.log(torch.clamp(proj_probs, min=1e-8, max=1.0 - 1e-8))
        results = torch.unbind(proj_logits, dim=0)

        for g, res in zip(groups, results):
            projections[g['base_name']] = res

    return projections

def gumbel_activation(x, layout, tau, hard):
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
    loss = torch.tensor(0.0, device=fake.device)
    if alpha <= 0 and beta <= 0:
        return loss
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
                if alpha > 0:
                    mse_sum += ((fake_chunk - real_chunk) ** 2).sum()
                    n_num_elements += fake_chunk.numel()
            else:
                if beta > 0:
                    b, w = fake_chunk.shape
                    fake_view = fake_chunk.reshape(-1, card)
                    real_view = real_chunk.reshape(-1, card)

                    r_idx = torch.argmax(real_view, dim=1)

                    ce = F.cross_entropy(fake_view, r_idx, reduction='sum')
                    ce_loss_sum += ce
                    n_cat_groups += (w // card) * b

        if alpha > 0 and n_num_elements > 0:
            loss += alpha * (mse_sum / n_num_elements)

        if beta > 0 and n_cat_groups > 0:
            loss += beta * (ce_loss_sum / n_cat_groups)

    elif mode == 'p1':
        if beta > 0:
            projections = project_categorical(fake, proj_groups)

            proj_loss_sum = torch.tensor(0.0, device=fake.device)
            total_obs = 0

            for group in proj_groups:
                base_name = group['base_name']
                if base_name in projections:
                    proj_logits = projections[base_name]
                    p1_start, p1_end = group['p1_range']
                    target_p1 = real[:, p1_start:p1_end]
                    card = group['card']
                    fake_view = proj_logits.reshape(-1, card)  # Log Probs
                    real_view = target_p1.reshape(-1, card)  # One-Hot

                    r_idx = torch.argmax(real_view, dim=1)
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