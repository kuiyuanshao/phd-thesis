import torch

@torch.jit.script
def fast_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    K_mean_row = K.mean(dim=0, keepdim=True)
    K_mean_col = K.mean(dim=1, keepdim=True)
    K_mean_all = K.mean()
    Kc = K - K_mean_row - K_mean_col + K_mean_all

    L_mean_row = L.mean(dim=0, keepdim=True)
    L_mean_col = L.mean(dim=1, keepdim=True)
    L_mean_all = L.mean()
    Lc = L - L_mean_row - L_mean_col + L_mean_all
    score = (Kc * Lc).sum()
    return score / ((n - 1.0) ** 2)

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
        x_out[:, mask.squeeze()] = (x[:, mask.squeeze()] - mean[:, mask.squeeze()]) / (4 * std[:, mask.squeeze()] + 1e-8)
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

    loss_set = fast_hsic(K_res, L_attr)

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
        loss_pairwise += fast_hsic(K_target, K_rest)

        current_idx += width
        n_blocks += 1

    if n_blocks > 1:
        loss_pairwise /= n_blocks

    return lambda_hsic * torch.relu(loss_set + loss_pairwise)