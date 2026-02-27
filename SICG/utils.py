from torch import pca_lowrank
import pandas as pd
import torch
import torch.nn.functional as F
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
            out_parts.append(torch.tanh(chunk))#.clamp(min=-4.0, max=4.0))
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

    alpha = torch.rand(real.size(0) // pack, 1, 1, device=device)
    alpha = alpha.repeat(1, pack, real.size(1))
    alpha = alpha.view(-1, real.size(1))

    interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

    disc_interpolates, _ = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(-1, pack * real.size(1))
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


def recon_loss(fake, real, mode='p2', layout=None, proj_groups=None, alpha=1.0, beta=1.0):
    loss = torch.tensor(0.0, device=fake.device)
    if alpha <= 0 and beta <= 0:
        return loss
    if mode == 'p2':
        assert layout is not None, "Layout required for P2 reconstruction"

        mse_sum = torch.tensor(0.0, device=fake.device)
        ce_loss_sum = torch.tensor(0.0, device=fake.device)
        n_cat_groups = 0
        n_num_elements = 0

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

def moment_matching_loss(real, fake):
    batch_size = real.size(0)
    loss_mean = torch.norm(
        torch.mean(fake.view(batch_size, -1), dim=0) - torch.mean(real.view(batch_size, -1), dim=0),
        1)
    loss_std = torch.norm(
        torch.std(fake.view(batch_size, -1), dim=0) - torch.std(real.view(batch_size, -1), dim=0),
        1)
    loss_info = loss_mean + loss_std
    return loss_info


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