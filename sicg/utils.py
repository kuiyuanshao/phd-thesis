import torch
import torch.nn.functional as F

def project_categorical(fake, proj_groups):
    """
    Vectorized projection of Phase-2 logits to Phase-1 space.
    Batches operations by variable cardinality to avoid Python loops.
    """
    projections = {}
    if not proj_groups:
        return projections

    # 1. Group by cardinality to enable tensor stacking
    # Structure: {cardinality: [group_dict_1, group_dict_2, ...]}
    grouped_by_card = {}
    for g in proj_groups:
        c = g['card']
        if c not in grouped_by_card:
            grouped_by_card[c] = []
        grouped_by_card[c].append(g)

    # 2. Process each cardinality batch
    for card, groups in grouped_by_card.items():
        # A. Stack Confusion Matrices -> Shape: (N_Groups, Card, Card)
        # All matrices should already be on the correct device from sicg.py
        matrices = torch.stack([g['matrix'] for g in groups], dim=0)

        # B. Gather all relevant slices from 'fake' -> Shape: (Batch, N_Groups, Card)
        # List comprehension + stack is efficient for gathering non-contiguous slices
        slices = [fake[:, g['p2_range'][0]:g['p2_range'][1]] for g in groups]
        fake_batch = torch.stack(slices, dim=1)

        # C. Compute Probabilities P(P2) -> Shape: (Batch, N_Groups, Card)
        probs = F.softmax(fake_batch, dim=2)

        # D. Project P(P2) * P(P1|P2) -> P(P1) using Batch Matrix Multiplication
        # Einsum dimensions:
        # b = batch, n = n_groups, c = card (p2), d = card (p1)
        # Note: matrices are (N, Card_P2, Card_P1)
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
        gradients = gradients.view(-1, pack * interpolates.size(1))

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
                # Numerical MSE (No Masking)
                if alpha > 0:
                    mse_sum += ((fake_chunk - real_chunk) ** 2).sum()
                    n_num_elements += fake_chunk.numel()
            else:
                # Categorical CE (No Masking)
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
        assert proj_groups is not None, "Projection groups required for P1 loss"

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

        if total_obs > 0:
            loss += (proj_loss_sum / total_obs)

    return loss