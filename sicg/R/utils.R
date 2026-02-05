pacman::p_load(torch)

reconLoss <- function(fake, true, fake_proj, true_proj, I, params, 
                      num_inds, cat_inds, 
                      ce_groups_p2, ce_groups_mode, total_cat_count){
  use_mm <- (length(num_inds) > 0L) && (params$alpha != 0)
  use_ce <- (length(cat_inds) > 0L) && (params$beta != 0)
  if (!use_mm && !use_ce)
    return (torch_tensor(0, device = fake$device))
  n_I <- I$sum()
  mm <- if (use_mm) {
    mse_sum <- nnf_mse_loss(fake[I, num_inds], true[I, num_inds], reduction = "sum")
    params$alpha * (mse_sum / (n_I + 1e-8))
  } else NULL
  ce <- if (use_ce) {
    params$beta *
      ceLoss(fake, true, fake_proj, true_proj, I, n_I, params, ce_groups_p2, ce_groups_mode, total_cat_count)
  } else NULL
  if (is.null(mm)) return(ce)
  if (is.null(ce)) return(mm)
  
  mm$add_(ce)
  return(mm)
}

infoLoss <- function(fake, true){
  f_flat <- fake$view(c(fake$size(1), -1))
  t_flat <- true$view(c(true$size(1), -1))
  
  return (torch_norm(torch_mean(f_flat, dim = 1) - torch_mean(t_flat, dim = 1), 2) +
            torch_norm(torch_std(f_flat, dim = 1) - torch_std(t_flat, dim = 1), 2))
}

ceLoss <- function(fake, true, fake_proj, A, I, n_I, params, ce_groups_p2, ce_groups_mode, total_cat_count){
  loss_sum <- torch_tensor(0, device = fake$device)
  
  notI <- I$logical_not()
  n_notI <- notI$sum()

  calc_batch_ce <- function(input, target_onehot, group, col_indices, mask_idx){
    inp_sub <- input[mask_idx, col_indices]
    tgt_sub <- target_onehot[mask_idx, col_indices]
    
    n_curr <- inp_sub$size(1)
    inp_view <- inp_sub$view(c(n_curr, group$k, group$L))$permute(c(1, 3, 2))
    tgt_idx <- torch_argmax(tgt_sub$view(c(n_curr, group$k, group$L)), dim = 3)
    
    return(nnf_cross_entropy(inp_view, tgt_idx, reduction = "sum"))
  }
  
  for (group in ce_groups_p2){
    if (params$cat == "projp1" | params$cat == "projp2"){
      idx_A <- if(!is.null(group$p1_idx)) group$p1_idx else group$p2_idx
      
      l1 <- calc_batch_ce(fake_proj, A, group, idx_A, notI)
      l2 <- calc_batch_ce(fake, true, group, group$p2_idx, I)
      term <- (params$proj_weight * l1 + l2) / (n_notI + n_I + 1e-8)
      loss_sum$add_(term)
    } else {
      l2 <- calc_batch_ce(fake, true, group, group$p2_idx, I)
      loss_sum$add_(l2 / (n_I + 1e-8))
    }
  }
  
  for (group in ce_groups_mode){
    l_mode <- calc_batch_ce(fake, true, group, group$idx, I)
    loss_sum$add_(l_mode / (n_I + 1e-8))
  }
  
  return (loss_sum / total_cat_count)
}

activationFun <- function(fake, cat_groups, all_nums, params, gen = F){
  hard_flag <- if (gen) TRUE else isTRUE(params$hard)
  
  for (group in cat_groups){
    subset <- fake[, group$indices]
    subset_view <- subset$view(c(fake$size(1), group$k, group$L))
    act <- nnf_gumbel_softmax(subset_view, tau = params$tau, hard = hard_flag, dim = 3)
    fake[, group$indices] <- act$view(c(fake$size(1), -1))
  }
  return (fake)
}

projCat <- function(fake, proj_groups){
  fake_result <- fake$clone()
  for (group in proj_groups) {
    subset <- fake[, group$indices]
    subset_view <- subset$view(c(fake$size(1), group$k, group$L))
    
    prob <- nnf_softmax(subset_view, dim = 3)
    proj <- torch_matmul(prob$unsqueeze(3), group$matrices)$squeeze(3)
    
    logits_obs <- torch_log(proj$clamp(1e-8, 1 - 1e-8))
    fake_result[, group$indices] <- logits_obs$view(c(fake$size(1), -1))
  }
  return (fake_result)
}

gradientPenalty <- function(D, real_samples, fake_samples, params, device, ones_buf = NULL) {
  batch_size <- real_samples$size(1)
  alp <- torch_rand(batch_size, 1, device = device)
  interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(TRUE)
  
  d_interpolates <- D(interpolates)
  
  fake <- if (!is.null(ones_buf) && ones_buf$size(1) == d_interpolates$size(1)) {
    ones_buf
  } else {
    torch_ones(d_interpolates$size(), device = device)
  }
  fake <- fake$detach()
  
  gradients <- torch::autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake,
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]
  
  if (params$pac > 1){
    gradients <- gradients$reshape(c(-1, params$pac * interpolates$size(2)))
  }
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  return (gradient_penalty)
}

create_data_loaders <- function(data_original, data_encode, data_info, params, 
                                phase1_rows, phase2_rows, p1set, p2set, 
                                p1b_t, p2b_t, epochs) {
  
  p1loader <- NULL
  p2loader <- NULL
  Dloader  <- NULL
  phase1_bins <- character(0)
  if (params$balancebatch) {
    p1_exclusive_vars <- data_info$cat_vars[!(data_info$cat_vars %in% data_info$phase2_vars)]
    
    if (length(p1_exclusive_vars) > 0) {
      phase1_bins <- p1_exclusive_vars[sapply(p1_exclusive_vars, function(col) {
        length(unique(data_original[phase1_rows, col])) > 1 && 
          length(unique(data_original[phase2_rows, col])) > 1
      })]
    }

    if (p1b_t > 0) {
      p1sampler <- BalancedSampler(data_original[phase1_rows, ], phase1_bins, p1b_t, epochs)
      p1loader <- dataloader(p1set, sampler = p1sampler, pin_memory = TRUE,
                             collate_fn = function(bl) bl[[1]])
    }
    
    p2sampler <- BalancedSampler(data_original[phase2_rows, ], phase1_bins, p2b_t, epochs)
    p2loader <- dataloader(p2set, sampler = p2sampler, pin_memory = TRUE,
                           collate_fn = function(bl) bl[[1]])
    D_sampler <- BalancedSampler(data_original[phase2_rows, ], phase1_bins, params$batch_size, epochs)
    Dloader <- dataloader(p2set, sampler = D_sampler, pin_memory = TRUE,
                          collate_fn = function(bl) bl[[1]])
    
  } else {
    if (p1b_t > 0) {
      p1sampler <- SRSSampler(length(phase1_rows), p1b_t, epochs)
      p1loader <- dataloader(p1set, sampler = p1sampler, pin_memory = TRUE,
                             collate_fn = function(bl) bl[[1]])
    }
    
    p2sampler <- SRSSampler(length(phase2_rows), p2b_t, epochs)
    p2loader <- dataloader(p2set, sampler = p2sampler, pin_memory = TRUE,
                           collate_fn = function(bl) bl[[1]])
    
    D_sampler <- SRSSampler(length(phase2_rows), params$batch_size, epochs)
    Dloader <- dataloader(p2set, sampler = D_sampler, pin_memory = TRUE,
                          collate_fn = function(bl) bl[[1]])
  }
  
  return(list(
    p1loader = p1loader,
    p2loader = p2loader,
    Dloader = Dloader,
    phase1_bins = phase1_bins
  ))
}