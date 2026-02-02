library(torch)
library(progress)
source("./mimegans/encoding.R")
masked_ce_categorical_fast <- function(X, G, M, cat_blocks, eps = 1e-8) {
  # X, G, M: [B, D] torch tensors (same device). cat_blocks: list of 1-based index vectors.
  B <- X$size(1); D <- X$size(2); K <- length(cat_blocks); dev <- X$device
  
  # column -> block id in 1..K (non-cat columns left as 1, but will be zero-masked)
  col2block <- rep.int(1L, D)
  is_cat_col <- rep(FALSE, D)
  for (k in seq_len(K)) { cols <- cat_blocks[[k]]; col2block[cols] <- k; is_cat_col[cols] <- TRUE }
  
  idx_bd   <- torch_tensor(col2block, dtype = torch_long(), device = dev)$unsqueeze(1)$expand(c(B, D))
  is_cat_bd <- torch_tensor(as.integer(is_cat_col), dtype = torch_float(), device = dev)$unsqueeze(1)$expand(c(B, D))
  
  # per-column CE: -y * log(p)  (assumes G are probs; for logits, apply softmax per block first)
  pred   <- torch_clamp(G, min = eps, max = 1.0)
  ce_col <- -(X * torch_log(pred)) * is_cat_bd
  
  # aggregate CE per block
  ce_per_block <- torch_zeros(c(B, K), device = dev)
  ce_per_block$scatter_add_(dim = 2, index = idx_bd, src = ce_col)
  
  # block mask: 1 if any one-hot in block is observed for the row
  m_col  <- (M > 0)$to(dtype = torch_float()) * is_cat_bd
  msum   <- torch_zeros(c(B, K), device = dev)
  msum$scatter_add_(dim = 2, index = idx_bd, src = m_col)
  m_blk  <- (msum > 0)$to(dtype = torch_float())
  
  # masked average over observed categorical blocks
  torch_sum(ce_per_block * m_blk) / (torch_sum(m_blk) + eps)
}

gain <- function(data, m = 20, data_info, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                 alpha = 100, beta = 10, n = 10000){
  device <- torch_device(device)
  loss_mat <- matrix(NA, nrow = n, ncol = 2)
  
  norm_result <- normalize(data, data_info$num_vars)
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  data_encode <- encode.onehot(norm_data, data_info$cat_vars, data_info$cat_vars, 
                               data_info$phase1_vars, data_info$phase2_vars)
  data_training <- data_encode$data
  
  num_inds <- which(names(data_training) %in% data_info$num_vars)
  cat_inds <- which(names(data_training) %in% unlist(data_encode$new_col_names))

  data_mask <- 1 - is.na(data_training)
  data_training[is.na(data_training)] <- 0
  data_mat <- as.matrix(data_training)
  
  nRow <- dim(data_training)[1]
  nCol <- dim(data_training)[2]
  
  X_t <- torch::torch_tensor(data_mat, device = device)
  M_t <- torch::torch_tensor(data_mask, device = device)
  
  GAIN_Generator <- torch::nn_module(
    initialize = function(nCol){
      self$seq <- torch::nn_sequential(nn_linear(nCol * 2, nCol),
                                       nn_relu(),
                                       nn_linear(nCol, nCol),
                                       nn_relu(),
                                       nn_linear(nCol, nCol),
                                       nn_sigmoid())
    }, 
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  GAIN_Discriminator <- torch::nn_module(
    initialize = function(nCol){
      self$seq <- torch::nn_sequential(nn_linear(nCol * 2, nCol),
                                       nn_relu(),
                                       nn_linear(nCol, nCol),
                                       nn_relu(),
                                       nn_linear(nCol, nCol),
                                       nn_sigmoid())
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  G_layer <- GAIN_Generator(nCol)$to(device = device)
  D_layer <- GAIN_Discriminator(nCol)$to(device = device)
  
  G_solver <- torch::optim_adam(G_layer$parameters)
  D_solver <- torch::optim_adam(D_layer$parameters)
  
  generator <- function(X, M){
    input <- torch_cat(list(X, M), dim = 2)
    return (G_layer(input))
  }
  discriminator <- function(X, H){
    input <- torch_cat(list(X, H), dim = 2)
    return (D_layer(input))
  }
  
  G_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    G_loss1 <- -torch_mean((1 - M) * torch_log(torch_clip(D_prob, 1e-8, 1)))

    mse_loss <- torch_mean((M[, num_inds, drop = F] * X[, num_inds, drop = F] - 
                             M[, num_inds, drop = F] * G_sample[, num_inds, drop = F]) ^ 2) / torch_mean(M[, num_inds, drop = F])
    # cross_entropy <- -torch_mean(X[, cat_inds, drop = F] * M[, cat_inds, drop = F] *
    #                                torch_log(torch_clip(G_sample[, cat_inds, drop = F], 1e-8, 1)) +
    #                                (1 - X[, cat_inds, drop = F]) * M[, cat_inds, drop = F] *
    #                                torch_log(torch_clip(1 - G_sample[, cat_inds, drop = F], 1e-8, 1.)))
    cross_entropy <- masked_ce_categorical_fast(X, G_sample, M, data_encode$binary_indices, eps = 1e-8)
    return (G_loss1 + alpha * mse_loss + beta * cross_entropy)
  }
  D_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    D_loss1 <- -torch_mean(M * torch_log(torch_clip(D_prob, 1e-8, 1)) + (1 - M) * torch_log(torch_clip(1 - D_prob, 1e-8, 1))) * 2
    return (D_loss1)
  }
  
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss",
    clear = FALSE, total = n, width = 100)
  
  for (i in 1:n){
    D_solver$zero_grad()
    
    ind_batch <- new_batch(X_t, M_t, nRow, batch_size, device)
    X_mb <- ind_batch[[1]]
    M_mb <- ind_batch[[2]]
    
    Z_mb <- ((-0.01) * torch::torch_rand(c(batch_size, nCol)) + 0.01)$to(device = device)
    H_mb <- 1 * (matrix(runif(batch_size * nCol, 0, 1), nrow = batch_size, ncol = nCol) > (1 - hint_rate))
    H_mb <- torch_tensor(H_mb, device = device)
    
    H_mb <- M_mb * H_mb
    X_mb <- M_mb * X_mb + (1 - M_mb) * Z_mb
    X_mb <- X_mb$to(device = device)
    
    d_loss <- D_loss(X_mb, M_mb, H_mb)
  
    d_loss$backward()
    D_solver$step()
    
    G_solver$zero_grad()
    
    ind_batch <- new_batch(X_t, M_t, nRow, batch_size, device)
    X_mb <- ind_batch[[1]]
    M_mb <- ind_batch[[2]]
    
    Z_mb <- ((-0.01) * torch::torch_rand(c(batch_size, nCol)) + 0.01)$to(device = device)
    H_mb <- 1 * (matrix(runif(batch_size * nCol, 0, 1), nrow = batch_size, ncol = nCol) > (1 - hint_rate))
    H_mb <- torch_tensor(H_mb, device = device)
    
    H_mb <- M_mb * H_mb
    X_mb <- M_mb * X_mb + (1 - M_mb) * Z_mb
    X_mb <- X_mb$to(device = device)
    
    g_loss <- G_loss(X_mb, M_mb, H_mb)
    
    g_loss$backward()
    G_solver$step()
    
    pb$tick(tokens = list(what = "GAIN   ", 
                          g_loss = sprintf("%.4f", g_loss$item()),
                          d_loss = sprintf("%.4f", d_loss$item())))
    Sys.sleep(1 / 10000)
    loss_mat[i, ] <- c(g_loss$item(), d_loss$item())
  }
  imputed_data_list <- vector("list", m)
  gsample_data_list <- vector("list", m)
  for (j in 1:m){
    Z <- ((-0.01) * torch::torch_rand(c(nRow, nCol)) + 0.01)$to(device = device)
    X <- M_t * X_t + (1 - M_t) * Z
    X <- X$to(device = device)
    M <- M_t
    
    G_sample <- generator(X, M)
    
    imputed_data <- M_t * X + (1 - M_t) * G_sample
    imputed_data <- imputed_data$detach()$cpu()
    imputed_data <- as.data.frame(as.matrix(imputed_data))
    names(imputed_data) <- names(data_training)
    
    imputed_data <- decode.onehot(imputed_data, data_encode)
    imputed_data <- renormalize(imputed_data, norm_parameters, data_info$num_vars)
    
    imputed_data_list[[j]] <- imputed_data
  }
  loss_mat <- as.data.frame(loss_mat)
  names(loss_mat) <- c("G_loss", "D_loss")
  return (list(imputation = imputed_data_list, 
               loss = loss_mat))
}