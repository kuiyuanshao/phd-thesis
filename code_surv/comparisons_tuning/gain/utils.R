

normalize <- function(data, numCol){
  norm_data <- data
  norm_parameters <- list()
  for (i in numCol){
    min_val <- min(norm_data[, i], na.rm = T)
    max_val <- max(norm_data[, i], na.rm = T) + 1e-6
    norm_data[, i] <- (norm_data[, i] - min_val) / max_val
    norm_parameters[[i]] <- c(min_val, max_val)
  }
  return (list(norm_data = norm_data, norm_parameters = norm_parameters))
}

renormalize <- function(norm_data, norm_parameters, numCol){
  renorm_data <- norm_data
  for (i in numCol){
    curr_param <- norm_parameters[[i]]
    renorm_data[, i] <- renorm_data[, i] * curr_param[2] + curr_param[1]
  }
  return (renorm_data)
}


new_batch <- function(norm_data, data_mask, nRow, batch_size, device = "cpu"){
  rows <- sample(nRow)
  inds <- rows[1:batch_size]
  norm_curr_batch <- norm_data[inds, ]
  mask_curr_batch <- data_mask[inds, ]
  
  return (list(norm_curr_batch, mask_curr_batch))
}









