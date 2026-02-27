initset <- dataset(
  name = "initData",
  initialize = function(data_training, rows, 
                        phase1_vars_encode, 
                        phase2_vars_encode, 
                        conditions_vars_encode,
                        device) {  # Added device argument
    
    subset <- data_training[rows, ]
    self$A_t <- torch_tensor(as.matrix(subset[, phase1_vars_encode]), device = device)
    X_mat <- as.matrix(subset[, phase2_vars_encode])
    self$M_t <- torch_tensor(1 - is.na(X_mat), dtype = torch_long(), device = device)
    X_mat[is.na(X_mat)] <- 0 
    self$X_t <- torch_tensor(X_mat, device = device)
    self$C_t <- torch_tensor(as.matrix(subset[, conditions_vars_encode]), device = device)
  },
  
  .getitem = function(i) {
    list(M = self$M_t[i, ], 
         C = self$C_t[i, ], 
         X = self$X_t[i, ], 
         A = self$A_t[i, ])
  },
  
  .length = function() {
    self$A_t$size(1)
  }
)

alloc_even <- function(total, levl) {
  k <- length(levl)
  if (k == 0L) return(integer(0))
  base <- total %/% k
  rem  <- total - base * k
  counts <- rep.int(base, k)
  if (rem > 0) counts[seq_len(rem)] <- counts[seq_len(rem)] + 1
  names(counts) <- levl
  counts
}

sample_from_cache <- function(grouped_indices, counts_vec) {
  total_req <- sum(counts_vec)
  res <- integer(total_req)
  cursor <- 1L
  
  for (grp in names(counts_vec)) {
    n_req <- counts_vec[[grp]]
    if (n_req > 0) {
      candidates <- grouped_indices[[grp]]
      n_cand <- length(candidates)
      if (n_cand > 0) {
        s_idx <- sample.int(n_cand, n_req, replace = n_req > n_cand)
        s <- candidates[s_idx]
        end <- cursor + n_req - 1L
        res[cursor:end] <- s
        cursor <- cursor + n_req
      } else {
        warning(paste("Skipping empty group:", grp))
      }
    }
  }
  if (cursor <= total_req) {
    res <- res[1:(cursor - 1L)]
  }
  return(res)
}

BalancedSampler <- sampler(
  "BalancedSampler",
  initialize = function(x, bin_cols, batch_size, epochs) {
    self$bs <- batch_size
    self$L <- length(bin_cols)
    self$epochs <- epochs
    self$bin_cols <- bin_cols
    
    self$indices_cache <- list()
    for (col in bin_cols) {
      self$indices_cache[[col]] <- split(seq_len(nrow(x)), as.factor(x[, col]), drop = TRUE)
    }
  },
  
  .iter = function() {
    cursor <- 0L
    function() {
      col_name <- self$bin_cols[(cursor %% self$L) + 1L]
      groups <- self$indices_cache[[col_name]]
      count_dist <- alloc_even(self$bs, names(groups))
      idx <- sample_from_cache(groups, count_dist)
      if (length(idx) > 0) {
        idx <- idx[sample.int(length(idx))]
      }
      cursor <<- cursor + 1L
      idx
    }
  },
  
  .length = function(){
    self$epochs
  }
)

SRSSampler <- sampler(
  name = "SRSSampler",
  initialize = function(n, batch_size, epochs) {
    self$n <- n
    self$bs <- batch_size
    self$replace <- n < batch_size
    self$epochs <- epochs
  },
  .iter = function() {
    function() {
      sample.int(self$n, self$bs, replace = self$replace)
    }
  },
  .length = function() {
    self$epochs
  }
)