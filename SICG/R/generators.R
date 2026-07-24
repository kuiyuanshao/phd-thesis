Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2, rate, ...){
    self$seq <- nn_sequential(
      nn_linear(dim1, dim2),
      nn_batch_norm1d(dim2),
      nn_elu(),
      nn_dropout(rate)
    )
  },
  forward = function(input){
    output <- self$seq(input)
    return (torch_cat(list(output, input), dim = 2))
  }
)

generator.mlp <- nn_module(
  "Generator",
  initialize = function(params, ...){
    self$nphase2 <- params$nphase2
    self$params <- params

    dim1 <- params$noise_dim + params$cond_dim

    self$dropout <- nn_dropout(params$g_dropout / 2)

    self$cond_encoder <- nn_sequential(
      nn_linear(params$ncols - params$nphase2, params$cond_dim),
      nn_batch_norm1d(params$cond_dim),
      nn_elu(0.2),
      nn_dropout(params$g_dropout / 2)
    )

    self$seq <- nn_sequential()
    for (i in 1:length(params$g_dim)){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, params$g_dim[i], params$g_dropout))
      dim1 <- dim1 + params$g_dim[i]
    }
    self$seq$add_module("Linear", nn_linear(dim1, params$nphase2))
  },
  forward = function(N, A, C, ...){
    cond <- self$dropout(torch_cat(list(A, C), dim = 2))
    cond <- self$cond_encoder(cond)
    input <- torch_cat(list(N, cond), dim = 2)
    X_fake <- self$seq(input)
    
    return (X_fake)
  }
)
