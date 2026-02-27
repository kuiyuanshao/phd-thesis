discriminator.mlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ...) {
    self$params <- params
    self$pacdim <- params$ncols * params$pac
    self$seq <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:length(params$d_dim)) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim[i]))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim[i]
    }
    self$seq$add_module("Linear", nn_linear(dim, 1))
  },
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    return (out)
  }
)

SpectralNormLinear <- nn_module(
  "SpectralNormLinear",
  initialize = function(in_features, out_features, bias = TRUE, 
                        power_iterations = 1, eps = 1e-12) {
    self$linear <- nn_linear(in_features, out_features, bias = bias)
    self$power_iterations <- power_iterations
    self$eps <- eps
    self$register_buffer("u", torch_randn(out_features))
    self$register_buffer("v", torch_randn(in_features))
  },
  
  compute_weight = function() {
    W <- self$linear$weight
    u <- self$u
    v <- self$v
    with_no_grad({
      for (i in 1:self$power_iterations) {
        v_s <- torch_mv(W$t(), u)
        v_norm <- v_s$norm()$clamp(min = self$eps)
        v <- v_s / v_norm
        u_s <- torch_mv(W, v)
        u_norm <- u_s$norm()$clamp(min = self$eps)
        u <- u_s / u_norm
      }
      self$u$copy_(u)
      self$v$copy_(v)
    })
    
    sigma <- torch_dot(u, torch_mv(W, v))
    W_sn <- W / sigma
    
    return(W_sn)
  },
  
  forward = function(x) {
    W_sn <- self$compute_weight()
    nnf_linear(x, W_sn, self$linear$bias)
  }
)