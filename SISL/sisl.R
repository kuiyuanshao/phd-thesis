## SISL imputer -- sample-likelihood multiple imputation under informative sampling.
##
## Point estimates come from the FULL phase-two data: a weight regression and, given the
## fitted inclusion surface, a sample-likelihood imputation model. Proper multiple imputation
## then draws the two-stage parameters from an estimated covariance and imputes each unsampled
## row from the non-selection density. Three constructions of that covariance are offered,
## selected by `variance`; they are the SISL-S / SISL-B / SISL-M variants of Chapter 5.
##
##   "sandwich"    (SISL-S): analytic covariances. HC0 sandwich for the weight regression and
##                 H^-1 J H^-1 (per-observation scores) for the imputation model, both at the
##                 full-data fits (requires numDeriv). Each imputation redraws the weight
##                 coefficients, refits the imputation model given the drawn surface, then
##                 redraws its coefficients, so the weight-surface uncertainty is carried in.
##   "bootstrap"   (SISL-B, default): as "sandwich" but both covariances are the covariance of
##                 K bootstrap refits of the phase-two rows, with the weight-model spline
##                 cut-points rebuilt on every resample.
##
## Weight model: log(W - 1) ~ bs(X,dfX) + ns(Y,dfY) + bs(X,dfXi):ns(Y,dfYi) + other predictors
## linear, by least squares; with tau^2 = var(residual) the fitted inclusion probability is
## pi(x,V) = 1/(1 + exp(eta(x,V) + tau^2/2)), the shifted log-normal form, inside (0,1) for
## every coefficient value. The interaction uses a coarser X-basis (dfXi) than the main effect
## (dfX). The spline surface is CLAMPED to the observed X-range wherever it is evaluated, which
## bounds the sample likelihood (without the clamp a sigma -> infinity optimum exists through
## B-spline extrapolation); a fit is rejected if optim fails, the objective is degenerate, or
## the fitted sigma exceeds 3 sd(X), and the imputation draw is retried up to max_retry times.
##
##   y     : variable to impute (X), NA where ry == 0
##   ry    : phase-two indicator (1 = X observed)
##   x     : matrix/data.frame of phase-one predictors; MUST contain a column named "Y" (the
##           outcome the design selects on); the rest enter the weight model linearly and the
##           imputation mean model through mean_formula.
##   invpi : released design weight W = 1/pi, for every row; must exceed 1 on observed rows.
##   variance : "sandwich", "bootstrap" (default), or "model-based" (see above).
##   K     : number of bootstrap refits for the "bootstrap" covariance.
sisl_impute <- function(y, ry, x, invpi, mean_formula = NULL,
                        dfX = 6L, dfXi = 3L, dfY = 5L, dfYi = 3L,
                        m = 20L, gh = 15L, n_grid = 512L, max_retry = 8L,
                        variance = c("bootstrap", "sandwich", "model-based"), K = 100L) {
  variance <- match.arg(variance)
  ns <- splines::ns; bs <- splines::bs
  ry <- as.logical(ry); Vdf <- as.data.frame(x)
  X <- as.numeric(y); W <- as.numeric(invpi)
  obs <- which(ry); mis <- which(!ry)
  n_obs <- length(obs); n_mis <- length(mis)
  if (n_mis == 0L) return(replicate(m, numeric(0), simplify = FALSE))
  if (!("Y" %in% names(Vdf))) stop("sisl_impute: x must contain a column named 'Y'")
  if (min(W[obs]) <= 1) stop("sisl_impute: observed design weights must exceed 1 (log(W-1) weight model)")

  if (is.null(mean_formula)) mean_formula <- stats::reformulate(names(Vdf))
  other <- setdiff(names(Vdf), "Y")
  Mm <- model.matrix(mean_formula, Vdf[mis, , drop = FALSE]); p <- ncol(Mm)
  gq <- statmod::gauss.quad.prob(gh, "normal"); zk <- gq$nodes; wk <- gq$weights

  ## ---- full-data fits: weight model, then imputation model given the fitted surface ----
  Xo <- X[obs]; Yo <- Vdf$Y[obs]; lWo <- log(W[obs] - 1)
  bXs <- bs(Xo, dfX); bXi <- if (dfXi == dfX) bXs else bs(Xo, dfXi)
  xr <- range(Xo); sdX <- sd(Xo)
  bYs <- ns(Yo, dfY); bYi <- ns(Yo, dfYi)
  Vlin_o <- if (length(other)) as.matrix(Vdf[obs, other, drop = FALSE]) else NULL
  Uo <- cbind(1, predict(bYs, Yo), Vlin_o); qU <- ncol(Uo)
  Um <- cbind(1, predict(bYs, Vdf$Y[mis]),
              if (length(other)) as.matrix(Vdf[mis, other, drop = FALSE]) else NULL)
  To <- lapply(1:dfYi, function(k) predict(bYi, Yo)[, k])
  Tm <- lapply(1:dfYi, function(k) predict(bYi, Vdf$Y[mis])[, k])
  BXo <- predict(bXs, Xo); BXio <- predict(bXi, Xo)
  Po <- cbind(Uo, BXo); for (j in 1:dfYi) Po <- cbind(Po, BXio * To[[j]])
  q <- ncol(Po); PtPi <- chol2inv(chol(crossprod(Po) + diag(1e-8, q)))
  g_hat <- as.vector(PtPi %*% crossprod(Po, lWo))
  res <- lWo - as.vector(Po %*% g_hat)
  s2 <- sum(res^2)/(n_obs - q); jt <- s2/2
  split_g <- function(gv) list(gU = gv[1:qU], gX = gv[qU + (1:dfX)],
                               Gl = lapply(1:dfYi, function(j) gv[qU + dfX + dfXi*(j-1) + (1:dfXi)]))
  eta_at <- function(xv, Urows, Trows, gs) {
    G <- nrow(xv); xc <- pmin(pmax(as.vector(xv), xr[1]), xr[2])              # CLAMP to X-range
    B <- predict(bXs, xc); Bi <- if (dfXi == dfX) B else predict(bXi, xc)
    out <- matrix(as.vector(B %*% gs$gX), G, ncol(xv)) + rep(as.vector(Urows %*% gs$gU), each = G)
    for (j in 1:dfYi) out <- out + matrix(as.vector(Bi %*% gs$Gl[[j]]), G, ncol(xv)) * rep(Trows[[j]], each = G)
    out }
  Mo <- model.matrix(mean_formula, Vdf[obs, , drop = FALSE])
  fit_theta_full <- function(rws, gs, start) {
    Xr <- Xo[rws]; Mor <- Mo[rws, , drop = FALSE]
    Ur <- Uo[rws, , drop = FALSE]; Tr <- lapply(To, function(v) v[rws])
    neg <- function(pp) { a <- pp[1:p]; sg <- exp(pp[p+1]); mu <- as.vector(Mor %*% a)
      pin <- 1/(1 + exp(eta_at(outer(sg*zk, mu, `+`), Ur, Tr, gs) + jt))
      D <- colSums(pin * wk)
      v <- -sum(dnorm(Xr, mu, sg, log = TRUE) - log(D)); if (is.finite(v)) v else 1e10 }
    optim(start, neg, method = "BFGS", hessian = identical(rws, seq_len(n_obs)),
          control = list(maxit = 500))
  }
  ok_fit <- function(op) !is.null(op) && all(is.finite(op$par)) && op$convergence == 0 &&
                         op$value < 1e9 && exp(op$par[p+1]) <= 3*sdX
  all_rows <- seq_len(n_obs); th0 <- c(qr.coef(qr(Mo), Xo), log(sd(Xo)))
  gs0 <- split_g(g_hat)
  op0 <- fit_theta_full(all_rows, gs0, th0)
  if (!ok_fit(op0)) stop("sisl_impute: full-data imputation-model fit failed")
  psd_root <- function(V) { V <- (V + t(V))/2; e <- eigen(V, symmetric = TRUE)
    e$vectors %*% (sqrt(pmax(e$values, 0)) * t(e$vectors)) }

  if (variance == "bootstrap") {
    Gm <- matrix(NA_real_, K, q); Th <- matrix(NA_real_, K, p + 1)
    for (k in 1:K) {
      rws <- sample(n_obs, n_obs, replace = TRUE)
      gk <- tryCatch({                       # weight refit, bases RE-BUILT on the resample
        Xk <- Xo[rws]; Yk <- Yo[rws]
        bXk <- bs(Xk, dfX); bXik <- if (dfXi == dfX) bXk else bs(Xk, dfXi)
        bYk <- ns(Yk, dfY); bYik <- ns(Yk, dfYi)
        Uk <- cbind(1, predict(bYk, Yk),
                    if (length(other)) Vlin_o[rws, , drop = FALSE] else NULL)
        Pk <- cbind(Uk, predict(bXk, Xk))
        Tk <- lapply(1:dfYi, function(kk) predict(bYik, Yk)[, kk])
        Bik <- predict(bXik, Xk)
        for (jj in 1:dfYi) Pk <- cbind(Pk, Bik * Tk[[jj]])
        as.vector(qr.solve(crossprod(Pk) + diag(1e-8, q), crossprod(Pk, lWo[rws])))
      }, error = function(e) NULL)
      if (!is.null(gk)) Gm[k, ] <- gk
      opk <- tryCatch(fit_theta_full(rws, gs0, op0$par), error = function(e) NULL)
      if (!is.null(opk) && all(is.finite(opk$par)) && opk$convergence == 0 && opk$value < 1e9)
        Th[k, ] <- opk$par
    }
    cg <- psd_root(cov(Gm[complete.cases(Gm), , drop = FALSE]))
    ct <- psd_root(cov(Th[complete.cases(Th), , drop = FALSE]))
  } else if (variance == "sandwich") {
    Vg <- PtPi %*% crossprod(Po * res) %*% PtPi
    ll_i <- function(pp) { a <- pp[1:p]; sg <- exp(pp[p+1]); mu <- as.vector(Mo %*% a)
      pin <- 1/(1 + exp(eta_at(outer(sg*zk, mu, `+`), Uo, To, gs0) + jt))
      D <- colSums(pin * wk)
      dnorm(Xo, mu, sg, log = TRUE) - log(D) }
    H0 <- solve((op0$hessian + t(op0$hessian))/2)
    sc <- numDeriv::jacobian(ll_i, op0$par)
    cg <- psd_root(Vg); ct <- psd_root(H0 %*% crossprod(sc) %*% H0)
  } else {
    cg <- psd_root(s2 * PtPi)
    ct <- psd_root(solve((op0$hessian + t(op0$hessian))/2))
  }

  draws <- vector("list", m); nretry <- 0L
  for (j in 1:m) {
    ok <- FALSE
    for (att in 1:max_retry) {
      gj <- g_hat + as.vector(cg %*% rnorm(q))
      op_j <- tryCatch(fit_theta_full(all_rows, split_g(gj), op0$par), error = function(e) NULL)
      if (!ok_fit(op_j)) { nretry <- nretry + 1L; next }
      ok <- TRUE; break
    }
    if (!ok) stop("sisl_impute: no acceptable redraw after retries")
    d <- as.vector(op_j$par + ct %*% rnorm(p + 1))
    a <- d[1:p]; sg <- exp(d[p+1]); mu <- as.vector(Mm %*% a)
    xg <- seq(min(mu) - 8*sg, max(mu) + 8*sg, length.out = n_grid)
    surv <- 1/(1 + exp(-(eta_at(matrix(xg, n_grid, n_mis), Um, Tm, split_g(gj)) + jt)))
    z <- (xg - rep(mu, each = n_grid))/sg; dim(z) <- c(n_grid, n_mis)
    dens <- surv * exp(-0.5*z*z)
    cdf <- colCumsum(dens); cdf <- cdf/rep(cdf[n_grid,], each = n_grid)
    u <- runif(n_mis)
    km <- .colSums(cdf < rep(u, each = n_grid), n_grid, n_mis)
    klo <- pmin(pmax(km, 1), n_grid - 1L)
    i0 <- klo + (seq_len(n_mis) - 1L) * n_grid; c0 <- cdf[i0]; c1 <- cdf[i0 + 1L]
    w2 <- (u - c0)/(c1 - c0); w2[!(c1 > c0)] <- 0; w2 <- pmin(pmax(w2, 0), 1)
    iv <- xg[klo] + w2*(xg[klo + 1L] - xg[klo])
    iv[km == 0L] <- xg[1L]; iv[km >= n_grid] <- xg[n_grid]
    draws[[j]] <- iv
  }
  attr(draws, "nretry") <- nretry; attr(draws, "variance") <- variance
  draws
}

## fast column cumulative sum; base fallback if Rfast/matrixStats absent
colCumsum <- local({
  have_rfast <- requireNamespace("Rfast", quietly = TRUE)
  have_ms <- requireNamespace("matrixStats", quietly = TRUE)
  if (have_rfast) function(M) Rfast::colCumSums(M)
  else if (have_ms) function(M) matrixStats::colCumsums(M)
  else function(M) apply(M, 2L, cumsum)
})
