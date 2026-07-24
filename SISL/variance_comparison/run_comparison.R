# SISL variance-source comparison.
#   - single point estimates ONLY from the full phase-two data (weight model and imputation
#     model each fit once on all 500 observed rows);
#   - the two covariances contribute to the MI redraws only, never to a point estimate;
#   - per imputation: draw weight coefficients ~ Normal(full-data fit, V_weight); refit the
#     imputation model on the FULL data with the drawn surface plugged in; draw its
#     coefficients ~ Normal(that refit, V_imputation); impute from the (1 - pi)-tilted density;
#   - Rubin's rules, m = 20.
# V_weight, V_imputation by variant:
#   dualboot     : covariance of K = 100 coefficient re-fits on i.i.d. resamples of the 500
#                  rows; the weight-model spline bases are RE-BUILT on each resample
#                  (cut-points from the resample's own quantiles); the imputation model is
#                  re-fit per resample with the full-data weight surface plugged in.
#   dualsandwich : HC0 sandwich for the weight regression; H^-1 J H^-1 (per-observation
#                  scores) for the imputation model, both at the full-data fits.
# Specification: weight model log(W-1) ~ bs(X,6) + ns(Y,5) + bs(X,4):ns(Y,3) + others,
# shift term tau^2/2 fixed at the full-data value; surface clamped to the observed X-range.
# Usage (idempotent PER VARIANT: a variant is skipped for a replicate when its file exists,
# so a new variant can be added without recomputing the ones already banked):
#   Rscript run_comparison.R <task_id>      task_id in 1..20, covering 500 replicates in
#                                           chunks of 25 (SLURM_ARRAY_TASK_ID is used when no
#                                           argument is given) -- see run_comparison.sl
#   Rscript run_comparison.R <from> <to>    explicit replicate range (local use)
#   SISL_VARIANTS=modelbased Rscript run_comparison.R <from> <to>   one variant only
# Seeding is controlled: replicate i always runs under set.seed(sim_seed[i]) with the seed
# vector stored in ../../code_infsamp/data/params/simulation_seed.RData, so results are
# reproducible regardless of chunking, reruns, or execution order.
# Saved outputs (harness convention):
#   ./simulations/<variant>/<digit>.RData    imp_list = the m completed datasets, xz
#   ./results/results_<variant>.csv          per-replicate pooled summaries
# for variant in {dualboot, dualsandwich, modelbased}. A variant counts as done for a
# replicate (and is skipped) when its own .RData file exists.
ns <- splines::ns; bs <- splines::bs
gq <- statmod::gauss.quad.prob(15, "normal"); zk <- gq$nodes; wk <- gq$weights
SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])))
HARNESS <- normalizePath(file.path(SCRIPT_DIR, "..", "..", "code_infsamp"))
OUT_DIR <- file.path(SCRIPT_DIR, "results")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
VARIANT_ALL <- c("dualboot", "dualsandwich", "modelbased")
OUT_FILES <- setNames(file.path(OUT_DIR, paste0("results_", VARIANT_ALL, ".csv")), VARIANT_ALL)
SIM_DIRS <- setNames(file.path(SCRIPT_DIR, "simulations", VARIANT_ALL), VARIANT_ALL)
for (d in SIM_DIRS) dir.create(d, recursive = TRUE, showWarnings = FALSE)
## Variants to attempt in this run; a variant already banked for a replicate is never
## recomputed, so adding a variant does not disturb the ones already on disk.
##   Rscript run_comparison.R <args>              all three variants
##   SISL_VARIANTS=modelbased Rscript ...         model-based arm only
WANT <- { e <- Sys.getenv("SISL_VARIANTS")
  if (nzchar(e)) intersect(trimws(strsplit(e, ",")[[1]]), VARIANT_ALL) else VARIANT_ALL }
if (!length(WANT)) stop("SISL_VARIANTS matched no known variant")
PRED <- c("Xt", "Z1", "Z2", "Z3", "Y"); dfXi <- 4L; dfYi <- 3L
load(file.path(HARNESS, "data/params/simulation_seed.RData"))
colCumsum <- if (requireNamespace("Rfast", quietly = TRUE)) function(M) Rfast::colCumSums(M) else function(M) apply(M, 2L, cumsum)
psd_root <- function(V) { V <- (V + t(V))/2; e <- eigen(V, symmetric = TRUE)
  e$vectors %*% (sqrt(pmax(e$values, 0)) * t(e$vectors)) }

## weight-model least squares on given rows, bases built ON THOSE ROWS (unfixed splines)
fit_weight <- function(X, Y, Vlin, lW) {
  bX6 <- bs(X, 6); bXi <- bs(X, dfXi); bY5 <- ns(Y, 5); bYi <- ns(Y, dfYi)
  U <- cbind(1, predict(bY5, Y), Vlin)
  P <- cbind(U, predict(bX6, X))
  Ti <- lapply(1:dfYi, function(k) predict(bYi, Y)[, k])
  Bi <- predict(bXi, X)
  for (j in 1:dfYi) P <- cbind(P, Bi * Ti[[j]])
  q <- ncol(P)
  g <- as.vector(qr.solve(crossprod(P) + diag(1e-8, q), crossprod(P, lW)))
  list(g = g, P = P, bX6 = bX6, bXi = bXi, bY5 = bY5, bYi = bYi, qU = ncol(U),
       res = lW - as.vector(P %*% g))
}

run_rep <- function(samp, seed, variants = VARIANT_ALL, m = 20L, n_grid = 512L, K = 100L,
                    max_retry = 8L) {
  Vdf <- samp[, PRED]; X <- samp$X; W <- samp$W
  ry <- samp$R == 1; obs <- which(ry); mis <- which(!ry)
  n_obs <- length(obs); n_mis <- length(mis)
  if (min(W[obs]) <= 1) stop("observed weights must exceed 1")
  other <- setdiff(PRED, "Y")
  Xo <- X[obs]; Yo <- Vdf$Y[obs]; lWo <- log(W[obs] - 1)
  Vlin_o <- as.matrix(Vdf[obs, other])
  ## full-data weight fit (the ONLY point estimate for the weight model)
  wf <- fit_weight(Xo, Yo, Vlin_o, lWo)
  g_hat <- wf$g; q <- length(g_hat); qU <- wf$qU
  s2 <- sum(wf$res^2)/(n_obs - q); jt <- s2/2
  xr <- range(Xo); sdX <- sd(Xo)
  split_g <- function(gv) list(gU = gv[1:qU], gX = gv[qU + (1:6)],
                               Gl = lapply(1:dfYi, function(j) gv[qU + 6 + dfXi*(j-1) + (1:dfXi)]))
  ## surface evaluation always uses the FULL-DATA bases (draws are applied to this basis)
  Uo <- cbind(1, predict(wf$bY5, Yo), Vlin_o)
  Um <- cbind(1, predict(wf$bY5, Vdf$Y[mis]), as.matrix(Vdf[mis, other]))
  To <- lapply(1:dfYi, function(k) predict(wf$bYi, Yo)[, k])
  Tm <- lapply(1:dfYi, function(k) predict(wf$bYi, Vdf$Y[mis])[, k])
  eta_at <- function(xv, Urows, Trows, gs) {
    G <- nrow(xv); xc <- pmin(pmax(as.vector(xv), xr[1]), xr[2])
    B6 <- predict(wf$bX6, xc); Bi <- predict(wf$bXi, xc)
    out <- matrix(as.vector(B6 %*% gs$gX), G, ncol(xv)) + rep(as.vector(Urows %*% gs$gU), each = G)
    for (j in 1:dfYi) out <- out + matrix(as.vector(Bi %*% gs$Gl[[j]]), G, ncol(xv)) * rep(Trows[[j]], each = G)
    out }
  Mo <- model.matrix(~ Xt + Z1 + Z2 + Z3 + Y, Vdf[obs, ]); p <- ncol(Mo)
  Mm <- model.matrix(~ Xt + Z1 + Z2 + Z3 + Y, Vdf[mis, ])
  fit_theta <- function(rws, gs, start) {
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
  op0 <- fit_theta(all_rows, gs0, th0)                # full-data imputation-model fit (the point)
  if (!ok_fit(op0)) stop("full-data imputation-model fit failed")
  ## ---- variance objects (only those the requested variants need) ----
  Vs <- list()
  if ("dualboot" %in% variants) {
    ## dual bootstrap, K = 100, splines re-built per resample
    Gmat <- matrix(NA_real_, K, q); THmat <- matrix(NA_real_, K, p + 1)
    for (k in 1:K) {
      rws <- sample(n_obs, n_obs, replace = TRUE)
      gk <- tryCatch(fit_weight(Xo[rws], Yo[rws], Vlin_o[rws, , drop = FALSE], lWo[rws])$g,
                     error = function(e) NULL)
      if (!is.null(gk)) Gmat[k, ] <- gk
      opk <- tryCatch(fit_theta(rws, gs0, op0$par), error = function(e) NULL)
      if (!is.null(opk) && all(is.finite(opk$par)) && opk$convergence == 0 &&
          opk$value < 1e9) THmat[k, ] <- opk$par
    }
    Vg_boot <- cov(Gmat[complete.cases(Gmat), , drop = FALSE])
    Vt_boot <- cov(THmat[complete.cases(THmat), , drop = FALSE])
    Vs$dualboot <- list(g = psd_root(Vg_boot), t = psd_root(Vt_boot))
  }
  H0 <- if (any(c("dualsandwich", "modelbased") %in% variants))
          solve((op0$hessian + t(op0$hessian))/2) else NULL
  PtPi <- if (any(c("dualsandwich", "modelbased") %in% variants))
            chol2inv(chol(crossprod(wf$P) + diag(1e-8, q))) else NULL
  if ("dualsandwich" %in% variants) {
    ## dual sandwich at the full-data fits
    Vg_sw <- PtPi %*% crossprod(wf$P * wf$res) %*% PtPi
    ll_i <- function(pp) { a <- pp[1:p]; sg <- exp(pp[p+1]); mu <- as.vector(Mo %*% a)
      pin <- 1/(1 + exp(eta_at(outer(sg*zk, mu, `+`), Uo, To, gs0) + jt))
      D <- colSums(pin * wk)
      dnorm(Xo, mu, sg, log = TRUE) - log(D) }
    sc <- numDeriv::jacobian(ll_i, op0$par)
    Vt_sw <- H0 %*% crossprod(sc) %*% H0
    Vs$dualsandwich <- list(g = psd_root(Vg_sw), t = psd_root(Vt_sw))
  }
  if ("modelbased" %in% variants)
    Vs$modelbased <- list(g = psd_root(s2 * PtPi), t = psd_root(H0))
  inv <- function(cdf, u, xg) { km <- .colSums(cdf < rep(u, each=n_grid), n_grid, n_mis); klo <- pmin(pmax(km,1), n_grid-1L)
    i0 <- klo + (seq_len(n_mis)-1L)*n_grid; c0 <- cdf[i0]; c1 <- cdf[i0+1L]
    w <- (u-c0)/(c1-c0); w[!(c1>c0)] <- 0; w <- pmin(pmax(w,0),1)
    iv <- xg[klo] + w*(xg[klo+1L]-xg[klo]); iv[km==0L] <- xg[1L]; iv[km>=n_grid] <- xg[n_grid]; iv }
  out <- list(); imps <- list()
  for (variant in names(Vs)) {
    cg <- Vs[[variant]]$g; ct <- Vs[[variant]]$t
    ## Reseed the model-based draws from the replicate seed so they do not depend on which
    ## other variants ran in this call. It is last in Vs, so earlier streams are unaffected.
    if (identical(variant, "modelbased")) set.seed(seed + 7919L)
    ests <- ses <- numeric(m); nretry <- 0L
    imp_list <- vector("list", m)
    for (j in 1:m) {
      ok <- FALSE
      for (att in 1:max_retry) {
        gj <- g_hat + as.vector(cg %*% rnorm(q))
        op_j <- tryCatch(fit_theta(all_rows, split_g(gj), op0$par), error = function(e) NULL)
        if (!ok_fit(op_j)) { nretry <- nretry + 1L; next }
        ok <- TRUE; break
      }
      if (!ok) stop("no acceptable draw after retries")
      d <- as.vector(op_j$par + ct %*% rnorm(p + 1))
      a <- d[1:p]; sg <- exp(d[p+1]); mu <- as.vector(Mm %*% a)
      xg <- seq(min(mu) - 8*sg, max(mu) + 8*sg, length.out = n_grid)
      surv <- 1/(1 + exp(-(eta_at(matrix(xg, n_grid, n_mis), Um, Tm, split_g(gj)) + jt)))
      z <- (xg - rep(mu, each = n_grid))/sg; dim(z) <- c(n_grid, n_mis)
      dens <- surv * exp(-0.5*z*z)
      cdf <- colCumsum(dens); cdf <- cdf/rep(cdf[n_grid,], each = n_grid)
      zd <- samp; zd$X[mis] <- inv(cdf, runif(n_mis), xg)
      imp_list[[j]] <- zd                              # completed dataset, kept
      f <- lm(Y ~ X + Z1 + Z2 + Z3, zd); s <- summary(f)$coefficients
      ests[j] <- s["X", 1]; ses[j] <- s["X", 2]
    }
    imps[[variant]] <- imp_list
    U <- mean(ses^2); B <- var(ests); Q <- mean(ests); Tt <- U + (1 + 1/m)*B
    df <- if (B <= 0) 1e6 else (m-1)*(1 + U/((1+1/m)*B))^2; tc <- qt(.975, df)
    out[[variant]] <- data.frame(variant = variant, Q = Q, U = U, B = B,
        lo = Q - tc*sqrt(Tt), hi = Q + tc*sqrt(Tt), nretry = nretry)
  }
  list(summary = do.call(rbind, out), imps = imps)
}

args <- commandArgs(trailingOnly = TRUE)
replicate_total <- 500L; n_chunks <- 20L; chunk_size <- ceiling(replicate_total / n_chunks)
if (length(args) >= 2) {                       # explicit range (local use)
  first_rep <- as.integer(args[1]); last_rep <- as.integer(args[2])
} else {                                       # task-id chunking (cluster / harness convention)
  task_env <- Sys.getenv("SLURM_ARRAY_TASK_ID")
  task_id <- if (length(args) == 1) as.integer(args[1]) else if (nzchar(task_env)) as.integer(task_env) else 1L
  first_rep <- (task_id - 1L) * chunk_size + 1L
  last_rep <- min(task_id * chunk_size, replicate_total)
}
## Per-variant idempotency: a variant is computed for replicate i only when its imputed-dataset
## file is absent, so variants already banked are never recomputed or overwritten.
variant_done <- function(v, i) file.exists(file.path(SIM_DIRS[[v]], sprintf("%04d.RData", i)))
summarized <- lapply(OUT_FILES, function(f)
  if (file.exists(f)) unique(read.csv(f)$rep) else integer(0))
for (i in first_rep:last_rep) {
  todo <- WANT[!vapply(WANT, variant_done, logical(1), i = i)]
  if (!length(todo)) next
  samp <- read.csv(sprintf("%s/data/Sample/InfSamp/%04d.csv", HARNESS, i))
  set.seed(sim_seed[i])
  r <- tryCatch(run_rep(samp, seed = sim_seed[i], variants = todo), error = function(e) NULL)
  if (is.null(r)) { cat(sprintf("[rep %d] FAILED\n", i)); next }
  digit <- sprintf("%04d", i)
  for (v in todo) {                                   # the imputed datasets, harness convention
    imp_list <- r$imps[[v]]
    save(imp_list, file = file.path(SIM_DIRS[[v]], paste0(digit, ".RData")), compress = "xz")
  }
  summ <- cbind(rep = i, r$summary)
  cat(sprintf("[rep %d] %s\n", i, paste(sprintf("%s Q=%.3f", summ$variant, summ$Q), collapse = "  ")))
  for (v in todo) {
    if (i %in% summarized[[v]]) next                  # summary row already banked (seeded rerun)
    rv <- summ[summ$variant == v, , drop = FALSE]
    ap <- file.exists(OUT_FILES[[v]])
    write.table(rv, OUT_FILES[[v]], sep = ",", append = ap, col.names = !ap, row.names = FALSE)
  }
}
