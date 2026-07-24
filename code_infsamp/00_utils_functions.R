suppressMessages(lapply(c("dplyr"), require, character.only = TRUE))

data_info_infsamp <- list(
  cat_vars = c("Z3", "R", "W"),
  num_vars = c("id", "Z1", "Z2", "X", "Xt", "Y", "pi"),
  phase1_vars = c("Xt"),
  phase2_vars = c("X")
)
DATA_INFOS <- list(infsamp = data_info_infsamp)
DESIGNS <- c("InfSamp")

IMP_PREDICTORS <- c("Xt", "Z1", "Z2", "Z3", "Y")
ANALYSIS_FORMULA <- "Y ~ X + Z1 + Z2 + Z3"

exactAllocation <- function(data, stratum_variable, target_variable, sample_size) {
  s <- as.factor(data[[stratum_variable]]); y <- data[[target_variable]]
  Nh <- tapply(y, s, length); Sh <- tapply(y, s, sd)
  if (sample_size < length(Nh) || sample_size > sum(Nh)) stop("Invalid sample size for allocation.")
  score <- as.numeric(Nh * Sh); score[!is.finite(score)] <- 0
  if (all(score <= 0)) score <- as.numeric(Nh)
  names(score) <- names(Nh)
  alloc <- setNames(rep(1L, length(Nh)), names(Nh))
  rem <- sample_size - sum(alloc)
  while (rem > 0) {
    available <- alloc < Nh
    if (!any(available)) stop("Unable to allocate requested sample size.")
    priority <- score / sqrt(alloc * (alloc + 1))
    priority[!available] <- -Inf
    h <- names(which.max(priority))
    alloc[h] <- alloc[h] + 1L
    rem <- rem - 1L
  }
  setNames(as.integer(alloc), names(Nh))
}

imp_gaussian <- function(samp, m, weighted = FALSE) {
  ry <- samp$R == 1; X <- samp$X
  P <- as.matrix(samp[, IMP_PREDICTORS, drop = FALSE])
  obs <- which(ry); mis <- which(!ry)
  D <- cbind(1, P); Do <- D[obs, , drop = FALSE]; Xo <- X[obs]; Dm <- D[mis, , drop = FALSE]
  p <- ncol(D); n <- length(obs)
  if (n <= p) stop("Not enough observed cases for Gaussian imputation.")
  if (weighted) {
    w <- as.numeric((1 / samp$pi[obs]) / mean(1 / samp$pi[obs]))
    if (any(!is.finite(w)) || any(w <= 0)) stop("Invalid imputation weights.")
  } else {
    XtX_inv <- solve(crossprod(Do))
    bh <- XtX_inv %*% crossprod(Do, Xo)
    r <- as.vector(Xo - Do %*% bh)
    rss <- sum(r^2)
    df <- n - p
  }
  lapply(1:m, function(j) {
    if (weighted) {
      bb <- rexp(n)
      om <- as.numeric(w * bb / mean(w * bb))
      bs <- solve(crossprod(Do, Do * om), crossprod(Do, Xo * om))
      rb <- as.vector(Xo - Do %*% bs)
      sg <- sqrt(sum(om * rb^2) / sum(om))
    } else {
      sg <- sqrt(rss / rchisq(1, df))
      bs <- bh + sg * (t(chol(XtX_inv)) %*% rnorm(p))
    }
    xi <- as.vector(Dm %*% bs) + sg * rnorm(length(mis))
    comp <- samp; comp$X[mis] <- xi; comp
  })
}

match_types <- function(df, template) {
  for (v in intersect(names(df), names(template))) {
    if (is.factor(template[[v]]) || is.character(template[[v]])) df[[v]] <- as.character(df[[v]])
    else df[[v]] <- as.numeric(df[[v]])
  }
  df
}
calcCICover <- function(lo, hi, truth) as.numeric(truth >= lo & truth <= hi)
expit <- function(x) 1 / (1 + exp(-x))
