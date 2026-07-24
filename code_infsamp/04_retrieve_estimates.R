suppressMessages(lapply(c("dplyr", "stringr", "survey"), require, character.only = TRUE))
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")
sim_root <- "./simulations"; out_dir <- file.path(sim_root, "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# All four analysis coefficients are retrieved (matching the nutri/surv/vccc harnesses), so the
# appendix table and figures can report the Phase-One covariates as the other chapters do:
#   X  = Phase-Two exposure (imputed / validated); UNVAL puts the naive proxy Xt in this slot.
#   Z1, Z2 = Phase-One continuous covariates.   Z3 = Phase-One binary covariate.
COEFS <- c("X", "Z1", "Z2", "Z3")

rubin <- function(est, se) {
  m <- length(est); Q <- mean(est); U <- mean(se^2); B <- var(est); Tt <- U + (1 + 1/m) * B
  df <- if (B <= 0) 1e6 else (m - 1) * (1 + U / ((1 + 1/m) * B))^2; tc <- qt(.975, df)
  c(Q, sqrt(Tt), Q - tc * sqrt(Tt), Q + tc * sqrt(Tt))
}
# fetch a coefficient row by exact name or factor-level prefix (Z3 may be "Z3" or "Z31")
grab <- function(s, ci, nm) {
  r <- if (nm %in% rownames(s)) nm else grep(paste0("^", nm), rownames(s), value = TRUE)[1]
  c(s[r, 1], s[r, 2], ci[r, 1], ci[r, 2])
}
# fit lm, return a named list COEF -> c(est, se, lo, hi); `expo` names the exposure term
fit_all <- function(formula, data, expo = "X") {
  f <- lm(formula, data); s <- summary(f)$coefficients; ci <- confint.default(f)
  setNames(lapply(c(expo, "Z1", "Z2", "Z3"), function(tm) grab(s, ci, tm)), COEFS)
}

replicate <- as.integer(Sys.getenv("NREP", "500")); rows <- list(); k <- 0L
for (i in 1:replicate) {
  digit <- str_pad(i, 4, pad = "0")
  tf <- paste0("./data/True/", digit, ".RData"); if (!file.exists(tf)) next
  load(tf); truedat <- as.data.frame(data)
  for (design in DESIGNS) {
    sf <- paste0("./data/Sample/", design, "/", digit, ".csv"); if (!file.exists(sf)) next
    samp <- read.csv(sf)
    add <- function(method, coef, v) { k <<- k + 1L; rows[[k]] <<- data.frame(
      Design = design, Method = method, Coef = coef, ID = i,
      Estimate = v[1], SE = v[2], CI_lo = v[3], CI_hi = v[4]) }
    add_all <- function(method, L) for (cf in COEFS) add(method, cf, L[[cf]])

    add_all("VAL",   fit_all(as.formula(ANALYSIS_FORMULA), truedat, "X"))
    add_all("UNVAL", fit_all(as.formula("Y ~ Xt + Z1 + Z2 + Z3"), truedat, "Xt"))
    add_all("CC",    fit_all(as.formula(ANALYSIS_FORMULA), subset(samp, R == 1), "X"))
    ccw <- tryCatch({                                   # design-based CC: released weights, svyglm
      dsn <- svydesign(ids = ~1, weights = ~W, data = subset(samp, R == 1))
      fit <- svyglm(as.formula(ANALYSIS_FORMULA), design = dsn)
      sm <- summary(fit)$coefficients; ci <- confint(fit)
      setNames(lapply(c("X", "Z1", "Z2", "Z3"), function(tm) grab(sm, ci, tm)), COEFS)
    }, error = function(e) setNames(rep(list(rep(NA_real_, 4)), 4), COEFS))
    add_all("CC-W", ccw)

    for (method in c("naive", "naive_w", "wl", "mice", "rf", "mixgb")) {  # SISL: variance_comparison
      f <- file.path(sim_root, design, method, paste0(digit, ".RData")); if (!file.exists(f)) next
      load(f)
      label <- c(mice = "MICE-PMM", rf = "MICE-RF", mixgb = "MIXGB", naive_w = "NAIVE+W")[method]
      if (is.na(label)) label <- toupper(method)
      fits <- lapply(imp_list, function(d) fit_all(as.formula(ANALYSIS_FORMULA), d, "X"))
      for (cf in COEFS) add(label, cf, rubin(sapply(fits, function(L) L[[cf]][1]),
                                             sapply(fits, function(L) L[[cf]][2])))
    }
    for (method in c("gain", "tabcsdi", "sicg", "sird")) {
      f <- file.path(sim_root, design, method, paste0(digit, ".parquet")); if (!file.exists(f)) next
      pq <- as.data.frame(arrow::read_parquet(f)); label <- toupper(method)
      if ("imp_id" %in% names(pq)) {
        fits <- lapply(split(pq, pq$imp_id), function(d) fit_all(as.formula(ANALYSIS_FORMULA), d, "X"))
        for (cf in COEFS) add(label, cf, rubin(sapply(fits, function(L) L[[cf]][1]),
                                               sapply(fits, function(L) L[[cf]][2])))
      } else {
        add_all(label, fit_all(as.formula(ANALYSIS_FORMULA), pq, "X"))
      }
    }
    rkf <- file.path(sim_root, design, "raking", paste0(digit, ".RData"))
    if (file.exists(rkf)) {
      load(rkf)
      sm <- summary(rakingest)$coefficients; ci <- confint(rakingest)
      add_all("RAKING", setNames(lapply(c("X", "Z1", "Z2", "Z3"),
                                        function(tm) grab(sm, ci, tm)), COEFS))
    }
  }
}
combined <- do.call(rbind, rows)
save(combined, file = file.path(out_dir, "results_COMBINED.RData"))
write.csv(combined, file = file.path(out_dir, "results_COMBINED.csv"), row.names = FALSE)
cat("Wrote", nrow(combined), "rows (", length(unique(combined$Method)), "methods x",
    length(COEFS), "coefs, ", length(unique(combined$ID)), "reps ) to results_COMBINED\n")
