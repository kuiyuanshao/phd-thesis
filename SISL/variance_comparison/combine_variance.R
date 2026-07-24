# Combine the SISL variance-source comparison (dualboot vs dualsandwich vs modelbased): read
# each saved imputed-dataset list, re-fit Y ~ X + Z1 + Z2 + Z3 on every completed dataset, and
# Rubin-pool ALL FOUR coefficients per replicate (X exposure + Z1,Z2 Phase-One continuous + Z3
# Phase-One binary), matching code_infsamp/04. Output carries a Coef column.
# A variant with no imputed-dataset files on disk is reported as absent and skipped, so this
# runs unchanged whether or not the model-based arm has been produced yet.
suppressMessages(library(stringr))
SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE)[1])))
truthX <- 0.48
VARIANTS <- c("dualboot", "dualsandwich", "modelbased")
ANALYSIS <- Y ~ X + Z1 + Z2 + Z3
COEFS <- c("X", "Z1", "Z2", "Z3")

grab <- function(s, nm) { r <- if (nm %in% rownames(s)) nm else grep(paste0("^", nm), rownames(s), value = TRUE)[1]; c(s[r, 1], s[r, 2]) }
pool_rep <- function(imp_list) {                 # Rubin's rules, all coefs, one replicate
  m <- length(imp_list)
  ests <- ses <- matrix(NA_real_, m, length(COEFS), dimnames = list(NULL, COEFS))
  for (j in seq_len(m)) { s <- summary(lm(ANALYSIS, imp_list[[j]]))$coefficients
    for (cf in COEFS) { g <- grab(s, cf); ests[j, cf] <- g[1]; ses[j, cf] <- g[2] } }
  out <- matrix(NA_real_, length(COEFS), 4, dimnames = list(COEFS, c("Est","SE","lo","hi")))
  for (cf in COEFS) { Q <- mean(ests[, cf]); U <- mean(ses[, cf]^2); B <- var(ests[, cf])
    Tt <- U + (1 + 1/m) * B; df <- if (B <= 0) 1e6 else (m-1)*(1+U/((1+1/m)*B))^2; tc <- qt(.975, df)
    out[cf, ] <- c(Q, sqrt(Tt), Q - tc*sqrt(Tt), Q + tc*sqrt(Tt)) }
  out
}

rows <- list()
for (v in VARIANTS) {
  vdir <- file.path(SCRIPT_DIR, "simulations", v)
  files <- sort(list.files(vdir, pattern = "^[0-9]{4}\\.RData$", full.names = TRUE))
  cat(sprintf("[%s] %d replicate files\n", v, length(files)))
  for (f in files) {
    e <- new.env(); load(f, envir = e); P <- pool_rep(e$imp_list)
    id <- as.integer(str_sub(basename(f), 1, 4))
    for (cf in COEFS) rows[[length(rows) + 1]] <- data.frame(variant = v, Coef = cf, ID = id,
      Estimate = P[cf, "Est"], SE = P[cf, "SE"], CI_lo = P[cf, "lo"], CI_hi = P[cf, "hi"], row.names = NULL)
  }
}
combined <- do.call(rbind, rows)
save(combined, file = file.path(SCRIPT_DIR, "results", "variance_COMBINED.RData"))
write.csv(combined, file.path(SCRIPT_DIR, "results", "variance_COMBINED.csv"), row.names = FALSE)

# exposure-only summary (unchanged headline table); only variants actually on disk are reported
xx <- combined[combined$Coef == "X", ]
PRESENT <- VARIANTS[VARIANTS %in% unique(xx$variant)]
summ <- do.call(rbind, lapply(PRESENT, function(v) { s <- xx[xx$variant == v, ]
  data.frame(Variant = v, n_rep = nrow(s), Truth = truthX, Mean_Est = mean(s$Estimate),
    Bias = mean(s$Estimate) - truthX, Pct_Bias = 100*(mean(s$Estimate)-truthX)/truthX,
    Emp_SE = sd(s$Estimate), Mean_SE = mean(s$SE), SE_Ratio = mean(s$SE)/sd(s$Estimate),
    RMSE = sqrt(mean((s$Estimate-truthX)^2)), Coverage = 100*mean(s$CI_lo<=truthX & truthX<=s$CI_hi)) }))
write.csv(summ, file.path(SCRIPT_DIR, "results", "variance_Summary_Table.csv"), row.names = FALSE)
cat("\n===== SISL variance-source comparison (exposure beta_X, 500 reps) =====\n")
print(as.data.frame(summ), digits = 4, row.names = FALSE)
cat("\ncoefs in output:", paste(unique(combined$Coef), collapse = ", "), "| rows:", nrow(combined), "\n")
