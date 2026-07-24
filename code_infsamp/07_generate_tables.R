## 07_generate_tables.R -- LaTeX tables for the informative-sampling design (Chapter 5, SISL).
##
## Emits, to ./simulations/results/tables/:
##   tab_sisl_rank.tex  the Chapter 5 benchmark block (exposure beta_X only)
##   tab_appc_sisl.tex  the Appendix C full table (all three coefficient groups)
## Both are pasted inline into the chapter/appendix .tex (thesis convention: no \input helpers).
##
## Inputs:
##   ./simulations/results/results_COMBINED.RData          every method except SISL  (`combined`)
##   ../SISL/variance_comparison/results/variance_COMBINED.RData  the SISL variance variants
## SISL is produced by the variance_comparison pipeline, not by 04, so the two sources are
## merged here. A method absent from disk is emitted as a "--" row rather than dropped.
##
## Metric definitions (the thesis-wide benchmark block, per code_nutri/05_generate_results.R):
##   Mean |Bias| (%) = |mean(Est) - truth| / |truth| * 100
##   Rel. RMSE (%)   = sqrt(mean((Est - truth)^2)) / |truth| * 100
##   Emp. SE         = sd(Est)
##   Mean SE         = mean(reported SE)
##   SE ratio        = Mean SE / Emp. SE
##   Nom. Cov.       = mean(truth inside the reported CI)
##   Orc. Cov. (Emp) = mean(truth inside Est +/- qnorm(.975) * Emp. SE)
##   Orc. Cov. (BC)  = mean(truth inside (Est - mean raw bias) +/- qnorm(.975) * reported SE)
##
## Pooling across the coefficients inside a group (verified against the deployed tables):
##   Mean |Bias|, Rel. RMSE : GEOMETRIC mean, i.e. exp(mean(log(.))), so that a coefficient
##                            whose true value is near zero cannot dominate. NOTE this is not
##                            the arithmetic mean used by 05_generate_results.R.
##   Emp. SE, Mean SE       : arithmetic mean
##   SE ratio               : pooled Mean SE / pooled Emp. SE
##   the three coverages    : arithmetic mean
##
## Shading: the three best distinct ranks per shaded column, ranked by |value - target|
## at the displayed precision (target 0 for bias and relative RMSE, 1 for the SE ratio,
## 0.95 for the coverages). Displayed ties receive the same shade. Emp. SE and Mean SE
## are not ranked.
suppressMessages(lapply(c("dplyr", "tibble"), require, character.only = TRUE))

RES     <- "./simulations/results"
OUT_DIR <- file.path(RES, "tables")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
VC <- "../SISL/variance_comparison/results/variance_COMBINED.RData"
Z  <- qnorm(0.975)

load("./data/params/truth.RData")                       # truth$beta_X
TRUTH <- c(X = truth$beta_X, Z1 = 0.30, Z2 = 0.20, Z3 = 0.30)   # 01_generate_data.R: bX, bet

## Coefficient groups, in table order.
GROUPS <- list("Continuous Phase-Two Covariates"          = "X",
               "Continuous Phase-One Covariates"          = c("Z1", "Z2"),
               "Binary \\& Categorical Phase-One Covariates" = "Z3")

## Table roster, in row order. VAL (oracle) and CC-W (design-weighted complete case, code-only)
## are deliberately not reported. `disp` is the printed label, `src` the harness label.
ROSTER <- tribble(
  ~src,          ~disp,        ~bold,
  "CC",          "CC",          FALSE,
  "UNVAL",       "UNVAL",       FALSE,
  "NAIVE",       "NAIVE",       FALSE,
  "NAIVE+W",     "NAIVE+W",     FALSE,
  "WL",          "WLMI",        FALSE,     # harness calls it WL; the thesis prints WLMI
  "RAKING",      "RAKING",      FALSE,
  "MICE-PMM",    "MICE-PMM",    FALSE,
  "MICE-RF",     "MICE-RF",     FALSE,
  "MIXGB",       "MIXGB",       FALSE,
  "GAIN",        "GAIN",        FALSE,
  "TABCSDI",     "TabCSDI",     FALSE,
  "SICG",        "SICG",        TRUE,
  "SIRD",        "SIRD",        TRUE,
  "SISL-S",      "SISL-S",      TRUE,
  "SISL-B",      "SISL-B",      TRUE)
## SISL-M (model-based variance) is deliberately NOT in ROSTER: it is not a deployed method but
## a diagnostic that shows what the inverse-information variance does here. It is emitted only in
## the small method-section table below, alongside SISL-S and SISL-B, never in the benchmark or
## appendix tables.

## ---- gather -------------------------------------------------------------------------------
load(file.path(RES, "results_COMBINED.RData"))          # `combined`
dat <- combined %>% transmute(Method = as.character(Method), Coef = as.character(Coef),
                              ID, Estimate, SE, CI_lo, CI_hi)
if (file.exists(VC)) {
  e <- new.env(); load(VC, envir = e)
  lab <- c(dualsandwich = "SISL-S", dualboot = "SISL-B", modelbased = "SISL-M")
  add <- e$combined %>% mutate(Method = unname(lab[as.character(variant)])) %>%
    filter(!is.na(Method)) %>%
    transmute(Method, Coef = as.character(Coef), ID, Estimate, SE, CI_lo, CI_hi)
  dat <- bind_rows(dat, add)
} else warning("variance_COMBINED.RData not found: the SISL rows will print as --")

## ---- per-coefficient metrics --------------------------------------------------------------
per <- dat %>% filter(Coef %in% names(TRUTH)) %>%
  mutate(Truth = TRUTH[Coef]) %>%
  group_by(Method, Coef) %>%
  mutate(EmpSE = sd(Estimate), MeanRawBias = mean(Estimate - Truth)) %>%
  summarise(n_rep  = dplyr::n(),
            MAB    = abs(mean(Estimate) - first(Truth)) / abs(first(Truth)) * 100,
            RRMSE  = sqrt(mean((Estimate - first(Truth))^2)) / abs(first(Truth)) * 100,
            EmpSE  = first(EmpSE),
            MeanSE = mean(SE),
            NomCov = mean(CI_lo <= first(Truth) & first(Truth) <= CI_hi),
            OrcE   = mean(abs(Estimate - first(Truth)) <= Z * first(EmpSE)),
            OrcB   = mean(abs((Estimate - first(MeanRawBias)) - first(Truth)) <= Z * SE),
            .groups = "drop")

pool <- function(g_coefs) {
  per %>% filter(Coef %in% g_coefs) %>% group_by(Method) %>%
    summarise(n_rep  = min(n_rep),
              MAB    = exp(mean(log(MAB))),          # geometric
              RRMSE  = exp(mean(log(RRMSE))),        # geometric
              EmpSE  = mean(EmpSE),
              MeanSE = mean(MeanSE),
              # SE ratio = the printed Mean SE / printed Emp SE (both rounded to 3 dp), so a
              # reader can divide the two SE columns and recover it exactly.
              SEr    = round(mean(MeanSE), 3) / round(mean(EmpSE), 3),
              NomCov = mean(NomCov), OrcE = mean(OrcE), OrcB = mean(OrcB),
              .groups = "drop")
}

## ---- LaTeX helpers ------------------------------------------------------------------------
## Rank displayed values by proximity to `target`; equal displayed distances share a rank.
shade <- function(v, target, dig) {
  shown <- round(v, dig)
  d <- round(abs(shown - target), dig); d[is.na(shown)] <- Inf
  best <- head(sort(unique(d[is.finite(d)])), 3L)
  out <- rep("", length(v))
  if (!length(best)) return(out)
  rank_names <- c("rankone", "ranktwo", "rankthree")
  for (i in seq_along(d)) {
    r <- match(d[[i]], best)
    if (!is.na(r)) out[[i]] <- rank_names[[r]]
  }
  out
}
cell <- function(v, sh, dig) ifelse(is.na(v), "--",
  paste0(ifelse(nzchar(sh), paste0("\\cellcolor{", sh, "}"), ""), formatC(v, format = "f", digits = dig)))

block_rows <- function(P) {
  M <- ROSTER %>% left_join(P, by = c("src" = "Method"))
  sh <- list(MAB = shade(M$MAB, 0, 1), RRMSE = shade(M$RRMSE, 0, 1),
             SEr = shade(M$SEr, 1, 2), NomCov = shade(M$NomCov, .95, 2),
             OrcE = shade(M$OrcE, .95, 2), OrcB = shade(M$OrcB, .95, 2))
  nm <- ifelse(M$bold, paste0("\\textbf{", M$disp, "}"), M$disp)
  paste0(nm, " & ", cell(M$MAB, sh$MAB, 1), " & ", cell(M$RRMSE, sh$RRMSE, 1), " & ",
         cell(M$EmpSE, "", 3), " & ", cell(M$MeanSE, "", 3), " & ", cell(M$SEr, sh$SEr, 2), " & ",
         cell(M$NomCov, sh$NomCov, 2), " & ", cell(M$OrcE, sh$OrcE, 2), " & ",
         cell(M$OrcB, sh$OrcB, 2), " \\\\")
}

HEAD <- paste("Method & \\multicolumn{1}{c}{Mean $|$Bias$|$ (\\%)} & \\multicolumn{1}{c}{Rel.\\ RMSE (\\%)}",
  "& \\multicolumn{1}{c}{Emp.\\ SE} & \\multicolumn{1}{c}{Mean SE} & \\multicolumn{1}{c}{SE ratio}",
  "& \\multicolumn{1}{c}{Nom.\\ Cov.} & \\multicolumn{1}{c}{Orc.\\ Cov.\\ (Emp)}",
  "& \\multicolumn{1}{c}{Orc.\\ Cov.\\ (BC)} \\\\")
DEFC <- paste("\\definecolor{rankone}{RGB}{33,113,181}\\definecolor{ranktwo}{RGB}{107,174,214}",
              "\\definecolor{rankthree}{RGB}{198,219,239}")

## ---- tab:sisl_rank (Chapter 5: exposure only) ---------------------------------------------
P1 <- pool(GROUPS[[1]])
t1 <- c("\\begin{table}[!htbp]\\centering", DEFC,
        "\\caption{PASTE THE DEPLOYED CAPTION HERE.}", "\\label{tab:sisl_rank}",
        "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\adjustbox{max width=\\linewidth}{%", "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
        HEAD, "\\midrule", block_rows(P1), "\\bottomrule", "\\end{tabular}}", "\\end{table}")
writeLines(t1, file.path(OUT_DIR, "tab_sisl_rank.tex"))

## ---- tab:appc_sisl (Appendix C: all three groups) -----------------------------------------
body <- unlist(lapply(seq_along(GROUPS), function(i) {
  r <- c(paste0("\\multicolumn{9}{l}{\\textbf{", names(GROUPS)[i], "}} \\\\"),
         "\\addlinespace[2pt]", block_rows(pool(GROUPS[[i]])))
  if (i < length(GROUPS)) c(r, "\\addlinespace[3pt]") else r
}))
t2 <- c(DEFC, "\\begin{longtable}{lrrrrrrrr}",
        "\\caption{PASTE THE DEPLOYED CAPTION HERE.}", "\\label{tab:appc_sisl} \\\\", "\\toprule",
        HEAD, "\\midrule \\endfirsthead",
        "\\multicolumn{9}{l}{\\footnotesize\\itshape Table~\\ref{tab:appc_sisl} (continued)} \\\\ \\toprule",
        HEAD, "\\midrule \\endhead",
        "\\midrule \\multicolumn{9}{r}{\\footnotesize\\itshape continued on next page} \\\\",
        "\\endfoot", "\\bottomrule \\endlastfoot", body, "\\end{longtable}")
writeLines(t2, file.path(OUT_DIR, "tab_appc_sisl.tex"))

## ---- tab:sisl_variance (Method section: the three variance variants) -----------------------
## Diagnostic table for Section 5.3. It compares model-based, sandwich and bootstrap covariance
## at both fitted stages. Only the variance and coverage columns are shown.
PV <- pool(GROUPS[[1]])
vroster <- tribble(
  ~src,      ~disp,
  "SISL-M",  "SISL-M (model-based)",
  "SISL-S",  "SISL-S (sandwich)",
  "SISL-B",  "SISL-B (bootstrap)")
MV <- vroster %>% left_join(PV, by = c("src" = "Method"))
vrows <- paste0(MV$disp, " & ", cell(MV$EmpSE, "", 3), " & ", cell(MV$MeanSE, "", 3), " & ",
                cell(MV$SEr, "", 2), " & ", cell(MV$NomCov, "", 2), " & ",
                cell(MV$OrcE, "", 2), " \\\\")
tv <- c("\\begin{table}[!htbp]\\centering",
        "\\caption{PASTE THE DEPLOYED CAPTION HERE.}", "\\label{tab:sisl_variance}",
        "\\small", "\\begin{tabular}{lrrrrr}", "\\toprule",
        paste("Variance source & \\multicolumn{1}{c}{Emp.\\ SE} & \\multicolumn{1}{c}{Mean SE}",
              "& \\multicolumn{1}{c}{SE ratio} & \\multicolumn{1}{c}{Nom.\\ Cov.}",
              "& \\multicolumn{1}{c}{Orc.\\ Cov.\\ (Emp)} \\\\"),
        "\\midrule", vrows, "\\bottomrule", "\\end{tabular}", "\\end{table}")
writeLines(tv, file.path(OUT_DIR, "tab_sisl_variance.tex"))

## ---- console report -----------------------------------------------------------------------
cat("\n=== replicate counts per method (exposure) ===\n")
print(as.data.frame(per %>% filter(Coef == "X") %>% select(Method, n_rep) %>% arrange(Method)),
      row.names = FALSE)
miss <- setdiff(ROSTER$src, unique(per$Method))
if (length(miss)) cat("\nNOT ON DISK (emitted as -- rows):", paste(miss, collapse = ", "), "\n")
cat("\n=== tab:sisl_rank (exposure beta_X, truth =", truth$beta_X, ") ===\n")
print(as.data.frame(ROSTER %>% left_join(P1, by = c("src" = "Method")) %>%
        select(Method = disp, n_rep, MAB, RRMSE, EmpSE, MeanSE, SEr, NomCov, OrcE, OrcB) %>%
        mutate(across(where(is.numeric), ~round(.x, 3)))), row.names = FALSE)
cat("\n=== tab:sisl_variance (method-section diagnostic: SISL-M vs SISL-S vs SISL-B) ===\n")
print(as.data.frame(MV %>% select(Variant = disp, EmpSE, MeanSE, SEr, NomCov, OrcE) %>%
        mutate(across(where(is.numeric), ~round(.x, 3)))), row.names = FALSE)
cat("\nwrote:", file.path(OUT_DIR, "tab_sisl_rank.tex"),
    "\n      ", file.path(OUT_DIR, "tab_appc_sisl.tex"),
    "\n      ", file.path(OUT_DIR, "tab_sisl_variance.tex"), "\n")
