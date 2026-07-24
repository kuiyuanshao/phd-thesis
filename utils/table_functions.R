## Shared table engine for the thesis result tables (Chapters 6-7 and Appendix C).
## Sourced by code_nutri/07_generate_tables.R, code_surv/07_generate_tables.R and
## code_vccc/07_generate_tables.R. Holds only reusable functions and configuration; every
## harness-specific detail (paths, truth vectors, coefficient lists, group maps, designs)
## lives in the per-harness 07 script that sources this file.
##
## Two table shapes are produced:
##   RANK table  (tab:sicg_rank_*, tab:sird_rank_*): methods as columns, six metric rows
##     (Mean |Bias|, Rel. RMSE, SE ratio, and the three coverages) nested under coefficient-
##     group headers, pooled across the setting's model x design combinations.
##   APPENDIX table (tab:appc_*): methods as columns, eight metric rows (adds Emp. SE and
##     Mean SE) nested under group -> design, one design block per sampling design.
##
## Metric definitions (thesis-wide benchmark block):
##   Mean |Bias| (%) = |mean(Est) - truth| / |truth| * 100          [per cell]
##   Rel. RMSE (%)   = sqrt(mean((Est-truth)^2)) / |truth| * 100     [per cell]
##   Emp. SE         = sd of the estimate;  Mean SE = mean model SE
##   SE ratio        = Mean SE / Emp. SE, both on the SAME scale as the model SE. In the Cox
##                     settings the estimate is stored as the hazard ratio exp(coef) while the
##                     model SE is the SE of coef, so the empirical SE entering the ratio is the
##                     sd of log(estimate) = sd(coef). Gaussian/Binomial store coef and its SE
##                     on one scale, so there raw and log-scale coincide in the ratio's intent.
##   Nom. Cov.       = truth inside the reported CI
##   Orc. Cov. (Emp) = truth inside Est +/- z * (estimate-scale empirical SE)
##   Orc. Cov. (BC)  = truth inside (Est - mean bias) +/- z * model SE; on the log scale for Cox
## Pooling across the cells of a group: Mean |Bias| and Rel. RMSE by the log-scale geometric
## mean (a near-zero coefficient cannot dominate); Emp. SE, Mean SE and the coverages by the
## arithmetic mean; SE ratio as the ratio of the pooled Mean SE and Emp. SE (so it equals the
## two SE rows the appendix also prints, and is consistent between the rank and appendix tables).

z <- qnorm(0.975)

methods_order <- c("CC", "UNVAL", "MICE-PMM", "MICE-RF", "MIXGB",
                   "TABCSDI", "GAIN", "RAKING", "SICG", "SIRD")
display_method <- c(CC = "CC", UNVAL = "UNVAL", "MICE-PMM" = "MICE-PMM", "MICE-RF" = "MICE-RF",
                    MIXGB = "MIXGB", TABCSDI = "TabCSDI", GAIN = "GAIN", RAKING = "RAKING",
                    SICG = "\\textbf{SICG}", SIRD = "\\textbf{SIRD}")

## metric key -> printed label; the rank and appendix tables select from this in row order.
metric_labels <- c(
  MeanAbsBias          = "Mean $|$Bias$|$ (\\%)",
  Rel_RMSE             = "Rel.\\ RMSE (\\%)",
  Empirical_SE         = "Emp.\\ SE",
  Mean_SE              = "Mean SE",
  SE_ratio             = "SE ratio",
  Nominal_Coverage     = "Nom.\\ Cov.",
  Oracle_Coverage      = "Orc.\\ Cov.\\ (Emp)",
  Oracle_Coverage_Bias = "Orc.\\ Cov.\\ (BC)")
rank_metrics <- c("MeanAbsBias", "Rel_RMSE", "SE_ratio",
                  "Nominal_Coverage", "Oracle_Coverage", "Oracle_Coverage_Bias")
appc_metrics <- c("MeanAbsBias", "Rel_RMSE", "Empirical_SE", "Mean_SE", "SE_ratio",
                  "Nominal_Coverage", "Oracle_Coverage", "Oracle_Coverage_Bias")

coverage_metrics <- c("Nominal_Coverage", "Oracle_Coverage", "Oracle_Coverage_Bias")
ratio_metrics    <- c("SE_ratio")
unshaded_metrics <- c("Empirical_SE", "Mean_SE")   # printed but not ranked
metric_target <- function(m) if (m %in% coverage_metrics) 0.95 else if (m %in% ratio_metrics) 1 else 0
metric_digits <- function(m) if (m %in% c("MeanAbsBias", "Rel_RMSE")) 1 else if (m %in% c("Empirical_SE", "Mean_SE")) 3 else 2

DEFCOLORS <- paste0("\\definecolor{rankone}{RGB}{33,113,181}",
                    "\\definecolor{ranktwo}{RGB}{107,174,214}",
                    "\\definecolor{rankthree}{RGB}{198,219,239}")

finite_or_na <- function(x) { x <- suppressWarnings(as.numeric(x)); x[is.infinite(x)] <- NA_real_; x }
mean_na   <- function(x) { x <- finite_or_na(x); if (!any(!is.na(x))) NA_real_ else mean(x, na.rm = TRUE) }
median_na <- function(x) { x <- finite_or_na(x); if (!any(!is.na(x))) NA_real_ else median(x, na.rm = TRUE) }
FLOOR_PCT <- 0.05                                   # guards log(0) in the geometric mean
logmean_na <- function(x) { x <- finite_or_na(x); x <- x[!is.na(x)]
  if (!length(x)) NA_real_ else exp(mean(log(pmax(x, FLOOR_PCT)))) }

normalise_method <- function(x) { y <- toupper(as.character(x))
  y[y == "MICE"] <- "MICE-PMM"; y[y == "RF"] <- "MICE-RF"; y[y == "TAB CSDI"] <- "TABCSDI"; y }
ci_col <- function(cov, bound) paste0(cov, ".", bound)

## Per (design, method, covariate) metrics from the wide coeff/se/ci frames. `cox` = TRUE marks
## the Cox settings, where the estimate is a hazard ratio: the SE-ratio empirical SE and the
## bias-corrected oracle coverage are then taken on the log scale to match the stored model SE.
summarise_wide <- function(coeff, se, ci, vars, truth_map, combo_name, designs_keep,
                           unval_design = "UNVAL", cox = FALSE) {
  design <- as.character(coeff$Design); method <- normalise_method(coeff$Method)
  id <- as.character(coeff$ID)
  keep <- ((design %in% designs_keep) | (method == "UNVAL" & design == unval_design)) &
          method %in% methods_order & method != "VAL"
  out <- list(); k <- 0
  for (cov in vars) {
    dat <- data.frame(Design = design, Method = method, ID = id, Covariate = cov,
                      Estimate = finite_or_na(coeff[[cov]]), ModelSE = finite_or_na(se[[cov]]),
                      Lower = finite_or_na(ci[[ci_col(cov, "lower")]]),
                      Upper = finite_or_na(ci[[ci_col(cov, "upper")]]),
                      Truth = truth_map[[cov]], stringsAsFactors = FALSE)[keep, , drop = FALSE]
    if (!nrow(dat)) next
    for (key in unique(paste(dat$Design, dat$Method, sep = "\r"))) {
      dd <- dat[paste(dat$Design, dat$Method, sep = "\r") == key, , drop = FALSE]
      dd <- dd[!is.na(dd$Estimate) & !is.na(dd$Truth), , drop = FALSE]
      if (!nrow(dd)) next
      emp_se <- sd(dd$Estimate, na.rm = TRUE)                 # estimate scale (Oracle_Coverage)
      raw_bias <- dd$Estimate - dd$Truth; mean_bias <- mean(raw_bias, na.rm = TRUE)
      pos <- dd$Estimate > 0 & !is.na(dd$Estimate)
      # empirical SE on the model-SE's scale, for the SE ratio
      emp_se_ratio <- if (cox) { if (sum(pos) > 1) sd(log(dd$Estimate[pos]), na.rm = TRUE) else NA_real_ } else emp_se
      if (cox) {                                              # bias-corrected oracle on log scale
        okl <- pos & !is.na(dd$ModelSE)
        mlb <- mean(log(dd$Estimate[okl]), na.rm = TRUE) - log(dd$Truth[1])
        orc_bc <- mean(log(dd$Truth[okl]) >= (log(dd$Estimate[okl]) - mlb) - z * dd$ModelSE[okl] &
                       log(dd$Truth[okl]) <= (log(dd$Estimate[okl]) - mlb) + z * dd$ModelSE[okl], na.rm = TRUE)
      } else {
        okb <- !is.na(dd$ModelSE)
        orc_bc <- mean(dd$Truth[okb] >= (dd$Estimate[okb] - mean_bias) - z * dd$ModelSE[okb] &
                       dd$Truth[okb] <= (dd$Estimate[okb] - mean_bias) + z * dd$ModelSE[okb], na.rm = TRUE)
      }
      k <- k + 1
      out[[k]] <- data.frame(Combo = combo_name, Design = dd$Design[1], Method = dd$Method[1],
        Covariate = cov,
        Mean_Percent_Bias = mean(raw_bias / dd$Truth * 100, na.rm = TRUE),
        Rel_RMSE = sqrt(mean(raw_bias^2, na.rm = TRUE)) / abs(dd$Truth[1]) * 100,
        Empirical_SE = emp_se, Mean_SE = mean(dd$ModelSE, na.rm = TRUE),
        Empirical_SE_ratio = emp_se_ratio,
        Nominal_Coverage = mean(dd$Truth >= dd$Lower & dd$Truth <= dd$Upper, na.rm = TRUE),
        Oracle_Coverage = mean(dd$Truth >= dd$Estimate - z * emp_se &
                               dd$Truth <= dd$Estimate + z * emp_se, na.rm = TRUE),
        Oracle_Coverage_Bias = orc_bc, stringsAsFactors = FALSE)
    }
  }
  if (!length(out)) data.frame() else do.call(rbind, out)
}

## Pool a set of per-cell rows (already restricted to the cells of one table entry) to the eight
## displayed metrics.
pool_cells <- function(dd) c(
  MeanAbsBias          = logmean_na(abs(dd$Mean_Percent_Bias)),
  Rel_RMSE             = logmean_na(dd$Rel_RMSE),
  # displayed Emp. SE is on the model-SE's scale (log for Cox) so it stays consistent with Mean
  # SE and with the SE ratio; equals sd(estimate) in the Gaussian/Binomial settings.
  Empirical_SE         = mean_na(dd$Empirical_SE_ratio),
  Mean_SE              = mean_na(dd$Mean_SE),
  # SE ratio = pooled Mean SE / pooled Emp. SE, i.e. the ratio of the two SE rows the appendix
  # prints; the same rule is used in the rank tables for consistency.
  SE_ratio             = mean_na(dd$Mean_SE) / mean_na(dd$Empirical_SE_ratio),
  Nominal_Coverage     = mean_na(dd$Nominal_Coverage),
  Oracle_Coverage      = mean_na(dd$Oracle_Coverage),
  Oracle_Coverage_Bias = mean_na(dd$Oracle_Coverage_Bias))

## na_str is what an absent/NA cell prints: the rank tables use "--", the appendix tables leave
## a fully-absent method column blank.
## The SE ratio is displayed as the ratio of the Emp. SE and Mean SE *as printed* (rounded to
## their 3-decimal display precision), so a reader can divide the two SE columns and recover it
## exactly. Using the printed values matters for the VCCC case study, whose empirical SEs are so
## small that full-precision and 3-decimal division disagree.
readable_se_ratio <- function(pc) {
  e <- round(pc[["Empirical_SE"]], 3); m <- round(pc[["Mean_SE"]], 3)
  if (is.na(e) || is.na(m) || e == 0) NA_real_ else m / e
}
metric_value <- function(pc, metric) if (metric == "SE_ratio") readable_se_ratio(pc) else pc[[metric]]

format_value <- function(v, m, na_str = "--") if (is.na(v) || !is.finite(v)) na_str else formatC(v, format = "f", digits = metric_digits(m))

## Rank shading over the ten methods. Ranking uses the displayed precision, retains the three
## best distinct distances from the target, and assigns the same shade to displayed ties.
rank_classes <- function(values, metric) {
  if (metric %in% unshaded_metrics) return(setNames(rep("", length(values)), names(values)))
  tgt <- metric_target(metric)
  cls <- setNames(rep("", length(values)), names(values))
  digits <- metric_digits(metric)
  shown <- round(values, digits)
  score <- round(abs(shown - tgt), digits)
  best <- head(sort(unique(score[is.finite(score)])), 3L)
  if (!length(best)) return(cls)
  ranks <- c("\\cellcolor{rankone}", "\\cellcolor{ranktwo}", "\\cellcolor{rankthree}")
  for (i in seq_along(values)) {
    r <- match(score[[i]], best)
    if (!is.na(r)) cls[[i]] <- ranks[[r]]
  }
  cls
}

## one metric row: `lead` is the leading cell(s) (e.g. blank, or a design label in the appendix).
metric_row <- function(vals, metric, lead, na_str = "--") {
  cls <- rank_classes(vals, metric)
  cells <- vapply(names(vals), function(m) paste0(cls[[m]], format_value(vals[[m]], metric, na_str)), character(1))
  paste0(lead, metric_labels[[metric]], " & ", paste(cells, collapse = " & "), " \\\\")
}

## RANK table: pool each (group, method) across all cells of the group; six metric rows.
rank_table <- function(cells, groups, label, caption, methods = methods_order) {
  disp <- display_method[methods]
  hdr <- paste0(" & ", paste(paste0("\\multicolumn{1}{c}{", disp, "}"), collapse = " & "), " \\\\")
  lines <- c("\\begin{table}[!htbp]\\centering", DEFCOLORS,
             paste0("\\caption{", caption, "}"), paste0("\\label{", label, "}"),
             "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}", "\\adjustbox{max width=\\linewidth}{%",
             paste0("\\begin{tabular}{l", paste(rep("r", length(methods)), collapse = ""), "}"),
             "\\toprule", hdr, "\\midrule")
  for (i in seq_along(groups)) {
    g <- groups[[i]]; gc <- cells[cells$Covariate %in% names(g), , drop = FALSE]
    grp <- unname(g[gc$Covariate]); gname <- unique(unname(g))[1]
    if (i > 1) lines <- c(lines, "\\midrule")
    lines <- c(lines, paste0("\\multicolumn{", length(methods) + 1, "}{l}{\\textit{", gname, "}} \\\\[2pt]"))
    for (metric in rank_metrics) {
      vals <- setNames(vapply(methods, function(m) {
        dd <- gc[grp == gname & gc$Method == m, , drop = FALSE]
        if (!nrow(dd)) NA_real_ else metric_value(pool_cells(dd), metric)
      }, numeric(1)), methods)
      lines <- c(lines, metric_row(vals, metric, "\\quad "))
    }
  }
  c(lines, "\\bottomrule", "\\end{tabular}}", "\\end{table}")
}

## APPENDIX longtable: per (group, design) block, eight metric rows; landscape.
appc_table <- function(cells, gmap, gorder, designs, design_disp, label, caption) {
  disp <- display_method[methods_order]
  hdr <- paste0("Design & Metric & ", paste(paste0("\\multicolumn{1}{c}{", disp, "}"), collapse = " & "), " \\\\")
  ncol <- length(methods_order) + 2
  head <- c("\\begin{landscape}", DEFCOLORS, "\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
            paste0("\\begin{longtable}{ll", paste(rep("r", length(methods_order)), collapse = ""), "}"),
            paste0("\\caption{", caption, "}"), paste0("\\label{", label, "} \\\\"),
            "\\toprule", hdr, "\\midrule \\endfirsthead",
            paste0("\\multicolumn{", ncol, "}{l}{\\footnotesize\\itshape Table~\\ref{", label, "} (continued)} \\\\"),
            "\\toprule", hdr, "\\midrule \\endhead",
            paste0("\\midrule \\multicolumn{", ncol, "}{r}{\\footnotesize\\itshape continued on next page} \\\\"),
            "\\endfoot", "\\bottomrule \\endlastfoot")
  present <- unique(cells$Covariate); body <- character(0)
  for (group in gorder) {
    covs <- intersect(gmap[[group]], present); if (!length(covs)) next
    body <- c(body, paste0("\\multicolumn{", ncol, "}{l}{\\textbf{", group, "}} \\\\"), "\\addlinespace[2pt]")
    for (d in designs) {
      dd <- cells[toupper(cells$Design) == toupper(d) & cells$Covariate %in% covs, , drop = FALSE]
      un <- cells[cells$Method == "UNVAL" & cells$Covariate %in% covs, , drop = FALSE]  # UNVAL is design-free
      for (mi in seq_along(appc_metrics)) {
        metric <- appc_metrics[mi]
        vals <- setNames(vapply(methods_order, function(m) {
          src <- if (m == "UNVAL") un else dd[dd$Method == m, , drop = FALSE]
          if (!nrow(src)) NA_real_ else metric_value(pool_cells(src), metric)
        }, numeric(1)), methods_order)
        lead <- if (mi == 1) paste0("\\quad \\texttt{", design_disp[[d]], "} & ") else " & "
        body <- c(body, metric_row(vals, metric, lead, na_str = ""))
      }
      body <- c(body, "\\addlinespace[2pt]")
    }
  }
  c(head, body, "\\end{longtable}", "\\end{landscape}")
}

write_table <- function(lines, dir, file) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, file.path(dir, file)); cat("  wrote", file.path(dir, file), "\n")
}
