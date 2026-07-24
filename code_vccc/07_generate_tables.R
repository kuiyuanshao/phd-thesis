## 07_generate_tables.R -- LaTeX result tables for the VCCC case study.
## Run from D:/phd-thesis/code_vccc. Emits to ./simulations/results/tables/:
##   tab_sicg_rank_vccc.tex, tab_sird_rank_vccc.tex                          (Chapters 6 and 7)
##   tab_appc_vccc_LM.tex, tab_appc_vccc_LOGISTIC.tex, tab_appc_vccc_COX.tex (Appendix C)
## Unlike the synthetic harnesses, VCCC stores results in long form and its 05 step already writes
## per-cell metrics (with the naive-proxy bias mapped into the validated-variable slot). This
## script therefore reads ./simulations/results/Summary_Table.csv and pools it, rather than
## recomputing from the raw fits. Shared engine: ../utils/table_functions.R.
suppressMessages(library(dplyr))
.core <- getwd()
while (!file.exists(file.path(.core, "utils", "table_functions.R")) && dirname(.core) != .core) .core <- dirname(.core)
source(file.path(.core, "utils", "table_functions.R"))

RES <- "./simulations/results"; OUT <- file.path(RES, "tables")
designs <- c("SRS", "RS", "WRS", "SFS", "ODS_TAIL", "NEYMAN_ODS", "NEYMAN_INF")
design_disp <- c(SRS = "SRS", RS = "RS", WRS = "WRS", SFS = "SFS", ODS_TAIL = "ODS-Tail",
                 NEYMAN_ODS = "Neyman-ODS", NEYMAN_INF = "Neyman-IF")

raw <- read.csv(file.path(RES, "Summary_Table.csv"), check.names = FALSE)
raw$Method <- normalise_method(raw$Method)
# keep the reported methods and the real designs plus the design-free UNVAL rows
raw <- raw[raw$Method %in% methods_order &
           (toupper(raw$Design) %in% designs | raw$Method == "UNVAL"), , drop = FALSE]
# cell frame in the shape pool_cells expects; metrics are precomputed by 05, so Empirical_SE_ratio
# is the Empirical_SE 05 stored (SE ratio = Mean SE / Emp SE, as the appendix reports both).
cells_all <- data.frame(
  Combo = raw$Model, Design = raw$Design, Method = raw$Method, Covariate = raw$Covariate,
  Mean_Percent_Bias = raw$Mean_Percent_Bias, Rel_RMSE = raw$Rel_RMSE,
  Empirical_SE = raw$Empirical_SE, Mean_SE = raw$Mean_SE, Empirical_SE_ratio = raw$Empirical_SE,
  Nominal_Coverage = raw$Nominal_Coverage, Oracle_Coverage = raw$Oracle_Coverage_SE,
  Oracle_Coverage_Bias = raw$Oracle_Coverage_Bias, stringsAsFactors = FALSE)

# Cox scale fix: 05 stores the Cox estimate as a hazard ratio but its model SE as the SE of the
# log hazard ratio, so Summary_Table.csv's Empirical_SE (an sd of hazard ratios) is on the wrong
# scale for the SE ratio and the displayed Emp. SE. Recompute the Cox empirical SE on the log
# scale, sd(log HR) = sd(coef), from the raw long results, matching the survival design. The
# hazard-ratio-scale Emp. SE is kept only where the coverage rows use it (via 05, unchanged).
e <- new.env(); load(file.path(RES, "results_COMBINED.RData"), envir = e)
cx <- e$combined_long[e$combined_long$Model == "COX", , drop = FALSE]
cx$Method <- normalise_method(cx$Method)
cx$Covariate <- sub("_nv$", "_v", cx$Covariate)   # UNVAL proxy slot -> validated-variable name
cx <- cx[cx$Coeff > 0 & is.finite(cx$Coeff), , drop = FALSE]
logse <- aggregate(Coeff ~ Design + Method + Covariate, data = cx,
                   FUN = function(v) if (length(v) > 1) sd(log(v)) else NA_real_)
names(logse)[names(logse) == "Coeff"] <- "logSE"
key <- function(df) paste(df$Design, df$Method, df$Covariate, sep = "\r")
isc <- cells_all$Combo == "COX"
m <- match(key(cells_all[isc, ]), key(logse))
cells_all$Empirical_SE_ratio[isc] <- logse$logSE[m]

P2C <- "Continuous Phase-Two Covariates"; P1C <- "Continuous Phase-One Covariates"
P1B <- "Binary \\& Categorical Phase-One Covariates"
rank_groups <- list(c("CD4_COUNT_BSL_sqrt_v" = P2C, "VL_COUNT_BSL_LOG_v" = P2C, "AGE_AT_MED_START_v" = P2C))
appc_order <- c(P2C, P1C, P1B)
appc_map <- list()
appc_map[[P2C]] <- c("CD4_COUNT_BSL_sqrt_v", "VL_COUNT_BSL_LOG_v", "AGE_AT_MED_START_v")
appc_map[[P1C]] <- "YEAR_OF_ENROLLMENT"
appc_map[[P1B]] <- "SEX"

CAPX <- "PASTE THE DEPLOYED CAPTION HERE."
cat("Generating VCCC case-study tables\n")
# rank pools all three models x seven designs (the 21 combinations)
write_table(rank_table(cells_all, rank_groups, "tab:sicg_rank_vccc", CAPX, setdiff(methods_order, "SIRD")), OUT, "tab_sicg_rank_vccc.tex")
write_table(rank_table(cells_all, rank_groups, "tab:sird_rank_vccc", CAPX, methods_order), OUT, "tab_sird_rank_vccc.tex")
for (mdl in list(c("LM", "tab:appc_vccc_LM"), c("LOGISTIC", "tab:appc_vccc_LOGISTIC"), c("COX", "tab:appc_vccc_COX"))) {
  cm <- cells_all[cells_all$Combo == mdl[1], , drop = FALSE]
  write_table(appc_table(cm, appc_map, appc_order, designs, design_disp, mdl[2], CAPX), OUT,
              paste0(sub(":", "_", mdl[2]), ".tex"))
}
