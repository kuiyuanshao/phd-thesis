## 07_generate_tables.R -- LaTeX result tables for Simulation Design II (nutrition).
## Run from D:/phd-thesis/code_nutri. Emits to ./simulations/results/tables/:
##   tab_sicg_rank_nutri.tex, tab_sird_rank_nutri.tex          (Chapters 6 and 7)
##   tab_appc_nutri_Gaussian.tex, tab_appc_nutri_Binomial.tex  (Appendix C)
## Reads ./simulations/results/results_COMBINED.RData (six objects combined_result*_{SBP,HTN}).
## Shared engine and metric/pooling definitions: ../utils/table_functions.R.
suppressMessages(library(dplyr))
.core <- getwd()
while (!file.exists(file.path(.core, "utils", "table_functions.R")) && dirname(.core) != .core) .core <- dirname(.core)
source(file.path(.core, "utils", "table_functions.R"))

RES <- "./simulations/results"; OUT <- file.path(RES, "tables")
e <- new.env(); load(file.path(RES, "results_COMBINED.RData"), envir = e)

vars <- c("Intercept", "ln_na_true", "c_age", "c_bmi", "high_chol",
          "usborn", "female", "bkg_pr", "bkg_o", "ln_na_true:c_age")
designs <- c("SRS", "RS", "WRS", "SFS", "ODS_TAIL", "NEYMAN_ODS", "NEYMAN_INF")
design_disp <- c(SRS = "SRS", RS = "RS", WRS = "WRS", SFS = "SFS", ODS_TAIL = "ODS-Tail",
                 NEYMAN_ODS = "Neyman-ODS", NEYMAN_INF = "Neyman-IF")
# DGP truths (05_generate_results.R), per outcome model
truth_sbp <- c(Intercept = 126.2498900, ln_na_true = 27.4725275, c_age = 0.4060319,
  c_bmi = 0.1703107, high_chol = 1.6108133, usborn = -0.5636760, female = -8.9318846,
  bkg_pr = -1.6697425, bkg_o = -2.3219322, "ln_na_true:c_age" = 0.15)
truth_htn <- c(Intercept = -1.91298009, ln_na_true = 1.43901944, c_age = 0.08715117,
  c_bmi = 0.05645171, high_chol = 0.37938242, usborn = -0.21958482, female = -0.11976800,
  bkg_pr = -0.17504836, bkg_o = -0.84484199, "ln_na_true:c_age" = 0.005)

cells <- function(combo) {
  tr <- if (combo == "SBP") truth_sbp else truth_htn
  summarise_wide(e[[paste0("combined_resultCoeff_", combo)]], e[[paste0("combined_resultStdError_", combo)]],
                 e[[paste0("combined_resultCI_", combo)]], vars, tr, combo, designs,
                 unval_design = "TRUE", cox = FALSE)
}
cells_sbp <- cells("SBP"); cells_htn <- cells("HTN")

# coefficient groups
P2C <- "Continuous Phase-Two Covariates"; P1C <- "Continuous Phase-One Covariates"
P1B <- "Binary \\& Categorical Phase-One Covariates"; INT <- "Interaction"
gmap <- c(setNames(rep(P2C, 1), "ln_na_true"),
          setNames(rep(P1C, 2), c("c_age", "c_bmi")),
          setNames(rep(P1B, 5), c("high_chol", "usborn", "female", "bkg_pr", "bkg_o")),
          setNames(rep(INT, 1), "ln_na_true:c_age"))
# rank table shows only the focal groups; each entry maps covariate -> group display name
rank_groups <- list(c("ln_na_true" = P2C), c("ln_na_true:c_age" = INT))
appc_order <- c(P2C, P1C, P1B, INT)
appc_map <- list(); appc_map[[P2C]] <- "ln_na_true"; appc_map[[P1C]] <- c("c_age", "c_bmi")
appc_map[[P1B]] <- c("high_chol", "usborn", "female", "bkg_pr", "bkg_o"); appc_map[[INT]] <- "ln_na_true:c_age"

CAPX <- "PASTE THE DEPLOYED CAPTION HERE."
cat("Generating Simulation Design II tables\n")
# Chapter 6 (SICG, no SIRD column) and Chapter 7 (SIRD included); rank pools the 14 combos.
allcells <- rbind(cells_sbp, cells_htn)
write_table(rank_table(allcells, rank_groups, "tab:sicg_rank_nutri", CAPX, setdiff(methods_order, "SIRD")), OUT, "tab_sicg_rank_nutri.tex")
write_table(rank_table(allcells, rank_groups, "tab:sird_rank_nutri", CAPX, methods_order), OUT, "tab_sird_rank_nutri.tex")
write_table(appc_table(cells_sbp, appc_map, appc_order, designs, design_disp, "tab:appc_nutri_Gaussian", CAPX), OUT, "tab_appc_nutri_Gaussian.tex")
write_table(appc_table(cells_htn, appc_map, appc_order, designs, design_disp, "tab:appc_nutri_Binomial", CAPX), OUT, "tab_appc_nutri_Binomial.tex")
