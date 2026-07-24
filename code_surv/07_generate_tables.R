## 07_generate_tables.R -- LaTeX result tables for Simulation Design III (survival / Cox).
## Run from D:/phd-thesis/code_surv. Emits to ./simulations/results/tables/:
##   tab_sicg_rank_surv.tex, tab_sird_rank_surv.tex             (Chapters 6 and 7)
##   tab_appc_surv_SampleE.tex, tab_appc_surv_SampleOE.tex      (Appendix C)
## Reads ./simulations/results/results_COMBINED.RData (six objects combined_result*_{SampleE,SampleOE}).
## Estimates are hazard ratios exp(coef); truth is the empirical mean of the VAL fit. cox = TRUE
## routes the SE-ratio empirical SE and the bias-corrected oracle coverage to the log scale.
## Shared engine: ../utils/table_functions.R.
suppressMessages(library(dplyr))
.core <- getwd()
while (!file.exists(file.path(.core, "utils", "table_functions.R")) && dirname(.core) != .core) .core <- dirname(.core)
source(file.path(.core, "utils", "table_functions.R"))

RES <- "./simulations/results"; OUT <- file.path(RES, "tables")
e <- new.env(); load(file.path(RES, "results_COMBINED.RData"), envir = e)

vars <- c("HbA1c", "rs4506565 1", "rs4506565 2", "Age", "eGFR", "Insulin", "BMI",
          "Sex TRUE", "Insurance TRUE", "Race AFR", "Race AMR", "Race SAS", "Race EAS",
          "Smoke 2", "Smoke 3", "Age:Insulin")
designs <- c("SRS", "BALANCE", "NEYMAN")
design_disp <- c(SRS = "SRS", BALANCE = "Balanced", NEYMAN = "Neyman-IF")

cells <- function(type) {
  co <- e[[paste0("combined_resultCoeff_", type)]]
  val <- co[normalise_method(co$Method) == "VAL", , drop = FALSE]        # truth = mean VAL fit
  truth <- setNames(vapply(vars, function(v) mean(finite_or_na(val[[v]]), na.rm = TRUE), numeric(1)), vars)
  summarise_wide(co, e[[paste0("combined_resultStdError_", type)]], e[[paste0("combined_resultCI_", type)]],
                 vars, truth, type, designs, unval_design = "UNVAL", cox = TRUE)
}
cells_E <- cells("SampleE"); cells_OE <- cells("SampleOE")

P2C <- "Continuous Phase-Two Covariates"; P2B <- "Binary \\& Categorical Phase-Two Covariates"
P1C <- "Continuous Phase-One Covariates"; P1B <- "Binary \\& Categorical Phase-One Covariates"
INT <- "Interaction"
rank_groups <- list(
  c("HbA1c" = P2C, "eGFR" = P2C, "Insulin" = P2C, "BMI" = P2C),
  c("rs4506565 1" = P2B, "rs4506565 2" = P2B, "Smoke 2" = P2B, "Smoke 3" = P2B),
  c("Age:Insulin" = INT))
appc_order <- c(P2C, P2B, P1C, P1B, INT)
appc_map <- list()
appc_map[[P2C]] <- c("HbA1c", "eGFR", "Insulin", "BMI")
appc_map[[P2B]] <- c("rs4506565 1", "rs4506565 2", "Smoke 2", "Smoke 3")
appc_map[[P1C]] <- "Age"
appc_map[[P1B]] <- c("Sex TRUE", "Insurance TRUE", "Race AFR", "Race AMR", "Race SAS", "Race EAS")
appc_map[[INT]] <- "Age:Insulin"

CAPX <- "PASTE THE DEPLOYED CAPTION HERE."
cat("Generating Simulation Design III tables\n")
allcells <- rbind(cells_E, cells_OE)     # rank pools the 6 type x design combinations
write_table(rank_table(allcells, rank_groups, "tab:sicg_rank_surv", CAPX, setdiff(methods_order, "SIRD")), OUT, "tab_sicg_rank_surv.tex")
write_table(rank_table(allcells, rank_groups, "tab:sird_rank_surv", CAPX, methods_order), OUT, "tab_sird_rank_surv.tex")
write_table(appc_table(cells_E, appc_map, appc_order, designs, design_disp, "tab:appc_surv_SampleE", CAPX), OUT, "tab_appc_surv_SampleE.tex")
write_table(appc_table(cells_OE, appc_map, appc_order, designs, design_disp, "tab:appc_surv_SampleOE", CAPX), OUT, "tab_appc_surv_SampleOE.tex")
