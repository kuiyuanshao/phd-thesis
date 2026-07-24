# Walk up to find utils/core_functions.R (sourced from varying subfolders).
.core_path <- getwd()
while (!file.exists(file.path(.core_path, "utils", "core_functions.R")) && dirname(.core_path) != .core_path) {
  .core_path <- dirname(.core_path)
}
source(file.path(.core_path, "utils", "core_functions.R"), local = TRUE)
rm(.core_path)

data_info_srs <- list(
  cat_vars = c(
    "idx", "usborn", "high_chol", "female",
    "bkg_pr", "bkg_o", "hypertension", "R",
    "W"
  ),
  num_vars = c(
    "id", "c_age", "c_bmi", "sbp",
    "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg",
    "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true"
  ),
  phase2_vars = c(
    "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true"
  ),
  phase1_vars = c(
    "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg"
  )
)

DATA_INFOS <- list(
  srs = data_info_srs,
  rs = list(
    cat_vars = data_info_srs$cat_vars,
    num_vars = c(data_info_srs$num_vars, "rs"),
    phase2_vars = data_info_srs$phase2_vars,
    phase1_vars = data_info_srs$phase1_vars
  ),
  wrs = list(
    cat_vars = data_info_srs$cat_vars,
    num_vars = c(data_info_srs$num_vars, "wrs"),
    phase2_vars = data_info_srs$phase2_vars,
    phase1_vars = data_info_srs$phase1_vars
  ),
  sfs = list(
    cat_vars = data_info_srs$cat_vars,
    num_vars = c(data_info_srs$num_vars, "score"),
    phase2_vars = data_info_srs$phase2_vars,
    phase1_vars = data_info_srs$phase1_vars
  ),
  ods_tail = data_info_srs,
  neyman_ods = list(
    cat_vars = c(data_info_srs$cat_vars, "outcome_strata"),
    num_vars = data_info_srs$num_vars,
    phase2_vars = data_info_srs$phase2_vars,
    phase1_vars = data_info_srs$phase1_vars
  ),
  neyman_inf = list(
    cat_vars = c(data_info_srs$cat_vars, "outcome_strata"),
    num_vars = c(data_info_srs$num_vars, "inf"),
    phase2_vars = data_info_srs$phase2_vars,
    phase1_vars = data_info_srs$phase1_vars
  )
)
