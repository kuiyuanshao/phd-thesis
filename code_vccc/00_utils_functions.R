# Walk up to find utils/core_functions.R: sourced from both dataset root and subfolders.
.core_path <- getwd()
while (!file.exists(file.path(.core_path, "utils", "core_functions.R")) && dirname(.core_path) != .core_path) {
  .core_path <- dirname(.core_path)
}
source(file.path(.core_path, "utils", "core_functions.R"), local = TRUE)
rm(.core_path)

data_info_vccc <- list(
  cat_vars = c(
    "SEX", "RACEETH", "RISK1", "RISK2", "DEATH_EVER",
    "GENERIC_ART_NAME_v", "GENERIC_ART_NAME_nv",
    "SOURCE_nv", "ART_SOURCE_nv",
    "DIAGNOSIS_v", "DIAGNOSIS_nv",
    "ANY_OI_v", "ANY_OI_nv"
  ),
  num_vars = c(
    "CFAR_PID", "AGE_AT_FIRST_VISIT", "AGE_AT_LAST_VISIT", "AGE_AT_DEATH",
    "YEAR_OF_ENROLLMENT", "YEAR_OF_LAST_VISIT", "YEAR_OF_DEATH", "YEARS_IN_STUDY",
    "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v",
    "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv",
    "AGE_AT_DX_ONSET_v", "YEAR_OF_DX_ONSET_v",
    "AGE_AT_DX_ONSET_nv", "YEAR_OF_DX_ONSET_nv",
    "CD4_COUNT_BSL_v", "CD4_COUNT_1Y_v", "VL_COUNT_BSL_v", "VL_COUNT_BSL_nv",
    "CD4_COUNT_BSL_sqrt_v", "CD4_COUNT_1Y_sqrt_v", "VL_COUNT_BSL_LOG_v", "VL_COUNT_BSL_LOG_nv"
  ),
  phase2_vars = c(
    "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v", "GENERIC_ART_NAME_v",
    "AGE_AT_DX_ONSET_v", "DIAGNOSIS_v", "YEAR_OF_DX_ONSET_v", "ANY_OI_v",
    "VL_COUNT_BSL_v",
    "VL_COUNT_BSL_LOG_v"
  ),
  phase1_vars = c(
    "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv", "GENERIC_ART_NAME_nv",
    "AGE_AT_DX_ONSET_nv", "DIAGNOSIS_nv", "YEAR_OF_DX_ONSET_nv", "ANY_OI_nv",
    "VL_COUNT_BSL_nv",
    "VL_COUNT_BSL_LOG_nv"
  )
)

## The sampled frames (data/Sample/<DESIGN>/*.csv) carry the phase-two indicator R and the
## design weight W, which the pre-sampling data_info_vccc above does not describe. W is a
## stratum label, so it is declared categorical here, matching the Python runners' data_info.
data_info_srs <- list(
  cat_vars = c(
    "SEX", "RACEETH", "RISK1", "RISK2", "DEATH_EVER",
    "GENERIC_ART_NAME_v", "GENERIC_ART_NAME_nv",
    "SOURCE_nv", "ART_SOURCE_nv",
    "DIAGNOSIS_v", "DIAGNOSIS_nv",
    "ANY_OI_v", "ANY_OI_nv",
    "W"
  ),
  num_vars = c(
    "CFAR_PID", "AGE_AT_FIRST_VISIT", "AGE_AT_LAST_VISIT", "AGE_AT_DEATH",
    "YEAR_OF_ENROLLMENT", "YEAR_OF_LAST_VISIT", "YEAR_OF_DEATH", "YEARS_IN_STUDY",
    "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v",
    "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv",
    "AGE_AT_DX_ONSET_v", "YEAR_OF_DX_ONSET_v",
    "AGE_AT_DX_ONSET_nv", "YEAR_OF_DX_ONSET_nv",
    "CD4_COUNT_BSL_v", "CD4_COUNT_1Y_v", "VL_COUNT_BSL_v", "VL_COUNT_BSL_nv",
    "CD4_COUNT_BSL_sqrt_v", "CD4_COUNT_1Y_sqrt_v", "VL_COUNT_BSL_LOG_v", "VL_COUNT_BSL_LOG_nv"
  ),
  phase2_vars = c(
    "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v", "GENERIC_ART_NAME_v",
    "AGE_AT_DX_ONSET_v", "DIAGNOSIS_v", "YEAR_OF_DX_ONSET_v", "ANY_OI_v",
    "VL_COUNT_BSL_v",
    "VL_COUNT_BSL_LOG_v"
  ),
  phase1_vars = c(
    "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv", "GENERIC_ART_NAME_nv",
    "AGE_AT_DX_ONSET_nv", "DIAGNOSIS_nv", "YEAR_OF_DX_ONSET_nv", "ANY_OI_nv",
    "VL_COUNT_BSL_nv",
    "VL_COUNT_BSL_LOG_nv"
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
