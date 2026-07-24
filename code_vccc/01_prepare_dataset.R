lapply(c("dplyr", "stringr"), require, character.only = T)

demo <- read.csv("./data/shepherd_demo_13feb2014.csv")
labs <- read.csv("./data/shepherd_labs_13feb2014.csv")
non_v_art <- read.csv("./data/shepherd_non_valid_art_13feb2014.csv")
non_v_ois <- read.csv("./data/shepherd_non_valid_ois_13feb2014.csv")
v_art <- read.csv("./data/shepherd_valid_art_13feb2014.csv")
v_ois <- read.csv("./data/shepherd_valid_ois_13feb2014.csv")
demo_spine <- demo %>%
  mutate(
    CFAR_PID = as.character(CFAR_PID),
    YEARS_IN_STUDY = YEAR_OF_LAST_VISIT - YEAR_OF_ENROLLMENT,
    DEATH_EVER = as.numeric(!is.na(AGE_AT_DEATH))
  )
art_v_exclude <- unique(v_art$CFAR_PID[is.na(v_art$AGE_AT_MED_START)])
v_art_first <- v_art %>%
  mutate(CFAR_PID = as.character(CFAR_PID)) %>%
  filter(!(CFAR_PID %in% art_v_exclude)) %>%
  arrange(CFAR_PID, AGE_AT_MED_START) %>%
  group_by(CFAR_PID) %>%
  slice(1) %>%
  ungroup() %>%
  rename_with(~paste0(., "_v"), -CFAR_PID)
art_nv_exclude <- unique(non_v_art$CFAR_PID[is.na(non_v_art$AGE_AT_MED_START)])
nv_art_first <- non_v_art %>%
  mutate(CFAR_PID = as.character(CFAR_PID)) %>%
  rename(ART_SOURCE = SOURCE) %>% 
  filter(!(CFAR_PID %in% art_nv_exclude)) %>%
  arrange(CFAR_PID, AGE_AT_MED_START) %>%
  group_by(CFAR_PID) %>%
  slice(1) %>%
  ungroup() %>%
  rename_with(~paste0(., "_nv"), -CFAR_PID)

v_ois_first <- v_ois %>%
  mutate(CFAR_PID = as.character(CFAR_PID)) %>%
  filter(!is.na(AGE_AT_DX_ONSET)) %>%
  arrange(CFAR_PID, AGE_AT_DX_ONSET) %>%
  group_by(CFAR_PID) %>%
  slice(1) %>%
  ungroup() %>%
  rename_with(~paste0(., "_v"), -CFAR_PID)
nv_ois_first <- non_v_ois %>%
  mutate(CFAR_PID = as.character(CFAR_PID)) %>%
  rename(OIS_SOURCE = SOURCE) %>%
  filter(!is.na(AGE_AT_DX_ONSET)) %>%
  arrange(CFAR_PID, AGE_AT_DX_ONSET) %>%
  group_by(CFAR_PID) %>%
  slice(1) %>%
  ungroup() %>%
  rename_with(~paste0(., "_nv"), -CFAR_PID)
nv_ois_first <- non_v_ois %>%
  mutate(CFAR_PID = as.character(CFAR_PID)) %>%
  filter(!is.na(AGE_AT_DX_ONSET)) %>%
  arrange(CFAR_PID, AGE_AT_DX_ONSET) %>%
  group_by(CFAR_PID) %>%
  slice(1) %>%
  ungroup() %>%
  rename_with(~paste0(., "_nv"), -CFAR_PID)
demo_events <- demo_spine %>%
  left_join(v_art_first, by = "CFAR_PID") %>%
  left_join(nv_art_first, by = "CFAR_PID") %>%
  left_join(v_ois_first, by = "CFAR_PID") %>%
  left_join(nv_ois_first, by = "CFAR_PID") %>%
  mutate(
    ANY_OI_v = ifelse(!is.na(AGE_AT_DX_ONSET_v), 1, 0),
    ANY_OI_nv = ifelse(!is.na(AGE_AT_DX_ONSET_nv), 1, 0)
  )
labs_cd4 <- labs %>% 
  filter(testName == 'CD4 COUNT')
labs_vl <- labs %>% 
  filter(testName != 'CD4 COUNT' & testName != 'CD4 PERCENT')
cd4_list <- split(labs_cd4, labs_cd4$CFAR_PID)
vl_list <- split(labs_vl, labs_vl$CFAR_PID)
get_closest_lab_buggy <- function(pid, target_age, baseline_age, lab_list, window_lower = -180/365.25, window_upper = 30/365.25) {
  if (is.na(target_age)) return(NA_real_)
  pid_labs <- lab_list[[as.character(pid)]]
  if (is.null(pid_labs) || nrow(pid_labs) == 0) return(NA_real_)
  valid_labs <- pid_labs %>%
    filter(AGE_AT_RESULT_DATE >= (target_age + window_lower) & 
             AGE_AT_RESULT_DATE <= (target_age + window_upper))
  if (nrow(valid_labs) == 0) return(NA_real_)
  closest_idx <- which.min(abs(valid_labs$AGE_AT_RESULT_DATE - baseline_age))
  return(valid_labs$RESULT_NUMERIC[closest_idx])
}
demo_events_strict <- demo_events %>%
  filter(!is.na(AGE_AT_MED_START_v) & !is.na(AGE_AT_MED_START_nv))
demo_mapped_strict <- demo_events_strict %>%
  rowwise() %>%
  mutate(
    CD4_COUNT_BSL_v = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_v, AGE_AT_MED_START_v, cd4_list),
    CD4_COUNT_1Y_v  = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_v + 1, AGE_AT_MED_START_v, cd4_list),
    VL_COUNT_BSL_v  = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_v, AGE_AT_MED_START_v, vl_list),
    VL_COUNT_BSL_nv = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_nv, AGE_AT_MED_START_nv, vl_list)
  ) %>%
  ungroup() %>%
  filter(
    !is.na(CD4_COUNT_BSL_v),
    !is.na(CD4_COUNT_1Y_v),
    !is.na(VL_COUNT_BSL_v),
    !is.na(VL_COUNT_BSL_nv)
  )
demo_final <- demo_mapped_strict %>%
  mutate(
    CD4_COUNT_BSL_sqrt_v = sqrt(CD4_COUNT_BSL_v),
    CD4_COUNT_1Y_sqrt_v  = sqrt(CD4_COUNT_1Y_v),
    VL_COUNT_BSL_LOG_v   = log(VL_COUNT_BSL_v),
    VL_COUNT_BSL_LOG_nv  = log(VL_COUNT_BSL_nv)
  )

mismatch_rate <- mean(demo_final$VL_COUNT_BSL_LOG_v != demo_final$VL_COUNT_BSL_LOG_nv)
cat("Mismatch rate (VL_v != VL_nv):", mismatch_rate, "\n")
cor_bsl <- cor(demo_final$VL_COUNT_BSL_LOG_v, demo_final$CD4_COUNT_BSL_sqrt_v)
cat("Correlation (VL_v vs CD4_BSL_v):", cor_bsl, "\n")
cor_1y <- cor(demo_final$VL_COUNT_BSL_LOG_v, demo_final$CD4_COUNT_1Y_sqrt_v)
cat("Correlation (VL_v vs CD4_1Y_v):", cor_1y, "\n")

demo_final$RISK2[demo_final$RISK2 == ""] <- "Not Specified"
demo_final <- demo_final %>%
  mutate(
    # Sentinel for missing: -1 (age) and 0 (year) are out-of-range flags
    AGE_AT_DEATH = ifelse(is.na(AGE_AT_DEATH), -1, AGE_AT_DEATH),
    YEAR_OF_DEATH = ifelse(is.na(YEAR_OF_DEATH), 0, YEAR_OF_DEATH)
  )
categorize_diagnosis <- function(diag_string) {
  
  safe_string <- str_replace_all(diag_string, "(?i)H/O TUBERCULOSIS|HX OF TUBERCULOSIS", "HX_TB_RECORDED")
  safe_string <- str_replace_all(safe_string, "(?i)H/O PCP|HX OF PCP", "HX_PCP_RECORDED")
  
  # Order matters: AIDS-defining events are matched before generic non-ADE fallbacks
  case_when(
    is.na(safe_string) | safe_string == "" ~ NA_character_,
    
    
    str_detect(safe_string, "(?i)PNEUMOCYST|PNEUMOCYT|\\bPCP\\b|JIROVECI") ~ "PCP",
    
    str_detect(safe_string, "(?i)\\bTUBERCULOSIS\\b|\\bPOTT\\b|\\bTUBERCLE\\b|\\bTUBERCULOUS") ~ "Active Tuberculosis (ADE)",
    str_detect(safe_string, "(?i)\\bMAC\\b|MYCOBACTERIUM AVIUM|\\bKANSASII\\b|\\bMYCOBACTERI|D/T MYCOBACTERIA") ~ "MAC/Atypical Mycobacteria",
    
    str_detect(safe_string, "(?i)\\bTOXO") ~ "Toxoplasmosis",
    str_detect(safe_string, "(?i)\\bCRYPTOCOCC") ~ "Cryptococcosis",
    str_detect(safe_string, "(?i)\\bHISTOPLASM") ~ "Histoplasmosis",
    str_detect(safe_string, "(?i)\\bCOCCIDIOID") ~ "Coccidioidomycosis",
    str_detect(safe_string, "(?i)\\bCRYPTOSPORIDI") ~ "Cryptosporidiosis",
    
    str_detect(safe_string, "(?i)\\bCMV\\b|CYTOMEGALO") & 
      str_detect(safe_string, "(?i)RETINITIS|RETINA|COLITIS|ESOPHAGITIS|ENCEPHALITIS|PNEUMONIA|ENTERITIS|DISEASE") ~ "CMV End-Organ Disease (ADE)",
    
    str_detect(safe_string, "(?i)HERPES SIMPLEX") ~ "Herpes Simplex (Severe/ADE)",
    
    str_detect(safe_string, "(?i)\\bKAPOSI") ~ "Kaposi Sarcoma",
    str_detect(safe_string, "(?i)LYMPHOMA") & 
      (!str_detect(safe_string, "(?i)\\bHODGKIN") | str_detect(safe_string, "(?i)NON[- ]?HODGKIN")) ~ "Lymphoma (AIDS-Defining)",
    
    str_detect(safe_string, "(?i)CANDID.*(ESOPHAG|LUNG|TRACHEA|BRONCH)|(ESOPHAG|LUNG|TRACHEA|BRONCH).*CANDID|CANDIDIASIS OF THE ESOPHAGUS") ~ "Candidiasis (ADE: Esophageal/Pulmonary)",
    
    str_detect(safe_string, "(?i)WASTING|CACHEXIA") ~ "HIV Wasting Syndrome (ADE)",
    str_detect(safe_string, "(?i)DEMENTIA|ENCEPHALOPATHY|\\bPML\\b|MULTIFOCAL|LEUKOENCEPHALOPATHY") ~ "Neuro/PML/Dementia (ADE)",
    
    str_detect(safe_string, "(?i)RECURR.*PNEUMONIA") ~ "Recurrent Bacterial Pneumonia (ADE)",
    str_detect(safe_string, "(?i)SALMONELLA.*(SEPTICEMIA|BACTEREMIA)") ~ "Salmonella Septicemia (ADE)",
    
    
    str_detect(safe_string, "(?i)\\bLTBI\\b|\\bPPD\\b|HX_TB_RECORDED") ~ "Latent TB / History (Non-ADE)",
    
    str_detect(safe_string, "(?i)CYTOMEGALOVIRUS INFECTION|CYTOMEGALOVIRUS VIREMIA|\\bCMV\\b|CYTOMEGALO") ~ "CMV Viremia/NOS (Non-ADE)",
    str_detect(safe_string, "(?i)CANDIDIASIS OF MOUTH|ORAL CANDID") ~ "Oral Candidiasis (Non-ADE)",
    str_detect(safe_string, "(?i)WEIGHT LOSS|LOSS OF WEIGHT") ~ "Generic Weight Loss (Non-Specific)",
    str_detect(safe_string, "(?i)ALTERED MENTAL STATUS") ~ "Altered Mental Status (Non-Specific)",
    
    str_detect(safe_string, "(?i)(BACTERIAL|LOBAR|PNEUMOCOCCAL|STAPHYLOCOCCUS).*PNEUMONIA|PNEUMONIA.*(BACTERIAL|LOBAR|PNEUMOCOCCAL|STAPHYLOCOCCUS)") ~ "Bacterial Pneumonia (Single/Unspecified)",
    
    str_detect(safe_string, "(?i)PNEUMONIA|PNEUMONITIS") ~ "Pneumonia (Other/NOS)",
    str_detect(safe_string, "(?i)MENINGITIS") ~ "Meningitis (Other/NOS)",
    str_detect(safe_string, "(?i)ESOPHAGITIS") ~ "Esophagitis (Other/NOS)",
    str_detect(safe_string, "(?i)COLITIS|GASTRITIS|SALMONELLA|ENTERITIS") ~ "Enteric/Gastrointestinal",
    str_detect(safe_string, "(?i)RETINITIS") ~ "Retinitis (NOS)",
    
    TRUE ~ "Other"
  )
}

demo_grouped <- demo_final %>%
  mutate(
    DIAGNOSIS_v = categorize_diagnosis(DIAGNOSIS_v),
    DIAGNOSIS_nv = categorize_diagnosis(DIAGNOSIS_nv)
  ) %>%
  mutate(
    DIAGNOSIS_v = ifelse(is.na(DIAGNOSIS_v), "Not Specified", DIAGNOSIS_v),
    DIAGNOSIS_nv = ifelse(is.na(DIAGNOSIS_nv), "Not Specified", DIAGNOSIS_nv),
    SOURCE_nv = ifelse(is.na(SOURCE_nv), "Not Specified", SOURCE_nv)
  )

demo_grouped <- demo_grouped %>%
  mutate(
    AGE_AT_MED_STOP_v  = coalesce(AGE_AT_MED_STOP_v, AGE_AT_LAST_VISIT),
    AGE_AT_MED_STOP_nv = coalesce(AGE_AT_MED_STOP_nv, AGE_AT_LAST_VISIT),
    YEAR_OF_RX_STOP_v  = coalesce(YEAR_OF_RX_STOP_v, YEAR_OF_LAST_VISIT),
    YEAR_OF_RX_STOP_nv = coalesce(YEAR_OF_RX_STOP_nv, YEAR_OF_LAST_VISIT),
    AGE_AT_DX_ONSET_v = ifelse(is.na(AGE_AT_DX_ONSET_v), -1, AGE_AT_DX_ONSET_v),
    YEAR_OF_DX_ONSET_v = ifelse(is.na(YEAR_OF_DX_ONSET_v), 0, YEAR_OF_DX_ONSET_v),
    AGE_AT_DX_ONSET_nv = ifelse(is.na(AGE_AT_DX_ONSET_nv), -1, AGE_AT_DX_ONSET_nv),
    YEAR_OF_DX_ONSET_nv = ifelse(is.na(YEAR_OF_DX_ONSET_nv), 0, YEAR_OF_DX_ONSET_nv)
  ) %>%
  select(
    -YEAR_OF_RX_STOP_v,
    -YEAR_OF_RX_STOP_nv,
    -YEAR_OF_RX_START_v,
    -YEAR_OF_RX_START_nv,
    -AGE_AT_DX_RESOLVED_nv,
    -YEAR_OF_DX_RESOLVED_nv
  )

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

library(survival)
data <- demo_grouped %>%
  mutate(
    TIME_v = ifelse(ANY_OI_v == 1, 
                    AGE_AT_DX_ONSET_v - AGE_AT_MED_START_v, 
                    AGE_AT_LAST_VISIT - AGE_AT_MED_START_v),
    
    TIME_nv = ifelse(ANY_OI_nv == 1, 
                     AGE_AT_DX_ONSET_nv - AGE_AT_MED_START_nv, 
                     AGE_AT_LAST_VISIT - AGE_AT_MED_START_nv)
  )

write.csv(data, file = "./data/data.csv")
fit_gauss_true <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_v + CD4_COUNT_BSL_sqrt_v + SEX +
                       AGE_AT_MED_START_v, 
                     data = data)

fit_gauss_err <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_nv + CD4_COUNT_BSL_sqrt_v + SEX +
                      AGE_AT_MED_START_nv, 
                    data = data)

coef_gauss_true <- coef(fit_gauss_true)
coef_gauss_err  <- coef(fit_gauss_err)
rel_bias_gauss <- (coef_gauss_err - coef_gauss_true) / coef_gauss_true
cat("\n--- Gaussian Model Relative Bias ---\n")
print(rel_bias_gauss)


fit_bin_true <- glm(ANY_OI_v ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                    family = binomial, 
                    data = data)

fit_bin_error <- glm(ANY_OI_nv ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, 
                     family = binomial, 
                     data = data)

coef_bin_true <- coef(fit_bin_true)
coef_bin_err  <- coef(fit_bin_error)

rel_bias_bin <- (coef_bin_err - coef_bin_true) / coef_bin_true
cat("\n--- Binomial Model Relative Bias ---\n")
print(rel_bias_bin)


fit_cox_true <- coxph(Surv(TIME_v, ANY_OI_v) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                      data = data)
fit_cox_error <- coxph(Surv(TIME_nv, ANY_OI_nv) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, 
                       data = data)
coef_cox_true <- coef(fit_cox_true)
coef_cox_err  <- coef(fit_cox_error)

rel_bias_cox <- (exp(coef_cox_err) - exp(coef_cox_true)) / exp(coef_cox_true)
cat("\n--- Cox Model Relative Bias ---\n")
print(rel_bias_cox)

jump_distance <- sqrt(200) - sqrt(100)
hr_true <- exp(coef(fit_cox_true)["CD4_COUNT_BSL_sqrt_v"] * jump_distance)
hr_error <- exp(coef(fit_cox_error)["CD4_COUNT_BSL_sqrt_v"] * jump_distance)
cat("True HR (100-cell jump):", hr_true, "\n")
cat("Error-Prone HR (100-cell jump):", hr_error, "\n")
