lapply(c("dplyr", "stringr"), require, character.only = T)

### demo: Demographics, lab: Laboratory results are error-free variables.
### non_v_art, non_v_ois: Tilde Variables, Error-Prone.
### v_art, v_ois: Validated Variables.
demo <- read.csv("./data/shepherd_demo_13feb2014.csv")
labs <- read.csv("./data/shepherd_labs_13feb2014.csv")
non_v_art <- read.csv("./data/shepherd_non_valid_art_13feb2014.csv")
non_v_ois <- read.csv("./data/shepherd_non_valid_ois_13feb2014.csv")
v_art <- read.csv("./data/shepherd_valid_art_13feb2014.csv")
v_ois <- read.csv("./data/shepherd_valid_ois_13feb2014.csv")
demo_spine <- demo %>%
  mutate(
    # Force CFAR_PID to character to prevent mismatched join errors later (as seen in Script 2)
    CFAR_PID = as.character(CFAR_PID),
    # Calculate overall study time using the exact variables from your demo dataset
    YEARS_IN_STUDY = YEAR_OF_LAST_VISIT - YEAR_OF_ENROLLMENT,
    # Create a binary mortality indicator (1 if deceased, 0 if alive) based on AGE_AT_DEATH
    DEATH_EVER = as.numeric(!is.na(AGE_AT_DEATH))
  )
# Process Antiretroviral Therapy (ART) Datasets
# Validated ART
art_v_exclude <- unique(v_art$CFAR_PID[is.na(v_art$AGE_AT_MED_START)])
v_art_first <- v_art %>%
  mutate(CFAR_PID = as.character(CFAR_PID)) %>%
  filter(!(CFAR_PID %in% art_v_exclude)) %>%      # Drop the entire patient if any NA exists
  arrange(CFAR_PID, AGE_AT_MED_START) %>%
  group_by(CFAR_PID) %>%
  slice(1) %>%
  ungroup() %>%
  rename_with(~paste0(., "_v"), -CFAR_PID)
# Non-Validated ART
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
  slice(1) %>%                                    # Keep only the index (first) infection
  ungroup() %>%
  rename_with(~paste0(., "_v"), -CFAR_PID)
# Non-Validated OIs
nv_ois_first <- non_v_ois %>%
  mutate(CFAR_PID = as.character(CFAR_PID)) %>%
  rename(OIS_SOURCE = SOURCE) %>% 
  filter(!is.na(AGE_AT_DX_ONSET)) %>%
  arrange(CFAR_PID, AGE_AT_DX_ONSET) %>%
  group_by(CFAR_PID) %>%
  slice(1) %>%
  ungroup() %>%
  rename_with(~paste0(., "_nv"), -CFAR_PID)
# Non-Validated OIs
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
  # Create simple binary indicators for whether they ever had an OI
  mutate(
    ANY_OI_v = ifelse(!is.na(AGE_AT_DX_ONSET_v), 1, 0),
    ANY_OI_nv = ifelse(!is.na(AGE_AT_DX_ONSET_nv), 1, 0)
  )
labs_cd4 <- labs %>% 
  filter(testName == 'CD4 COUNT')
labs_vl <- labs %>% 
  filter(testName != 'CD4 COUNT' & testName != 'CD4 PERCENT')
# Looking up a patient's labs in a list is near-instantaneous compared to filtering a massive dataframe.
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
    # Baseline variables: target and baseline are the same
    CD4_COUNT_BSL_v = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_v, AGE_AT_MED_START_v, cd4_list),
    # 1-Year variable: window is at +1, but distance is minimized against baseline (AGE_AT_MED_START_v)
    CD4_COUNT_1Y_v  = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_v + 1, AGE_AT_MED_START_v, cd4_list),
    VL_COUNT_BSL_v  = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_v, AGE_AT_MED_START_v, vl_list),
    VL_COUNT_BSL_nv = get_closest_lab_buggy(CFAR_PID, AGE_AT_MED_START_nv, AGE_AT_MED_START_nv, vl_list)
  ) %>%
  ungroup() %>%
  # Force the Complete Case Lab Filter
  filter(
    !is.na(CD4_COUNT_BSL_v),
    !is.na(CD4_COUNT_1Y_v),
    !is.na(VL_COUNT_BSL_v),
    !is.na(VL_COUNT_BSL_nv)
  )
# Perform transformations and verify the hard match
demo_final <- demo_mapped_strict %>%
  mutate(
    CD4_COUNT_BSL_sqrt_v = sqrt(CD4_COUNT_BSL_v),
    CD4_COUNT_1Y_sqrt_v  = sqrt(CD4_COUNT_1Y_v),
    VL_COUNT_BSL_LOG_v   = log(VL_COUNT_BSL_v),
    VL_COUNT_BSL_LOG_nv  = log(VL_COUNT_BSL_nv)
  )

# Check 1: Proportion of mismatched log viral loads
mismatch_rate <- mean(demo_final$VL_COUNT_BSL_LOG_v != demo_final$VL_COUNT_BSL_LOG_nv)
cat("Mismatch rate (VL_v != VL_nv):", mismatch_rate, "\n")
# Check 2: Correlation between True Baseline VL and True Baseline CD4
cor_bsl <- cor(demo_final$VL_COUNT_BSL_LOG_v, demo_final$CD4_COUNT_BSL_sqrt_v)
cat("Correlation (VL_v vs CD4_BSL_v):", cor_bsl, "\n")
# Check 3: Correlation between True Baseline VL and True 1-Year CD4
cor_1y <- cor(demo_final$VL_COUNT_BSL_LOG_v, demo_final$CD4_COUNT_1Y_sqrt_v)
cat("Correlation (VL_v vs CD4_1Y_v):", cor_1y, "\n")

demo_final$RISK2[demo_final$RISK2 == ""] <- "Not Specified"
demo_final <- demo_final %>%
  mutate(
    # Use -1 as a sentinel value. Deep learning models can easily learn that 
    # if DEATH_EVER == 0, the specific value of -1 is ignored.
    AGE_AT_DEATH = ifelse(is.na(AGE_AT_DEATH), -1, AGE_AT_DEATH),
    # Similarly, use 0 or -1. Since year is a large positive number, 
    # 0 is a clear out-of-bounds indicator.
    YEAR_OF_DEATH = ifelse(is.na(YEAR_OF_DEATH), 0, YEAR_OF_DEATH)
  )
categorize_diagnosis <- function(diag_string) {
  
  # ---------------------------------------------------------
  # PRE-PROCESSING: NEUTRALIZE LATENT/HISTORY KEYWORDS
  # ---------------------------------------------------------
  safe_string <- str_replace_all(diag_string, "(?i)H/O TUBERCULOSIS|HX OF TUBERCULOSIS", "HX_TB_RECORDED")
  safe_string <- str_replace_all(safe_string, "(?i)H/O PCP|HX OF PCP", "HX_PCP_RECORDED")
  
  case_when(
    # Handle NAs/Blanks first
    is.na(safe_string) | safe_string == "" ~ NA_character_,
    
    # ---------------------------------------------------------
    # TIER 1: DEFINITIVE AIDS-DEFINING EVENTS (ADEs)
    # All severe diseases evaluated FIRST.
    # ---------------------------------------------------------
    
    # 1. PCP
    str_detect(safe_string, "(?i)PNEUMOCYST|PNEUMOCYT|\\bPCP\\b|JIROVECI") ~ "PCP",
    
    # 2. Active Mycobacteria
    str_detect(safe_string, "(?i)\\bTUBERCULOSIS\\b|\\bPOTT\\b|\\bTUBERCLE\\b|\\bTUBERCULOUS") ~ "Active Tuberculosis (ADE)",
    str_detect(safe_string, "(?i)\\bMAC\\b|MYCOBACTERIUM AVIUM|\\bKANSASII\\b|\\bMYCOBACTERI|D/T MYCOBACTERIA") ~ "MAC/Atypical Mycobacteria",
    
    # 3. Fungal & Protozoal ADEs
    str_detect(safe_string, "(?i)\\bTOXO") ~ "Toxoplasmosis",
    str_detect(safe_string, "(?i)\\bCRYPTOCOCC") ~ "Cryptococcosis",
    str_detect(safe_string, "(?i)\\bHISTOPLASM") ~ "Histoplasmosis",
    str_detect(safe_string, "(?i)\\bCOCCIDIOID") ~ "Coccidioidomycosis",
    str_detect(safe_string, "(?i)\\bCRYPTOSPORIDI") ~ "Cryptosporidiosis",
    
    # 4. End-Organ CMV ADEs
    str_detect(safe_string, "(?i)\\bCMV\\b|CYTOMEGALO") & 
      str_detect(safe_string, "(?i)RETINITIS|RETINA|COLITIS|ESOPHAGITIS|ENCEPHALITIS|PNEUMONIA|ENTERITIS|DISEASE") ~ "CMV End-Organ Disease (ADE)",
    
    # 5. Other Viral ADEs
    str_detect(safe_string, "(?i)HERPES SIMPLEX") ~ "Herpes Simplex (Severe/ADE)",
    
    # 6. AIDS-Defining Malignancies
    str_detect(safe_string, "(?i)\\bKAPOSI") ~ "Kaposi Sarcoma",
    str_detect(safe_string, "(?i)LYMPHOMA") & 
      (!str_detect(safe_string, "(?i)\\bHODGKIN") | str_detect(safe_string, "(?i)NON[- ]?HODGKIN")) ~ "Lymphoma (AIDS-Defining)",
    
    # 7. Candidiasis (Esophageal/Pulmonary are ADEs)
    str_detect(safe_string, "(?i)CANDID.*(ESOPHAG|LUNG|TRACHEA|BRONCH)|(ESOPHAG|LUNG|TRACHEA|BRONCH).*CANDID|CANDIDIASIS OF THE ESOPHAGUS") ~ "Candidiasis (ADE: Esophageal/Pulmonary)",
    
    # 8. Wasting / Severe Neuro ADEs
    str_detect(safe_string, "(?i)WASTING|CACHEXIA") ~ "HIV Wasting Syndrome (ADE)",
    str_detect(safe_string, "(?i)DEMENTIA|ENCEPHALOPATHY|\\bPML\\b|MULTIFOCAL|LEUKOENCEPHALOPATHY") ~ "Neuro/PML/Dementia (ADE)",
    
    # 9. Bacterial ADEs
    str_detect(safe_string, "(?i)RECURR.*PNEUMONIA") ~ "Recurrent Bacterial Pneumonia (ADE)",
    str_detect(safe_string, "(?i)SALMONELLA.*(SEPTICEMIA|BACTEREMIA)") ~ "Salmonella Septicemia (ADE)",
    
    # ---------------------------------------------------------
    # TIER 2: NON-ADES, SYMPTOMS, AND GENERIC INFECTIONS
    # Evaluated LAST so they only trigger if no ADE was found.
    # ---------------------------------------------------------
    
    # Safely catches our neutralized latent strings from the pre-processing step
    str_detect(safe_string, "(?i)\\bLTBI\\b|\\bPPD\\b|HX_TB_RECORDED") ~ "Latent TB / History (Non-ADE)",
    
    str_detect(safe_string, "(?i)CYTOMEGALOVIRUS INFECTION|CYTOMEGALOVIRUS VIREMIA|\\bCMV\\b|CYTOMEGALO") ~ "CMV Viremia/NOS (Non-ADE)",
    str_detect(safe_string, "(?i)CANDIDIASIS OF MOUTH|ORAL CANDID") ~ "Oral Candidiasis (Non-ADE)",
    str_detect(safe_string, "(?i)WEIGHT LOSS|LOSS OF WEIGHT") ~ "Generic Weight Loss (Non-Specific)",
    str_detect(safe_string, "(?i)ALTERED MENTAL STATUS") ~ "Altered Mental Status (Non-Specific)",
    
    # Generic Bacterial Pneumonias
    str_detect(safe_string, "(?i)(BACTERIAL|LOBAR|PNEUMOCOCCAL|STAPHYLOCOCCUS).*PNEUMONIA|PNEUMONIA.*(BACTERIAL|LOBAR|PNEUMOCOCCAL|STAPHYLOCOCCUS)") ~ "Bacterial Pneumonia (Single/Unspecified)",
    
    # Generic Organ Systems
    str_detect(safe_string, "(?i)PNEUMONIA|PNEUMONITIS") ~ "Pneumonia (Other/NOS)",
    str_detect(safe_string, "(?i)MENINGITIS") ~ "Meningitis (Other/NOS)",
    str_detect(safe_string, "(?i)ESOPHAGITIS") ~ "Esophagitis (Other/NOS)",
    str_detect(safe_string, "(?i)COLITIS|GASTRITIS|SALMONELLA|ENTERITIS") ~ "Enteric/Gastrointestinal",
    str_detect(safe_string, "(?i)RETINITIS") ~ "Retinitis (NOS)",
    
    # Fallback Catch-all
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
  weight_var = "W",
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
    # Calculate True Survival Time (Phase 2 / Validated)
    # Time from ART Start to either OI Onset or Last Visit
    TIME_v = ifelse(ANY_OI_v == 1, 
                    AGE_AT_DX_ONSET_v - AGE_AT_MED_START_v, 
                    AGE_AT_LAST_VISIT - AGE_AT_MED_START_v),
    
    # Calculate Error-Prone Survival Time (Phase 1 / Non-Validated)
    TIME_nv = ifelse(ANY_OI_nv == 1, 
                     AGE_AT_DX_ONSET_nv - AGE_AT_MED_START_nv, 
                     AGE_AT_LAST_VISIT - AGE_AT_MED_START_nv)
  )

write.csv(data, file = "./data/data.csv")
#### 1. Gaussian Model (Linear Regression) ################################
fit_gauss_true <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_v + CD4_COUNT_BSL_sqrt_v + SEX +
                       AGE_AT_MED_START_v, 
                     data = data)

fit_gauss_err <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_nv + CD4_COUNT_BSL_sqrt_v + SEX +
                      AGE_AT_MED_START_nv, 
                    data = data)

# Extract full coefficients
coef_gauss_true <- coef(fit_gauss_true)
coef_gauss_err  <- coef(fit_gauss_err)
# Calculate and output Relative Bias
rel_bias_gauss <- (coef_gauss_err - coef_gauss_true) / coef_gauss_true
cat("\n--- Gaussian Model Relative Bias ---\n")
print(rel_bias_gauss)


#### 2. Binomial Model (Logistic Regression) ##############################
fit_bin_true <- glm(ANY_OI_v ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                    family = binomial, 
                    data = data)

# ERROR-PRONE (Non-Validated) Binomial Model
fit_bin_error <- glm(ANY_OI_nv ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, 
                     family = binomial, 
                     data = data)

# Extract full coefficients
coef_bin_true <- coef(fit_bin_true)
coef_bin_err  <- coef(fit_bin_error)

# Calculate and output Relative Bias
rel_bias_bin <- (coef_bin_err - coef_bin_true) / coef_bin_true
cat("\n--- Binomial Model Relative Bias ---\n")
print(rel_bias_bin)


#### 3. Cox Model (Proportional Hazards) ##################################
fit_cox_true <- coxph(Surv(TIME_v, ANY_OI_v) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                      data = data)
fit_cox_error <- coxph(Surv(TIME_nv, ANY_OI_nv) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, 
                       data = data)
# Extract full coefficients (Note: Cox models do not have an intercept)
coef_cox_true <- coef(fit_cox_true)
coef_cox_err  <- coef(fit_cox_error)

# Calculate and output Relative Bias
rel_bias_cox <- (exp(coef_cox_err) - exp(coef_cox_true)) / exp(coef_cox_true)
cat("\n--- Cox Model Relative Bias ---\n")
print(rel_bias_cox)

jump_distance <- sqrt(200) - sqrt(100) # approx 4.14
# Calculate the scaled Hazard Ratios
hr_true <- exp(coef(fit_cox_true)["CD4_COUNT_BSL_sqrt_v"] * jump_distance)
hr_error <- exp(coef(fit_cox_error)["CD4_COUNT_BSL_sqrt_v"] * jump_distance)
cat("True HR (100-cell jump):", hr_true, "\n")
cat("Error-Prone HR (100-cell jump):", hr_error, "\n")
