#################################### Generate Population ####################################
# T2D across the whole population
lapply(c("LDlinkR", "dplyr", "purrr", "mvtnorm", "stringr", "tidyr", "haven"), require, character.only = T)
source("00_utils_functions.R")
generateData <- function(n, seed){
  set.seed(seed)
  n <- n
  data <- data.frame(ID = as.character(1:n))
  # AGE: 18 ~ 90
  data$AGE <- pmin(90, pmax(18, rnorm(n, 50, 15)))
  # SEX: 0 For 45% Female, 1 For 55% Male
  data$SEX <- as.logical(rbinom(n, 1, 0.55))
  # RACE: 50% EUR, 20% AFR, 10% EAS, 10% SAS, 10% AMR
  data$RACE <- sample(c("EUR", "AFR", "AMR", "SAS", "EAS"), 
                      size = n, replace = T, c(0.5, 0.2, 0.1, 0.1, 0.1)) 
  data$RACE <- factor(data$RACE, levels = c("EUR", "AFR", "AMR", "SAS", "EAS"))
  # INSURANCE: 70% has insurance
  data$INSURANCE <- as.logical(rbinom(n, 1, prob = 0.7))
  # REGION & URBAN:
  getRandProp <- function(i){
    set.seed(1)
    v <- abs(rnorm(i))
    v <- v / sum(v)
    return (v)
  }
  data$REGION <- sample(as.character(1:20), size = n,
                        replace = T, prob = getRandProp(20))
  data$URBAN <- data$REGION %in% as.character(1:15)
  # HOUSEHOLD INCOME:
  data$INCOME <- sample(as.character(1:5), size = n,
                        replace = T, prob = getRandProp(5))
  # EDUCATION YEARS:
  data$EDU <- rpois(n, 4) + 6
  # 15 Genotypes
  genoInfo <- loadGenotypeInfo()
  data <- simGenotypes(data, genoInfo)
  # Load Covariates Info
  covarInfo <- loadCovarInfo(genoInfo)
  # HEIGHT: 149 ~ 
  data$HEIGHT <- pmax(149, simCovs(data, covarInfo$formu$HEIGHT, covarInfo$betas$HEIGHT, covarInfo$type$HEIGHT, covarInfo$sigma$HEIGHT))
  # SMOKE: 
  data$SMOKE <- simCovs(data, covarInfo$formu$SMOKE, covarInfo$betas$SMOKE, covarInfo$type$SMOKE)
  # EXER:
  data$EXER <- simCovs(data, covarInfo$formu$EXER, covarInfo$betas$EXER, covarInfo$type$EXER)
  # ALC:
  data$ALC <- simCovs(data, covarInfo$formu$ALC, covarInfo$betas$ALC, covarInfo$type$ALC)
  # BMI: 10 ~ 
  data$BMI <- simCovs(data, covarInfo$formu$BMI, covarInfo$betas$BMI, covarInfo$type$BMI, covarInfo$sigma$BMI)
  # MARRIAGE:
  data$MARRIAGE <- simCovs(data, covarInfo$formu$MARRIAGE, covarInfo$betas$MARRIAGE, covarInfo$type$MARRIAGE)
  # Medication Count:
  data$MED_Count <- simCovs(data, covarInfo$formu$MED_Count, covarInfo$betas$MED_Count, covarInfo$type$MED_Count)
  # RENAL GROUP: 
  data <- cbind(data, simCovs(data, covarInfo$formu$RENAL, covarInfo$betas$RENAL, covarInfo$type$RENAL, covarInfo$sigma$RENAL))
  # LIPIDS: LDL is in inv-norm scale
  data <- cbind(data, simCovs(data, covarInfo$formu$LIPIDS, covarInfo$betas$LIPIDS, covarInfo$type$LIPIDS, covarInfo$sigma$LIPIDS))
  # HEMA: HCT is in inv-norm scale
  data <- cbind(data, simCovs(data, covarInfo$formu$HEMA, covarInfo$betas$HEMA, covarInfo$type$HEMA, covarInfo$sigma$HEMA))
  # NUTRIENTS:
  data <- cbind(data, simCovs(data, covarInfo$formu$NUTRIENTS, covarInfo$betas$NUTRIENTS, covarInfo$type$NUTRIENTS, covarInfo$sigma$NUTRIENTS))
  # LIVER:
  data <- cbind(data, simCovs(data, covarInfo$formu$LIVER, covarInfo$betas$LIVER, covarInfo$type$LIVER, covarInfo$sigma$LIVER))
  # DIABETE MEASUREMENTS:
  data <- cbind(data, simCovs(data, covarInfo$formu$DIABETE, 
                              covarInfo$betas$DIABETE, covarInfo$type$DIABETE, covarInfo$sigma$DIABETE))
  # Ferritin:
  data$Ferritin <- simCovs(data, covarInfo$formu$FERRITIN, 
                           covarInfo$betas$FERRITIN, covarInfo$type$FERRITIN, covarInfo$sigma$FERRITIN)
  # VITAL SIGNS:
  data <- cbind(data, simCovs(data, covarInfo$formu$VITALS, covarInfo$betas$VITALS, covarInfo$type$VITALS, covarInfo$sigma$VITALS))
  
  for (i in c(29, (32:76)[!(32:76) %in% (56:59)])){
    data[, i] <- exp(data[, i]) - 1
  }
  # HYPENTERSION
  data$SBP <- pmin(pmax(data$SBP, 90), 200)
  data$HYPERTENSION <- with(data, SBP >= 140)
  data$Glucose <- data$Glucose + 10
  data$F_Glucose <- data$F_Glucose + 6
  data$HbA1c <- data$HbA1c + 40
  # WEIGHT:
  data$WEIGHT <- round(data$BMI * (data$HEIGHT / 100)^2, 3)
  
  age_penalty <- 0.5 * (data$AGE - 40) * (data$AGE > 40) # Extra creatinine for age > 40
  data$Creatinine <- data$Creatinine + 20 + age_penalty
  
  calculate_eGFR <- function(creat, age, sex) {
    scr <- creat / 88.4 # convert to mg/dL
    k <- ifelse(sex, 0.9, 0.7) 
    alpha <- ifelse(sex, -0.411, -0.329)
    factor_sex <- ifelse(sex, 1, 1.012) 
    egfr <- 142 * (pmin(scr / k, 1) ^ alpha) * (pmax(scr / k, 1) ^ -1.200) * (0.9938 ^ age) * factor_sex
    egfr <- pmin(round(egfr, 0), 140)
    return(egfr)
  }
  data$eGFR <- calculate_eGFR(data$Creatinine, data$AGE, data$SEX)
  
  # Adding Measurement Errors:
  # GENOTYPES:
  GENO_M <- matrix(
    c(0.99, 0.005, 0.005,
      0.005, 0.99, 0.005,
      0.005, 0.005, 0.99),
    nrow = 3, byrow = TRUE,
    dimnames = list(0:2, 
                    0:2)
  )
  for (name in genoInfo$rsids$refsnp_id){
    data[[paste0(name, "_STAR")]] <- sapply(data[[name]], 
                                            function(true_val) {sample(as.character(0:2), 
                                                                       size = 1, 
                                                                       prob = GENO_M[true_val, ])})
  }
  
  # T_I: Self-Reported Time Interval between Treatment Initiation SGLT2 and T2D Diagnosis (Months)
  # 1. SCALING 
  # ----------------------------------------------------------------------------
  A1c_sc <- (data$HbA1c - 50) / 15 
  Age_sc <- (data$AGE - 60) / 15
  eGFR_sc <- (data$eGFR - 60) / 20
  BMI_sc <- (data$BMI - 30) / 5
  geno_eff <- as.numeric(data$rs4506565) 
  
  # 2. PROPENSITY FOR ACUTE SWITCH
  # Boosted coefficients to ensure clean separation and significance
  prob_acute <- plogis(-5.5 +                
                         0.6 * A1c_sc +        
                         0.3 * BMI_sc +        
                         0.3 * geno_eff +      
                         0.6 * as.numeric(data$INSURANCE) +
                         # RACE Effect: Asians/Blacks might have higher acute risk
                         0.4 * as.numeric(data$RACE %in% c("AFR", "SAS")))
  
  is_acute <- (runif(n) < prob_acute)
  
  # 3. LINEAR PREDICTOR (Hazard Strength)
  
  # RACE Effect Vector
  # EUR=0, AFR=0.3, AMR=0.1, SAS=0.4, EAS=0.2
  race_eff <- case_when(
    data$RACE == "AFR" ~ 0.4,
    data$RACE == "SAS" ~ 0.5,
    data$RACE == "EAS" ~ 0.3,
    data$RACE == "AMR" ~ 0.2,
    TRUE ~ 0.0
  )
  
  eta_I <- (0.2 * A1c_sc + 0.6 * A1c_sc^2) +      # HbA1c: Quadratic
    (0.5 * eGFR_sc) +                      # eGFR: Linear
    (0.4 * BMI_sc) +                       # BMI: Linear
    (-0.4 * A1c_sc * Age_sc) +             # Interaction
    (0.5 * geno_eff) +                     # Genotype
    (0.6 * as.numeric(data$SEX)) +         # Sex
    (0.7 * as.numeric(data$SMOKE == 1)) +  # SMOKE (Current=1): Strong effect
    race_eff                               # Race Effect
  
  # 4. TIME GENERATION
  T_acute <- (12 * exp(-0.3 * eta_I)) * (-log(runif(n)))^(1/1.2)
  
  T_routine <- rlnorm(n, meanlog = log(26) - 0.2 * eta_I, sdlog = 0.5)
  
  T_I <- ifelse(is_acute, T_acute, T_routine)
  
  # 5. CENSORING & TRUE OBSERVED
  # Weaker censoring to ensure we have enough events for significance
  c_rate <- 0.02 * exp(-0.01 * (data$AGE - 60) + 0.05 * as.numeric(data$URBAN))
  C <- pmin(rexp(n, c_rate), 48) 
  
  # TRUE OUTCOMES (For "True Model" Benchmark)
  data$T_I <- pmin(T_I, C)
  data$C <- C
  data$EVENT <- as.integer(T_I <= C)
  data <- add_measurement_errors(data)
  
  cols_to_transform <- setdiff(data_info_srs$cat_vars, c("R", "EVENT", "EVENT_STAR"))
  data[cols_to_transform] <- lapply(data[cols_to_transform], as.character)
  
  return (data)
}

add_measurement_errors <- function(data) {
  n <- nrow(data)
  # Linear Measurement Error Model: W = a0 + a1*X + a2*Z1 + ... + e
  apply_linear_error <- function(true_vec, slope = 1.0, intercept = 0.0, 
                                 z_list = NULL, z_coeffs = NULL, noise_sd = 0.1) {
    val <- intercept + slope * true_vec
    if (!is.null(z_list) && !is.null(z_coeffs)) {
      for (i in 1:length(z_list)) {
        val <- val + z_coeffs[i] * as.numeric(z_list[[i]]) 
      }
    }
    val <- val + rnorm(length(true_vec), mean = 0, sd = noise_sd)
    return (pmax(min(true_vec), val))
  }
  apply_cat_error <- function(true_vec, levels_vec, z_list = NULL, z_coeffs = NULL, base_accuracy = 0.80) {
    n_obs <- length(true_vec)
    logits <- rep(qlogis(base_accuracy), n_obs)
    if (!is.null(z_list) && !is.null(z_coeffs)) {
      for (i in 1:length(z_list)) {
        logits <- logits + z_coeffs[i] * as.vector(scale(as.numeric(z_list[[i]])))
      }
    }
    p_correct <- plogis(logits)
    res <- true_vec
    is_error <- runif(n_obs) > p_correct
    if (sum(is_error) > 0) {
      err_idx <- which(is_error)
      for (idx in err_idx) {
        others <- levels_vec[levels_vec != true_vec[idx]]
        res[idx] <- if (length(others) == 1) others else sample(others, 1)
      }
    }
    return(res)
  }
  
  # ============================================================================
  # 1. PRIMARY PREDICTORS 
  # ============================================================================
  # HbA1c_STAR ~ alpha_0 + alpha_1 * HbA1c + alphas * [URBAN, AGE, RACE, BMI, INSURANCE] + U
  data$HbA1c_STAR <- apply_linear_error(data$HbA1c, 
                                        slope = 1.5, intercept = -10.0,
                                        z_list = list(1 - data$URBAN, 
                                                      data$AGE - mean(data$AGE),
                                                      data$RACE %in% c("AMR", "AFR"),
                                                      data$BMI, 1- data$INSURANCE),
                                        z_coeffs = c(-1, -0.2, -5, -0.2, -2), 
                                        noise_sd = sd(data$HbA1c) * 0.33)
  
  # eGFR: Recalculated from a very noisy Creatinine
  data$Creatinine_STAR <- apply_linear_error(data$Creatinine, slope = 1.2, 
                                             noise_sd = sd(data$Creatinine) * 0.40)
  scr <- data$Creatinine_STAR / 88.4
  k <- ifelse(data$SEX, 0.9, 0.7); alpha <- ifelse(data$SEX, -0.411, -0.329)
  sex_f <- ifelse(data$SEX, 1.0, 1.018)
  data$eGFR_STAR <- 141 * (pmin(scr/k, 1)^alpha) * (pmax(scr/k, 1)^-1.209) * (0.993^data$AGE) * sex_f
  data$eGFR_STAR <- pmin(data$eGFR_STAR, 140)
  # BMI Components: Heavy systematic bias
  TRUE_WEIGHT <- data$BMI * (data$HEIGHT / 100)^2
  data$WEIGHT_STAR <- apply_linear_error(TRUE_WEIGHT, slope = 1.05, intercept = -3.0,
                                         z_list = list(data$BMI, data$SEX),
                                         z_coeffs = c(-0.5, 0.5), noise_sd = 1)
  data$HEIGHT_STAR <- apply_linear_error(data$HEIGHT, slope = 1.05, intercept = 3.0,
                                         z_list = list(data$AGE), z_coeffs = c(0.2), noise_sd = 0.5)
  data$BMI_STAR <- data$WEIGHT_STAR / (data$HEIGHT_STAR / 100)^2
  
  # ============================================================================
  # 2. NECESSARY CATEGORICAL CONTROLS & AUXILIARY Z
  # ============================================================================
  
  # SMOKE: Reduced base accuracy to 65% (Strong impact on Cox model)
  data$SMOKE_STAR <- apply_cat_error(data$SMOKE, sort(unique(data$SMOKE)), 
                                     z_list = list(data$EDU), z_coeffs = c(0.5), base_accuracy = 0.75)
  
  data$INCOME_STAR <- apply_cat_error(data$INCOME, sort(unique(data$INCOME)), 
                                      z_list = list(data$EDU), z_coeffs = c(0.5), base_accuracy = 0.70)
  data$ALC_STAR <- apply_cat_error(data$ALC, sort(unique(data$ALC)), base_accuracy = 0.70)
  data$EXER_STAR <- apply_cat_error(data$EXER, sort(unique(data$EXER)), base_accuracy = 0.70)
  
  # EDU (Numeric Z): Biased reporting
  data$EDU_STAR <- round(apply_linear_error(data$EDU, slope = 1.0, z_list = list(data$INCOME), 
                                            z_coeffs = c(0.4), noise_sd = 1))
  data$EDU_STAR <- pmax(0, data$EDU_STAR)
  
  # SBP & Triglyceride (Keep as they are common Cox covariates)
  data$SBP_STAR <- apply_linear_error(data$SBP, slope = 1.0, intercept = 10.0, 
                                      z_list = list(data$AGE), z_coeffs = c(0.3), noise_sd = 3)
  data$Triglyceride_STAR <- apply_linear_error(data$Triglyceride, slope = 1.0, 
                                               noise_sd = sd(data$Triglyceride) * 0.40)
  
  # ============================================================================
  # 3. SURVIVAL
  # ============================================================================
  
  access_multiplier <- 1 + 0.15 * 
    as.numeric(!data$INSURANCE) + 0.10 * as.numeric(!data$URBAN)
  severity_buffer <- 0.05 * pmax(0, (data$HbA1c - 50) / 10)
  final_multiplier <- pmax(1.01, access_multiplier - severity_buffer)
  T_candidate <- data$T_I * final_multiplier + rexp(n, rate = 1.0)
  
  dropout_risk <- 0.1 * as.numeric(!data$INSURANCE) + 0.05 * as.numeric(!data$URBAN)
  is_dropout <- runif(n) < dropout_risk
  
  data$C_STAR <- ifelse(is_dropout, 
                        pmax(0.1, data$C - 5 + rnorm(n, 0, 1)), 
                        pmax(0.1, data$C + rnorm(n, 0, 1)))
  
  data$T_I_STAR <- pmin(T_candidate, data$C_STAR)
  data$EVENT_STAR <- as.integer(T_candidate <= data$C_STAR)

  logit_miss <- -2 + 0.5 * as.numeric(!data$INSURANCE) - 0.2 * (data$HbA1c - 40)/10
  miss_prob <- plogis(logit_miss)
  
  is_missed <- (data$EVENT_STAR == 1) & (runif(n) < miss_prob)
  data$EVENT_STAR[is_missed] <- 0
  data$T_I_STAR[is_missed] <- data$C_STAR[is_missed]
  
  return(data)
}

loadCovarInfo <- function(genoInfo){
  if (!file.exists("./data/covarInfo.RData")){
    load("./data/labtest_parameters.RData")
    # HEIGHT ~ SEX + Genos
    formu_HEIGHT <- as.formula(paste0(" ~ SEX + ", paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HEIGHT", ]$Query, collapse = " + ")))
    betas_HEIGHT <- c(162, 13, transform_betas(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HEIGHT", ]$Beta))
    sigma_HEIGHT <- 7
    # SMOKE ~ AGE + SEX + RACE (Current Smoker, Ex-Smoker, Never Smoked)
    formu_SMOKE <- as.formula(paste0("~", Mod_list$SMOKE$formula[3]))
    betas_SMOKE <- Mod_list$SMOKE$coeff
    # EXER ~ AGE + SEX + RACE + SMOKE (Normal, Low, High)
    formu_EXER <- as.formula(~ AGE + SEX + RACE + SMOKE)
    betas_EXER <- rbind(
      c(0.5, 0.01, -0.2, 0.1, -0.1, 0.05, 0.08, 0.4, 0.2), # Low vs. Normal
      c(-0.5, -0.02, 0.3, -0.1, 0.1, -0.05, -0.08, -0.3, -0.1) # High vs. Normal
    )
    # ALC ~ AGE + SEX + SMOKE + EXER (Moderate, None, Heavy)
    formu_ALC <- as.formula(~ AGE + SEX + RACE + SMOKE + EXER)
    betas_ALC <- rbind(
      c(-1.0, 0.01, -0.5, 0.2, 0.3, 0.1, 0.15, -0.4, -0.2,  0.3, -0.3), # None vs. Moderate
      c(-1.5, -0.02, 0.6, 0.1, -0.2, 0.05, 0.1, 0.5, 0.2, 0.2, -0.4) # Heavy vs. Moderate
    )
    # MARRIAGE ~ AGE + SEX + RACE + BMI (None, Married, Divorced)
    formu_MARRIAGE <- as.formula(~ AGE + SEX + RACE + BMI)
    betas_MARRIAGE <- rbind(
      c(-2.0, 0.05, -0.3, -0.2, 0.1, -0.1, -0.05, -0.02), # Married vs. None
      c(-3.0, 0.03, 0.2, 0.1, -0.1, 0.05, 0.08, 0.01) # Divorced vs. None
    )
    # BMI ~ SEX + SMOKE + RACE + AGE + GENOs
    formu_BMI <- as.formula(paste0("~",  Mod_list$BMI$BMI$formula[3], " + ", 
                                   paste0(unique(genoInfo$genoTrait$Query[genoInfo$genoTrait$PHENO == "BMI"]), 
                                          collapse = " + ")))
    betas_BMI <- c(Mod_list$BMI$BMI$coeff, 
                   transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "BMI"]))
    sigma_BMI <- Sigma$BMI
    # Medication Count:
    formu_MED_Count <- as.formula(paste0("~", Mod_list$MED_Count$MED_Count$formula[3]))
    betas_MED_Count <- Mod_list$MED_Count$MED_Count$coeff
    # Diabetes
    formu_DIABETE <- list(Glucose = as.formula(paste("~", Mod_list$Diabete$Glucose$formula[3], "+", 
                                                     paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Glucose", ]$Query, collapse = " + "))),
                          F_Glucose = as.formula(paste("~", Mod_list$Diabete$F_Glucose$formula[3], "+", 
                                                       paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "F_Glucose", ]$Query, collapse = " + "))),
                          HbA1c = as.formula(paste("~", Mod_list$Diabete$HbA1c$formula[3], "+", 
                                                   paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HbA1c", ]$Query, collapse = " + "))),
                          Insulin = as.formula(paste("~", Mod_list$Diabete$Insulin$formula[3], "+", 
                                                     paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Insulin", ]$Query, collapse = " + "))))
    betas_DIABETE <- list(Glucose = c(Mod_list$Diabete$Glucose$coeff, 
                                      transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Glucose"])), 
                          F_Glucose = c(Mod_list$Diabete$F_Glucose$coeff,
                                        transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "F_Glucose"])),
                          HbA1c = c(Mod_list$Diabete$HbA1c$coeff,
                                    transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "HbA1c"])),
                          Insulin = c(Mod_list$Diabete$Insulin$coeff,
                                      transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Insulin"])))  # Insulin pmol/L
    sigma_DIABETE <- Sigma$Diabete
    # Nutrients
    formu_NUTRIENTS <- list(Na_INTAKE = as.formula(~ AGE + SEX + BMI + RACE),
                            K_INTAKE = as.formula(~ AGE + SEX + BMI + RACE),
                            KCAL_INTAKE = as.formula(~ AGE + SEX + BMI + RACE),
                            PROTEIN_INTAKE = as.formula(~ AGE + SEX + BMI + RACE))
    betas_NUTRIENTS <- list(Na_INTAKE = c(7.66, -0.00084, 0.27, 0.0177, 0.009, -0.10, -0.15, 0.03),
                            K_INTAKE = c(7.2, 0.0075, 0.177, 0.00956, 0.04, -0.08, -0.12, 0.02),
                            KCAL_INTAKE = c(7.4, -0.00256, 0.16, 0.014, 0.20, -0.4, -0.6, 0.12),
                            PROTEIN_INTAKE = c(4, 0.000459, 0.187, 0.0166, 0.02, -0.13, -0.14, 0.1))
    sigma_NUTRIENTS <- matrix(c(0.1820, 0.0350, 0.0250, 0.0360,
                                0.0350, 0.1530, 0.0120, 0.0330,
                                0.0250, 0.0120, 0.1070, 0.0170,
                                0.0360, 0.0330, 0.0170, 0.1220), nrow = 4, byrow = TRUE)
    # LIVER
    formu_LIVER <- list(AST = as.formula(paste("~", Mod_list$Liver$AST$formula[3])), 
                        ALT = as.formula(paste("~", Mod_list$Liver$ALT$formula[3], "+", 
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "ALT", ]$Query, collapse = " + "))),
                        ALP = as.formula(paste("~", Mod_list$Liver$ALP$formula[3], "+", 
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "ALP", ]$Query, collapse = " + "))), 
                        GGT = as.formula(paste("~", Mod_list$Liver$GGT$formula[3])), 
                        Bilirubin = as.formula(paste("~", Mod_list$Liver$Bilirubin$formula[3])), 
                        Albumin = as.formula(paste("~", Mod_list$Liver$Albumin$formula[3], "+", 
                                                   paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Albumin", ]$Query, collapse = " + "))),
                        Globulin = as.formula(paste("~", Mod_list$Liver$Globulin$formula[3])), 
                        Protein = as.formula(paste("~", Mod_list$Liver$Protein$formula[3])))
    betas_LIVER <- list(AST = Mod_list$Liver$AST$coeff,
                        ALT = c(Mod_list$Liver$ALT$coeff,
                                transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "ALT"])),
                        ALP = c(Mod_list$Liver$ALP$coeff,
                                transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "ALP"])), 
                        GGT = Mod_list$Liver$GGT$coeff,
                        Bilirubin = Mod_list$Liver$Bilirubin$coeff,
                        Albumin = c(Mod_list$Liver$Albumin$coeff,
                                    transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Albumin"])),
                        Globulin = Mod_list$Liver$Globulin$coeff,
                        Protein = Mod_list$Liver$Protein$coeff)
    sigma_LIVER <- Sigma$Liver
    
    # INFLAM
    formu_FERRITIN <- as.formula(paste("~", Mod_list$Ferritin$Ferritin$formula[3]))
    betas_FERRITIN <- Mod_list$Ferritin$Ferritin$coeff
    sigma_FERRITIN <- Sigma$Ferritin
    
    # RENAL
    formu_RENAL <- list(Creatinine = as.formula(paste("~", Mod_list$Renal$Creatinine$formula[3])),
                        Urea = as.formula(paste("~", Mod_list$Renal$Urea$formula[3])),
                        Potassium = as.formula(paste("~", Mod_list$Renal$Potassium$formula[3])),
                        Sodium = as.formula(paste("~", Mod_list$Renal$Sodium$formula[3])),
                        Chloride = as.formula(paste("~", Mod_list$Renal$Chloride$formula[3])),
                        Bicarbonate = as.formula(paste("~", Mod_list$Renal$Bicarbonate$formula[3])),
                        Calcium = as.formula(paste("~", Mod_list$Renal$Calcium$formula[3])),
                        Magnesium = as.formula(paste("~", Mod_list$Renal$Magnesium$formula[3])), 
                        Phosphate = as.formula(paste("~", Mod_list$Renal$Phosphate$formula[3])))
    betas_RENAL <- list(Creatinine = Mod_list$Renal$Creatinine$coeff,
                        Urea = Mod_list$Renal$Urea$coeff,
                        Potassium = Mod_list$Renal$Potassium$coeff,
                        Sodium = Mod_list$Renal$Sodium$coeff,
                        Chloride = Mod_list$Renal$Chloride$coeff,
                        Bicarbonate = Mod_list$Renal$Bicarbonate$coeff,
                        Calcium = Mod_list$Renal$Calcium$coeff,
                        Magnesium = Mod_list$Renal$Magnesium$coeff, 
                        Phosphate = Mod_list$Renal$Phosphate$coeff)
    sigma_RENAL <- Sigma$Renal
    # LIPIDS
    formu_LIPIDS <- list(Triglyceride = as.formula(paste("~", Mod_list$Lipid$Triglyceride$formula[3], " + ",
                                                         paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Triglyceride", ]$Query, collapse = " + "))), 
                         HDL = as.formula(paste("~", Mod_list$Lipid$HDL$formula[3])), 
                         LDL = as.formula(paste("~", Mod_list$Lipid$LDL$formula[3], " + ",
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "LDL", ]$Query, collapse = " + "))))
    betas_LIPIDS <- list(Triglyceride = c(Mod_list$Lipid$Triglyceride$coeff, 
                                          transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Triglyceride"])), 
                         HDL = Mod_list$Lipid$HDL$coeff, 
                         LDL = c(Mod_list$Lipid$LDL$coeff, 
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "LDL"])))
    sigma_LIPIDS <- Sigma$Lipid
    # HEMATOLOGY
    formu_HEMA <- list(Hb = as.formula(paste("~", Mod_list$Haematology$Hb$formula[3])), 
                       HCT = as.formula(paste("~", Mod_list$Haematology$HCT$formula[3], " + ",
                                              paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HCT", ]$Query, collapse = " + "))),
                       RBC = as.formula(paste("~", Mod_list$Haematology$RBC$formula[3])), 
                       WBC = as.formula(paste("~", Mod_list$Haematology$WBC$formula[3])), 
                       Platelet = as.formula(paste("~", Mod_list$Haematology$Platelet$formula[3])), 
                       MCV = as.formula(paste("~", Mod_list$Haematology$MCV$formula[3])), 
                       RDW = as.formula(paste("~", Mod_list$Haematology$RDW$formula[3])), 
                       Neutrophils = as.formula(paste("~", Mod_list$Haematology$Neutrophils$formula[3])), 
                       Lymphocytes = as.formula(paste("~", Mod_list$Haematology$Lymphocytes$formula[3])), 
                       Monocytes = as.formula(paste("~", Mod_list$Haematology$Monocytes$formula[3])), 
                       Eosinophils = as.formula(paste("~", Mod_list$Haematology$Eosinophils$formula[3])), 
                       Basophils = as.formula(paste("~", Mod_list$Haematology$Basophils$formula[3])))
    
    betas_HEMA <- list(Hb = Mod_list$Haematology$Hb$coeff, 
                       HCT = c(Mod_list$Haematology$HCT$coeff, 
                               transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "HCT"])),
                       RBC = Mod_list$Haematology$RBC$coeff,
                       WBC = Mod_list$Haematology$WBC$coeff,
                       Platelet = Mod_list$Haematology$Platelet$coeff,
                       MCV = Mod_list$Haematology$MCV$coeff,
                       RDW = Mod_list$Haematology$RDW$coeff,
                       Neutrophils = Mod_list$Haematology$Neutrophils$coeff,
                       Lymphocytes = Mod_list$Haematology$Lymphocytes$coeff,
                       Monocytes = Mod_list$Haematology$Monocytes$coeff,
                       Eosinophils = Mod_list$Haematology$Eosinophils$coeff,
                       Basophils = Mod_list$Haematology$Basophils$coeff)
    sigma_HEMA <- Sigma$Haematology
    
    # Vital Signs
    formu_VITALS <- list(SBP = as.formula(paste("~", Mod_list$Vital$SBP$formula[3], " + ",
                                                paste0(unique(genoInfo$genoTrait$Query[genoInfo$genoTrait$PHENO == "SBP"]), collapse = " + "))), 
                         Temperature = as.formula(paste("~", Mod_list$Vital$Temperature$formula[3])), 
                         HR = as.formula(paste("~", Mod_list$Vital$HR$formula[3])), 
                         SpO2 = as.formula(paste("~", Mod_list$Vital$SpO2$formula[3])))
    betas_VITALS <- list(SBP = c(Mod_list$Vital$SBP$coeff,
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "SBP"])),
                         Temperature = Mod_list$Vital$Temperature$coeff,
                         HR = Mod_list$Vital$HR$coeff,
                         SpO2 = Mod_list$Vital$SpO2$coeff)
    sigma_VITALS <- Sigma$Vital
    # PACK UP
    formu <- list(HEIGHT = formu_HEIGHT, SMOKE = formu_SMOKE, EXER = formu_EXER, 
                  ALC = formu_ALC, BMI = formu_BMI, MED_Count = formu_MED_Count, MARRIAGE = formu_MARRIAGE,
                  VITALS = formu_VITALS, RENAL = formu_RENAL, LIPIDS = formu_LIPIDS,
                  HEMA = formu_HEMA, NUTRIENTS = formu_NUTRIENTS, LIVER = formu_LIVER,
                  DIABETE = formu_DIABETE, FERRITIN = formu_FERRITIN)
    betas <- list(HEIGHT = betas_HEIGHT, SMOKE = betas_SMOKE, EXER = betas_EXER, 
                  ALC = betas_ALC, BMI = betas_BMI, MED_Count = betas_MED_Count, MARRIAGE = betas_MARRIAGE,
                  VITALS = betas_VITALS, RENAL = betas_RENAL, LIPIDS = betas_LIPIDS,
                  HEMA = betas_HEMA, NUTRIENTS = betas_NUTRIENTS, LIVER = betas_LIVER,
                  DIABETE = betas_DIABETE, FERRITIN = betas_FERRITIN)
    sigma <- list(HEIGHT = sigma_HEIGHT, BMI = sigma_BMI,
                  VITALS = sigma_VITALS, RENAL = sigma_RENAL, LIPIDS = sigma_LIPIDS,
                  HEMA = sigma_HEMA, NUTRIENTS = sigma_NUTRIENTS, LIVER = sigma_LIVER,
                  DIABETE = sigma_DIABETE, FERRITIN = sigma_FERRITIN)
    type <- list(HEIGHT = "Norm", SMOKE = "Cat", EXER = "Cat", 
                 ALC = "Cat", BMI = "Norm", MED_Count = "Pois", MARRIAGE = "Cat",
                 VITALS = "MVNorm", RENAL = "MVNorm", LIPIDS = "MVNorm",
                 HEMA = "MVNorm", NUTRIENTS = "MVNorm", LIVER = "MVNorm",
                 DIABETE = "MVNorm", FERRITIN = "Norm")
    save(formu, betas, sigma, type, file = "./data/covarInfo.RData")
  }else{
    load("./data/covarInfo.RData")
  }
  return (list(formu = formu, betas = betas, sigma = sigma, type = type))
}

simCovs <- function(data, formu, betas, type, sds = NULL){
  if (type == "Cat"){
    mm <- model.matrix(formu, data)
    lp <- exp(cbind(rep(0, nrow(data)), mm %*% t(betas)))
    probmat <- lp / rowSums(lp)
    cummat <- t(apply(probmat, 1, cumsum))
    result <- as.factor(max.col(cummat >= runif(nrow(data)), ties.method = "first"))
  }else if (type == "Norm"){
    mm <- model.matrix(formu, data)
    means <- mm %*% betas
    result <- round(rnorm(nrow(data), means, sds), 3)
  }else if (type == "MVNorm"){
    means <- lapply(1:length(formu), function(i) model.matrix(formu[[i]], data) %*% betas[[i]]) %>% bind_cols
    result <- t(apply(means, 1, function(mu) round(rmvnorm(1, mu, sds), 3)))
    result <- as.data.frame(result)
    names(result) <- gsub("formu_", "", names(formu))
  }else if (type == "Pois"){
    lambda <- exp(model.matrix(formu, data) %*% betas)
    result <- rpois(nrow(data), lambda = lambda)
  }
  return (result)
}

simGenotypes <- function(data, genoInfo){
  df <- data.frame()
  race <- unique(data$RACE)
  for (r in race){
    ind <- which(data$RACE == r)
    curr_subset <- subset(data, RACE == r)
    genoMat <- matrix(0, nrow = length(ind), ncol = 15)
    covnames <- c()
    m <- 1
    for (c in unique(genoInfo$rsids$chr)){
      fsub <- subset(genoInfo$genoFreq, Race == r & chr == c)
      probs <- as.numeric(fsub$Frequency[seq(2, length(fsub$Frequency), by = 2)])
      names(probs) <- fsub$refsnp_id[seq(2, length(fsub$Frequency), by = 2)]
      if (as.character(c) %in% names(genoInfo$genoCorr)){
        rsub <- subset(genoInfo$genoCorr[[as.character(c)]], Race == r)
        Sigma <- as.matrix(rsub[, -which(!names(rsub) %in% unique(fsub$refsnp_id))])
        rownames(Sigma) <- colnames(Sigma)
        Sigma <- Sigma[names(probs), names(probs)]
        
        U <- pnorm(rmvnorm(length(ind), sigma = Sigma))
        G <- lapply(1:ncol(U), function(i){
          as.factor(qbinom(U[, i], 2, probs[i]))
        }) %>% bind_cols
      }else{
        G <- rbinom(length(ind), 2, probs)
      }
      genoMat[, m:(m + length(unique(fsub$refsnp_id)) - 1)] <- as.matrix(G)
      m <- m + length(unique(fsub$refsnp_id))
      covnames <- c(covnames, fsub$refsnp_id[seq(2, length(fsub$Frequency), by = 2)])
    }
    genoMat <- data.frame(genoMat)
    names(genoMat) <- covnames
    curr_subset <- cbind(curr_subset, data.frame(genoMat))
    df <- rbind(df, curr_subset)
  }
  return (df)
}

transform_betas <- function(input_vector) {
  transformed_betas <- numeric(length(input_vector) * 2)
  for (i in 1:length(input_vector)) {
    transformed_betas[2 * i - 1] <- input_vector[i]
    transformed_betas[2 * i] <- input_vector[i] * 2
  }
  return(transformed_betas)
}

fetch_parameters <- function(){
  desired_labtests <- list(
    # ====== Liver-function panel (enzymes & plasma proteins) ======================
    AST        = c("AST"),
    ALT        = c("ALT", "ALT plasma"),
    ALP        = c("Alkaline Phosphatase"),
    GGT        = c("GGT"),
    Bilirubin  = c("Bilirubin", "Total Bilirubin"),
    Albumin    = c("Albumin"),
    Globulin   = c("Globulin plasma", "Globulin"),
    Protein    = c("Protein", "Total Protein"),
    
    # ====== Renal / electrolyte + acid–base panel =================================
    Creatinine = c("Creatinine", "Whole Blood Creatinine"),
    Urea       = c("Urea", "Whole Blood Urea"),
    Potassium  = c("Potassium", "Potassium blood", "Potassium whole blood"),
    Sodium     = c("Sodium", "Sodium blood", "Sodium whole blood"),
    Chloride   = c("Chloride blood", "Chloride"),
    Bicarbonate= c("Bicarbonate", "Actual Bicarbonate", "Standard Bicarbonate",
                   "Gases Standard Bicarbonate", "Gases Actual Bicarbonate"),
    Calcium    = c("Calcium", "Adjusted Calcium"),
    Magnesium  = c("Magnesium"),
    Phosphate  = c("Phosphate"),
    
    # ====== Haematology / full-blood-count panel ==================================
    Hb         = c("Hb  - Haemoglobin", "Hb - Haemoglobin",
                   "Blood Haemoglobin", "Haemoglobin"),
    HCT        = c("Hct - Haematocrit", "HCT", "Haematocrit"),
    RBC        = c("RBC", "RBC - Red Cell Count"),
    WBC        = c("WBC - White Cell Count", "WBC"),
    Platelet   = c("Platelet Count", "Platelets"),
    MCV        = c("MCV", "MCV - Mean Cell Volume"),
    RDW        = c("RDW"),
    Neutrophils= c("Neutrophils"),
    Lymphocytes= c("Lymphocytes"),
    Monocytes  = c("Monocytes"),
    Eosinophils= c("Eosinophils"),
    Basophils  = c("Basophils"),
    
    # ====== Lipid panel ===========================================================
    Triglyceride = c("Triglyceride"),
    HDL          = c("HDL Cholesterol", "Cholesterol (HDL)"),
    LDL          = c("LDL cholesterol",
                     "Cholesterol (LDL)(calculated)", "LDL Cholesterol"),
    
    # ====== Glucose / diabetes panel =============================================
    Glucose     = c("Glucose", "Glucose blood", "Glucose whole blood",
                    "Glucose whole Blood"),
    F_Glucose   = c("Fasting Glucose"),
    HbA1c       = c("HbA1c", "Haemoglobin A1c (%)"),
    Insulin     = c("Insulin"),
    
    # ====== Iron-storage marker ===================================================
    Ferritin    = c("Ferritin")
  )
  cohort <- read_sas("cohort_29nov2022.sas7bdat", "formats.sas7bcat")
  cohort <- cohort %>% select(patient_reference_no, gender, 
                              smoking_status, ethnic_gp, age_g, BMI_g, 
                              medication_count_t, SBP_ews, 
                              temperature_ews,
                              HR_ews, O2sat_ews)
  bmi_bounds <- list(`1` = c(15, 18.49), `2` = c(18.5, 24.99),
                     `3` = c(25, 29.99), `4` = c(30, 34.99),
                     `5` = c(35, 39.99), `6` = c(40, 60))
  sbp_bounds <- list(`0` = c(111, 219), `1` = c(101, 110),
                     `2` = c(91, 100), `3` = c(60, 90), 
                     `4` = c(220, 260))
  temp_bounds <- list(`0` = c(36.1, 38.0), `1` = c(35.1, 36.0),
                      `2` = c(38.1, 40.5), `3` = c(32.0, 35.0))
  hr_bounds <- list(`0` = c(51, 90), `1` = c(91, 110),
                    `2` = c(111, 130), `3` = c(30, 40),
                    `4` = c(131, 220))
  spo2_bounds <- list(`0` = c(96, 100), `1` = c(94, 95),
                      `2` = c(92, 93), `3` = c(85, 91))
  bounds_sample = function(code, bounds) {
    if (is.na(code) || !as.character(code) %in% names(bounds)) 
      return(NA_real_)
    rng = bounds[[as.character(code)]]
    runif(1, rng[1], rng[2])
  }
  cohort <- cohort %>% 
    mutate(BMI = map_dbl(BMI_g, ~ bounds_sample(.x, bmi_bounds)),
           SBP = map_dbl(SBP_ews, ~ bounds_sample(.x, sbp_bounds)),
           Temperature = map_dbl(temperature_ews, ~ bounds_sample(.x, temp_bounds)),
           HR = map_dbl(HR_ews, ~ bounds_sample(.x, hr_bounds)),
           SpO2 = map_dbl(O2sat_ews, ~ bounds_sample(.x, spo2_bounds))) %>%
    select(patient_reference_no, gender, 
           smoking_status, ethnic_gp, age_g, BMI, 
           medication_count_t, SBP, 
           Temperature, HR, SpO2) %>%
    group_by(patient_reference_no) %>%
    summarise(across(c(BMI, SBP, Temperature, HR, SpO2), 
                     ~ mean(.x, na.rm = TRUE)), 
              gender = first(gender),
              smoking_status = first(smoking_status),
              ethnic_gp = first(ethnic_gp),
              age_g = first(age_g), 
              medication_count_t = first(medication_count_t),
              .groups = "drop")
  labtest <- read_sas("labtest2.sas7bdat")
  labtest <- labtest %>% 
    select(patient_reference_no, 
           test_result_date, 
           result_test_name, 
           test_result)
  alias_tbl <- imap_dfr(desired_labtests, ~ tibble(alias = .x, canonical = .y)) %>% 
    mutate(alias_lower = str_to_lower(str_squish(alias))) %>%
    distinct(alias_lower, canonical)   
  labtest <- labtest %>%
    mutate(test_result_date = as.Date(test_result_date),
           alias_lower = str_to_lower(str_squish(result_test_name))) %>%
    left_join(alias_tbl, by = "alias_lower") %>%
    filter(result_test_name %in% unlist(desired_labtests)) %>%
    mutate(result_test_name = coalesce(canonical, result_test_name)) %>% 
    select(-alias_lower, -canonical) %>%
    filter(result_test_name != "" & !is.na(result_test_name))
  labtest_wide <- labtest %>%
    pivot_wider(id_cols = c(patient_reference_no, test_result_date),
                names_from = result_test_name,
                values_from = test_result,
                values_fn = list(test_result = ~ first(na.omit(.))),
                values_fill = NA)
  # Eliminate some extreme values, since it's ED dataset
  for (i in names(desired_labtests)){
    labtest_wide[[i]] <- as.numeric(labtest_wide[[i]])
  }
  extreme_range <- with(labtest_wide, list(AST = AST > 1000,
                                           ALT = ALT > 1000,
                                           ALP = ALP > 2000,
                                           GGT = GGT > 1000,
                                           Bilirubin = Bilirubin > 200,
                                           Creatinine = Creatinine > 2000,
                                           Urea = Urea > 50,
                                           WBC = WBC > 100,
                                           Neutrophils = Neutrophils > 50,
                                           Lymphocytes = Lymphocytes > 20,
                                           Monocytes = Monocytes > 10,
                                           Eosinophils = Eosinophils > 10,
                                           Basophils = Basophils > 10,
                                           Triglyceride = Triglyceride > 20,
                                           Glucose = Glucose > 20,
                                           Ferritin = Ferritin > 1000))
  for (i in names(extreme_range)){
    labtest_wide[[i]][extreme_range[[i]]] <- NA
  }
  bindedframe <- labtest_wide %>%
    left_join(cohort %>% distinct(patient_reference_no, .keep_all = TRUE),
              by = "patient_reference_no")
  # Lots of Skewed densities with outliers, retrieve parameters on log scale:
  for (i in c(names(desired_labtests), "SBP", "Temperature", "HR", "SpO2", "BMI")){
    bindedframe[[i]] <- log(bindedframe[[i]] + 1)
  }
  
  # save(bindedframe, file = "joinedData.RData")
  load("joinedData.RData")
  bindedframe$BMI <- log(bindedframe$BMI + 1)
  bindedframe$smoking_status[bindedframe$smoking_status == "Question not asked"] <- 
    sample(c("Never Smoked", "Ex Smoker", "Current Smoker (within 4wks)"), 1)
  bindedframe$ethnic_gp <- as.factor(bindedframe$ethnic_gp)
  bindedframe <- bindedframe %>%
    rename(SEX = gender, 
           SMOKE = smoking_status,
           RACE = ethnic_gp,
           MED_Count = medication_count_t,
           AGE = age_g)
  # Retrieve COV for each group
  ensure_psd <- function(mat, tol = 1e-8) {
    is_psd <- function(mat, tol = 1e-8) {
      min(eigen(mat, symmetric = TRUE)$values) >= -tol
    }
    if (!is_psd(mat)){
      e <- eigen(mat, symmetric = TRUE)
      vals <- pmax(e$values, tol)
      m2 <- e$vectors %*% diag(vals) %*% t(e$vectors)
      return ((m2 + t(m2)) / 2)
    }else{
      return (mat)
    }
  }
  Vital_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = c("SBP", "Temperature", "HR", "SpO2"))),
                     use = "pairwise.complete.obs"))
  Liver_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[1:8])),
                     use = "pairwise.complete.obs"))
  Renal_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[9:18])),
                     use = "pairwise.complete.obs"))
  Haematology_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[19:30])),
                           use = "pairwise.complete.obs"))
  Lipid_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[31:33])),
                     use = "pairwise.complete.obs"))
  Diabete_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[34:37])),
                       use = "pairwise.complete.obs"))
  Ferritin_Sigma <- sd(as.matrix(subset(bindedframe, select = names(desired_labtests)[38])), na.rm = T) ^ 2
  BMI_Sigma <- sd(as.matrix(subset(bindedframe, select = "BMI")), na.rm = T) ^ 2
  # # Retrieve Mean for each group
  # Vital_MU <- colMeans(as.matrix(subset(bindedframe, select = c("SBP", "Temperature", "HR", "SpO2"))), na.rm = T)
  # Liver_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[1:8])), na.rm = T)
  # Renal_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[9:18])), na.rm = T)
  # Haematology_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[19:30])), na.rm = T)
  # Lipid_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[31:33])), na.rm = T)
  # Diabete_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[34:37])), na.rm = T)
  # Ferritin_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[38])), na.rm = T)
  # Retrieve Coefficients:
  retrieveCoeffs <- function(data, formula, family = "gaussian") {
    mod <- glm(formula, data = data, family = family)
    sumry <- summary(mod)
    ct <- sumry$coefficients
    
    pcol <- if ("Pr(>|t|)" %in% colnames(ct)) "Pr(>|t|)" else "Pr(>|z|)"
    pvalues <- ct[, pcol]
    estimates <- ct[, "Estimate"]
    
    terms <- attr(terms(mod), "term.labels")
    keep <- character()
    for (t in terms) {
      if (is.factor(data[[t]]) || is.character(data[[t]])) {
        cn <- grep(paste0("^", t), rownames(ct), value = TRUE)
        pv <- pvalues[cn]
        es <- estimates[cn]
        if (any(pv <= 0.05)) {
          keep <- c(keep, t)
          for (lev in cn) {
            if (!(pvalues[lev] <= 0.05)) {
              mod$coefficients[lev] <- 0
            }
          }
        }
      } else {
        if (pvalues[t] <= 0.05) {
          keep <- c(keep, t)
        }
      }
    }
    resp <- deparse(formula[[2]])
    if (length(keep) > 0) {
      newf <- as.formula(paste0(resp, " ~ ", paste(keep, collapse = " + ")))
    } else {
      newf <- as.formula(paste0(resp, " ~ 1"))
    }
    mod2 <- glm(newf, data = data, family = family)
    coefs <- coef(mod2)
    race_idx <- grep("^RACE", names(coefs))
    if (length(race_idx) > 0) {
      coefs[race_idx] <- coefs[race_idx] +
        rnorm(length(race_idx), mean = 0, sd = 0.05) 
      # Added Small Random Noise to Coefficients belong to RACE
      # Masking the True Coefficients, hence protects data privacy.
    }
    list(coeff = coefs, formula = as.character(newf))
  }
  Liver_mod_list <- lapply(names(desired_labtests)[1:8], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Liver_mod_list) <- names(desired_labtests)[1:8]
  Renal_mod_list <- lapply(names(desired_labtests)[9:18], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Renal_mod_list) <- names(desired_labtests)[9:18]
  Haematology_mod_list <- lapply(names(desired_labtests)[19:30], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Haematology_mod_list) <- names(desired_labtests)[19:30]
  Lipid_mod_list <- lapply(names(desired_labtests)[31:33], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Lipid_mod_list) <- names(desired_labtests)[31:33]
  Diabete_mod_list <- lapply(names(desired_labtests)[34:37], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Diabete_mod_list) <- names(desired_labtests)[34:37]
  Ferritin_mod_list <- lapply(names(desired_labtests)[38], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Ferritin_mod_list) <- names(desired_labtests)[38]
  Vital_mod_list <- lapply(c("SBP", "Temperature", "HR", "SpO2"), function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Vital_mod_list) <- c("SBP", "Temperature", "HR", "SpO2")
  MED_Count_mod_list <- lapply("MED_Count", function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI")), family = "poisson")
  })
  names(MED_Count_mod_list) <- c("MED_Count")
  BMI_mod_list <- lapply("BMI", function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE")))
  })
  names(BMI_mod_list) <- c("BMI")
  
  SMOKE_mod <- multinom(SMOKE ~ AGE + SEX + RACE, data = bindedframe)
  SMOKE_mod_list <- list(coeff = as.matrix(coef(SMOKE_mod)), 
                         formula = as.character(as.formula("SMOKE ~ AGE + SEX + RACE")))
  
  Mod_list <- list(Vital = Vital_mod_list,
                   Liver = Liver_mod_list,
                   Renal = Renal_mod_list,
                   Haematology = Haematology_mod_list,
                   Lipid = Lipid_mod_list,
                   Diabete = Diabete_mod_list,
                   Ferritin = Ferritin_mod_list,
                   BMI = BMI_mod_list,
                   SMOKE = SMOKE_mod_list,
                   MED_Count = MED_Count_mod_list)
  Sigma <- list(Vital = Vital_Sigma, 
                Liver = Liver_Sigma,
                Renal = Renal_Sigma, 
                Haematology = Haematology_Sigma,
                Lipid = Lipid_Sigma,
                Diabete = Diabete_Sigma,
                Ferritin = Ferritin_Sigma,
                BMI = BMI_Sigma)
  save(Mod_list, Sigma, file = "./data/labtest_parameters.RData",
       compress = TRUE)
}

####### STARTING SIMULATION.  SAVING FILES ########
if(!dir.exists('./data')){dir.create("./data")}
if(!dir.exists('./data/True')){dir.create("./data/True")}
replicate <- 500
n <- 2e4
if (file.exists("./data/data_generation_seed.RData")){
  load("./data/data_generation_seed.RData")
}else{
  seed <- sample(1:100000, 500)
  save(seed, file = "./data/data_generation_seed.RData")
}

for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  data <- suppressMessages({generateData(n, seed[i])})
  save(data, file = paste0("./data/True/", digit, ".RData"))
}


data$HbA1c_c  <- (data$HbA1c - 50) / 15
data$eGFR_c <- (data$eGFR - 60) / 20
data$BMI_c <- (data$BMI - 30) / 5
data$AGE_c <- (data$AGE - 60) / 15
data$SMOKE <- as.character(data$SMOKE)
data$HbA1c_STAR_c  <- (data$HbA1c_STAR - 50) / 15
data$eGFR_STAR_c <- (data$eGFR_STAR - 60) / 20
data$BMI_STAR_c <- (data$BMI_STAR - 30) / 5
data$SMOKE_STAR <- as.character(data$SMOKE_STAR)

fit.STAR <- coxph(Surv(T_I_STAR, EVENT_STAR) ~
                    poly(HbA1c_STAR_c, 2, raw = TRUE) + eGFR_STAR_c + BMI_STAR_c +
                    rs4506565_STAR + AGE_c + SEX +
                    INSURANCE + RACE + SMOKE_STAR +
                    HbA1c_STAR_c:AGE_c,
                  data = data)
fit.TRUE <- coxph(Surv(T_I, EVENT) ~
                    poly(HbA1c_c, 2, raw = TRUE) + eGFR_c + BMI_c +
                    rs4506565 + AGE_c + SEX +
                    INSURANCE + RACE + SMOKE +
                    HbA1c_c:AGE_c,
                  data = data)
exp(coef(fit.TRUE)) - exp(coef(fit.STAR))
ggplot(data) + geom_point(aes(x = T_I, y = T_I_STAR))
ggplot(data) +
  geom_point(aes(x = HbA1c, y = HbA1c_STAR))

