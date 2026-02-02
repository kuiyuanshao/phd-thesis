#################################### Generate Population ####################################
# T2D across the whole population
lapply(c("LDlinkR", "dplyr", "purrr", "mvtnorm", "stringr", "tidyr", "haven"), require, character.only = T)
source("00_utils_functions.R")
generateData <- function(n, seed){
  set.seed(seed)
  n <- n
  data <- data.frame(ID = as.character(1:n))
  # AGE: 18 ~ 90
  data$AGE <- pmin(90, pmax(25, rnorm(n, 62, 10)))
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
  data <- scaleVariables(data)
  data$WEIGHT <- round(data$BMI * (data$HEIGHT / 100)^2, 3)
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
  # GENOTYPES Measurement Errors:
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
  # T_I: Time Interval between Treatment Initiation SGLT2 and T2D Diagnosis (Months)
  mm_T_I <- model.matrix(~ I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) + 
                                rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) + 
                                SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE + 
                           I((HbA1c - 50) / 5):I((AGE - 50) / 5), data = data)
  betas_T_I <- log(c(1, 1.25, 1.1, 1.02, 1.04, 1.05, 0.9,
                     1.1, 1.2, 0.90, 0.90, 1, 0.9, 1.1, 0.85, 0.9, 0.85))
  eta_I <- as.vector(mm_T_I %*% betas_T_I)
  k <- 2
  lambda <- log(2) / (25 ^ k)
  T_I <- (-log(runif(n)) / (lambda*exp(eta_I)))^(1/k)
  scale_vec <- 10 * exp(0.05 * (data$AGE - 50) - 0.1 * as.numeric(data$URBAN))
  C_drop <- rweibull(n, shape = 1.5, scale = scale_vec)
  C <- pmin(C_drop, 24)
  # True Event Time
  data$T_I <- T_I
  data$C <- C_drop
  data$EVENT <- as.integer(T_I <= C)
  data <- add_measurement_errors(data)
  data$T_I <- pmin(T_I, C)
  data$C <- C
  
  cols_to_transform <- setdiff(data_info_srs$cat_vars, c("R", "EVENT", "EVENT_STAR"))
  data[cols_to_transform] <- lapply(data[cols_to_transform], as.factor)
  
  return (data)
}

add_measurement_errors <- function(data) {
  n <- nrow(data)
  # Linear Measurement Error Model: W = a0 + a1*X + a2*Z1 + ... + e
  apply_linear_error <- function(true_vec, slope = 1.0, intercept = 0.0, 
                                 z_list = NULL, z_coeffs = NULL, noise_sd = 0.1,
                                 min_val = 0.1) {
    
    mu_orig <- mean(true_vec, na.rm = TRUE)
    sd_orig <- sd(true_vec, na.rm = TRUE)
    val <- as.vector(scale(true_vec)) + intercept
    if (!is.null(z_list) && !is.null(z_coeffs)) {
      for (i in 1:length(z_list)) {
        vec_raw <- as.vector(as.numeric(z_list[[i]]))
        is_cat <- length(unique(vec_raw)) <= 20
        if (is_cat) {
          cov_z <- vec_raw
        } else {
          cov_z <- as.vector(scale(vec_raw))
        }
        val <- val + z_coeffs[i] * cov_z
      }
    }
    val <- (val + rnorm(length(true_vec), mean = 0, sd = noise_sd)) / slope
    val <- (val * sd_orig) + mu_orig
    k <- 2 
    val_smooth <- min_val + log1p(exp(k * (val - min_val))) / k
    val_final <- ifelse(val > min_val, val, val_smooth)
    
    return(val_final)
  }
  apply_cat_error <- function(true_vec, levels_vec, z_list = NULL, z_coeffs = NULL, base_accuracy = 0.80) {
    n_obs <- length(true_vec)
    logits <- rep(qlogis(base_accuracy), n_obs)
    for (i in 1:length(z_list)) {
      logits <- logits + z_coeffs[i] * as.vector(scale(as.numeric(z_list[[i]])))
    }
    p_correct <- plogis(logits)
    res <- true_vec
    is_error <- runif(n_obs) > p_correct
    if (sum(is_error) > 0) {
      err_idx <- which(is_error)
      for (idx in err_idx) {
        others <- levels_vec[levels_vec != true_vec[idx]]
        res[idx] <- others[sample(length(others), 1)]
      }
    }
    return(res)
  }
  # HbA1c_STAR ~
  data$HbA1c_STAR <- exp(apply_linear_error(
    true_vec = log(data$HbA1c), 
    slope = 1.2, 
    intercept = -2, 
    z_list = list(data$SEX, data$BMI, 
                  data$EDU,
                  as.numeric(data$INSURANCE == F), 
                  data$MED_Count,
                  as.numeric(data$RACE %in% c("SAS", "AMR"))),
    z_coeffs = c(-0.2, -0.2, 0.15, 0.1, -0.2, -0.2), 
    noise_sd = 0.5,
    min_val = 3.8))
  
  # eGFR: Recalculated from a very noisy Creatinine
  data$Creatinine_STAR <- data$Creatinine + rnorm(n, 0, sd(data$Creatinine) * 0.2)
  
  scr <- data$Creatinine_STAR / 88.4
  k <- ifelse(data$SEX, 0.9, 0.7); alpha <- ifelse(data$SEX, -0.411, -0.329)
  sex_f <- ifelse(data$SEX, 1.0, 1.018)
  data$eGFR_STAR <- 141 * (pmin(scr/k, 1)^alpha) * (pmax(scr/k, 1)^-1.209) * (0.993^data$AGE) * sex_f
  data$eGFR_STAR <- pmin(data$eGFR_STAR, 140)
  
  data$WEIGHT_STAR <- exp(apply_linear_error(log(data$WEIGHT), 
                                         slope = 1.1, 
                                         intercept = -1,
                                         z_list = list(data$BMI, 
                                                       data$AGE,
                                                       as.numeric(data$SEX == 1)),
                                         z_coeffs = c(0.05, 0.05, 0.05), 
                                         noise_sd = 0.35,
                                         min_val = 3.8))
  
  data$BMI_STAR <- data$WEIGHT_STAR / (data$HEIGHT / 100)^2
  
  data$Glucose_STAR <- exp(apply_linear_error(log(data$Glucose), 
                                          slope = 1.1, 
                                          intercept = 0, 
                                          z_list = list(data$HbA1c, 
                                                        data$BMI), 
                                          z_coeffs = c(-0.1, -0.05), 
                                          noise_sd = 0.5,
                                          min_val = 1.5))
  
  data$F_Glucose_STAR <- exp(apply_linear_error(log(data$F_Glucose), 
                                            slope = 1.05, 
                                            intercept = 0, 
                                            z_list = list(data$Insulin, 
                                                          data$AGE), 
                                            z_coeffs = c(-0.03, -0.01), 
                                            noise_sd = 0.6,
                                            min_val = 1.3))
  
  data$Insulin_STAR <- exp(apply_linear_error(log(data$Insulin), 
                                          slope = 1.05, 
                                          intercept = 0.05, 
                                          z_list = list(data$BMI, 
                                                        data$Glucose), 
                                          z_coeffs = c(0.04, 0.05), 
                                          noise_sd = 0.55,
                                          min_val = 0.7))
  
  data$Na_INTAKE_STAR <- exp(apply_linear_error(log(data$Na_INTAKE), 
                                            slope = 1.1, 
                                            intercept = -1, 
                                            z_list = list(data$BMI, 
                                                          as.numeric(data$HYPERTENSION)), 
                                            z_coeffs = c(0.05, 0.03), 
                                            noise_sd = 0.5,
                                            min_val = 0.7))
  
  data$K_INTAKE_STAR <- apply_linear_error(data$K_INTAKE, 
                                           slope = 1.2, 
                                           intercept = -1, 
                                           z_list = list(data$EDU), 
                                           z_coeffs = c(0.05), 
                                           noise_sd = 0.55,
                                           min_val = 6.9)
  
  data$KCAL_INTAKE_STAR <- apply_linear_error(data$KCAL_INTAKE, 
                                              slope = 1.3, 
                                              intercept = -0.5, 
                                              z_list = list(data$BMI, 
                                                            as.numeric(data$SEX == 2),
                                                            data$AGE), 
                                              z_coeffs = c(0.02, 0.15, 0.005), 
                                              noise_sd = 0.5,
                                              min_val = 7.1)
  
  data$PROTEIN_INTAKE_STAR <- apply_linear_error(data$PROTEIN_INTAKE, 
                                                 slope = 1.1, 
                                                 intercept = -0.5, 
                                                 z_list = list(data$BMI, 
                                                               as.numeric(data$SEX == 2)), 
                                                 z_coeffs = c(0.1, 0.1), 
                                                 noise_sd = 0.4,
                                                 min_val = 3.7)
  
  data$SBP_STAR <- apply_linear_error(data$SBP, 
                                      slope = 1.1, 
                                      intercept = 0, 
                                      z_list = list(data$AGE, 
                                                    data$BMI,
                                                    as.numeric(data$HYPERTENSION)), 
                                      z_coeffs = c(0.1, 0.2, 0.1), 
                                      noise_sd = 0.85,
                                      min_val = 90.0)
  
  data$HYPERTENSION_STAR <- with(data, SBP_STAR >= 140)
  
  data$Triglyceride_STAR <- exp(log(data$Triglyceride) + rnorm(dim(data)[1], 0, 0.2))
  
  data$EDU_STAR <- apply_linear_error(data$EDU, 
                                      slope = 1.2, 
                                      intercept = 1, 
                                      z_list = list(data$AGE, 
                                                    as.numeric(data$URBAN),
                                                    data$INCOME < 3), 
                                      z_coeffs = c(-0.03, -0.1, 0.05), 
                                      noise_sd = 0.7)
  data$EDU_STAR <- pmax(0, as.integer(data$EDU_STAR))
  # ============================================================================
  # NECESSARY CATEGORICAL CONTROLS & AUXILIARY Z
  # ============================================================================
  data$SMOKE_STAR <- apply_cat_error(data$SMOKE, 
                                     levels_vec = sort(unique(data$SMOKE)),
                                     base_accuracy = 0.75, 
                                     z_list = list(data$AGE - mean(data$AGE), 
                                                   as.numeric(data$SEX == 1),
                                                   data$EDU - mean(data$EDU)), 
                                     z_coeffs = c(0.5, -0.5, -0.5))
  data$INCOME_STAR <- apply_cat_error(data$INCOME, 
                                      levels_vec = sort(unique(data$INCOME)), 
                                      base_accuracy = 0.75, 
                                      z_list = list(data$EDU - mean(data$EDU), 
                                                    data$AGE - mean(data$AGE),
                                                    as.numeric(data$MARRIAGE == 1)), 
                                      z_coeffs = c(0.5, 0.2, -0.3))
  data$ALC_STAR <- apply_cat_error(data$ALC, 
                                   levels_vec = sort(unique(data$ALC)), 
                                   base_accuracy = 0.80, 
                                   z_list = list(as.numeric(data$SEX == 1), 
                                                 data$AGE - mean(data$AGE),
                                                 data$SMOKE == 1), 
                                   z_coeffs = c(-0.5, 0.2, 0.3))
  data$EXER_STAR <- apply_cat_error(data$EXER, 
                                    levels_vec = sort(unique(data$EXER)), 
                                    base_accuracy = 0.75, 
                                    z_list = list(data$BMI - mean(data$BMI), 
                                                  data$EDU - mean(data$EDU),
                                                  data$AGE - mean(data$AGE)), 
                                    z_coeffs = c(-0.4, 0.3, 0.2))
  ######## Survival
  C_STAR <- apply_linear_error(data$C, 
                               slope = 1.1, 
                               intercept = 0,
                               z_list = list(data$URBAN, 
                                             data$AGE,
                                             as.numeric(data$SMOKE == 1),
                                             data$BMI,
                                             as.numeric(data$INSURANCE == 1)),
                               z_coeffs = c(0.1, 0.05, 0.05, 0.02, 0.01), 
                               noise_sd = 0.25,
                               min_val = 0)
  C_STAR <- pmin(C_STAR, 24)
  data$C_STAR <- C_STAR
  data$T_I_STAR <- pmin(data$T_I, C_STAR)
  data$EVENT_STAR <- as.integer(data$T_I <= C_STAR)
  
  return(data)
}

scaleVariables <- function(data){
  # --- Helper Function: Soft Range Fitting ---
  # This maps a standardized variable (Z-score) to a strictly bounded range [min, max]
  # It uses a sigmoid curve to ensure no hard cuts, while adjusting the slope and offset
  
  soft_fit <- function(val, target_mean, target_sd, min_val, max_val) {
    z <- as.numeric(scale(val))
    range_width <- max_val - min_val
    p_mean <- (target_mean - min_val) / range_width
    p_mean <- pmin(pmax(p_mean, 0.001), 0.999)
    
    offset <- qlogis(p_mean) 
    
    slope <- target_sd / (range_width * p_mean * (1 - p_mean))
    return(min_val + range_width * plogis(offset + z * slope))
  }
  
  # --- SoftMax Helper for Glucose Dependency ---
  # Smoothly ensures Random Glucose >= Fasting Glucose without a hard cut
  soft_max_2var <- function(a, b) {
    k <- pmax(a, b)
    return(k + log(exp(a - k) + exp(b - k)))
  }
  # Vitals
  data$SBP <- soft_fit(data$SBP, target_mean = 138, target_sd = 15, min = 90, max = 190)
  data$HYPERTENSION <- with(data, SBP >= 140)
  
  data$Temperature <- soft_fit(data$Temperature, target_mean = 36.8, target_sd = 0.4, min = 35.5, max = 39.0)
  data$HR          <- soft_fit(data$HR, target_mean = 76, target_sd = 12, min = 45, max = 120)
  data$SpO2        <- soft_fit(data$SpO2, target_mean = 97, target_sd = 1.5, min = 88, max = 100)
  
  # Labs - Glucose & Metabolism
  data$Ferritin    <- soft_fit(data$Ferritin, target_mean = 150, target_sd = 100, min = 10, max = 1000)
  
  # Note: Added biological Max for Glucose/Insulin to facilitate the smooth curve (S-shape)
  data$F_Glucose   <- soft_fit(data$F_Glucose, target_mean = 9.5, target_sd = 2.5, min = 4.0, max = 40.0)
  
  # First fit Glucose loosely
  data$Glucose     <- soft_fit(data$Glucose, target_mean = 11, target_sd = 3.5, min = 4.5, max = 50.0)
  # Then apply soft constraint: Glucose must be >= F_Glucose
  data$Glucose     <- soft_max_2var(data$Glucose, data$F_Glucose)
  
  data$HbA1c       <- soft_fit(data$HbA1c, target_mean = 65, target_sd = 10, min = 48, max = 150) # Max 150 added
  data$Insulin     <- soft_fit(data$Insulin, target_mean = 18, target_sd = 10, min = 2.0, max = 100.0) # Max 100 added
  
  # Kidney Function
  data$Creatinine  <- soft_fit(data$Creatinine, target_mean = 88.0, target_sd = 25.0, min = 45.0, max = 300.0)
  data$Urea        <- soft_fit(data$Urea, target_mean = 7.0, target_sd = 3.0, min = 2.0, max = 50.0) # Max 50 added
  
  # Electrolytes
  data$Potassium   <- soft_fit(data$Potassium, target_mean = 4.2, target_sd = 0.4, min = 3.0, max = 6.5)
  data$Sodium      <- soft_fit(data$Sodium, target_mean = 139.0, target_sd = 3.0, min = 125.0, max = 155.0)
  data$Chloride    <- soft_fit(data$Chloride, target_mean = 102.0, target_sd = 3.0, min = 85.0, max = 120.0)
  data$Bicarbonate <- soft_fit(data$Bicarbonate, target_mean = 25.0, target_sd = 2.5, min = 15.0, max = 35.0)
  data$Calcium     <- soft_fit(data$Calcium, target_mean = 2.4, target_sd = 0.1, min = 1.8, max = 3.0)
  data$Magnesium   <- soft_fit(data$Magnesium, target_mean = 0.8, target_sd = 0.1, min = 0.4, max = 1.5)
  data$Phosphate   <- soft_fit(data$Phosphate, target_mean = 1.1, target_sd = 0.2, min = 0.5, max = 2.5)
  
  # Lipids
  data$Triglyceride <- soft_fit(data$Triglyceride, target_mean = 2.1, target_sd = 1.0, min = 0.5, max = 15.0)
  data$HDL          <- soft_fit(data$HDL, target_mean = 1.1, target_sd = 0.3, min = 0.5, max = 5.0)
  data$LDL          <- soft_fit(data$LDL, target_mean = 2.8, target_sd = 0.8, min = 1.0, max = 10.0)
  
  # Blood Count (CBC)
  data$Hb          <- soft_fit(data$Hb, target_mean = 135.0, target_sd = 15.0, min = 80.0, max = 180.0)
  data$HCT         <- soft_fit(data$HCT, target_mean = 0.41, target_sd = 0.04, min = 0.25, max = 0.55)
  data$RBC         <- soft_fit(data$RBC, target_mean = 4.8, target_sd = 0.5, min = 3.0, max = 7.0)
  data$WBC         <- soft_fit(data$WBC, target_mean = 7.5, target_sd = 2.0, min = 2.0, max = 30.0)
  data$Platelet    <- soft_fit(data$Platelet, target_mean = 250.0, target_sd = 60.0, min = 50.0, max = 800.0)
  data$MCV         <- soft_fit(data$MCV, target_mean = 90.0, target_sd = 5.0, min = 60.0, max = 120.0)
  data$RDW         <- soft_fit(data$RDW, target_mean = 13.5, target_sd = 1.5, min = 10.0, max = 25.0)
  
  # Differential Count
  data$Neutrophils <- soft_fit(data$Neutrophils, target_mean = 4.5, target_sd = 1.5, min = 1.0, max = 15.0)
  data$Lymphocytes <- soft_fit(data$Lymphocytes, target_mean = 2.2, target_sd = 0.6, min = 0.5, max = 10.0)
  data$Monocytes   <- soft_fit(data$Monocytes, target_mean = 0.6, target_sd = 0.2, min = 0.1, max = 3.0)
  data$Eosinophils <- soft_fit(data$Eosinophils, target_mean = 0.2, target_sd = 0.1, min = 0.0, max = 2.0)
  data$Basophils   <- soft_fit(data$Basophils, target_mean = 0.05, target_sd = 0.02, min = 0.0, max = 0.5)
  
  # Liver Function
  data$AST         <- soft_fit(data$AST, target_mean = 32, target_sd = 18, min = 10, max = 150)
  data$ALT         <- soft_fit(data$ALT, target_mean = 38, target_sd = 22, min = 10, max = 180)
  data$ALP         <- soft_fit(data$ALP, target_mean = 85, target_sd = 25, min = 30, max = 200)
  data$GGT         <- soft_fit(data$GGT, target_mean = 55, target_sd = 40, min = 10, max = 300)
  data$Bilirubin   <- soft_fit(data$Bilirubin, target_mean = 12, target_sd = 6, min = 3, max = 50)
  data$Albumin     <- soft_fit(data$Albumin, target_mean = 42, target_sd = 4, min = 25, max = 55)
  data$Globulin    <- soft_fit(data$Globulin, target_mean = 28, target_sd = 4, min = 15, max = 45)
  data$Protein     <- soft_fit(data$Protein, target_mean = 70, target_sd = 5, min = 50, max = 90)
  
  # Dietary Intake
  data$Na_INTAKE      <- soft_fit(data$Na_INTAKE, target_mean = 8.2, target_sd = 0.4, min = 5, max = 9.2)
  data$K_INTAKE       <- soft_fit(data$K_INTAKE, target_mean = 7.9, target_sd = 0.3, min = 5, max = 8.5)
  data$KCAL_INTAKE    <- soft_fit(data$KCAL_INTAKE, target_mean = 7.7, target_sd = 0.2, min = 5, max = 8.3)
  data$PROTEIN_INTAKE <- soft_fit(data$PROTEIN_INTAKE, target_mean = 4.45, target_sd = 0.25, min = 3.7, max = 7)
  
  return (data)
}

loadGenotypeInfo <- function(){
  rsids <- data.frame(refsnp_id = c("rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                    "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                    "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806"),
                      chr = c(9, 6, 3, 3, 9, 
                              10, 3, 3, 6, 11,
                              10, 6, 10, 11, 3))
  unique_chrs <- unique(rsids$chr)
  race <- c(unique(list_pop()$super_pop_code)[-1])
  retrieveFreq <- function(rsid, race){
    LDhap(snps = rsid,
          pop = race,
          token = "979f246a6b57")
  }
  retrieveCorr <- function(rsids, chr, race){
    LDmatrix(snp = rsids$refsnp_id[rsids$chr == chr],
             pop = race, 
             r2d = "r2",
             token = "979f246a6b57") 
  }
  if (!file.exists("./data/genoFreq.RData")){
    genoFreq <- vector("list", 150)
    m <- 1
    for (i in rsids$refsnp_id){
      for (j in race){
        mat <- retrieveFreq(i, j) %>%
          dplyr::rename(allele = i) %>%
          mutate(Race = j, refsnp_id = i)
        genoFreq[[m]] <- mat
        m <- m + 1
      }
    }
    genoFreq <- bind_rows(genoFreq)
    genoFreq <- merge(genoFreq, rsids)
    save(genoFreq, file = "./data/genoFreq.RData")
  }else{
    load("./data/genoFreq.RData")
  }
  if (!file.exists("./data/genoCorr.RData")){
    genoCorr <- list()
    for (i in unique_chrs){
      chr_list <- vector("list", 5)
      if (sum(rsids$chr == i) > 1){
        for (j in race){
          mat <- retrieveCorr(rsids, i, j) %>%
            mutate(Race = j, chr = i)
          chr_list[[j]] <- mat
        }
        genoCorr[[as.character(i)]] <- bind_rows(chr_list) 
      }
    }
    save(genoCorr, file = "./data/genoCorr.RData")
  }else{
    load("./data/genoCorr.RData")
  }
  if (!file.exists("./data/genoTrait.RData")){
    pat <- "^[+-]?[0-9]+(?:\\.[0-9]+)?-[+-]?[0-9]+(?:\\.[0-9]+)?"
    trait <- LDtrait(snps = rsids$refsnp_id,
                     pop = "ALL",
                     win_size = "1",
                     token = "979f246a6b57")
    trait <- trait %>%
      mutate(Beta_or_OR = str_extract(Beta_or_OR, pat)) %>%
      separate(Beta_or_OR, into = c("low", "high"), sep = "-", convert = TRUE) %>%
      mutate(Beta = (low + high) / 2)
    phenos <- list(
      BMI    = "Body mass index",
      T2D    = c("Type 2 diabetes",
                 "Type ii diabetes",
                 "Type 2 diabetes (PheCode 250.2)",
                 "Type 2 diabetes with renal manifestations (PheCode 250.22)",
                 "Type 2 diabetes with ophthalmic manifestations (PheCode 250.23)",
                 "Type 2 diabetes with neurological manifestations (PheCode 250.24)"),
      HCT    = "hematocrit (maximum, inv-norm transformed)",
      SBP    = "Systolic blood pressure",
      Med    = c("Medication use (drugs used in diabetes)",
                 "Takes medication for Diabetes/sugar?"),
      Glucose = "Glucose (finger stick, mean, inv-norm transformed)",
      F_Glucose= c("Fasting blood glucose",
                   "Fasting plasma glucose",
                   "Fasting glucose"),
      LDL    = c("low density lipoprotein cholesterol (LDLC, mean, inv-norm transformed)"),
      HbA1c  = "Hemoglobin A1c (HbA1c, mean, inv-norm transformed)",
      HEIGHT = "Height",
      Triglyceride = "Triglyceride levels",
      ALT    = c("Alanine levels (UKB data field 23460)", "Alanine levels",
                 "Alanine aminotransferase levels"),
      ALP    = "Serum alkaline phosphatase levels",
      Albumin= "Serum albumin levels",
      Insulin= "Fasting insulin"
    )
    all_mats <- imap(phenos, function(trait_names, pheno_code) {
      trait %>%
        filter(GWAS_Trait %in% trait_names) %>%
        group_by(Query) %>%
        summarise(Beta = mean(Beta, na.rm = TRUE), .groups = "drop") %>%
        mutate(PHENO = pheno_code)
    })
    genoTrait <- bind_rows(all_mats)
    save(genoTrait, file = "./data/genoTrait.RData")
  }else{
    load("./data/genoTrait.RData")
  }
  return (list(rsids = rsids, genoFreq = genoFreq, genoCorr = genoCorr, genoTrait = genoTrait))
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


# fit.STAR <- coxph(Surv(T_I_STAR, EVENT_STAR) ~
#                     I((HbA1c_STAR - 50) / 5) + I(I((HbA1c_STAR - 50) / 5)^2) +
#                     I((HbA1c_STAR - 50) / 5):I((AGE - 50) / 5) +
#                     rs4506565_STAR + I((AGE - 50) / 5) + I((eGFR_STAR - 90) / 10) +
#                     SEX + INSURANCE + RACE + I(BMI_STAR / 5) + SMOKE_STAR,
#                   data = data)
# fit.TRUE <- coxph(Surv(T_I, EVENT) ~
#                     I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
#                     I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
#                     rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
#                     SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
#                   data = data)
# exp(coef(fit.TRUE)) - exp(coef(fit.STAR))

