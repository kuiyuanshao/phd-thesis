calibrateFun <- function(samp, design = "Else", outcome = "gauss"){
  dimnames.twophase2 <- function(object,...) dimnames(object$phase1$sample$variables)
  samp$R <- as.logical(samp$R)
  
  if (design %in% c("Neyman_ODS", "Neyman_INF", "Neyman_ODS_UNVAL", "Neyman_INF_UNVAL")) {
    n_sampled <- table(samp$outcome_strata, samp$R)
    if (any(n_sampled[, 2] == 0)){
      idx <- which(n_sampled[, 2] == 0)
      samp$outcome_strata[samp$outcome_strata == rownames(n_sampled)[idx]] <- rownames(n_sampled)[idx - 1]
    }
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~outcome_strata), 
                             subset = ~R, data = samp)
  } else {
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, NULL), 
                             subset = ~R, data = samp)
  }
  twophase_des$variables <- twophase_des$phase1$sample$variables
  # =====================================================================
  # UNIFIED IMPUTATION MODELS (Shared across all outcomes)
  # =====================================================================
  # We include CD4_COUNT_1Y_sqrt_v, TIME_nv, and ANY_OI_nv in the predictors 
  # to ensure congeniality for the Gauss, Cox, and Binomial models simultaneously.
  
  modimp.AGE <- svyglm(AGE_AT_MED_START_v ~ AGE_AT_MED_START_nv + ART_SOURCE_nv + 
                         CD4_COUNT_1Y_sqrt_v + TIME_nv + ANY_OI_nv + 
                         CD4_COUNT_BSL_sqrt_v + SEX + YEAR_OF_ENROLLMENT, 
                       family = "gaussian", design = twophase_des)
  samp$AGE_impute <- as.vector(predict(modimp.AGE, newdata = samp, type = "response", se.fit = FALSE))
  
  modimp.VL <- svyglm(VL_COUNT_BSL_LOG_v ~ VL_COUNT_BSL_LOG_nv + AGE_AT_MED_START_nv + 
                        CD4_COUNT_1Y_sqrt_v + TIME_nv + ANY_OI_nv + 
                        CD4_COUNT_BSL_sqrt_v + SEX + YEAR_OF_ENROLLMENT, 
                      family = "gaussian", design = twophase_des)
  samp$VL_impute <- as.vector(predict(modimp.VL, newdata = samp, type = "response", se.fit = FALSE))
  
  modimp.OI <- svyglm(ANY_OI_v ~ ANY_OI_nv + AGE_AT_MED_START_nv + 
                        CD4_COUNT_1Y_sqrt_v + TIME_nv + 
                        CD4_COUNT_BSL_sqrt_v + SEX + YEAR_OF_ENROLLMENT, 
                      family = "binomial", design = twophase_des)
  samp$OI_impute <- round(as.vector(predict(modimp.OI, newdata = samp, type = "response", se.fit = FALSE)))
  
  modimp.TIME <- svyglm(TIME_v ~ TIME_nv + ANY_OI_nv + AGE_AT_MED_START_nv + 
                          CD4_COUNT_1Y_sqrt_v + 
                          CD4_COUNT_BSL_sqrt_v + SEX + YEAR_OF_ENROLLMENT, 
                        family = "gaussian", design = twophase_des)
  samp$TIME_impute <- as.vector(predict(modimp.TIME, newdata = samp, type = "response", se.fit = FALSE))
  
  
  # =====================================================================
  # PHASE 1 TARGET WORKING MODELS
  # =====================================================================
  if (outcome == "gauss") {
    phase1model_imp <- glm(CD4_COUNT_1Y_sqrt_v ~ VL_impute + CD4_COUNT_BSL_sqrt_v + SEX + AGE_impute, 
                           data = samp, family = "gaussian")
    inffun_imp <- dfbeta(phase1model_imp)
    
  } else if (outcome == "bin") {
    phase1model_imp <- glm(OI_impute ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_impute + YEAR_OF_ENROLLMENT, 
                           data = samp, family = "binomial")
    inffun_imp <- dfbeta(phase1model_imp)
    
  } else if (outcome == "cox") {
    phase1model_imp <- coxph(Surv(TIME_impute, OI_impute) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_impute + YEAR_OF_ENROLLMENT, 
                             data = samp)
    inffun_imp <- residuals(phase1model_imp, type = "dfbeta")
  } else {
    stop("Outcome must be 'gauss', 'bin', or 'cox'")
  }
  
  
  # =====================================================================
  # CALIBRATION & RAKING
  # =====================================================================
  colnames(inffun_imp) <- paste0("if", 1:ncol(inffun_imp))
  
  if (design %in% c("Neyman_ODS", "Neyman_INF", "Neyman_ODS_UNVAL", "Neyman_INF_UNVAL")){
    twophase_des_imp <- twophase(id = list(~1, ~1), strata = list(NULL, ~outcome_strata), 
                                 subset = ~R, data = cbind(samp, inffun_imp))
  } else {
    twophase_des_imp <- twophase(id = list(~1, ~1), strata = list(NULL, NULL), 
                                 subset = ~R, data = cbind(samp, inffun_imp))
  }
  
  califormu <- make.formula(colnames(inffun_imp)) 
  cali_twophase_imp <- survey::calibrate(twophase_des_imp, califormu, phase = 2, calfun = "raking")
  
  if (outcome == "gauss") {
    rakingest <- svyglm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_v + CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v, 
                        design = cali_twophase_imp, family = "gaussian")
  } else if (outcome == "bin") {
    rakingest <- svyglm(ANY_OI_v ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                        design = cali_twophase_imp, family = "binomial")
  } else if (outcome == "cox") {
    rakingest <- svycoxph(Surv(TIME_v, ANY_OI_v) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                          design = cali_twophase_imp)
  }
  
  return(rakingest)
}