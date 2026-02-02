dimnames.twophase2 <- function(object,...) dimnames(object$phase1$sample$variables)
calibrateFun <- function(samp, design = "Else"){
  dimnames.twophase2 <- function(object,...) dimnames(object$phase1$sample$variables)
  samp$R <- as.logical(samp$R)
  
  if (design != "SRS"){
    n_sampled <- table(samp$STRATA, samp$R)
    if (any(n_sampled[, 2] == 0)){
      idx <- which(n_sampled[, 2] == 0)
      samp$STRATA[samp$STRATA == rownames(n_sampled)[idx]] <- rownames(n_sampled)[idx - 1]
    }
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~STRATA), 
                             subset = ~R, data = samp)
  }else{
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, NULL), 
                             subset = ~R, data = samp)
  }
  #twophase_des_2 <- subset(svydesign(ids = ~1, strata = ~STRATA, weights = ~W, data = samp), R)
  
  modimp.HbA1c <- svyglm(HbA1c ~ HbA1c_STAR + AGE + SEX + RACE + BMI + SMOKE_STAR + SBP +
                           Glucose_STAR + F_Glucose_STAR + Insulin + INSURANCE + T_I_STAR +
                           EVENT_STAR + MED_Count + rs10811661_STAR + rs11708067_STAR + eGFR, 
                         family = "gaussian", design = twophase_des)
  samp$HbA1c_impute <- as.vector(predict(modimp.HbA1c, newdata = samp, 
                                         type = "response", se.fit = FALSE))
  modimp.SMOKE <- svy_vglm(SMOKE ~ SMOKE_STAR + AGE + SEX + RACE + EXER_STAR + ALC_STAR +
                             BMI + HbA1c_STAR + Ferritin + INSURANCE + T_I_STAR + 
                             Protein + Globulin + Albumin + Bilirubin + GGT + ALP + 
                             ALT + AST + Basophils + Eosinophils + Monocytes + Lymphocytes +
                             Neutrophils + RDW + MCV + Platelet + WBC + RBC + HCT + Hb +
                             LDL + HDL + Triglyceride + Phosphate + Magnesium + Calcium + 
                             Bicarbonate + Potassium + Urea + eGFR + Creatinine + SpO2 + 
                             HR + Temperature + SBP + MED_Count, 
                           family = "multinomial", design = twophase_des)
  samp$SMOKE_impute <- apply(predict(modimp.SMOKE$fit, newdata = samp, 
                                     type = "response", se.fit = FALSE), 1, which.max)
  modimp.rs4506565 <- svy_vglm(rs4506565 ~ rs4506565_STAR + T_I_STAR + 
                                 AGE + SEX + RACE + INSURANCE + HbA1c_STAR + 
                                 SMOKE_STAR + EVENT_STAR + eGFR + F_Glucose_STAR +
                                 Albumin + BMI, 
                               family = "multinomial", design = twophase_des)
  samp$rs4506565_impute <- apply(predict(modimp.rs4506565$fit, newdata = samp, 
                                         type = "response", se.fit = FALSE), 1, which.max)
  modimp.EVENT <- svyglm(EVENT ~ EVENT_STAR + HbA1c_STAR + rs4506565_STAR + 
                           AGE + SEX + INSURANCE + RACE + BMI + SMOKE_STAR + T_I_STAR + eGFR, 
                         family = "binomial", design = twophase_des)
  samp$EVENT_impute <- round(as.vector(predict(modimp.EVENT, newdata = samp, 
                                               type = "response", se.fit = FALSE)))
  
  modimp.T_I <- svyglm(T_I ~ EVENT_STAR + HbA1c_STAR + rs4506565_STAR + 
                         AGE + SEX + INSURANCE + RACE + BMI + SMOKE_STAR + T_I_STAR + eGFR, 
                       family = "gaussian", design = twophase_des)
  samp$T_I_impute <- as.vector(predict(modimp.T_I, newdata = samp, 
                                       type = "response", se.fit = FALSE))
  
  phase1model_imp <- coxph(Surv(T_I_impute, EVENT_impute) ~ I((HbA1c_impute - 50) / 5) + 
                             rs4506565_impute + I((AGE - 50) / 5) + I((eGFR - 90) / 10) + SEX + INSURANCE + 
                             RACE + I(BMI / 5) + SMOKE_impute, data = samp)
  inffun_imp <- residuals(phase1model_imp, type = "dfbeta")
  colnames(inffun_imp) <- paste0("if", 1:ncol(inffun_imp))
  
  if (design != "SRS"){
    twophase_des_imp <- twophase(id = list(~1, ~1), strata = list(NULL, ~STRATA), 
                                 subset = ~R, data = cbind(samp, inffun_imp))
  }else{
    twophase_des_imp <- twophase(id = list(~1, ~1), strata = list(NULL, NULL), 
                                 subset = ~R, data = cbind(samp, inffun_imp))
  }
  califormu <- make.formula(colnames(inffun_imp)) 
  cali_twophase_imp <- survey::calibrate(twophase_des_imp, califormu, phase = 2, calfun = "raking")
  
  rakingest <- svycoxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) + SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE, design = cali_twophase_imp)
  
  return (rakingest)
}
