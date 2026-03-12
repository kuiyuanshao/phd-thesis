calibrateFunOE <- function(samp, design = "Else"){
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
  twophase_des$variables <- twophase_des$phase1$sample$variables
  # eGFR: Driven by Creatinine, AGE, SEX.
  # Creatinine driven by SEX, SMOKE, RACE, AGE, BMI, MED_Count.
  modimp.eGFR <- svyglm(eGFR ~ eGFR_STAR + Creatinine_STAR + AGE + SEX + RACE +
                          SMOKE_STAR + BMI_STAR + MED_Count,
                        family = "gaussian", design = twophase_des)
  samp$eGFR_impute <- as.vector(predict(modimp.eGFR, newdata = samp,
                                        type = "response", se.fit = FALSE))

  # BMI: True ~ SEX+SMOKE+RACE+AGE+Genotypes.
  # Error (WEIGHT_STAR) depends on EXER, INCOME, HEIGHT.
  modimp.BMI <- svyglm(BMI ~ BMI_STAR + SEX + SMOKE_STAR + RACE + AGE + HEIGHT +
                         EXER_STAR + INCOME_STAR + rs10811661_STAR +
                         rs11708067_STAR + rs4506565_STAR + rs5219_STAR,
                       family = "gaussian", design = twophase_des)
  samp$BMI_impute <- as.vector(predict(modimp.BMI, newdata = samp,
                                       type = "response", se.fit = FALSE))

  # Insulin: True ~ rs11708067.
  # Error depends on BMI, Glucose, AGE. also invovle the interaction term.
  modimp.Insulin <- svyglm(Insulin ~ Insulin_STAR + rs11708067_STAR +
                             BMI_STAR + Glucose_STAR + F_Glucose_STAR + AGE +
                             T_I_STAR * AGE + EVENT_STAR * AGE,
                           family = "gaussian", design = twophase_des)
  samp$Insulin_impute <- as.vector(predict(modimp.Insulin, newdata = samp,
                                           type = "response", se.fit = FALSE))

  # HbA1c: True ~ SEX+SMOKE+RACE+AGE+BMI+MED_Count+Genotypes.
  # Error depends on INSURANCE, INCOME.
  # Also predicts Glucose_STAR and F_Glucose_STAR (so we use them to reverse-infer).
  modimp.HbA1c <- svyglm(HbA1c ~ HbA1c_STAR + SEX + SMOKE_STAR + RACE + AGE +
                           BMI_STAR + MED_Count + rs10811661_STAR + rs11708067_STAR +
                           Glucose_STAR + F_Glucose_STAR + INSURANCE + INCOME_STAR +
                           SBP + KCAL_INTAKE_STAR + Insulin_STAR,
                         family = "gaussian", design = twophase_des)
  samp$HbA1c_impute <- as.vector(predict(modimp.HbA1c, newdata = samp,
                                         type = "response", se.fit = FALSE))

  # SMOKE: True ~ AGE+SEX+RACE.
  # Error depends on EDU, INCOME, SpO2, ALC.
  # Predicts EXER, BMI, MED_Count.
  modimp.SMOKE <- svy_vglm(SMOKE ~ SMOKE_STAR + AGE + SEX + RACE + EDU_STAR +
                             INCOME_STAR + SpO2 + ALC_STAR + EXER_STAR +
                             BMI_STAR + MED_Count,
                           family = "multinomial", design = twophase_des)
  samp$SMOKE_impute <- apply(predict(modimp.SMOKE$fit, newdata = samp,
                                     type = "response", se.fit = FALSE), 1, which.max)

  # rs4506565: Correlated with RACE.
  # Downstream causes BMI, Albumin, F_Glucose, and T_I (EVENT).
  modimp.rs4506565 <- svy_vglm(rs4506565 ~ rs4506565_STAR + RACE + BMI_STAR +
                                 Albumin + F_Glucose_STAR + T_I_STAR + EVENT_STAR,
                               family = "multinomial", design = twophase_des)
  samp$rs4506565_impute <- apply(predict(modimp.rs4506565$fit, newdata = samp,
                                         type = "response", se.fit = FALSE), 1, which.max)

  # EVENT & T_I: True ~ HbA1c, rs4506565, AGE, eGFR, Insulin, BMI, SEX, INSURANCE, RACE, SMOKE.
  # Error (C_STAR) depends on URBAN, AGE, BMI, INSURANCE.
  modimp.EVENT <- svyglm(EVENT ~ EVENT_STAR + T_I_STAR + HbA1c_STAR + rs4506565_STAR +
                           AGE + eGFR_STAR + Insulin_STAR + BMI_STAR + SEX +
                           INSURANCE + RACE + SMOKE_STAR + URBAN + AGE:Insulin_STAR,
                         family = "binomial", design = twophase_des)
  samp$EVENT_impute <- round(as.vector(predict(modimp.EVENT, newdata = samp,
                                               type = "response", se.fit = FALSE)))

  modimp.T_I <- svyglm(T_I ~ EVENT_STAR + T_I_STAR + HbA1c_STAR + rs4506565_STAR +
                         AGE + eGFR_STAR + Insulin_STAR + BMI_STAR + SEX +
                         INSURANCE + RACE + SMOKE_STAR + URBAN + AGE:Insulin_STAR,
                       family = "gaussian", design = twophase_des)
  samp$T_I_impute <- as.vector(predict(modimp.T_I, newdata = samp,
                                       type = "response", se.fit = FALSE))

  phase1model_imp <- coxph(Surv(T_I_impute, EVENT_impute) ~ I((HbA1c_impute - 50) / 5) +
                             rs4506565_impute + I((AGE - 60) / 5) + I((eGFR_impute - 75) / 10) +
                             I((Insulin_impute - 15) / 2) + I((BMI_impute - 28) / 2) + SEX + INSURANCE +
                             RACE + SMOKE_impute + I((AGE - 60) / 5):I((Insulin_impute - 15) / 2),
                           data = samp)

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
                          rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                          I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE +
                          RACE + SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2),
                        design = cali_twophase_imp)

  return (rakingest)
}


calibrateFunE <- function(samp, design = "Else"){
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
  twophase_des$variables <- twophase_des$phase1$sample$variables
  # eGFR: Driven by Creatinine, AGE, SEX.
  # Creatinine driven by SEX, SMOKE, RACE, AGE, BMI, MED_Count.
  modimp.eGFR <- svyglm(eGFR ~ eGFR_STAR + Creatinine_STAR + AGE + SEX + RACE +
                          SMOKE_STAR + BMI_STAR + MED_Count,
                        family = "gaussian", design = twophase_des)
  samp$eGFR_impute <- as.vector(predict(modimp.eGFR, newdata = samp,
                                        type = "response", se.fit = FALSE))

  # BMI: True ~ SEX+SMOKE+RACE+AGE+Genotypes.
  # Error (WEIGHT_STAR) depends on EXER, INCOME, HEIGHT.
  modimp.BMI <- svyglm(BMI ~ BMI_STAR + SEX + SMOKE_STAR + RACE + AGE + HEIGHT +
                         EXER_STAR + INCOME_STAR + rs10811661_STAR +
                         rs11708067_STAR + rs4506565_STAR + rs5219_STAR,
                       family = "gaussian", design = twophase_des)
  samp$BMI_impute <- as.vector(predict(modimp.BMI, newdata = samp,
                                       type = "response", se.fit = FALSE))

  # Insulin: True ~ rs11708067.
  # Error depends on BMI, Glucose, AGE. also invovle the interaction term.
  modimp.Insulin <- svyglm(Insulin ~ Insulin_STAR + rs11708067_STAR +
                             BMI_STAR + Glucose_STAR + F_Glucose_STAR + AGE +
                             T_I * AGE + EVENT * AGE,
                           family = "gaussian", design = twophase_des)
  samp$Insulin_impute <- as.vector(predict(modimp.Insulin, newdata = samp,
                                           type = "response", se.fit = FALSE))

  # HbA1c: True ~ SEX+SMOKE+RACE+AGE+BMI+MED_Count+Genotypes.
  # Error depends on INSURANCE, INCOME.
  # Also predicts Glucose_STAR and F_Glucose_STAR (so we use them to reverse-infer).
  modimp.HbA1c <- svyglm(HbA1c ~ HbA1c_STAR + SEX + SMOKE_STAR + RACE + AGE +
                           BMI_STAR + MED_Count + rs10811661_STAR + rs11708067_STAR +
                           Glucose_STAR + F_Glucose_STAR + INSURANCE + INCOME_STAR +
                           SBP + KCAL_INTAKE_STAR + Insulin_STAR,
                         family = "gaussian", design = twophase_des)
  samp$HbA1c_impute <- as.vector(predict(modimp.HbA1c, newdata = samp,
                                         type = "response", se.fit = FALSE))

  # SMOKE: True ~ AGE+SEX+RACE.
  # Error depends on EDU, INCOME, SpO2, ALC.
  # Predicts EXER, BMI, MED_Count.
  modimp.SMOKE <- svy_vglm(SMOKE ~ SMOKE_STAR + AGE + SEX + RACE + EDU_STAR +
                             INCOME_STAR + SpO2 + ALC_STAR + EXER_STAR +
                             BMI_STAR + MED_Count,
                           family = "multinomial", design = twophase_des)
  samp$SMOKE_impute <- apply(predict(modimp.SMOKE$fit, newdata = samp,
                                     type = "response", se.fit = FALSE), 1, which.max)

  # rs4506565: Correlated with RACE.
  # Downstream causes BMI, Albumin, F_Glucose, and T_I (EVENT).
  modimp.rs4506565 <- svy_vglm(rs4506565 ~ rs4506565_STAR + RACE + BMI_STAR +
                                 Albumin + F_Glucose_STAR + T_I + EVENT,
                               family = "multinomial", design = twophase_des)
  samp$rs4506565_impute <- apply(predict(modimp.rs4506565$fit, newdata = samp,
                                         type = "response", se.fit = FALSE), 1, which.max)

  phase1model_imp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c_impute - 50) / 5) +
                             rs4506565_impute + I((AGE - 60) / 5) + I((eGFR_impute - 75) / 10) +
                             I((Insulin_impute - 15) / 2) + I((BMI_impute - 28) / 2) + SEX + INSURANCE +
                             RACE + SMOKE_impute + I((AGE - 60) / 5):I((Insulin_impute - 15) / 2),
                           data = samp)

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
                          rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                          I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE +
                          RACE + SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2),
                        design = cali_twophase_imp)

  return (rakingest)
}