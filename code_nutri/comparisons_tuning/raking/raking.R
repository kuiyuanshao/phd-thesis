calibrateFun <- function(samp, design = "Else", outcome = "sbp"){
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

  modimp.ln_na <- svyglm(ln_na_true ~ ln_na_avg + c_age + c_bmi + high_chol + usborn +
                           female + bkg_o + bkg_pr + sbp * c_age + hypertension * c_age,
                         family = "gaussian", design = twophase_des)

  samp$ln_na_true_impute <- as.vector(predict(modimp.ln_na, newdata = samp,
                                              type = "response", se.fit = FALSE))

  if (outcome == "sbp") {
    phase1model_imp <- glm(sbp ~ ln_na_true_impute * c_age + c_bmi + high_chol +
                             usborn + female + bkg_o + bkg_pr,
                           data = samp, family = "gaussian")
  } else if (outcome == "hypertension") {
    phase1model_imp <- glm(hypertension ~ ln_na_true_impute * c_age + c_bmi + high_chol +
                             usborn + female + bkg_o + bkg_pr,
                           data = samp, family = "quasibinomial")
  } else {
    stop("Outcome must be either 'sbp' or 'hypertension'")
  }

  inffun_imp <- dfbeta(phase1model_imp)
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

  if (outcome == "sbp") {
    rakingest <- svyglm(sbp ~ ln_na_true * c_age + c_bmi + high_chol + 
                          usborn + female + bkg_o + bkg_pr, 
                        design = cali_twophase_imp, family = "gaussian")
  } else {
    rakingest <- svyglm(hypertension ~ ln_na_true * c_age + c_bmi + high_chol + 
                          usborn + female + bkg_o + bkg_pr, 
                        design = cali_twophase_imp, family = "binomial")
  }
  
  return(rakingest)
}