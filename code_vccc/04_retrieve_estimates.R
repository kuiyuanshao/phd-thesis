lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools"), require, character.only = TRUE)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")

sim_root <- "./simulations"
data_root <- "./data"

retrieveEst <- function(method) {
  resultCoeff <- resultStdError <- resultCI <- NULL
  sampling_designs <- toupper(c("srs", "rs", "wrs", "sfs", "ods_tail", "neyman_ods", "neyman_inf", "neyman_ods_unval", "neyman_inf_unval"))
  weighted_designs <- toupper(c("neyman_ods", "neyman_inf", "neyman_ods_unval", "neyman_inf_unval"))
  for (i in 1:500) {
    digit <- stringr::str_pad(i, 4, pad = "0")
    cat("Current:", digit, "\n")

    load(file.path(data_root, "True", paste0(digit, ".RData")))
    jump_distance <- sqrt(200) - sqrt(100)
    if (method == "VAL") {
      fit_gauss <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_v + CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v, data = data)
      fit_bin <- glm(ANY_OI_v ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, family = binomial, data = data)
      fit_cox <- coxph(Surv(TIME_v, ANY_OI_v) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, data = data)

      resultCoeff <- rbind(resultCoeff, c(coef(fit_gauss), toupper(method), toupper(method), digit), "LM")
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_gauss))), toupper(method), toupper(method), digit), "LM")
      resultCI <- rbind(resultCI, c(confint(fit_gauss)[, 1], confint(fit_gauss)[, 2], toupper(method), toupper(method), digit), "LM")

      resultCoeff <- rbind(resultCoeff, c(coef(fit_bin), toupper(method), toupper(method), digit), "LOGISTIC")
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_bin))), toupper(method), toupper(method), digit), "LOGISTIC")
      resultCI <- rbind(resultCI, c(confint(fit_bin)[, 1], confint(fit_bin)[, 2], toupper(method), toupper(method), digit), "LOGISTIC")

      resultCoeff <- rbind(resultCoeff, c(exp(coef(fit_cox)) * jump_distance, toupper(method), toupper(method), digit), "COX")
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_cox))), toupper(method), toupper(method), digit), "COX")
      resultCI <- rbind(resultCI, c(confint(fit_cox)[, 1], confint(fit_cox)[, 2], toupper(method), toupper(method), digit), "COX")

    } else if (method == "UNVAL") {
      fit_gauss <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_nv + CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv, data = data)
      fit_bin <- glm(ANY_OI_nv ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, family = binomial, data = data)
      fit_cox <- coxph(Surv(TIME_nv, ANY_OI_nv) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, data = data)

      resultCoeff <- rbind(resultCoeff, c(coef(fit_gauss), toupper(method), toupper(method), digit), "LM")
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_gauss))), toupper(method), toupper(method), digit), "LM")
      resultCI <- rbind(resultCI, c(confint(fit_gauss)[, 1], confint(fit_gauss)[, 2], toupper(method), toupper(method), digit), "LM")

      resultCoeff <- rbind(resultCoeff, c(coef(fit_bin), toupper(method), toupper(method), digit), "LOGISTIC")
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_bin))), toupper(method), toupper(method), digit), "LOGISTIC")
      resultCI <- rbind(resultCI, c(confint(fit_bin)[, 1], confint(fit_bin)[, 2], toupper(method), toupper(method), digit), "LOGISTIC")

      resultCoeff <- rbind(resultCoeff, c(exp(coef(fit_cox)) * jump_distance, toupper(method), toupper(method), digit), "COX")
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_cox))), toupper(method), toupper(method), digit), "COX")
      resultCI <- rbind(resultCI, c(confint(fit_cox)[, 1], confint(fit_cox)[, 2], toupper(method), toupper(method), digit), "COX")
    } else {
      for (j in sampling_designs) {
        if (method == "CC") {
          samp <- read.csv(file.path(data_root, j, paste0(digit, ".csv")))
          samp <- match_types(samp, data)

          if (j %in% weighted_designs) {
            design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, data = samp)
            fit_gauss <- svyglm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_nv + CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv, family = gaussian, design)
            fit_bin <- svyglm(ANY_OI_nv ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, family = binomial, data = design)
            fit_cox <- svycoxph(Surv(TIME_nv, ANY_OI_nv) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, data = design)
          } else {
            fit_gauss <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_nv + CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv, data = data)
            fit_bin <- glm(ANY_OI_nv ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, family = binomial, data = data)
            fit_cox <- coxph(Surv(TIME_nv, ANY_OI_nv) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_nv + YEAR_OF_ENROLLMENT, data = data)
          }

          resultCoeff <- rbind(resultCoeff, c(coef(fit_gauss), toupper(method), toupper(method), digit), "LM")
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_gauss))), toupper(method), toupper(method), digit), "LM")
          resultCI <- rbind(resultCI, c(confint(fit_gauss)[, 1], confint(fit_gauss)[, 2], toupper(method), toupper(method), digit), "LM")

          resultCoeff <- rbind(resultCoeff, c(coef(fit_bin), toupper(method), toupper(method), digit), "LOGISTIC")
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_bin))), toupper(method), toupper(method), digit), "LOGISTIC")
          resultCI <- rbind(resultCI, c(confint(fit_bin)[, 1], confint(fit_bin)[, 2], toupper(method), toupper(method), digit), "LOGISTIC")

          resultCoeff <- rbind(resultCoeff, c(exp(coef(fit_cox)) * jump_distance, toupper(method), toupper(method), digit), "COX")
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(fit_cox))), toupper(method), toupper(method), digit), "COX")
          resultCI <- rbind(resultCI, c(confint(fit_cox)[, 1], confint(fit_cox)[, 2], toupper(method), toupper(method), digit), "COX")

        } else if (method == "raking") {
          load(file.path(sim_root, sample_type, j, method, paste0(digit, ".RData")))
          cox.mod <- rakingest
          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), toupper(method), digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), toupper(method), digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod))[, 1], exp(confint(cox.mod))[, 2], toupper(j), toupper(method), digit))

        } else {
          if (!(method %in% c("sicg", "sird"))) {
            temp_env <- new.env()
            load(file.path(sim_root, sample_type, j, method, paste0(digit, ".RData")), envir = temp_env)
            multi_impset <- temp_env[[ls(temp_env)[1]]]
          } else {
            multi_impset <- read_parquet(file.path(sim_root, sample_type, j, method, paste0(digit, ".parquet")))
            multi_impset <- multi_impset %>% group_split(imp_id)
            multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
          }

          if (method %in% c("sicg", "sird", "mixgb")) {
            multi_impset <- lapply(multi_impset, function(dat) {
              match_types(dat, data)
            })
            imp.mids <- imputationList(multi_impset)
          } else if (method == "mice") {
            multi_impset <- mice::complete(multi_impset, "all")
            multi_impset <- lapply(multi_impset, function(dat) {
              match_types(dat, data)
            })
            imp.mids <- imputationList(multi_impset)
          }

          cox.mod <- with(data = imp.mids,
                          exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) +
                                        I((eGFR - 75) / 10) + I((Insulin - 15) / 2) + I((BMI - 28) / 2) +
                                        SEX + INSURANCE + RACE + SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2)))
          pooled <- MIcombine(cox.mod)
          capture.output(sumry <- summary(pooled), file = nullfile())
          resultCoeff <- rbind(resultCoeff, c(exp(sumry$results), toupper(j), toupper(method), digit))
          resultStdError <- rbind(resultStdError, c(sumry$se, toupper(j), toupper(method), digit))
          resultCI <- rbind(resultCI, c(exp(sumry$`(lower`), exp(sumry$`upper)`), toupper(j), toupper(method), digit))
        }
      }
    }
  }

  vars_vec <- c("HbA1c", "rs4506565 1", "rs4506565 2", "Age",
                "eGFR", "Insulin", "BMI", "Sex TRUE", "Insurance TRUE",
                "Race AFR", "Race AMR", "Race SAS", "Race EAS", "Smoke 2", "Smoke 3",
                "Age:Insulin")

  resultCoeff <- as.data.frame(resultCoeff)
  names(resultCoeff) <- c(vars_vec, "Design", "Method", "ID")

  resultStdError <- as.data.frame(resultStdError)
  names(resultStdError) <- c(vars_vec, "Design", "Method", "ID")

  resultCI <- as.data.frame(resultCI)
  names(resultCI) <- c(paste0(vars_vec, ".lower"),
                       paste0(vars_vec, ".upper"),
                       "Design", "Method", "ID")

  save(resultCoeff, resultStdError, resultCI,
       file = file.path(sim_root, sample_type, paste0("results_", toupper(method), ".RData")))
}

methods <- c("true", "me", "complete_case", "mice", "mixgb", "raking", "sicg", "sird", "tabcsdi")
for (method in methods) {
  retrieveEst(method)
}

combine <- function() {
  filenames <- file.path(sim_root, sample_type, paste0("results_", toupper(methods), ".RData"))
  list_coeff <- list()
  list_ci <- list()
  list_se <- list()

  for (f in filenames) {
    if (file.exists(f)) {
      temp_env <- new.env()
      load(f, envir = temp_env)
      list_coeff[[f]] <- temp_env$resultCoeff
      list_ci[[f]] <- temp_env$resultCI
      list_se[[f]] <- temp_env$resultStdError
    }
  }

  combined_resultCoeff <- do.call(rbind, list_coeff)
  combined_resultCI <- do.call(rbind, list_ci)
  combined_resultStdError <- do.call(rbind, list_se)

  rownames(combined_resultCoeff) <- NULL
  rownames(combined_resultCI) <- NULL
  rownames(combined_resultStdError) <- NULL

  save(combined_resultCoeff, combined_resultCI, combined_resultStdError,
       file = file.path(sim_root, sample_type, "results_COMBINED.RData"))
}

combine()