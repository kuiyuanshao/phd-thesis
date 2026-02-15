lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools"), require, character.only = T)
source("00_utils_functions.R")

options(survey.lonely.psu = "certainty")

retrieveEst <- function(method){
  resultCoeff <- resultStdError <- resultCI <- NULL
  sampling_designs <- c("SRS")#, "Balance", "Neyman")
  for (i in 1:8){
    digit <- stringr::str_pad(i, 4, pad = 0)
    cat("Current:", digit, "\n")
    load(paste0("./data/True/", digit, ".RData"))
    if (method == "true"){
      cox.mod <- coxph(Surv(T_I, EVENT) ~
                         I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                         I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                         rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                         SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                       data = data)
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(method), toupper(method), digit))
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(method), toupper(method), digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), toupper(method), toupper(method), digit))
    }else if (method == "me"){
      cox.mod <- coxph(Surv(T_I_STAR, EVENT_STAR) ~
                         I((HbA1c_STAR - 50) / 5) + I(I((HbA1c_STAR - 50) / 5)^2) +
                         I((HbA1c_STAR - 50) / 5):I((AGE - 50) / 5) +
                         rs4506565_STAR + I((AGE - 50) / 5) + I((eGFR_STAR - 90) / 10) +
                         SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE_STAR,
                       data = data)
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(method), toupper(method), digit))
      resultStdError<- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(method), toupper(method), digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), 
                                    exp(confint(cox.mod)[, 2]), toupper(method), toupper(method), digit))
    }else{
      for (j in sampling_designs){
        if (method == "complete_case"){
          samp <- read.csv(paste0("./data/Sample/", j, "/", digit, ".csv"))
          samp <- match_types(samp, data)
          if (j %in% c("Balance", "Neyman")){
            design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                                data = samp)
            cox.mod <- svycoxph(Surv(T_I, EVENT) ~
                                  I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                                  I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                                  rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                  SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                                design)
          }else{
            cox.mod <- coxph(Surv(T_I, EVENT) ~
                               I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                               I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                               rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                               SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                             data = data)
          }
          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), toupper(method), digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), toupper(method), digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), 
                                        exp(confint(cox.mod)[, 2]), toupper(j), toupper(method), digit))
        }else if (method == "raking"){
          load(paste0("./simulations/", j, "/", method, "/", digit, ".RData"))
          cox.mod <- rakingest
          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), toupper(method), digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), toupper(method), digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod))[, 1], exp(confint(cox.mod))[, 2], toupper(j), toupper(method), digit))
        }else{
          if (!(method %in% c("sicg", "sird"))){
            temp_env <- new.env()
            load(paste0("./simulations/", j, "/", method, "/", digit, ".RData"), envir = temp_env)
            multi_impset <- temp_env[[ls(temp_env)[1]]]
          }else{
            multi_impset <- read_parquet(paste0("./simulations/", j, "/", method, "/", digit, ".parquet"))
            multi_impset <- multi_impset %>% group_split(imp_id)
            multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
          }
          
          if (method %in% c("sicg", "sird", "mixgb")){
            multi_impset <- lapply(multi_impset, function(dat){
              match_types(dat, data)
            })
            imp.mids <- imputationList(multi_impset)
          }else if (method == "mice"){
            multi_impset <- mice::complete(multi_impset, "all")
            multi_impset <- lapply(multi_impset, function(dat){
              match_types(dat, data)
            })
            imp.mids <- imputationList(multi_impset)
          }
          cox.mod <- with(data = imp.mids, 
                          exp = coxph(Surv(T_I, EVENT) ~
                                        I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                                        I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                                        rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                        SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
          pooled <- MIcombine(cox.mod)
          capture.output(sumry <- summary(pooled), file = "NUL")
          resultCoeff <- rbind(resultCoeff, c(exp(sumry$results), toupper(j), toupper(method), digit))
          resultStdError <- rbind(resultStdError, c(sumry$se, toupper(j), toupper(method), digit))
          resultCI <- rbind(resultCI, c(exp(sumry$`(lower`), exp(sumry$`upper)`), toupper(j), toupper(method), digit))
        }
      }
    }
  }
  vars_vec <- c("HbA1c", "HbA1c^2", "rs4506565 1", "rs4506565 2", "AGE",
                "eGFR", "BMI", "SEX TRUE", "INSURANCE TRUE",
                "RACE AFR", "RACE AMR", "RACE SAS", "RACE EAS", "SMOKE 2", "SMOKE 3",
                "AGE:HbA1c")
  resultCoeff <- as.data.frame(resultCoeff)
  names(resultCoeff) <- c(vars_vec, "Design", "Method", "ID")
  resultStdError <- as.data.frame(resultStdError)
  names(resultStdError) <- c(vars_vec, "Design", "Method", "ID")
  resultCI <- as.data.frame(resultCI)
  names(resultCI) <- c(paste0(vars_vec, ".lower"), 
                       paste0(vars_vec, ".upper"),
                       "Design", "Method", "ID")
  save(resultCoeff, resultStdError, resultCI, 
       file = paste0("./simulations/results_", toupper(method),".RData"))
}

#methods <- c("true", "me", "complete_case", "raking",
#             "mice", "mixgb", "sicg", "sird")
#methods <- c("true", "me", "complete_case", "sicg")
methods <- c("sicg")
for (method in methods){
  retrieveEst(method)
}
methods <- c("true", "me", "complete_case", "sicg", "sird")
combine <- function(){
  filenames <- paste0("./simulations/results_", toupper(methods), ".RData")
  list_coeff <- list()
  list_ci <- list()
  list_se <- list()
  
  for (f in filenames) {
    temp_env <- new.env()
    load(f, envir = temp_env)
    list_coeff[[f]] <- temp_env$resultCoeff
    list_ci[[f]] <- temp_env$resultCI
    list_se[[f]] <- temp_env$resultStdError
  }
  combined_resultCoeff <- do.call(rbind, list_coeff)
  combined_resultCI <- do.call(rbind, list_ci)
  combined_resultStdError <- do.call(rbind, list_se)
  rownames(combined_resultCoeff) <- NULL
  rownames(combined_resultCI) <- NULL
  rownames(combined_resultStdError) <- NULL
  
  save(combined_resultCoeff, combined_resultCI, combined_resultStdError,
       file = "./simulations/results_COMBINED.RData")
}

combine()

