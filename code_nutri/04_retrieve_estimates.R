lapply(c("dplyr", "stringr", "survey", "mice", "arrow", "mitools", "brglm2"), require, character.only = TRUE)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")

is_nesi <- grepl("nesi.org.nz", Sys.info()["nodename"]) || dir.exists("/opt/nesi")

sim_root <- "./simulations"
data_root <- "./data"

out_dir <- file.path(sim_root, "results")
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

cat(sprintf("Environment: %s\n", ifelse(is_nesi, "VM", "Local")))
cat(sprintf("Simulation root: %s\n", sim_root))

f_sbp <- as.formula("sbp ~ ln_na_true * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")
f_htn <- as.formula("hypertension ~ ln_na_true * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")

f_sbp_me <- as.formula("sbp ~ ln_na_avg * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")
f_htn_me <- as.formula("hypertension ~ ln_na_avg * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")

method_mapping <- c(
  "true" = "VAL",
  "me" = "UNVAL",
  "complete_case" = "CC",
  "ipw" = "IPW",
  "mice" = "MICE-PMM",
  "rf" = "MICE-RF",
  "mixgb" = "MIXGB",
  "raking" = "RAKING",
  "sicg" = "SICG",
  "sird" = "SIRD",
  "tabcsdi" = "TABCSDI",
  "gain" = "GAIN"
)

retrieveEst <- function(method) {
  res_sbp_coeff <- res_sbp_se <- res_sbp_ci <- NULL
  res_htn_coeff <- res_htn_se <- res_htn_ci <- NULL
  
  out_method <- unname(method_mapping[method])
  sampling_designs <- c("SRS", "RS", "WRS", "SFS", "ODS_TAIL", "Neyman_ODS", "Neyman_INF")
  
  for (i in 1:500) {
    digit <- stringr::str_pad(i, 4, pad = "0")
    cat("Current:", digit, "\n")
    true_file <- file.path(data_root, "True", paste0(digit, ".RData"))
    if (!file.exists(true_file)) next
    
    temp_env_true <- new.env()
    load(true_file, envir = temp_env_true)
    data <- temp_env_true$data
    
    if (method == "true") {
      mod_sbp <- lm(f_sbp, data = data)
      mod_htn <- glm(f_htn, data = data, family = binomial)
      
      res_sbp_coeff <- rbind(res_sbp_coeff, c(coef(mod_sbp), "TRUE", out_method, digit))
      res_sbp_se <- rbind(res_sbp_se, c(sqrt(diag(vcov(mod_sbp))), "TRUE", out_method, digit))
      res_sbp_ci <- rbind(res_sbp_ci, c(suppressMessages(confint(mod_sbp))[, 1], suppressMessages(confint(mod_sbp))[, 2], "TRUE", out_method, digit))
      
      res_htn_coeff <- rbind(res_htn_coeff, c(coef(mod_htn), "TRUE", out_method, digit))
      res_htn_se <- rbind(res_htn_se, c(sqrt(diag(vcov(mod_htn))), "TRUE", out_method, digit))
      res_htn_ci <- rbind(res_htn_ci, c(suppressMessages(confint(mod_htn))[, 1], suppressMessages(confint(mod_htn))[, 2], "TRUE", out_method, digit))
      
    } else if (method == "me") {
      mod_sbp <- lm(f_sbp_me, data = data)
      mod_htn <- glm(f_htn_me, data = data, family = binomial)
      
      res_sbp_coeff <- rbind(res_sbp_coeff, c(coef(mod_sbp), "TRUE", out_method, digit))
      res_sbp_se <- rbind(res_sbp_se, c(sqrt(diag(vcov(mod_sbp))), "TRUE", out_method, digit))
      res_sbp_ci <- rbind(res_sbp_ci, c(suppressMessages(confint(mod_sbp))[, 1], suppressMessages(confint(mod_sbp))[, 2], "TRUE", out_method, digit))
      
      res_htn_coeff <- rbind(res_htn_coeff, c(coef(mod_htn), "TRUE", out_method, digit))
      res_htn_se <- rbind(res_htn_se, c(sqrt(diag(vcov(mod_htn))), "TRUE", out_method, digit))
      res_htn_ci <- rbind(res_htn_ci, c(suppressMessages(confint(mod_htn))[, 1], suppressMessages(confint(mod_htn))[, 2], "TRUE", out_method, digit))
      
    } else {
      for (j in sampling_designs) {
        samp_file <- file.path(data_root, "Sample", j, paste0(digit, ".csv"))
        if (!file.exists(samp_file)) next
        
        if (method == "complete_case") {
          # Genuine complete case: UNWEIGHTED analysis of the validated rows, every design.
          samp <- read.csv(samp_file)
          samp <- match_types(samp, data)
          samp <- samp[!is.na(samp$ln_na_true), ]

          mod_sbp <- lm(f_sbp, data = samp)
          mod_htn <- glm(f_htn, data = samp, family = binomial)

          res_sbp_coeff <- rbind(res_sbp_coeff, c(coef(mod_sbp), toupper(j), out_method, digit))
          res_sbp_se <- rbind(res_sbp_se, c(sqrt(diag(vcov(mod_sbp))), toupper(j), out_method, digit))
          res_sbp_ci <- rbind(res_sbp_ci, c(suppressMessages(confint(mod_sbp))[, 1], suppressMessages(confint(mod_sbp))[, 2], toupper(j), out_method, digit))
          
          res_htn_coeff <- rbind(res_htn_coeff, c(coef(mod_htn), toupper(j), out_method, digit))
          res_htn_se <- rbind(res_htn_se, c(sqrt(diag(vcov(mod_htn))), toupper(j), out_method, digit))
          res_htn_ci <- rbind(res_htn_ci, c(suppressMessages(confint(mod_htn))[, 1], suppressMessages(confint(mod_htn))[, 2], toupper(j), out_method, digit))
          
        } else if (method == "ipw") {
          # Design-weighted (Horvitz-Thompson) complete case = IPW; kept in the pipeline
          # for the record but NOT reported in the thesis (raking is the design-based
          # comparator there). Reproduces the previous design-weighted "complete_case".
          samp <- read.csv(samp_file)
          samp <- match_types(samp, data)
          samp <- samp[!is.na(samp$ln_na_true), ]

          if (j %in% c("Neyman_ODS", "Neyman_INF")) {
            design_obj <- svydesign(ids = ~1, weights = ~W, data = samp)
            mod_sbp <- svyglm(f_sbp, design = design_obj)
            mod_htn <- svyglm(f_htn, design = design_obj, family = quasibinomial)
          } else {
            mod_sbp <- lm(f_sbp, data = samp)
            mod_htn <- glm(f_htn, data = samp, family = binomial)
          }

          res_sbp_coeff <- rbind(res_sbp_coeff, c(coef(mod_sbp), toupper(j), out_method, digit))
          res_sbp_se <- rbind(res_sbp_se, c(sqrt(diag(vcov(mod_sbp))), toupper(j), out_method, digit))
          res_sbp_ci <- rbind(res_sbp_ci, c(suppressMessages(confint(mod_sbp))[, 1], suppressMessages(confint(mod_sbp))[, 2], toupper(j), out_method, digit))

          res_htn_coeff <- rbind(res_htn_coeff, c(coef(mod_htn), toupper(j), out_method, digit))
          res_htn_se <- rbind(res_htn_se, c(sqrt(diag(vcov(mod_htn))), toupper(j), out_method, digit))
          res_htn_ci <- rbind(res_htn_ci, c(suppressMessages(confint(mod_htn))[, 1], suppressMessages(confint(mod_htn))[, 2], toupper(j), out_method, digit))

        } else if (method == "raking") {
          res_file <- file.path(sim_root, j, method, paste0(digit, ".RData"))
          if (!file.exists(res_file)) next

          temp_env <- new.env()
          load(res_file, envir = temp_env)
          mod_sbp <- temp_env$rakingest$sbp
          mod_htn <- temp_env$rakingest$hypertension
          
          res_sbp_coeff <- rbind(res_sbp_coeff, c(coef(mod_sbp), toupper(j), out_method, digit))
          res_sbp_se <- rbind(res_sbp_se, c(sqrt(diag(vcov(mod_sbp))), toupper(j), out_method, digit))
          res_sbp_ci <- rbind(res_sbp_ci, c(suppressMessages(confint(mod_sbp))[, 1], suppressMessages(confint(mod_sbp))[, 2], toupper(j), out_method, digit))
          
          res_htn_coeff <- rbind(res_htn_coeff, c(coef(mod_htn), toupper(j), out_method, digit))
          res_htn_se <- rbind(res_htn_se, c(sqrt(diag(vcov(mod_htn))), toupper(j), out_method, digit))
          res_htn_ci <- rbind(res_htn_ci, c(suppressMessages(confint(mod_htn))[, 1], suppressMessages(confint(mod_htn))[, 2], toupper(j), out_method, digit))
          
        } else if (method == "gain") {
          # GAIN is single imputation: use model SE/CI directly, no Rubin pooling.
          res_file <- file.path(sim_root, j, method, paste0(digit, ".parquet"))
          if (!file.exists(res_file)) next
          single <- match_types(as.data.frame(read_parquet(res_file)), data)
          mod_sbp <- lm(f_sbp, data = single)
          if (sum(unique(single$hypertension)) == 3) single$hypertension <- single$hypertension - 1
          
          n_par <- 10L
          htn_fit <- tryCatch(
            {
              m <- glm(f_htn, data = single, family = binomial)
              ci <- suppressMessages(confint(m))
              list(coeff = coef(m), se = sqrt(diag(vcov(m))),
                   lower = ci[, 1], upper = ci[, 2], separated = FALSE)
            },
            error = function(e) {
              list(coeff = rep(NA_real_, n_par), se = rep(NA_real_, n_par),
                   lower = rep(NA_real_, n_par), upper = rep(NA_real_, n_par), separated = TRUE)
            }
          )
          
          res_sbp_coeff <- rbind(res_sbp_coeff, c(coef(mod_sbp), toupper(j), out_method, digit))
          res_sbp_se <- rbind(res_sbp_se, c(sqrt(diag(vcov(mod_sbp))), toupper(j), out_method, digit))
          res_sbp_ci <- rbind(res_sbp_ci, c(suppressMessages(confint(mod_sbp))[, 1], suppressMessages(confint(mod_sbp))[, 2], toupper(j), out_method, digit))
          
          res_htn_coeff <- rbind(res_htn_coeff, c(htn_fit$coeff, toupper(j), out_method, digit))
          res_htn_se <- rbind(res_htn_se, c(htn_fit$se, toupper(j), out_method, digit))
          res_htn_ci <- rbind(res_htn_ci, c(htn_fit$lower, htn_fit$upper, toupper(j), out_method, digit))

        } else {
          if (method %in% c("sicg", "sird", "tabcsdi")) {
            res_file <- file.path(sim_root, j, method, paste0(digit, ".parquet"))
            if (!file.exists(res_file)) next
            multi_impset <- read_parquet(res_file)
            multi_impset <- multi_impset %>% group_split(imp_id)
            multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
          } else {
            res_file <- file.path(sim_root, j, method, paste0(digit, ".RData"))
            if (!file.exists(res_file)) next
            temp_env <- new.env()
            load(res_file, envir = temp_env)
            obj_name <- ls(temp_env)[1]
            multi_impset <- temp_env[[obj_name]]
          }
          
          if (method %in% c("sicg", "sird", "tabcsdi", "mixgb")) {
            multi_impset <- lapply(multi_impset, function(dat) match_types(as.data.frame(dat), data))
            imp.mids <- imputationList(multi_impset)
          } else if (method %in% c("mice", "rf")) {
            if (inherits(multi_impset, "mids")) {
              multi_impset_list <- mice::complete(multi_impset, "all")
              multi_impset_list <- lapply(multi_impset_list, function(dat) match_types(dat, data))
              imp.mids <- imputationList(multi_impset_list)
            } else {
              next
            }
          } else {
            next
          }
          
          mod_sbp <- lapply(imp.mids$imputations, function(d) lm(f_sbp, data = d))
          mod_htn <- lapply(imp.mids$imputations, function(d){
            if (sum(unique(d$hypertension)) == 3){
              d$hypertension <- d$hypertension - 1
            }
            glm(f_htn, data = d, family = binomial)
          })
          
          pooled_sbp <- MIcombine(mod_sbp)
          pooled_htn <- MIcombine(mod_htn)
          
          capture.output(sumry_sbp <- summary(pooled_sbp), file = nullfile())
          capture.output(sumry_htn <- summary(pooled_htn), file = nullfile())
          
          res_sbp_coeff <- rbind(res_sbp_coeff, c(sumry_sbp$results, toupper(j), out_method, digit))
          res_sbp_se <- rbind(res_sbp_se, c(sumry_sbp$se, toupper(j), out_method, digit))
          res_sbp_ci <- rbind(res_sbp_ci, c(sumry_sbp$`(lower`, sumry_sbp$`upper)`, toupper(j), out_method, digit))
          
          res_htn_coeff <- rbind(res_htn_coeff, c(sumry_htn$results, toupper(j), out_method, digit))
          res_htn_se <- rbind(res_htn_se, c(sumry_htn$se, toupper(j), out_method, digit))
          res_htn_ci <- rbind(res_htn_ci, c(sumry_htn$`(lower`, sumry_htn$`upper)`, toupper(j), out_method, digit))
        }
      }
    }
  }
  
  vars_vec <- c("Intercept", "ln_na_true", "c_age", "c_bmi", "high_chol", "usborn", "female", "bkg_pr", "bkg_o", "ln_na_true:c_age")
  
  save_results <- function(coeff, se, ci, prefix) {
    if (!is.null(coeff)) {
      coeff <- as.data.frame(coeff, stringsAsFactors = FALSE)
      names(coeff) <- c(vars_vec, "Design", "Method", "ID")
      
      se <- as.data.frame(se, stringsAsFactors = FALSE)
      names(se) <- c(vars_vec, "Design", "Method", "ID")
      
      ci <- as.data.frame(ci, stringsAsFactors = FALSE)
      names(ci) <- c(paste0(vars_vec, ".lower"), paste0(vars_vec, ".upper"), "Design", "Method", "ID")
      
      save(coeff, se, ci, file = file.path(out_dir, paste0("results_", prefix, "_", out_method, ".RData")))
    }
  }
  
  save_results(res_sbp_coeff, res_sbp_se, res_sbp_ci, "SBP")
  save_results(res_htn_coeff, res_htn_se, res_htn_ci, "HTN")
  cat(sprintf("Method %s retrieved\n", out_method))
}

methods <- c("complete_case", "ipw")
for (method in methods) {
  retrieveEst(method)
}

combine <- function() {
  list_coeff_sbp <- list()
  list_ci_sbp <- list()
  list_se_sbp <- list()
  
  list_coeff_htn <- list()
  list_ci_htn <- list()
  list_se_htn <- list()
  
  out_methods <- unname(method_mapping)
  
  filenames_sbp <- file.path(out_dir, paste0("results_SBP_", out_methods, ".RData"))
  for (f in filenames_sbp) {
    if (file.exists(f)) {
      temp_env <- new.env()
      load(f, envir = temp_env)
      list_coeff_sbp[[f]] <- temp_env$coeff
      list_ci_sbp[[f]] <- temp_env$ci
      list_se_sbp[[f]] <- temp_env$se
    }
  }
  
  filenames_htn <- file.path(out_dir, paste0("results_HTN_", out_methods, ".RData"))
  for (f in filenames_htn) {
    if (file.exists(f)) {
      temp_env <- new.env()
      load(f, envir = temp_env)
      list_coeff_htn[[f]] <- temp_env$coeff
      list_ci_htn[[f]] <- temp_env$ci
      list_se_htn[[f]] <- temp_env$se
    }
  }
  
  combined_resultCoeff_SBP <- do.call(rbind, list_coeff_sbp)
  combined_resultCI_SBP <- do.call(rbind, list_ci_sbp)
  combined_resultStdError_SBP <- do.call(rbind, list_se_sbp)
  
  combined_resultCoeff_HTN <- do.call(rbind, list_coeff_htn)
  combined_resultCI_HTN <- do.call(rbind, list_ci_htn)
  combined_resultStdError_HTN <- do.call(rbind, list_se_htn)
  
  if (!is.null(combined_resultCoeff_SBP)) rownames(combined_resultCoeff_SBP) <- NULL
  if (!is.null(combined_resultCI_SBP)) rownames(combined_resultCI_SBP) <- NULL
  if (!is.null(combined_resultStdError_SBP)) rownames(combined_resultStdError_SBP) <- NULL
  
  if (!is.null(combined_resultCoeff_HTN)) rownames(combined_resultCoeff_HTN) <- NULL
  if (!is.null(combined_resultCI_HTN)) rownames(combined_resultCI_HTN) <- NULL
  if (!is.null(combined_resultStdError_HTN)) rownames(combined_resultStdError_HTN) <- NULL
  
  save(combined_resultCoeff_SBP, combined_resultCI_SBP, combined_resultStdError_SBP,
       combined_resultCoeff_HTN, combined_resultCI_HTN, combined_resultStdError_HTN,
       file = file.path(out_dir, "results_COMBINED.RData"))
  cat("Combined results saved\n")
}

combine()

combine_times <- function() {
  time_dir <- file.path(sim_root, "time")
  if (!dir.exists(time_dir)) { cat("No time/ directory; skipping timing bind.\n"); return(invisible(NULL)) }
  time_methods <- c("mice", "rf", "mixgb", "raking", "gain", "sicg", "sird", "tabcsdi")
  recs <- list()
  for (m in time_methods) {
    fs <- list.files(time_dir, pattern = sprintf("(^|_)%s_.*iteration_times.*\\.csv$", m), full.names = TRUE)
    for (f in fs) {
      d <- tryCatch(read.csv(f), error = function(e) NULL)
      if (is.null(d) || !("Time_Seconds" %in% names(d))) next
      recs[[length(recs) + 1]] <- data.frame(
        Method = unname(method_mapping[m]),
        Design = if ("Design" %in% names(d)) as.character(d$Design) else NA_character_,
        Time_Seconds = as.numeric(d$Time_Seconds),
        stringsAsFactors = FALSE
      )
    }
  }
  if (!length(recs)) { cat("No timing files found in", time_dir, "\n"); return(invisible(NULL)) }
  combined_times <- do.call(rbind, recs)
  rownames(combined_times) <- NULL
  save(combined_times, file = file.path(out_dir, "results_TIMES.RData"))
  write.csv(combined_times, file = file.path(out_dir, "results_TIMES.csv"), row.names = FALSE)
  cat(sprintf("Bound %d timing datapoints across %d methods -> %s\n",
              nrow(combined_times), length(unique(combined_times$Method)),
              file.path(out_dir, "results_TIMES.RData")))
}

combine_times()