lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools"), require, character.only = TRUE)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")

is_nesi <- grepl("nesi.org.nz", Sys.info()["nodename"]) || dir.exists("/opt/nesi")

if (is_nesi) {
  sim_root <- path.expand("~/00_nesi_projects/uoa03789_nobackup/simulations/code_surv")
} else {
  sim_root <- "./simulations"
}
data_root <- "./data"

args <- commandArgs(trailingOnly = TRUE)
type_env <- Sys.getenv("TYPE")
sample_types <- if (length(args) >= 1) {
  args[1]
} else if (nzchar(type_env)) {
  type_env
} else {
  c("SampleOE", "SampleE")
}

cat(sprintf("Environment: %s\n", ifelse(is_nesi, "VM", "Local")))
cat(sprintf("Simulation root: %s\n", sim_root))
cat(sprintf("Sample types to evaluate: %s\n", paste(sample_types, collapse = ", ")))

out_dir <- file.path(sim_root, "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

all_sample_types <- c("SampleOE", "SampleE")

f_surv <- as.formula(
  "Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) +
   I((eGFR - 75) / 10) + I((Insulin - 15) / 2) + I((BMI - 28) / 2) +
   SEX + INSURANCE + RACE + SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2)")

.me_rhs <- paste(
  "I((HbA1c_STAR - 50) / 5) + rs4506565_STAR + I((AGE - 60) / 5) +",
  "I((eGFR_STAR - 75) / 10) + I((Insulin_STAR - 15) / 2) + I((BMI_STAR - 28) / 2) +",
  "SEX + INSURANCE + RACE + SMOKE_STAR + I((AGE - 60) / 5):I((Insulin_STAR - 15) / 2)")
f_surv_me_oe <- as.formula(paste("Surv(T_I_STAR, EVENT_STAR) ~", .me_rhs))
f_surv_me_e  <- as.formula(paste("Surv(T_I, EVENT) ~", .me_rhs))

method_mapping <- c(
  "true" = "VAL", "me" = "UNVAL", "complete_case" = "CC", "ipw" = "IPW",
  "mice" = "MICE-PMM", "rf" = "MICE-RF", "mixgb" = "MIXGB", "raking" = "RAKING",
  "sicg" = "SICG", "sird" = "SIRD", "tabcsdi" = "TABCSDI", "gain" = "GAIN")

safe_load <- function(file, envir = parent.frame())
  tryCatch({ load(file, envir = envir); TRUE }, error = function(e) FALSE)

retrieveEst <- function(method) {
  resultCoeff <- resultStdError <- resultCI <- NULL
  out_method <- unname(method_mapping[method])
  sampling_designs <- c("SRS", "Balance", "Neyman")

  for (i in 1:500) {
    digit <- stringr::str_pad(i, 4, pad = "0")
    cat("Current:", digit, "\n")

    if (!safe_load(file.path(data_root, "True", paste0(digit, ".RData")))) next

    if (method == "true") {
      cox.mod <- coxph(f_surv, data = data)
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), out_method, out_method, digit))
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), out_method, out_method, digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), out_method, out_method, digit))

    } else if (method == "me") {
      surv_form <- if (sample_type == "SampleOE") f_surv_me_oe else f_surv_me_e
      cox.mod <- coxph(surv_form, data = data)
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), out_method, out_method, digit))
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), out_method, out_method, digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), out_method, out_method, digit))

    } else {
      for (j in sampling_designs) {
        if (method == "complete_case") {
          # Genuine complete case: UNWEIGHTED coxph on the validated rows, every design.
          samp <- read.csv(file.path(data_root, sample_type, j, paste0(digit, ".csv")))
          samp <- match_types(samp, data)

          cox.mod <- coxph(f_surv, data = samp)

          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), out_method, digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), out_method, digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), toupper(j), out_method, digit))

        } else if (method == "ipw") {
          # Design-weighted complete case = IPW; kept for the record, NOT in the thesis.
          samp <- read.csv(file.path(data_root, sample_type, j, paste0(digit, ".csv")))
          samp <- match_types(samp, data)

          if (j %in% c("Balance", "Neyman")) {
            design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, data = samp)
            cox.mod <- svycoxph(f_surv, design)
          } else {
            cox.mod <- coxph(f_surv, data = samp)
          }

          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), out_method, digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), out_method, digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), toupper(j), out_method, digit))

        } else if (method == "raking") {
          if (!safe_load(file.path(sim_root, sample_type, j, method, paste0(digit, ".RData")))) next
          cox.mod <- rakingest
          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), out_method, digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), out_method, digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod))[, 1], exp(confint(cox.mod))[, 2], toupper(j), out_method, digit))

        } else {
          if (!(method %in% c("sicg", "sird", "tabcsdi"))) {
            temp_env <- new.env()
            if (!safe_load(file.path(sim_root, sample_type, j, method, paste0(digit, ".RData")), temp_env)) next
            multi_impset <- temp_env[[ls(temp_env)[1]]]
          } else {
            multi_impset <- tryCatch(
              read_parquet(file.path(sim_root, sample_type, j, method, paste0(digit, ".parquet"))),
              error = function(e) NULL)
            if (is.null(multi_impset)) next
            multi_impset <- multi_impset %>% group_split(imp_id)
            multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
          }

          if (method %in% c("sicg", "sird", "mixgb", "tabcsdi")) {
            multi_impset <- lapply(multi_impset, function(dat) {
              match_types(dat, data)
            })
            imp.mids <- imputationList(multi_impset)
          } else if (method %in% c("mice", "rf")) {
            multi_impset <- mice::complete(multi_impset, "all")
            multi_impset <- lapply(multi_impset, function(dat) {
              match_types(dat, data)
            })
            imp.mids <- imputationList(multi_impset)
          } else {
            stop(sprintf("MI branch: no imp.mids constructor for method '%s'", method))
          }

          # data = d forces coxph to find outcome vars in each imputation, not f_surv's global env
          cox.mod <- lapply(imp.mids$imputations, function(d) coxph(f_surv, data = d))
          pooled <- MIcombine(cox.mod)
          capture.output(sumry <- summary(pooled), file = nullfile())
          resultCoeff <- rbind(resultCoeff, c(exp(sumry$results), toupper(j), out_method, digit))
          resultStdError <- rbind(resultStdError, c(sumry$se, toupper(j), out_method, digit))
          resultCI <- rbind(resultCI, c(exp(sumry$`(lower`), exp(sumry$`upper)`), toupper(j), out_method, digit))
        }
      }
    }
  }

  vars_vec <- c("HbA1c", "rs4506565 1", "rs4506565 2", "Age",
                "eGFR", "Insulin", "BMI", "Sex TRUE", "Insurance TRUE",
                "Race AFR", "Race AMR", "Race SAS", "Race EAS", "Smoke 2", "Smoke 3",
                "Age:Insulin")

  if (!is.null(resultCoeff)) {
    coeff <- as.data.frame(resultCoeff, stringsAsFactors = FALSE)
    names(coeff) <- c(vars_vec, "Design", "Method", "ID")

    se <- as.data.frame(resultStdError, stringsAsFactors = FALSE)
    names(se) <- c(vars_vec, "Design", "Method", "ID")

    ci <- as.data.frame(resultCI, stringsAsFactors = FALSE)
    names(ci) <- c(paste0(vars_vec, ".lower"), paste0(vars_vec, ".upper"),
                   "Design", "Method", "ID")

    save(coeff, se, ci,
         file = file.path(out_dir, paste0("results_", sample_type, "_", out_method, ".RData")))
  }
  cat(sprintf("Method %s (%s) retrieved\n", out_method, sample_type))
}

methods <- c("true", "me", "complete_case", "ipw", "mice", "rf", "mixgb", "raking", "sicg", "sird", "tabcsdi", "gain")
run_methods <- c("complete_case", "ipw")   # only these are re-computed; combine() re-uses the rest

combine <- function() {
  save_objs <- character(0)
  for (st in all_sample_types) {
    lc <- lci <- lse <- list()
    files <- file.path(out_dir, paste0("results_", st, "_", unname(method_mapping[methods]), ".RData"))
    for (f in files) {
      if (file.exists(f)) {
        e <- new.env(); load(f, envir = e)
        lc[[f]] <- e$coeff; lci[[f]] <- e$ci; lse[[f]] <- e$se
      }
    }
    cc <- do.call(rbind, lc); ci <- do.call(rbind, lci); se <- do.call(rbind, lse)
    if (!is.null(cc)) rownames(cc) <- NULL
    if (!is.null(ci)) rownames(ci) <- NULL
    if (!is.null(se)) rownames(se) <- NULL
    assign(paste0("combined_resultCoeff_", st),    cc)
    assign(paste0("combined_resultCI_", st),       ci)
    assign(paste0("combined_resultStdError_", st), se)
    save_objs <- c(save_objs, paste0("combined_resultCoeff_", st),
                   paste0("combined_resultCI_", st), paste0("combined_resultStdError_", st))
  }
  save(list = save_objs, file = file.path(out_dir, "results_COMBINED.RData"))
}

combine_times <- function() {
  time_methods <- c("mice", "rf", "mixgb", "raking", "gain", "sicg", "sird", "tabcsdi")
  recs <- list()
  for (st in all_sample_types) {
    time_dir <- file.path(sim_root, "time", st)
    if (!dir.exists(time_dir)) { cat("No time/", st, "directory; skipping.\n"); next }
    for (m in time_methods) {
      fs <- list.files(time_dir, pattern = sprintf("(^|_)%s_.*iteration_times.*\\.csv$", m), full.names = TRUE)
      for (f in fs) {
        d <- tryCatch(read.csv(f), error = function(e) NULL)
        if (is.null(d) || !("Time_Seconds" %in% names(d))) next
        recs[[length(recs) + 1]] <- data.frame(
          SampleType = st, Method = unname(method_mapping[m]),
          Design = if ("Design" %in% names(d)) as.character(d$Design) else NA_character_,
          Time_Seconds = as.numeric(d$Time_Seconds), stringsAsFactors = FALSE
        )
      }
    }
  }
  if (!length(recs)) { cat("No timing files found.\n"); return(invisible(NULL)) }
  combined_times <- do.call(rbind, recs); rownames(combined_times) <- NULL
  save(combined_times, file = file.path(out_dir, "results_TIMES.RData"))
  write.csv(combined_times, file = file.path(out_dir, "results_TIMES.csv"), row.names = FALSE)
  cat(sprintf("Bound %d timing datapoints across %d methods -> %s\n",
              nrow(combined_times), length(unique(combined_times$Method)),
              file.path(out_dir, "results_TIMES.RData")))
}

for (sample_type in sample_types) {
  cat(sprintf("\n===== Retrieving estimates for %s =====\n", sample_type))
  for (method in run_methods) retrieveEst(method)
}
combine()
combine_times()
