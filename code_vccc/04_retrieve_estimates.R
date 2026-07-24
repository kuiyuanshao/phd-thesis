lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools"), require, character.only = TRUE)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")

sim_root  <- "./simulations"
data_root <- "./data"
n_rep <- 500

out_dir <- file.path(sim_root, "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Cox CD4 effect reported as hazard ratio per sqrt-CD4 increment 100 -> 200 cells/uL.
jump_distance <- sqrt(200) - sqrt(100)

designs <- toupper(c("srs", "rs", "wrs", "sfs", "ods_tail",
                     "neyman_ods", "neyman_inf"))
weighted_designs <- toupper(c("neyman_ods", "neyman_inf"))

f_lm  <- function(s) as.formula(sprintf(
  "CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG%s + CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START%s", s, s))
f_bin <- function(s) as.formula(sprintf(
  "ANY_OI%s ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START%s + YEAR_OF_ENROLLMENT", s, s))
f_cox <- function(s) as.formula(sprintf(
  "Surv(TIME%s, ANY_OI%s) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START%s + YEAR_OF_ENROLLMENT", s, s, s))

make_rows <- function(model, design, method, id, cf, se, lo, hi) {
  data.frame(Model = model, Design = design, Method = method, ID = id,
             Covariate = names(cf), Coeff = unname(cf), SE = unname(se),
             Lower = unname(lo), Upper = unname(hi), stringsAsFactors = FALSE)
}

# For COX, estimate and CI are reported as exp(coef)*jump; SE stays on log-HR scale.
single_extract <- function(mod, model) {
  cf <- coef(mod)
  se <- sqrt(diag(vcov(mod)))
  ci <- suppressMessages(confint(mod))
  if (is.null(dim(ci))) ci <- matrix(ci, ncol = 2)
  lo <- ci[, 1]; hi <- ci[, 2]
  if (model == "COX") { cf <- exp(cf) * jump_distance; lo <- exp(lo) * jump_distance; hi <- exp(hi) * jump_distance }
  list(cf = cf, se = se, lo = lo, hi = hi)
}

pool_three <- function(imps, design, method, id) {
  # complete() can return ANY_OI_v as factor level codes 1/2; force binary back to 0/1.
  to01 <- function(y) { y <- suppressWarnings(as.numeric(as.character(y)))
                        if (any(y == 2, na.rm = TRUE)) y - 1 else y }
  fits <- list(
    LM       = function(d) lm(f_lm("_v"), data = d),
    LOGISTIC = function(d) { d$ANY_OI_v <- to01(d$ANY_OI_v); glm(f_bin("_v"), data = d, family = binomial) },
    COX      = function(d) { d$ANY_OI_v <- to01(d$ANY_OI_v); coxph(f_cox("_v"), data = d) }
  )
  out <- list()
  for (mt in names(fits)) {
    mods <- lapply(imps, fits[[mt]])
    pooled <- MIcombine(mods)
    capture.output(s <- summary(pooled), file = nullfile())
    nm <- names(coef(mods[[1]]))
    cf <- setNames(s$results, nm); se <- setNames(s$se, nm)
    lo <- setNames(s$`(lower`, nm); hi <- setNames(s$`upper)`, nm)
    if (mt == "COX") { cf <- exp(cf) * jump_distance; lo <- exp(lo) * jump_distance; hi <- exp(hi) * jump_distance }
    out[[mt]] <- make_rows(mt, design, method, id, cf, se, lo, hi)
  }
  do.call(rbind, out)
}

retrieveEst <- function(method) {
  rows <- list()
  add  <- function(df) rows[[length(rows) + 1]] <<- df

  data <- read.csv(file.path(data_root, "data.csv"))

  # VAL / UNVAL: computed once on the full real data (no replicate, no design).
  if (method %in% c("VAL", "UNVAL")) {
    sfx <- if (method == "VAL") "_v" else "_nv"
    mods <- list(LM = lm(f_lm(sfx), data = data),
                 LOGISTIC = glm(f_bin(sfx), data = data, family = binomial),
                 COX = coxph(f_cox(sfx), data = data))
    for (mt in names(mods)) {
      ex <- single_extract(mods[[mt]], mt)
      add(make_rows(mt, method, method, "0001", ex$cf, ex$se, ex$lo, ex$hi))
    }
    result_long <- do.call(rbind, rows)
    save(result_long, file = file.path(out_dir, paste0("results_", toupper(method), ".RData")))
    cat(sprintf("Method %s retrieved (%d rows)\n", method, nrow(result_long)))
    return(invisible(NULL))
  }

  for (i in 1:n_rep) {
    digit <- str_pad(i, 4, pad = "0")
    cat("Current:", digit, "\n")

    for (j in designs) {
      if (method == "CC") {
        sf <- file.path(data_root, "Sample", j, paste0(digit, ".csv"))
        if (!file.exists(sf)) next
        samp <- match_types(read.csv(sf), data)
        val <- samp[complete.cases(samp[, c("VL_COUNT_BSL_LOG_v", "CD4_COUNT_1Y_sqrt_v",
                                             "ANY_OI_v", "TIME_v", "AGE_AT_MED_START_v")]), ]
        # Genuine complete case: UNWEIGHTED fits on the validated rows, every design.
        mods <- list(LM = lm(f_lm("_v"), data = val),
                     LOGISTIC = glm(f_bin("_v"), data = val, family = binomial),
                     COX = coxph(f_cox("_v"), data = val))
        for (mt in names(mods)) {
          ex <- single_extract(mods[[mt]], mt)
          add(make_rows(mt, j, method, digit, ex$cf, ex$se, ex$lo, ex$hi))
        }

      } else if (method == "IPW") {
        # Design-weighted complete case = IPW; kept for the record, NOT in the thesis.
        sf <- file.path(data_root, "Sample", j, paste0(digit, ".csv"))
        if (!file.exists(sf)) next
        samp <- match_types(read.csv(sf), data)
        val <- samp[complete.cases(samp[, c("VL_COUNT_BSL_LOG_v", "CD4_COUNT_1Y_sqrt_v",
                                             "ANY_OI_v", "TIME_v", "AGE_AT_MED_START_v")]), ]
        if (j %in% weighted_designs) {
          des <- svydesign(ids = ~1, strata = ~outcome_strata, weights = ~W, data = val)
          mods <- list(LM = svyglm(f_lm("_v"), design = des, family = gaussian()),
                       LOGISTIC = svyglm(f_bin("_v"), design = des, family = quasibinomial()),
                       COX = svycoxph(f_cox("_v"), design = des))
        } else {
          mods <- list(LM = lm(f_lm("_v"), data = val),
                       LOGISTIC = glm(f_bin("_v"), data = val, family = binomial),
                       COX = coxph(f_cox("_v"), data = val))
        }
        for (mt in names(mods)) {
          ex <- single_extract(mods[[mt]], mt)
          add(make_rows(mt, j, method, digit, ex$cf, ex$se, ex$lo, ex$hi))
        }

      } else if (method == "raking") {
        rf <- file.path(sim_root, j, "raking", paste0(digit, ".RData"))
        if (!file.exists(rf)) next
        re <- new.env(); load(rf, envir = re)
        comp <- list(LM = re$rakingest$gauss, LOGISTIC = re$rakingest$bin, COX = re$rakingest$cox)
        for (mt in names(comp)) {
          ex <- single_extract(comp[[mt]], mt)
          add(make_rows(mt, j, method, digit, ex$cf, ex$se, ex$lo, ex$hi))
        }

      } else if (method == "gain") {
        # GAIN is single imputation: one completed dataset, fit once.
        pf <- file.path(sim_root, j, "gain", paste0(digit, ".parquet"))
        if (!file.exists(pf)) next
        single <- match_types(as.data.frame(read_parquet(pf)), data)
        mods <- list(LM = lm(f_lm("_v"), data = single),
                     LOGISTIC = glm(f_bin("_v"), data = single, family = binomial),
                     COX = coxph(f_cox("_v"), data = single))
        for (mt in names(mods)) {
          ex <- single_extract(mods[[mt]], mt)
          add(make_rows(mt, j, method, digit, ex$cf, ex$se, ex$lo, ex$hi))
        }

      } else {
        if (method %in% c("sicg", "sird", "tabcsdi")) {
          pf <- file.path(sim_root, j, method, paste0(digit, ".parquet"))
          if (!file.exists(pf)) next
          imps <- read_parquet(pf) %>% group_split(imp_id) %>%
            lapply(function(d) d %>% select(-imp_id)) %>%
            lapply(function(d) match_types(as.data.frame(d), data))
        } else {  # mice / rf (mids) ; mixgb (list)
          rdf <- file.path(sim_root, j, method, paste0(digit, ".RData"))
          if (!file.exists(rdf)) next
          me <- new.env(); load(rdf, envir = me)
          obj <- me[[ ls(me)[ which(vapply(ls(me), function(n) inherits(me[[n]], c("mids","list")), logical(1)))[1] ] ]]
          if (method %in% c("mice", "rf")) {
            if (!inherits(obj, "mids")) next
            imps <- lapply(mice::complete(obj, "all"), function(d) match_types(d, data))
          } else {
            imps <- lapply(obj, function(d) match_types(as.data.frame(d), data))
          }
        }
        add(pool_three(imps, j, method, digit))
      }
    }
  }

  result_long <- if (length(rows)) do.call(rbind, rows) else NULL
  save(result_long, file = file.path(out_dir, paste0("results_", toupper(method), ".RData")))
  cat(sprintf("Method %s retrieved (%d rows)\n", method, if (is.null(result_long)) 0L else nrow(result_long)))
}

methods <- c("VAL", "UNVAL", "CC", "IPW", "raking", "mice", "rf", "mixgb", "sicg", "sird", "tabcsdi", "gain")
run_methods <- c("CC", "IPW")   # only these are re-computed; combine() re-uses the rest
for (m in run_methods) retrieveEst(m)

combine <- function() {
  files <- file.path(out_dir, paste0("results_", toupper(methods), ".RData"))
  parts <- list()
  for (f in files) {
    if (file.exists(f)) {
      e <- new.env(); load(f, envir = e)
      if (!is.null(e$result_long)) parts[[f]] <- e$result_long
    }
  }
  combined_long <- do.call(rbind, parts)
  rownames(combined_long) <- NULL
  save(combined_long, file = file.path(out_dir, "results_COMBINED.RData"))
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
        Method = toupper(m),
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
