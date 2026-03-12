lapply(c("mice", "dplyr", "stringr"), require, character.only = TRUE)
source("00_utils_functions.R")
source("../SIWL/mice.impute.siwl.R")
is_nesi <- grepl("nesi.org.nz", Sys.info()["nodename"]) || dir.exists("/opt/nesi")

sim_root <- "./simulations"
sub_dirs <- c(
  "Neyman_ODS_UNVAL/siwl_oracle",
  "Neyman_INF_UNVAL/siwl_oracle",
  "Neyman_ODS_UNVAL/siwl",
  "Neyman_INF_UNVAL/siwl"
)

for (d in sub_dirs) {
  full_dir <- file.path(sim_root, d)
  if (!dir.exists(full_dir)) {
    dir.create(full_dir, recursive = TRUE)
  }
}

args <- commandArgs(trailingOnly = TRUE)
task_id_env <- Sys.getenv("SLURM_ARRAY_TASK_ID")
task_id <- if (length(args) >= 1) {
  as.integer(args[1])
} else if (nzchar(task_id_env)) {
  as.integer(task_id_env)
} else {
  1L
}

samp_env <- Sys.getenv("SAMP")
sampling_design <- if (length(args) >= 2) {
  args[2]
} else if (nzchar(samp_env)) {
  samp_env
} else {
  "All"
}

replicate <- 500
n_chunks <- 20
chunk_size <- ceiling(replicate / n_chunks)
first_rep <- (task_id - 1) * chunk_size + 1
last_rep <- min(task_id * chunk_size, replicate)

cat(sprintf("Environment: %s\n", ifelse(is_nesi, "VM", "Local")))
cat(sprintf("Simulation root: %s\n", sim_root))
cat(sprintf("Task %d handling replicates %d to %d for sampling design: %s\n",
            task_id, first_rep, last_rep, sampling_design))

configs <- readRDS("./data/Config/best_config_mice.rds")

do_siwl <- function(dat, nm, digit) {
  dat$sbp_c_age <- as.numeric(dat$sbp) * as.numeric(dat$c_age)
  dat$htn_c_age <- as.numeric(as.character(dat$hypertension)) * as.numeric(dat$c_age)

  ini <- mice(dat, maxit = 0, print = FALSE)
  pred <- ini$predictorMatrix
  meth <- ini$method

  pred[,] <- 0

  base_covars <- c("c_age", "c_bmi", "female", "usborn", "high_chol", "bkg_pr", "bkg_o")

  pred["ln_na_true", c("ln_na_avg", base_covars, "sbp", "hypertension", "sbp_c_age", "htn_c_age")] <- 1
  pred["ln_k_true", c("ln_k_avg", base_covars)] <- 1
  pred["ln_kcal_true", c("ln_kcal_avg", base_covars)] <- 1
  pred["ln_protein_true", c("ln_protein_avg", base_covars)] <- 1

  meth["ln_na_true"] <- "siwl"

  weights <- dat$W
  strata <- dat$outcome_strata
  split_q <- quantile(dat[["ln_na_avg"]], c(0.19, 0.81))
  alloc_p <- table(dat$R, strata)[2,] / colSums(table(dat$R, strata))

  siwl_oracle_imp <- tryCatch(
      mice(dat, m = 20, print = FALSE, maxit = 1,
           predictorMatrix = pred,
           method = meth,
           remove.collinear = FALSE,
           maxcor = 1.0001,
           eps = 0,
           sampleweights = weights, strata = strata, split_q = split_q, alloc_p = alloc_p,
           by = "sampling", pifun = pifun),
      error = identity
    )

  qp <- quickpred(dat, mincor = configs$mincor)

  if ("ln_na_true" %in% rownames(qp)) {
    qp["ln_na_true", "sbp_c_age"] <- 1
    qp["ln_na_true", "htn_c_age"] <- 1
  }
  tm <- system.time({
    siwl_imp <- tryCatch(
      mice(dat, m = 20, print = TRUE, maxit = 25,
           predictorMatrix = qp,
           meth = meth,
           sampleweights = weights, strata = strata, split_q = split_q, alloc_p = alloc_p,
           by = "sampling", pifun = pifun,
           remove.collinear = FALSE),
      error = identity
    )
  })

  if (!inherits(siwl_imp, "error")) {
    cat(sprintf("System time: user=%.3fs sys=%.3fs elapsed=%.3fs\n",
                tm[["user.self"]], tm[["sys.self"]], tm[["elapsed"]]))

    save_file_1 <- file.path(sim_root, nm, "siwl_oracle", paste0(digit, ".RData"))
    save_file_2 <- file.path(sim_root, nm, "siwl", paste0(digit, ".RData"))
    save(siwl_oracle_imp, file = save_file_1, compress = "xz", compression_level = 6)
    save(siwl_imp, file = save_file_2, compress = "xz", compression_level = 6)
    return(tm[["elapsed"]])
  }

  cat(sprintf("siwl_imp failed (%s)\n", siwl_imp$message))
  return(NA)
}

process_design <- function(design_name, digit, complete_data) {
  out_path_1 <- file.path(sim_root, design_name, "siwl_oracle", paste0(digit, ".RData"))
  out_path_2 <- file.path(sim_root, design_name, "siwl", paste0(digit, ".RData"))
  if (!file.exists(out_path_1) & !file.exists(out_path_2)) {
    info_key <- tolower(design_name)
    data_info <- DATA_INFOS[[info_key]]

    samp <- read.csv(file.path("./data/Sample", design_name, paste0(digit, ".csv"))) %>%
      match_types(complete_data) %>%
      mutate(across(all_of(data_info$cat_vars), as.factor),
             across(all_of(data_info$num_vars), safenumeric))

    elapsed_time <- do_siwl(samp, design_name, digit)
    return(data.frame(Design = design_name, Replicate = as.integer(digit), Time_Seconds = elapsed_time, stringsAsFactors = FALSE))
  }
  return(NULL)
}

for (i in first_rep:last_rep) {
  digit <- stringr::str_pad(i, 4, pad = "0")
  cat(sprintf("Replicate %s\n", digit))

  load(file.path("./data/True", paste0(digit, ".RData")))

  if (sampling_design %in% c("Neyman_ODS_UNVAL", "All")) {
    process_design("Neyman_ODS_UNVAL", digit, data)
  }
  if (sampling_design %in% c("Neyman_INF_UNVAL", "All")) {
    process_design("Neyman_INF_UNVAL", digit, data)
  }
}