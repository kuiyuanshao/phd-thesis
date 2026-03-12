lapply(c("mice", "dplyr", "stringr"), require, character.only = TRUE)
source("00_utils_functions.R")

is_nesi <- grepl("nesi.org.nz", Sys.info()["nodename"]) || dir.exists("/opt/nesi")

sim_root <- "./simulations"
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

sub_dirs <- c(
  "Neyman_ODS_UNVAL/siwl",
  "Neyman_INF_UNVAL/siwl"
)

for (d in sub_dirs) {
  full_dir <- file.path(sim_root, d)
  if (!dir.exists(full_dir)) {
    dir.create(full_dir, recursive = TRUE)
  }
}

replicate <- 500
n_chunks <- 20
chunk_size <- ceiling(replicate / n_chunks)
first_rep <- (task_id - 1) * chunk_size + 1
last_rep <- min(task_id * chunk_size, replicate)

cat(sprintf("Environment: %s\n", ifelse(is_nesi, "VM", "Local")))
cat(sprintf("Simulation root: %s\n", sim_root))
cat(sprintf("Task %d handling replicates %d to %d for design: %s\n",
            task_id, first_rep, last_rep, sampling_design))

configs <- readRDS("./data/Config/best_config_mice.rds")

do_siwl <- function(dat, nm, digit, best_config) {
  mincor <- best_config$mincor

  ini <- mice(dat, maxit = 0, print = FALSE)
  meth <- ini$method
  meth["VL_COUNT_BSL_LOG_v"] <- "siwl"

  repeat {
    cat(sprintf("Trying mincor = %.2f\n", mincor))
    qp <- quickpred(dat, mincor = mincor)

    weights <- dat$W
    strata <- dat$outcome_strata
    split_q <- quantile(dat[["VL_COUNT_BSL_LOG_nv"]], c(0.19, 0.81))
    alloc_p <- table(dat$R, strata)[2,] / colSums(table(dat$R, strata))

    tm <- system.time({
      siwl_imp <- tryCatch(
        mice(dat, m = 20, print = TRUE, maxit = 25,
             method = meth, ridge = best_config$ridge,
             predictorMatrix = qp,
             sampleweights = weights, strata = strata, split_q = split_q, alloc_p = alloc_p,
             by = "sampling", pifun = pifun, 
             remove.collinear = FALSE),
        error = identity
      )
    })

    if (!inherits(siwl_imp, "error")) {
      cat(sprintf("[system.time] user=%.3fs sys=%.3fs elapsed=%.3fs\n",
                  tm[["user.self"]], tm[["sys.self"]], tm[["elapsed"]]))

      save_file <- file.path(sim_root, nm, "siwl", paste0(digit, ".RData"))
      save(siwl_imp, tm, file = save_file, compress = "xz", compression_level = 6)
      return(tm[["elapsed"]])
    }
    message(sprintf("      mice() failed (%s)", siwl_imp$message))

    mincor <- mincor + 0.05
  }
}

process_design <- function(design_name, digit, complete_data) {
  out_path <- file.path(sim_root, design_name, "mice", paste0(digit, ".RData"))

  if (!file.exists(out_path)) {
    info_key <- tolower(design_name)
    data_info <- DATA_INFOS[[info_key]]

    samp <- read.csv(file.path("./data/Sample", design_name, paste0(digit, ".csv"))) %>%
      match_types(complete_data) %>%
      mutate(across(all_of(data_info$cat_vars), as.factor),
             across(all_of(data_info$num_vars), safenumeric))

    elapsed_time <- do_mice(samp, design_name, digit, configs)
    return(data.frame(Design = design_name, Replicate = as.integer(digit), Time_Seconds = elapsed_time, stringsAsFactors = FALSE))
  }
  return(NULL)
}

timing_records <- list()

for (i in first_rep:last_rep) {
  digit <- stringr::str_pad(i, 4, pad = "0")
  cat("Current:", digit, "\n")

  data <- read.csv("./data/data.csv")

  if (sampling_design %in% c("Neyman_ODS_UNVAL", "All")) {
    res <- process_design("Neyman_ODS_UNVAL", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("Neyman_INF_UNVAL", "All")) {
    res <- process_design("Neyman_INF_UNVAL", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
}

if (length(timing_records) > 0) {
  timing_df <- do.call(rbind, timing_records)
  csv_file <- file.path(sim_root, sprintf("siwl_iteration_times_task_%d.csv", task_id))
  write.csv(timing_df, file = csv_file, row.names = FALSE)
  cat(sprintf("Saved timing data to %s\n", csv_file))
}