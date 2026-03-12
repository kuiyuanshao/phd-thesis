lapply(c("mixgb", "dplyr", "stringr"), require, character.only = TRUE)
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

type_env <- Sys.getenv("TYPE")
type <- if (length(args) >= 2) {
  args[2]
} else if (nzchar(type_env)) {
  type_env
} else {
  "SampleOE"
}

samp_env <- Sys.getenv("SAMP")
sampling_design <- if (length(args) >= 3) {
  args[3]
} else if (nzchar(samp_env)) {
  samp_env
} else {
  "All"
}

sub_dirs <- c(
  "SRS/mixgb",
  "Balance/mixgb",
  "Neyman/mixgb"
)

for (d in sub_dirs) {
  full_dir <- file.path(sim_root, type, d)
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
cat(sprintf("Task %d handling replicates %d to %d for type: %s, design: %s\n",
            task_id, first_rep, last_rep, type, sampling_design))

configs <- readRDS("./data/Config/best_config_mixgb.rds")

do_mixgb <- function(samp, params, nm, digit, complete_data) {
  tm <- system.time({
    mixgb_imp <- mixgb(samp, m = 5, xgb.params = params$params,
                       initial.fac = "sample", nrounds = params$nrounds)
  })
  mixgb_imp <- lapply(mixgb_imp, function(dat){
    match_types(as.data.frame(dat), complete_data)
  })
  save_file <- file.path(sim_root, type, nm, "mixgb", paste0(digit, ".RData"))
  save(mixgb_imp, file = save_file, compress = "xz", compression_level = 6)

  return(tm[["elapsed"]])
}

process_design <- function(design_name, digit, complete_data) {
  out_path <- file.path(sim_root, type, design_name, "mixgb", paste0(digit, ".RData"))

  if (!file.exists(out_path)) {
    info_key <- tolower(design_name)
    data_info <- DATA_INFOS[[type]][[info_key]]

    samp <- read.csv(file.path("./data", type, design_name, paste0(digit, ".csv"))) %>%
      match_types(complete_data) %>%
      mutate(across(all_of(data_info$cat_vars), as.factor, .names = "{.col}"),
             across(all_of(data_info$num_vars), safenumeric, .names = "{.col}"))

    elapsed_time <- do_mixgb(samp, configs, design_name, digit, complete_data)
    return(data.frame(Design = design_name, Replicate = as.integer(digit), Time_Seconds = elapsed_time, stringsAsFactors = FALSE))
  }
  return(NULL)
}

timing_records <- list()

for (i in first_rep:last_rep) {
  digit <- stringr::str_pad(i, 4, pad = "0")
  cat(sprintf("Current: %s\n", digit))

  load(file.path("./data/True", paste0(digit, ".RData")))

  if (sampling_design %in% c("SRS", "All")) {
    res <- process_design("SRS", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("Balance", "All")) {
    res <- process_design("Balance", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("Neyman", "All")) {
    res <- process_design("Neyman", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
}

if (length(timing_records) > 0) {
  timing_df <- do.call(rbind, timing_records)
  csv_file <- file.path(sim_root, sprintf("%s_mixgb_iteration_times_task_%d.csv", type, task_id))
  write.csv(timing_df, file = csv_file, row.names = FALSE)
  cat(sprintf("Saved timing data to %s\n", csv_file))
}