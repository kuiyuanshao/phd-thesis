lapply(c("survey", "svyVGAM", "dplyr", "stringr"), require, character.only = TRUE)
lapply(paste0("./comparisons_tuning/raking/", list.files("./comparisons_tuning/raking/")), source)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")

is_nesi <- grepl("nesi.org.nz", Sys.info()["nodename"]) || dir.exists("/opt/nesi")

sim_root <- "./simulations"

sub_dirs <- c(
  "SRS/raking",
  "RS/raking",
  "WRS/raking",
  "SFS/raking",
  "ODS_TAIL/raking",
  "Neyman_ODS/raking",
  "Neyman_INF/raking",
  "Neyman_ODS_UNVAL/raking",
  "Neyman_INF_UNVAL/raking"
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

process_design <- function(design_name, digit, complete_data) {
  out_path <- file.path(sim_root, design_name, "raking", paste0(digit, ".RData"))

  if (!file.exists(out_path)) {
    samp <- read.csv(file.path("./data/Sample", design_name, paste0(digit, ".csv")))
    samp <- match_types(samp, complete_data)

    tm <- system.time({
      rakingest <- list(
        sbp = calibrateFun(samp, design = design_name, outcome = "sbp"),
        hypertension = calibrateFun(samp, design = design_name, outcome = "hypertension")
      )
    })

    save(rakingest, file = out_path, compress = "xz", compression_level = 6)
    return(data.frame(Design = design_name, Replicate = as.integer(digit), Time_Seconds = tm[["elapsed"]], stringsAsFactors = FALSE))
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
  if (sampling_design %in% c("RS", "All")) {
    res <- process_design("RS", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("WRS", "All")) {
    res <- process_design("WRS", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("SFS", "All")) {
    res <- process_design("SFS", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("ODS_TAIL", "All")) {
    res <- process_design("ODS_TAIL", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("Neyman_ODS", "All")) {
    res <- process_design("Neyman_ODS", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
  if (sampling_design %in% c("Neyman_INF", "All")) {
    res <- process_design("Neyman_INF", digit, data)
    if (!is.null(res)) timing_records[[length(timing_records) + 1]] <- res
  }
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
  csv_file <- file.path(sim_root, sprintf("raking_iteration_times_task_%d.csv", task_id))
  write.csv(timing_df, file = csv_file, row.names = FALSE)
  cat(sprintf("Saved timing data to %s\n", csv_file))
}