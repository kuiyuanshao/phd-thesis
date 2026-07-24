# SICG between-imputation-variance (B) analysis: MC-Dropout vs Bootstrap vs SWAG vs None.

if (basename(getwd()) == "bvar") setwd("..")
stopifnot(dir.exists("bvar"), file.exists("../../utils/ablation_analysis.R"))

suppressMessages(source("../../utils/ablation_analysis.R"))

bvar_dir <- "bvar/simulations"
if (!dir.exists(bvar_dir)) dir.create(bvar_dir, recursive = TRUE)

bvar_path <- function(method, model = NULL, sim_dir = bvar_dir) {
  tag <- if (is.null(model)) toupper(method) else paste0(toupper(model), "_", toupper(method))
  file.path(sim_dir, paste0("results_SICG_BVAR_", tag, ".RData"))
}

mi_label <- function(method) {
  m <- toupper(method)
  dplyr::recode(m, UNPACK = "None", NONE = "None", BOOTSTRAP = "Bootstrap",
                DROPOUT = "Dropout", SWAG = "SWAG", .default = m)
}
mi_levels <- c("None", "Bootstrap", "Dropout", "SWAG")

cox_type <- c(
  "HbA1c" = "continuous", "rs4506565 1" = "categorical", "rs4506565 2" = "categorical",
  "Age" = "observed", "eGFR" = "continuous", "Insulin" = "continuous", "BMI" = "continuous",
  "Sex TRUE" = "observed", "Insurance TRUE" = "observed",
  "Race AFR" = "observed", "Race AMR" = "observed", "Race SAS" = "observed", "Race EAS" = "observed",
  "Smoke 2" = "categorical", "Smoke 3" = "categorical", "Age:Insulin" = "continuous"
)
nutri_vars <- c("Intercept", "ln_na_true", "c_age", "c_bmi", "high_chol",
                "usborn", "female", "bkg_pr", "bkg_o", "ln_na_true:c_age")
nutri_type <- c(
  "Intercept" = "observed", "ln_na_true" = "continuous", "c_age" = "observed",
  "c_bmi" = "observed", "high_chol" = "observed", "usborn" = "observed",
  "female" = "observed", "bkg_pr" = "observed", "bkg_o" = "observed",
  "ln_na_true:c_age" = "continuous"
)

f_sbp <- as.formula("sbp ~ ln_na_true * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")
f_htn <- as.formula("hypertension ~ ln_na_true * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")


.bm_one_method <- function(method, sim_dir, true_dir, vars_vec, fitter, n_rep) {
  resultBm <- NULL
  for (i in seq_len(n_rep)) {
    digit <- str_pad(i, 4, pad = "0")
    true_file <- file.path(true_dir, paste0(digit, ".RData"))
    if (!file.exists(true_file)) next
    parquet_file <- file.path(sim_dir, toupper(method), paste0(digit, ".parquet"))
    if (!file.exists(parquet_file)) next

    e <- new.env(); load(true_file, envir = e); data <- e$data
    imps <- read_parquet(parquet_file) %>%
      group_split(imp_id) %>%
      lapply(function(d) d %>% select(-imp_id)) %>%
      lapply(function(d) match_types(as.data.frame(d), data))

    betas <- tryCatch(do.call(rbind, lapply(imps, fitter)),
                      error = function(err) NULL)
    if (is.null(betas) || ncol(betas) != length(vars_vec)) next
    resultBm <- rbind(resultBm, c(apply(betas, 2, var), toupper(method), digit))
  }
  if (is.null(resultBm)) return(NULL)
  setNames(as.data.frame(resultBm, stringsAsFactors = FALSE), c(vars_vec, "Method", "ID"))
}

retrieveBm_cox <- function(method, overwrite = FALSE,
                           sim_dir  = "./simulations",
                           true_dir = "../../code_surv/data/True",
                           n_rep = 500) {
  out_file <- bvar_path(method)
  if (!overwrite && file.exists(out_file)) { cat(sprintf("[skip] cox B %s (%s)\n", method, out_file)); return(invisible(out_file)) }
  resultBm <- .bm_one_method(method, sim_dir, true_dir, ablation_vars_vec,
                             fitter = function(d) coef(coxph(cox_formula_true, data = d)),
                             n_rep = n_rep)
  if (is.null(resultBm)) { cat(sprintf("[warn] no surv reps for %s; nothing saved.\n", method)); return(invisible(NULL)) }
  save(resultBm, file = out_file)
  cat(sprintf("cox B %s -> %s (%d reps)\n", method, out_file, nrow(resultBm)))
  invisible(out_file)
}

retrieveBm_nutri <- function(method, overwrite = FALSE,
                             sim_dir  = bvar_dir,
                             true_dir = "../../code_nutri/data/True",
                             n_rep = 20) {
  out_sbp <- bvar_path(method, "SBP"); out_htn <- bvar_path(method, "HTN")
  if (!overwrite && file.exists(out_sbp) && file.exists(out_htn)) {
    cat(sprintf("[skip] nutri B %s\n", method)); return(invisible(c(out_sbp, out_htn)))
  }
  Bm_sbp <- .bm_one_method(method, sim_dir, true_dir, nutri_vars,
                           fitter = function(d) coef(lm(f_sbp, data = d)), n_rep = n_rep)
  Bm_htn <- .bm_one_method(method, sim_dir, true_dir, nutri_vars,
                           fitter = function(d) {
                             if (sum(unique(d$hypertension)) == 3) d$hypertension <- d$hypertension - 1
                             coef(glm(f_htn, data = d, family = binomial))
                           }, n_rep = n_rep)
  if (is.null(Bm_sbp) && is.null(Bm_htn)) { cat(sprintf("[warn] no nutri reps for %s; nothing saved.\n", method)); return(invisible(NULL)) }
  if (!is.null(Bm_sbp)) { resultBm <- Bm_sbp; save(resultBm, file = out_sbp); cat(sprintf("nutri SBP B %s -> %s (%d reps)\n", method, out_sbp, nrow(Bm_sbp))) }
  if (!is.null(Bm_htn)) { resultBm <- Bm_htn; save(resultBm, file = out_htn); cat(sprintf("nutri HTN B %s -> %s (%d reps)\n", method, out_htn, nrow(Bm_htn))) }
  invisible(c(out_sbp, out_htn))
}


load_bm_long <- function(model, file, vars_vec, type_map) {
  if (!file.exists(file)) return(NULL)
  e <- new.env(); load(file, envir = e); df <- e$resultBm
  df %>%
    pivot_longer(all_of(vars_vec), names_to = "Covariate", values_to = "B") %>%
    mutate(Model = model,
           MI_Method = factor(mi_label(Method), levels = mi_levels),
           B = as.numeric(B),
           ImputeType = unname(type_map[Covariate])) %>%
    select(Model, MI_Method, Covariate, ImputeType, ID, B)
}

assemble_long <- function(methods_surv = c("UNPACK", "BOOTSTRAP", "DROPOUT", "SWAG"),
                          methods_nutri = c("NONE", "BOOTSTRAP", "DROPOUT", "SWAG")) {
  parts <- list()
  for (m in methods_surv)  parts[[length(parts) + 1]] <- load_bm_long("COX", bvar_path(m), ablation_vars_vec, cox_type)
  for (m in methods_nutri) parts[[length(parts) + 1]] <- load_bm_long("SBP_GAUSSIAN", bvar_path(m, "SBP"), nutri_vars, nutri_type)
  for (m in methods_nutri) parts[[length(parts) + 1]] <- load_bm_long("HTN_BINOMIAL", bvar_path(m, "HTN"), nutri_vars, nutri_type)
  bind_rows(parts)
}


summarise_bvar <- function(long) {
  imp <- long %>% filter(ImputeType %in% c("continuous", "categorical"))

  per_cov <- imp %>%
    group_by(Model, MI_Method, ImputeType, Covariate) %>%
    summarise(Mean_B = mean(B, na.rm = TRUE), Median_B = median(B, na.rm = TRUE),
              SD_B = sd(B, na.rm = TRUE), n_rep = sum(!is.na(B)), .groups = "drop") %>%
    arrange(Model, ImputeType, Covariate, MI_Method)

  by_type <- imp %>%
    group_by(Model, MI_Method, ImputeType) %>%
    summarise(Mean_B = mean(B, na.rm = TRUE), Median_B = median(B, na.rm = TRUE),
              n_cov = n_distinct(Covariate), n_rep = n_distinct(ID), .groups = "drop") %>%
    arrange(Model, ImputeType, MI_Method)

  list(per_cov = per_cov, by_type = by_type, imp_long = imp)
}

plot_bvar_box <- function(imp_long) {
  ggplot(imp_long, aes(x = MI_Method, y = B, fill = ImputeType)) +
    geom_boxplot(alpha = 0.6, outlier.size = 0.6) +
    scale_y_log10() +
    facet_wrap(~ Model + Covariate, scales = "free_y", ncol = 4) +
    scale_fill_brewer(palette = "Set2") +
    labs(x = NULL, y = expression("Between-imputation variance " * italic(B) * " (log"[10] * " scale)"),
         fill = "Imputed type") +
    theme_ablation() +
    theme(legend.position = "top")
}


SURV_METHODS  <- c("UNPACK", "BOOTSTRAP", "DROPOUT", "SWAG")   # UNPACK = None
NUTRI_METHODS <- c("NONE", "BOOTSTRAP", "DROPOUT", "SWAG")
run_stage <- tolower(Sys.getenv("BVAR_RUN", "all"))

if (run_stage %in% c("surv", "all")) {
  cat("\n=== Stage 1a: surv Cox B (existing parquets) ===\n")
  for (m in SURV_METHODS) retrieveBm_cox(m)
}

if (run_stage %in% c("nutri", "all")) {
  cat("\n=== Stage 1b: nutri sbp/htn B (light run parquets, if present) ===\n")
  for (m in NUTRI_METHODS) {
    if (dir.exists(file.path(bvar_dir, toupper(m)))) retrieveBm_nutri(m) else
      cat(sprintf("[skip] nutri %s -- no simulation folder yet\n", m))
  }
}

if (run_stage == "surv") {
  cat("\nStage 1a complete (surv only). Re-run with BVAR_RUN=all after the nutri light run finishes.\n")
  quit(save = "no")
}

cat("\n=== Stage 2: assemble long B table ===\n")
long <- assemble_long(SURV_METHODS, NUTRI_METHODS)
cat("Models present:", paste(unique(long$Model), collapse = ", "), "\n")

cat("\n=== Stage 3: summary CSVs + boxplot ===\n")
res <- summarise_bvar(long)
write.csv(res$per_cov, file.path(bvar_dir, "results_SICG_BVAR_SUMMARY.csv"), row.names = FALSE)
write.csv(res$by_type, file.path(bvar_dir, "results_SICG_BVAR_BYTYPE.csv"), row.names = FALSE)
png_path <- file.path(bvar_dir, "results_SICG_BVAR_BOXPLOT.png")
ggsave(png_path, plot_bvar_box(res$imp_long), width = 16, height = 11, limitsize = FALSE)

cat(sprintf("\nSummary (per covariate) -> %s\n", file.path(bvar_dir, "results_SICG_BVAR_SUMMARY.csv")))
cat(sprintf("Summary (by type)       -> %s\n", file.path(bvar_dir, "results_SICG_BVAR_BYTYPE.csv")))
cat(sprintf("Boxplot                 -> %s\n", png_path))
cat("\n--- by-type B (mean across imputed covariates & replicates) ---\n")
print(as.data.frame(res$by_type))
