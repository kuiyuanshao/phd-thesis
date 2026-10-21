# =============================================================================
# 08_generate_dispersion_figure.R
#
# Companion to Figure 3.1 (fig:gain_underdispersion), extended to every
# imputation method under Simulation Design II. For a single replicate under
# simple random sampling, it plots the Gaussian outcome against the (imputed)
# exposure for the unvalidated units, one panel per method, together with the
# true reference fit. Output: simulations/results/simII_imputation_dispersion.pdf,
# included in Appendix C.
#
# The panel content and styling deliberately mirror Figure 3.1: translucent
# grey points, a solid within-panel linear fit, and the dashed true-regression
# line repeated in every panel so the change in slope is directly readable.
# =============================================================================

suppressMessages({
  library(mice)
  library(arrow)
  library(ggplot2)
})

dir.create("simulations/results", recursive = TRUE, showWarnings = FALSE)

rep_id  <- "0001"                 # first replicate
sim_dir <- file.path("simulations", "SRS")
out_pdf <- "simulations/results/simII_imputation_dispersion.pdf"

n_show  <- 2500                   # unvalidated units shown per panel (same units across panels)
set.seed(1)

# ---- truth (full population for this replicate) -----------------------------
true_env <- new.env()
load(file.path("data", "True", paste0(rep_id, ".RData")), envir = true_env)
truth <- true_env$data            # true ln_na_true, sbp, id (no R column)

# ---- validation indicator from an imputed dataset (shared sampling) ---------
mice_env <- new.env()
load(file.path(sim_dir, "mice", paste0(rep_id, ".RData")), envir = mice_env)
mice_d1  <- complete(mice_env$mice_imp, 1)          # id, R, ln_na_true (imputed), sbp
r0_ids   <- mice_d1$id[mice_d1$R == 0]
show_ids <- sample(r0_ids, min(n_show, length(r0_ids)))

# unvalidated units of one completed dataset: (imputed) exposure and outcome
get_unval <- function(df) {
  df <- df[df$R == 0 & df$id %in% show_ids, ]
  data.frame(id = df$id, exposure = df$ln_na_true, sbp = df$sbp)
}
read_first_imp <- function(name) {
  d <- as.data.frame(read_parquet(file.path(sim_dir, name, paste0(rep_id, ".parquet"))))
  if ("imp_id" %in% names(d)) d <- d[d$imp_id == min(d$imp_id), ]   # first imputation
  d
}

# ---- reference (true exposure) for the same unvalidated units ---------------
truth_r <- merge(truth[, c("id", "ln_na_true", "sbp")], mice_d1[, c("id", "R")], by = "id")
ref_df  <- data.frame(id = truth_r$id[truth_r$R == 0],
                      exposure = truth_r$ln_na_true[truth_r$R == 0],
                      sbp = truth_r$sbp[truth_r$R == 0])
ref_df  <- ref_df[ref_df$id %in% show_ids, ]

# reference regression: fit on the true exposure of the unvalidated units
ref_fit <- lm(sbp ~ exposure, data = ref_df)
ref_int <- unname(coef(ref_fit)[1])
ref_slp <- unname(coef(ref_fit)[2])

# ---- assemble one long data frame, one block per panel ----------------------
rf_env <- new.env(); load(file.path(sim_dir, "rf",    paste0(rep_id, ".RData")), envir = rf_env)
mx_env <- new.env(); load(file.path(sim_dir, "mixgb", paste0(rep_id, ".RData")), envir = mx_env)

blocks <- list(
  data.frame(method = "Reference (True Value)", ref_df[, c("exposure", "sbp")]),
  data.frame(method = "MICE-PMM", get_unval(mice_d1)[, c("exposure", "sbp")]),
  data.frame(method = "MICE-RF",  get_unval(complete(rf_env$mice_imp, 1))[, c("exposure", "sbp")]),
  data.frame(method = "MIXGB",    get_unval(mx_env$mixgb_imp[[1]])[, c("exposure", "sbp")]),
  data.frame(method = "TabCSDI",  get_unval(read_first_imp("tabcsdi"))[, c("exposure", "sbp")]),
  data.frame(method = "GAIN",     get_unval(read_first_imp("gain"))[, c("exposure", "sbp")]),
  data.frame(method = "SICG",     get_unval(read_first_imp("sicg"))[, c("exposure", "sbp")]),
  data.frame(method = "SIRD",     get_unval(read_first_imp("sird"))[, c("exposure", "sbp")])
)
plot_df <- do.call(rbind, blocks)
plot_df$method <- factor(plot_df$method,
  levels = c("Reference (True Value)", "MICE-PMM", "MICE-RF", "MIXGB",
             "TabCSDI", "GAIN", "SICG", "SIRD"))

cat("panel counts:\n"); print(table(plot_df$method))

# ---- plot (mirrors Figure 3.1) ----------------------------------------------
p <- ggplot(plot_df, aes(x = exposure, y = sbp)) +
  geom_point(colour = "grey55", alpha = 0.15, size = 0.45, shape = 16) +
  geom_abline(intercept = ref_int, slope = ref_slp,
              linetype = "dashed", colour = "black", linewidth = 0.6) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = "#2166AC", linewidth = 1) +
  facet_wrap(~ method, ncol = 4) +
  labs(x = "Exposure (Log Sodium)",
       y = "Systolic Blood Pressure (mmHg)",
       title = "Simulation Design II — Imputed Exposure and the Exposure–Outcome Relationship") +
  theme_bw(base_size = 12) +
  theme(strip.text  = element_text(face = "bold"),
        plot.title  = element_text(face = "bold", size = 13),
        legend.position = "none",
        panel.spacing = grid::unit(0.7, "lines"))

ggsave(out_pdf, p, width = 13, height = 7, device = cairo_pdf)
cat("written:", out_pdf, "\n")
