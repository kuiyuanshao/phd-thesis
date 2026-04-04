lapply(c("tidyr", "dplyr", "GGally", 'ggplot2', 'ggrepel'), require, character.only = TRUE)

analyze_tuning <- function(file_prefix, cat_params = c(), cont_params = c()) {
  results_file <- paste0(file_prefix, "_tuning_results.csv")
  importance_file <- paste0(file_prefix, "_tuning_results_importance.csv")
  
  tuning_results <- read.csv(results_file) %>% 
    drop_na(avg_mse, avg_ce, avg_ed) %>%
    mutate(
      metric_1 = avg_mse + avg_ce, 
      metric_2 = avg_ed,
      total_combined_loss = metric_1 + metric_2
    )

  importance <- read.csv(importance_file) %>% 
    arrange(desc(importance))
  
  ordered_params <- trimws(importance$parameter)
  tuning_results_clean <- tuning_results %>%
    rowwise() %>%
    mutate(is_pareto = !any(
      tuning_results$metric_1 <= metric_1 & 
        tuning_results$metric_2 <= metric_2 & 
        (tuning_results$metric_1 < metric_1 | tuning_results$metric_2 < metric_2)
    )) %>%
    ungroup()
  
  pareto_front <- tuning_results_clean %>% filter(is_pareto == TRUE)
  
  min_m1 <- min(pareto_front$metric_1)
  max_m1 <- max(pareto_front$metric_1)
  min_m2 <- min(pareto_front$metric_2)
  max_m2 <- max(pareto_front$metric_2)
  
  tuning_results_clean <- tuning_results_clean %>%
    mutate(
      norm_m1 = (metric_1 - min_m1) / (max_m1 - min_m1 + 1e-9),
      norm_m2 = (metric_2 - min_m2) / (max_m2 - min_m2 + 1e-9),
      dist_to_utopia = sqrt((norm_m1 - 0)^2 + (norm_m2 - 0)^2) 
    )
  
  top_models <- tuning_results_clean %>%
    arrange(dist_to_utopia) %>%
    head(5)
  
  cat("\n=== Parameter Summaries (Ordered by Importance) ===\n")
  
  for (p in ordered_params) {
    if (p %in% colnames(top_models)) {
      if (p %in% cat_params) {
        cat("\nTable for", p, ":\n")
        print(table(top_models[[p]]))
      } else {
        cat("\nMedian", p, ":", median(top_models[[p]], na.rm = TRUE), "\n")
      }
    }
  }
  
  p1 <- ggplot(tuning_results_clean, aes(x = metric_1, y = metric_2)) +
    geom_point(aes(color = is_pareto, alpha = 0.7)) +
    geom_point(data = head(top_models, 5), color = "red", shape = 1, size = 6, stroke = 1.5) +
    scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "blue")) +
    labs(
      title = paste("Macro View:", file_prefix),
      subtitle = "Blue points form the Pareto Front. Red circles indicate the Top 5 optimal candidates.",
      x = "avg_mse + avg_ce (Minimize)",
      y = "avg_ed (Minimize)",
      color = "Is Pareto Optimal?"
    ) +
    theme_minimal()
  print(p1)
  
  top_5_long <- head(top_models, 5) %>%
    select(trial_number, avg_mse, avg_ce, avg_ed) %>%
    pivot_longer(cols = c(avg_mse, avg_ce, avg_ed), 
                 names_to = "loss_component", 
                 values_to = "value") %>%
    mutate(trial_number = as.factor(trial_number))
  
  p2 <- ggplot(top_5_long, aes(x = reorder(trial_number, value), y = value, fill = loss_component)) +
    geom_bar(stat = "identity", position = "dodge") + 
    scale_fill_brewer(palette = "Set2", labels = c("Categorical Error", "Energy Distance", "MSE")) +
    labs(
      title = paste("Loss Breakdown of Top 5 Models:", file_prefix),
      x = "Trial Number",
      y = "Raw Loss Value",
      fill = "Loss Component"
    ) +
    theme_minimal()
  print(p2)
  
  for (p in cat_params) {
    if (p %in% colnames(tuning_results)) {
      p_box <- ggplot(tuning_results) + 
        geom_boxplot(aes(x = as.factor(.data[[p]]), y = total_combined_loss)) +
        labs(x = p, title = paste("Boxplot of Combined Loss by", p)) +
        theme_minimal()
      print(p_box)
    }
  }
  
  for (p in cont_params) {
    if (p %in% colnames(tuning_results)) {
      p_smooth <- ggplot(tuning_results, aes(x = .data[[p]], y = total_combined_loss)) + 
        geom_point() +
        geom_smooth() +
        labs(x = p, title = paste("Trend of Combined Loss vs", p)) +
        theme_minimal()
      print(p_smooth)
    }
  }
  
  invisible(list(clean_data = tuning_results_clean, top_models = top_models))
}

unpack_categorical <- c(
  "hidden_dim", 
  "layers", 
  "batch_size", 
  "scale_layer", 
  "scale_hidden_dim", 
  "scale_lr", 
  "discriminator_steps"
)
unpack_continuous <- c(
  "lr", 
  "weight_decay", 
  "dropout", 
  "tau"
)
results_unpack <- analyze_tuning(
  file_prefix = "./tuning_results/unpack", 
  cat_params = unpack_categorical, 
  cont_params = unpack_continuous
)

plot(results_unpack$clean_data$lr, results_unpack$clean_data$avg_total_loss)




pack_categorical <- c(
  "hidden_dim", 
  "layers", 
  "batch_size", 
  "scale_layer", 
  "scale_hidden_dim", 
  "scale_lr", 
  "discriminator_steps",
  "pack"
)
pack_continuous <- c(
  "lr", 
  "weight_decay", 
  "dropout", 
  "tau"
)
results_pack <- analyze_tuning(
  file_prefix = "./tuning_results/pack", 
  cat_params = pack_categorical, 
  cont_params = pack_continuous
)

unpack_categorical <- c(
  "loss_info"
)
results_unpack_info <- analyze_tuning(
  file_prefix = "./tuning_results/unpack_info", 
  cat_params = unpack_categorical, 
  cont_params = NULL
)

pack_categorical <- c(
  "loss_mml"
)
results_pack_mml <- analyze_tuning(
  file_prefix = "./tuning_results/pack_mml", 
  cat_params = pack_categorical, 
  cont_params = NULL
)




lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools"), require, character.only = T)
source("../../code_surv/00_utils_functions.R")

options(survey.lonely.psu = "certainty")

retrieveEst <- function(method){
  resultCoeff <- resultStdError <- resultCI <- NULL
  for (i in 1:500){
    digit <- stringr::str_pad(i, 4, pad = 0)
    cat("Current:", digit, "\n")
    load(paste0("../../code_surv/data/True/", digit, ".RData"))
    if (method == "VAL"){
      cox.mod <- coxph(Surv(T_I, EVENT) ~
                         I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                         I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                         SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2), data = data)
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(method), digit))
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(method), digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), toupper(method), digit))
    }else if (method == "UNVAL"){
      cox.mod <- coxph(Surv(T_I, EVENT) ~
                         I((HbA1c_STAR - 50) / 5) + rs4506565_STAR + I((AGE - 60) / 5) + I((eGFR_STAR - 75) / 10) +
                         I((Insulin_STAR - 15) / 2) + I((BMI_STAR - 28) / 2) + SEX + INSURANCE + RACE +
                         SMOKE_STAR + I((AGE - 60) / 5):I((Insulin_STAR - 15) / 2), data = data)
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(method), digit))
      resultStdError<- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(method), digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), 
                                    exp(confint(cox.mod)[, 2]), toupper(method), digit))
    }else{
      if (method == "CC"){
        samp <- read.csv(paste0("../../code_surv/data/SampleE/SRS/", digit, ".csv"))
        samp <- match_types(samp, data)
        cox.mod <- coxph(Surv(T_I, EVENT) ~
                           I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                           I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                           SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2), data = samp)
        resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(method), digit))
        resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(method), digit))
        resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), 
                                      exp(confint(cox.mod)[, 2]), toupper(method), digit))
      }else{
        if (!file.exists(paste0("./simulations/", method, "/", digit, ".parquet"))){
          next
        }
        multi_impset <- read_parquet(paste0("./simulations/", method, "/", digit, ".parquet"))
        multi_impset <- multi_impset %>% group_split(imp_id)
        multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
        
        multi_impset <- lapply(multi_impset, function(dat){
          match_types(dat, data)
        })
        imp.mids <- imputationList(multi_impset)
        
        cox.mod <- with(data = imp.mids, 
                        exp = coxph(Surv(T_I, EVENT) ~
                                      I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                                      I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                                      SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2)))
        pooled <- MIcombine(cox.mod)
        capture.output(sumry <- summary(pooled), file = "NUL")
        resultCoeff <- rbind(resultCoeff, c(exp(sumry$results), toupper(method), digit))
        resultStdError <- rbind(resultStdError, c(sumry$se, toupper(method), digit))
        resultCI <- rbind(resultCI, c(exp(sumry$`(lower`), exp(sumry$`upper)`), toupper(method), digit))
      }
    }
  }
  vars_vec <- c("HbA1c", "rs4506565 1", "rs4506565 2", "Age",
                "eGFR", "Insulin", "BMI", "Sex TRUE", "Insurance TRUE",
                "Race AFR", "Race AMR", "Race SAS", "Race EAS", "Smoke 2", "Smoke 3",
                "Age:Insulin")
  resultCoeff <- as.data.frame(resultCoeff)
  names(resultCoeff) <- c(vars_vec, "Method", "ID")
  resultStdError <- as.data.frame(resultStdError)
  names(resultStdError) <- c(vars_vec, "Method", "ID")
  resultCI <- as.data.frame(resultCI)
  names(resultCI) <- c(paste0(vars_vec, ".lower"), 
                       paste0(vars_vec, ".upper"),
                       "Method", "ID")
  save(resultCoeff, resultStdError, resultCI, 
       file = paste0("./simulations/results_", toupper(method),".RData"))
}

methods <- c("Pack_mml_semisupervised")#c("VAL", "UNVAL", "CC", "Pack", "Unpack", "Pack_mml", "Unpack_info"，"Pack_mml_ce")
for (method in methods){
  retrieveEst(method)
}
methods <- c("VAL", "UNVAL", "CC", "Pack", "Unpack", "Pack_mml", "Unpack_info",
             "Pack_mml_ce", "Pack_mml_semisupervised")

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


lapply(c("ggplot2", "dplyr", "tidyr", "RColorBrewer", "ggh4x", "extrafont", "patchwork"), require, character.only = T)
source("../../code_surv/00_utils_functions.R")
# font_import()
# loadfonts(device="win") 

load("./simulations/results_COMBINED.RData")
vars_vec <- c("HbA1c", "rs4506565 1", "rs4506565 2", "Age",
              "eGFR", "Insulin", "BMI", "Sex TRUE", "Insurance TRUE",
              "Race AFR", "Race AMR", "Race SAS", "Race EAS", "Smoke 2", "Smoke 3",
              "Age:Insulin")

truth <- c(1.25, 1.075, 1.15, 1.1, 0.9,
           1.1, 1.1, 0.85, 0.9, 0.90, 1, 0.9, 0.8,
           0.85, 0.9, 1.1)

truth_df <- data.frame(
  Covariate = factor(vars_vec, levels = vars_vec),
  Truth = truth
)

combined_resultCoeff_long <- combined_resultCoeff %>% 
  pivot_longer(
    cols = 1:16,
    names_to = "Covariate", 
    values_to = "Coefficient"
  ) %>%
  mutate(Coefficient = as.numeric(Coefficient), 
         Method = factor(Method, levels = toupper(methods)),
         Covariate = factor(Covariate, levels = vars_vec)) %>%
  left_join(truth_df, by = "Covariate") %>%
  mutate(Percent_Bias = (Coefficient - Truth) / Truth * 100)

combined_resultStdError_long <- combined_resultStdError %>% 
  pivot_longer(
    cols = 1:16,
    names_to = "Covariate", 
    values_to = "StdError"
  ) %>%
  mutate(StdError = as.numeric(StdError), 
         Method = factor(Method, levels = toupper(methods)),
         Covariate = factor(Covariate, levels = vars_vec))

means.coef <- combined_resultCoeff_long %>% 
  filter(Method == "VAL") %>%
  select(-c("Method", "ID")) %>% 
  group_by(Covariate) %>%
  summarise(mean = mean(Coefficient))

true.coeff <- combined_resultCoeff %>% filter(Method == "VAL") %>%
  select(-c("Method", "ID")) %>%
  mutate(across(everything(), as.numeric))

cols <- intersect(names(true.coeff), names(combined_resultCoeff))

rmse_result <- combined_resultCoeff %>%
  filter(!Method %in% c("UNVAL", "VAL")) %>%
  select(Method, any_of("ID"), all_of(cols)) %>%
  mutate(across(all_of(cols), as.numeric)) %>%
  group_by(Method) %>%
  summarise(across(
    all_of(cols),
    ~ {
      t <- true.coeff[[cur_column()]]
      sqrt(mean((.x - t)^2, na.rm = TRUE))
    }
  ), .groups = "drop")

diffCoeff <- combined_resultCoeff %>%
  filter(!Method %in% c("UNVAL", "VAL")) %>%
  select(Method, any_of("ID"), all_of(cols)) %>%
  mutate(across(all_of(cols), as.numeric)) %>%
  group_by(Method) %>%
  summarise(across(
    all_of(cols),
    ~ {
      t <- true.coeff[[cur_column()]]
      (.x - t)
    }
  ), .groups = "drop")

rmse_result_long <- rmse_result %>% 
  pivot_longer(
    cols = 2:17,
    names_to = "Covariate", 
    values_to = "Coefficient"
  ) %>%
  mutate(Coefficient = as.numeric(Coefficient), 
         Method = factor(Method, levels = toupper(methods)),
         Covariate = factor(Covariate, levels = vars_vec))

CIcoverage <- NULL
for (method in unique(combined_resultCI$Method)){
  ind <- which(combined_resultCI$Method == method)
  curr_g <- NULL
  for (i in ind){
    curr.lower <- combined_resultCI[i, 1:16]
    curr.upper <- combined_resultCI[i, 17:33]
    curr_g <- rbind(curr_g, c(calcCICover(truth, curr.lower, curr.upper), method))
  }
  CIcoverage <- rbind(CIcoverage, curr_g)
}
CIcoverage <- as.data.frame(CIcoverage)
names(CIcoverage) <- c(names(combined_resultCoeff)[1:16],
                       "Method")

CIcoverage <- CIcoverage %>%
  select(Method, all_of(cols)) %>%
  mutate(across(all_of(names(.)[2:17]), as.logical)) %>%
  group_by(Method) %>%
  summarise(across(all_of(cols), ~ mean(.x)), .groups = "drop")

CIcoverage_long <- CIcoverage %>%
  pivot_longer(
    cols = 2:17,
    names_to = "Covariate", 
    values_to = "Coverage"
  ) %>%
  mutate(Coverage = as.numeric(Coverage), 
         Method = factor(Method, levels = toupper(methods)),
         Covariate = factor(Covariate, levels = vars_vec))


oracle_sds <- combined_resultCoeff %>%
  select(Method, all_of(cols)) %>%
  mutate(across(all_of(cols), as.numeric)) %>%
  group_by(Method) %>%
  summarise(across(all_of(cols), ~ sd(.x, na.rm = TRUE)), .groups = "drop")


truth_df_join <- truth_df %>% mutate(Covariate = as.character(Covariate))

oracle_CIcoverage_long <- combined_resultCoeff %>%
  select(Method, any_of("ID"), all_of(cols)) %>%
  pivot_longer(
    cols = all_of(cols), 
    names_to = "Covariate", 
    values_to = "Estimate"
  ) %>%
  mutate(Estimate = as.numeric(Estimate)) %>%
  left_join(
    oracle_sds %>% pivot_longer(cols = all_of(cols), names_to = "Covariate", values_to = "Oracle_SD"),
    by = c("Method", "Covariate")
  ) %>%
  left_join(truth_df_join, by = "Covariate") %>%
  mutate(
    Lower = Estimate - 1.96 * Oracle_SD,
    Upper = Estimate + 1.96 * Oracle_SD,
    Covered = Truth >= Lower & Truth <= Upper
  ) %>%
  group_by(Method, Covariate) %>%
  summarise(Coverage = mean(Covered, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    Method = factor(Method, levels = toupper(methods)),
    Covariate = factor(Covariate, levels = vars_vec),
    Coverage_Type = "Oracle"
  )


range_coef <- list(
  Covariate == "HbA1c"          ~ scale_y_continuous(limits = c(-15, 15)),
  Covariate == "rs4506565 1"    ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "rs4506565 2"    ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Age"            ~ scale_y_continuous(limits = c(-30, 30)),
  Covariate == "eGFR"           ~ scale_y_continuous(limits = c(-10, 10)),
  Covariate == "Insulin"        ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "BMI"            ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Sex TRUE"       ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Insurance TRUE" ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Race AFR"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Race AMR"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Race SAS"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Race EAS"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Smoke 2"        ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Smoke 3"        ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Age:Insulin"    ~ scale_y_continuous(limits = c(-10, 10))
)

# dev.off()

ggplot(CIcoverage_long) + 
  geom_col(aes(x = Method, y = Coverage, fill = Method), position = "dodge", alpha = 0.8) + 
  geom_hline(yintercept = 0.95, lty = 2, color = "red", linewidth = 0.8) + 
  facet_wrap(~ Covariate, scales = "free", ncol = 4) + 
  scale_fill_viridis_d(option = "cividis") +
  labs(x = NULL, y = "Coverage") +
  theme_bw(base_size = 14, base_family = "Times New Roman") + 
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95")
  ) + 
  coord_cartesian(ylim = c(0, 1.1))

ggsave("./simulations/Imputation_NominalCoverage_Barchart.png", width = 18, height = 12, limitsize = FALSE)

ggplot(oracle_CIcoverage_long) + 
  geom_col(aes(x = Method, y = Coverage, fill = Method), position = "dodge", alpha = 0.8) + 
  geom_hline(yintercept = 0.95, lty = 2, color = "red", linewidth = 0.8) + 
  facet_wrap(~ Covariate, scales = "free", ncol = 4) + 
  scale_fill_viridis_d(option = "cividis") +
  labs(x = NULL, y = "Coverage") +
  theme_bw(base_size = 14, base_family = "Times New Roman") + 
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95")
  ) + 
  coord_cartesian(ylim = c(0, 1.1))

ggsave("./simulations/Imputation_OracleCoverage_Barchart.png", width = 18, height = 12, limitsize = FALSE)

ggplot(rmse_result_long) + 
  geom_col(aes(x = Method, y = Coefficient, fill = Method), position = "dodge", alpha = 0.8) + 
  facet_wrap(~ Covariate, scales = "free", ncol = 4) + 
  scale_fill_viridis_d(option = "cividis") +
  labs(x = NULL, y = "Coefficient") +
  theme_bw(base_size = 14, base_family = "Times New Roman") + 
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95")
  )

ggsave("./simulations/Imputation_Coeff_Barchart.png", width = 18, height = 12, limitsize = FALSE)

ggplot(combined_resultCoeff_long) + 
  geom_boxplot(aes(x = Method, y = Percent_Bias, fill = Method), alpha = 0.6, outlier.size = 0.8) + 
  geom_hline(yintercept = 0, lty = 2, color = "red", linewidth = 0.8) + 
  facet_wrap(~ Covariate, scales = "free", ncol = 4) + 
  scale_fill_viridis_d(option = "cividis") +
  labs(x = NULL, y = "Percentage Bias (%)") +
  theme_bw(base_size = 14, base_family = "Times New Roman") + 
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95")
  ) + 
  facetted_pos_scales(y = range_coef)

ggsave("./simulations/Imputation_Bias_Boxplot.png", width = 18, height = 12, limitsize = FALSE)




