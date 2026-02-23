lapply(c("ggplot2", "dplyr", "tidyr", "RColorBrewer", "ggh4x", "extrafont", "patchwork"), require, character.only = T)
source("00_utils_functions.R")
# font_import()
# loadfonts(device="win") 
load("./simulations/results_COMBINED.RData")
methods <- c("true", "me", "complete_case", "raking",
             "mice", "mixgb", "sicg", "sird")
vars_vec <- c("HbA1c", "rs4506565 1", "rs4506565 2", "AGE",
              "eGFR", "BMI", "SEX TRUE", "INSURANCE TRUE",
              "RACE AFR", "RACE AMR", "RACE SAS", "RACE EAS", "SMOKE 2", "SMOKE 3")
combined_resultCoeff_long <- combined_resultCoeff %>% 
  pivot_longer(
    cols = 1:14,
    names_to = "Covariate", 
    values_to = "Coefficient"
  ) %>%
  mutate(Coefficient = as.numeric(Coefficient), 
         Method = factor(Method, levels = toupper(methods)),
         `Sampling Design` = factor(Design, levels = c("SRS", "BALANCE", "NEYMAN")),
         Covariate = factor(Covariate, levels = vars_vec))

combined_resultStdError_long <- combined_resultStdError %>% 
  pivot_longer(
    cols = 1:14,
    names_to = "Covariate", 
    values_to = "StdError"
  ) %>%
  mutate(StdError = as.numeric(StdError), 
         Method = factor(Method, levels = toupper(methods)),
         `Sampling Design` = factor(Design, levels = c("SRS", "BALANCE", "NEYMAN")),
         Covariate = factor(Covariate, levels = vars_vec))

means.coef <- combined_resultCoeff_long %>% 
  filter(Method == "TRUE") %>%
  select(-c("Design", "Method", "ID")) %>% 
  group_by(Covariate) %>%
  summarise(mean = mean(Coefficient))

true.coeff <- combined_resultCoeff %>% filter(Method == "TRUE") %>%
  select(-c("Design", "Method", "ID")) %>%
  mutate(across(everything(), as.numeric))
cols <- intersect(names(true.coeff), names(combined_resultCoeff))
rmse_result <- combined_resultCoeff %>%
  filter(!Method %in% c("ME", "TRUE")) %>%
  select(Method, Design, any_of("ID"), all_of(cols)) %>%
  mutate(across(all_of(cols), as.numeric)) %>%
  group_by(Method, Design) %>%
  summarise(across(
    all_of(cols),
    ~ {
      t <- true.coeff[[cur_column()]]
      sqrt(mean((.x - t)^2, na.rm = TRUE))
    }
  ), .groups = "drop")

diffCoeff <- combined_resultCoeff %>%
  filter(!Method %in% c("ME", "TRUE")) %>%
  select(Method, Design, any_of("ID"), all_of(cols)) %>%
  mutate(across(all_of(cols), as.numeric)) %>%
  group_by(Method, Design) %>%
  summarise(across(
    all_of(cols),
    ~ {
      t <- true.coeff[[cur_column()]]
      (.x - t)
    }
  ), .groups = "drop")

rmse_result_long <- rmse_result %>% 
  pivot_longer(
    cols = 3:16,
    names_to = "Covariate", 
    values_to = "Coefficient"
  ) %>%
  mutate(Coefficient = as.numeric(Coefficient), 
         Method = factor(Method, levels = toupper(methods)),
         `Sampling Design` = factor(Design, levels = c("SRS", "BALANCE", "NEYMAN")),
         Covariate = factor(Covariate, levels = vars_vec))

truth <- colMeans(true.coeff)
CIcoverage <- NULL
for (design in unique(combined_resultCI$Design)){
  for (method in unique(combined_resultCI$Method)){
    ind <- which(combined_resultCI$Design == design & combined_resultCI$Method == method)
    curr_g <- NULL
    for (i in ind){
      curr.lower <- combined_resultCI[i, 1:14]
      curr.upper <- combined_resultCI[i, 15:31]
      curr_g <- rbind(curr_g, c(calcCICover(truth, curr.lower, curr.upper), design, method))
    }
    CIcoverage <- rbind(CIcoverage, curr_g)
  }
}
CIcoverage <- as.data.frame(CIcoverage)
names(CIcoverage) <- c(names(combined_resultCoeff)[1:14],
                      "Design", "Method")

CIcoverage <- CIcoverage %>%
  select(Method, Design, all_of(cols)) %>%
  mutate(across(all_of(names(.)[3:16]), as.logical)) %>%
  group_by(Design, Method) %>%
  summarise(across(all_of(cols), ~ mean(.x)), .groups = "drop")

CIcoverage_long <- CIcoverage %>%
  pivot_longer(
    cols = 3:16,
    names_to = "Covariate", 
    values_to = "Coverage"
  ) %>%
  mutate(Coverage = as.numeric(Coverage), 
         Method = factor(Method, levels = toupper(methods)),
         `Sampling Design` = factor(Design, levels = c("SRS", "BALANCE", "NEYMAN")),
         Covariate = factor(Covariate, levels = vars_vec))

range_coef <- list(
  Covariate == "HbA1c"          ~ scale_y_continuous(limits = c(means.coef$mean[1] - 0.15,  means.coef$mean[1] + 0.15)),
  Covariate == "rs4506565 1"    ~ scale_y_continuous(limits = c(means.coef$mean[2] - 0.25,  means.coef$mean[2] + 0.25)),
  Covariate == "rs4506565 2"    ~ scale_y_continuous(limits = c(means.coef$mean[3] - 0.25,  means.coef$mean[3] + 0.25)),
  Covariate == "AGE"            ~ scale_y_continuous(limits = c(means.coef$mean[4] - 0.5,   means.coef$mean[4] + 0.5)),
  Covariate == "eGFR"           ~ scale_y_continuous(limits = c(means.coef$mean[5] - 0.1,   means.coef$mean[5] + 0.1)),
  Covariate == "BMI"            ~ scale_y_continuous(limits = c(means.coef$mean[6] - 0.25,  means.coef$mean[6] + 0.25)),
  Covariate == "SEX TRUE"       ~ scale_y_continuous(limits = c(means.coef$mean[7] - 0.2,   means.coef$mean[7] + 0.2)),
  Covariate == "INSURANCE TRUE" ~ scale_y_continuous(limits = c(means.coef$mean[8] - 0.5,   means.coef$mean[8] + 0.5)),
  Covariate == "RACE AFR"       ~ scale_y_continuous(limits = c(means.coef$mean[9] - 0.5,   means.coef$mean[9] + 0.5)),
  Covariate == "RACE AMR"       ~ scale_y_continuous(limits = c(means.coef$mean[10] - 0.25, means.coef$mean[10] + 0.25)),
  Covariate == "RACE SAS"       ~ scale_y_continuous(limits = c(means.coef$mean[11] - 0.25, means.coef$mean[11] + 0.25)),
  Covariate == "RACE EAS"       ~ scale_y_continuous(limits = c(means.coef$mean[12] - 0.5,  means.coef$mean[12] + 0.5)),
  Covariate == "SMOKE 2"        ~ scale_y_continuous(limits = c(means.coef$mean[13] - 0.25, means.coef$mean[13] + 0.25)),
  Covariate == "SMOKE 3"        ~ scale_y_continuous(limits = c(means.coef$mean[14] - 0.25, means.coef$mean[14] + 0.25)),
  Covariate == "AGE:HbA1c"      ~ scale_y_continuous(limits = c(means.coef$mean[15] - 0.15, means.coef$mean[15] + 0.15))
)
dev.off()

ggplot(CIcoverage_long) + 
  geom_col(aes(x = Method, 
               y = Coverage,
               fill = `Sampling Design`), position = "dodge") + 
  geom_hline(aes(yintercept = 0.95), lty = 2) + 
  facet_wrap(~ Covariate, scales = "free") + 
  theme_minimal() + 
  theme(text = element_text(family = "Times New Roman")) + 
  scale_fill_manual(
    values = c("SRS" = "red", "BALANCE" = "green", "NEYMAN" = "blue", "NA" = "black"),
    breaks = c("SRS", "BALANCE", "NEYMAN")) +
  ylim(0, 1.25)
ggsave("./simulations/Imputation_Coverage_Barchart.png", width = 30, height = 10, limitsize = F)

ggplot(rmse_result_long) + 
  geom_col(aes(x = Method, 
               y = Coefficient,
               fill = `Sampling Design`), position = "dodge") + 
  facet_wrap(~ Covariate, scales = "free") + 
  theme_minimal() + 
  theme(text = element_text(family = "Times New Roman")) + 
  scale_fill_manual(
    values = c("SRS" = "red", "BALANCE" = "green", "NEYMAN" = "blue", "NA" = "black"),
    breaks = c("SRS", "BALANCE", "NEYMAN"))
  #facetted_pos_scales(y = range_coef)

ggsave("./simulations/Imputation_Coeff_Barchart.png", width = 30, height = 10, limitsize = F)


ggplot(combined_resultCoeff_long) + 
  geom_boxplot(aes(x = Method, 
                   y = Coefficient,
                   colour = `Sampling Design`)) + 
  geom_hline(data = means.coef, aes(yintercept = mean), lty = 2) + 
  facet_wrap(~ Covariate, scales = "free") + 
  theme_minimal() + 
  theme(text = element_text(family = "Times New Roman")) + 
  scale_fill_manual(
    values = c("SRS" = "red", "BALANCE" = "green", "NEYMAN" = "blue", "NA" = "black"),
    breaks = c("SRS", "BALANCE", "NEYMAN")) + 
  facetted_pos_scales(y = range_coef)

ggsave("./simulations/Imputation_Coeff_Boxplot.png", width = 30, height = 10, limitsize = F)

ggplot(combined_resultStdError_long) + 
  geom_boxplot(aes(x = factor(Method, levels = toupper(methods)), 
                   y = StdError,
                   colour = factor(Design, levels = c("SRS", "BALANCE", "NEYMAN")))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Standard Errors") + 
  theme(text = element_text(family = "Times New Roman")) + 
  facet_wrap(~ Covariate, scales = "free") + 
  scale_colour_manual(
    name = "Sampling Design",
    values = c("SRS" = "red", "BALANCE" = "green", "NEYMAN" = "blue", "NA" = "black"),
    breaks = c("SRS", "BALANCE", "NEYMAN")
  )

ggsave("./simulations/Imputation_StdError_Boxplot.png", width = 30, height = 10, limitsize = F)



### Fill for V and UnV, Lines for imputations, lty differences
generateNumD <- function(data, imp_list, info){
  # 1. Variable Filtering
  info$num_vars <- info$num_vars[!(info$num_vars %in% c("EDU", "EDU_STAR"))]
  phase2_target <- info$phase2_vars[info$phase2_vars %in% info$num_vars]
  phase1_target <- info$phase1_vars[info$phase1_vars %in% info$num_vars]
  
  base_cols <- c("#EFC000", "black", "#00A087", "#3C5488", "#E64B35", "#4DBBD5")
  
  imp_names <- names(imp_list)
  all_labels <- c("Validated", "Unvalidated", imp_names)
  
  # Self-adjusting color mapping
  # This ensures colors are assigned correctly even if the number of datasets changes
  color_map <- setNames(rep(base_cols, length.out = length(all_labels)), all_labels)
  
  plot_list <- list()
  
  for (i in seq_along(phase2_target)){
    v2_name <- phase2_target[i]
    v1_name <- phase1_target[i]
    
    # 3. Prepare the additional datasets for plotting
    # We combine them into one 'long' dataframe with a 'Source' identifier
    extra_data_list <- lapply(imp_names, function(nm) {
      df <- imp_list[[nm]][, v2_name, drop = FALSE]
      df$Source <- nm
      colnames(df)[1] <- "Value"
      return(df)
    })
    combined_extra <- do.call(rbind, extra_data_list)
    
    # 4. Calculate global range for x-axis
    all_vals <- c(data[[v2_name]])
    val_range <- range(all_vals, na.rm = TRUE)
    
    # 5. Build the Plot
    p <- ggplot() +
      # Layer 1: Validated
      geom_density(data = data, aes(x = .data[[v2_name]], colour = "Validated"), 
                   alpha = 0.4) +
      # Layer 2: Unvalidated
      geom_density(data = data, aes(x = .data[[v1_name]], colour = "Unvalidated"), 
                   alpha = 0.4) +
      # Layer 3: All elements from the list (handled in one call for legend accuracy)
      geom_density(data = combined_extra, aes(x = Value, colour = Source), 
                   alpha = 0.3) +
      
      # Formatting
      xlim(val_range[1], val_range[2]) +
      labs(x = "", y = "", title = v2_name) +
      scale_colour_manual(name = "Type", values = color_map) +
      theme_minimal() + 
      theme(
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(hjust = 0.5),
        legend.position = "bottom"
      )
    
    plot_list[[i]] <- p
  }
  
  return(plot_list)
}
load("./data/True/0001.RData")
sird_set <- read_parquet(paste0("./simulations/SRS/sird/0001.parquet"))
sird_set <- sird_set %>% group_split(imp_id)
sird_set <- lapply(sird_set, function(d) d %>% select(-imp_id))
sird_set <- lapply(sird_set, function(dat){
  match_types(dat, data)
})
sicg_set <- read_parquet(paste0("./simulations/SRS/sicg/0001.parquet"))
sicg_set <- sicg_set %>% group_split(imp_id)
sicg_set <- lapply(sicg_set, function(d) d %>% select(-imp_id))
sicg_set <- lapply(sicg_set, function(dat){
  match_types(dat, data)
})
imp_list <- list(SIRD = sird_set[[1]], SICG = sicg_set[[1]])
p_list <- generateNumD(data, imp_list, data_info_srs)
wrap_plots(p_list, ncol = 4) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom",
        text = element_text(family = "Times New Roman"))
ggsave("./simulations/RawDensity.png", width = 20, height = 20, limitsize = F)



generateErrorD <- function(data, imp, info){
  info$num_vars <- info$num_vars[!(info$num_vars %in% c("EDU", "EDU_STAR"))]
  phase2_target <- info$phase2_vars[info$phase2_vars %in% info$num_vars]
  phase1_target <- info$phase1_vars[info$phase1_vars %in% info$num_vars]
  
  npg_cols <- c("#E64B35", "#4DBBD5")
  plot_list <- list()
  
  for (i in seq_along(phase2_target)){
    v2_name <- phase2_target[i]
    v1_name <- phase1_target[i]
    val_range <- range(c(data[[v2_name]] - data[[v1_name]]), na.rm = TRUE)
    
    p <- ggplot(data) + 
      geom_density(aes(x = .data[[v2_name]] - .data[[v1_name]], fill = "Validated"), 
                   alpha = 0.5, color = NA) +
      geom_density(data = imp, aes(x = .data[[v2_name]] - .data[[v1_name]], fill = "Imputation"), 
                   alpha = 0.5, color = NA) +
      xlim(val_range[1], val_range[2]) +
      labs(x = "Value", y = "Density", title = v2_name) +
      
      scale_fill_manual(name = "Type",
                        values = c("Validated" = npg_cols[1], 
                                   "Imputation" = npg_cols[2])) +
      theme_minimal() + 
      theme(
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(hjust = 0.5)
      )
    plot_list[[i]] <- p
  }
  
  return(plot_list)
}

load("./data/True/0001.RData")
multi_impset <- read_parquet(paste0("./simulations/SRS/sird/0001.parquet"))
multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
p_list <- generateErrorD(data, multi_impset[[1]], data_info_srs)
wrap_plots(p_list, ncol = 4) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom",
        text = element_text(family = "Times New Roman"))
ggsave("./simulations/ErrorDensity.png", width = 20, height = 20, limitsize = F)
