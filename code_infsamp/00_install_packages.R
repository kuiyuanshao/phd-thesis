pkgs <- c("pacman", "data.table", "dplyr", "stringr", "mice", "survey", "mitools",
          "ggplot2", "tidyr", "patchwork",
          "ranger", "mixgb", "arrow", "statmod",
          "mlr3mbo", "mlr3", "mlr3learners", "bbotk", "DiceKriging",
          "paradox", "Matrix", "expm", "fields")
new <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org")
