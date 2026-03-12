import argparse
import pandas as pd
import sys
import os
import yaml
import time
import csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tuner import BivariateTuner

data_info = {
    "weight_var": "W",
    "cat_vars": [
        "idx", "usborn", "high_chol", "female",
        "bkg_pr", "bkg_o", "hypertension", "R"
    ],
    "num_vars": [
        "id", "c_age", "c_bmi", "sbp",
        "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg",
        "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true",
        "W"
    ],
    "phase2_vars": [
        "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true"
    ],
    "phase1_vars": [
        "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg"
    ]
}

CONFIG_PROFILES = {
    "multinomial": {
        "yaml_file": "base_config_sird.yaml",
        "output_csv": "sird_tuning_results.csv",
        "tuning_grid": {
            "lr": ("log_float", 1e-4, 1e-2),
            "channels": ("int_exp2", 7, 10),
            "layers": ("int", 2, 7),
            "weight_decay": ("log_float", 1e-6, 1e-3),
            "sum_scale": ("float", 0.01, 0.25),
            "dropout": ("float", 0, 0.5),
            "batch_size": ("int_exp2", 5, 8),
            "loss_num": ("int", 1, 50),
            "loss_cat": ("int", 1, 50),
            "num_steps": ("int", 20, 75),
            "diffusion_embedding_dim": ("int_exp2", 6, 8),
        }
    }
}


def main():
    parser = argparse.ArgumentParser(description="Run tuner with selected configuration profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="multinomial",
        choices=list(CONFIG_PROFILES.keys()),
        help="Choose the tuning profile to run."
    )
    args = parser.parse_args()

    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    tuning_grid = selected["tuning_grid"]
    output_csv = selected["output_csv"]

    num_params = len(tuning_grid)
    total_trials = max(30, 7 * num_params)
    print(f"[Task] Starting Tuning using profile: {args.profile}")
    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    file_path = "../../data/Sample/SRS/0001.csv"
    df = pd.read_csv(file_path).loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]
    
    start_time = time.time()
    
    tuner = BivariateTuner(df, base_config, tuning_grid, data_info, n_splits=3)
    tuner.tune(n_trials=total_trials, output_csv=output_csv)
    
    end_time = time.time()
    iteration_time = end_time - start_time

    csv_file_path = f"sird_tuning_time.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time_Seconds"])
        writer.writerow([iteration_time])
if __name__ == "__main__":
    main()
