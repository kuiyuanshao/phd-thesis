import argparse
import pandas as pd
import sys
import os
import yaml
import time
import csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../benchmarks/TabCSDI")))
from tuner import BivariateTuner

data_info = {
    "weight_var": "W",
    "cat_vars": [
        "SEX", "RACEETH", "RISK1", "RISK2", "DEATH_EVER",
        "GENERIC_ART_NAME_v", "GENERIC_ART_NAME_nv",
        "SOURCE_nv", "ART_SOURCE_nv",
        "DIAGNOSIS_v", "DIAGNOSIS_nv",
        "ANY_OI_v", "ANY_OI_nv"
    ],
    "num_vars": [
        "CFAR_PID", "AGE_AT_FIRST_VISIT", "AGE_AT_LAST_VISIT", "AGE_AT_DEATH",
        "YEAR_OF_ENROLLMENT", "YEAR_OF_LAST_VISIT", "YEAR_OF_DEATH", "YEARS_IN_STUDY",
        "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v",
        "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv",
        "AGE_AT_DX_ONSET_v", "YEAR_OF_DX_ONSET_v",
        "AGE_AT_DX_ONSET_nv", "YEAR_OF_DX_ONSET_nv",
        "CD4_COUNT_BSL_v", "CD4_COUNT_1Y_v", "VL_COUNT_BSL_v", "VL_COUNT_BSL_nv",
        "CD4_COUNT_BSL_sqrt_v", "CD4_COUNT_1Y_sqrt_v", "VL_COUNT_BSL_LOG_v", "VL_COUNT_BSL_LOG_nv"
    ],
    "phase2_vars": [
        "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v", "GENERIC_ART_NAME_v",
        "AGE_AT_DX_ONSET_v", "DIAGNOSIS_v", "YEAR_OF_DX_ONSET_v", "ANY_OI_v",
        "VL_COUNT_BSL_v",
        "VL_COUNT_BSL_LOG_v"
    ],
    "phase1_vars": [
        "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv", "GENERIC_ART_NAME_nv",
        "AGE_AT_DX_ONSET_nv", "DIAGNOSIS_nv", "YEAR_OF_DX_ONSET_nv", "ANY_OI_nv",
        "VL_COUNT_BSL_nv",
        "VL_COUNT_BSL_LOG_nv"
    ]
}

CONFIG_PROFILES = {
    "tabcsdi": {
        "yaml_file": "base_config_tabcsdi.yaml",
        "output_csv": "tabcsdi_tuning_results.csv",
        "tuning_grid": {
            "lr": ("log_float", 1e-4, 4e-3),
            "channels": ("int_exp2", 6, 9),
            "nheads": ("cat", [2, 4, 8]),
            "layers": ("int", 2, 5),
            "batch_size": ("int_exp2", 4, 8),
            "num_steps": ("int", 25, 100),
            "diffusion_embedding_dim": ("int_exp2", 5, 7),
            "featureemb": ("int", 2, 16)
        }
    },
    "tabcsdi_epoch": {
        "yaml_file": "base_config_tabcsdi_epoch.yaml",
        "output_csv": "tabcsdi_epoch_tuning_results.csv",
        "tuning_grid": {
            "epochs": ("int", 300, 5000)
        }
    }
}


def main():
    parser = argparse.ArgumentParser(description="Run tuner with selected configuration profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="tabcsdi_epoch",
        choices=list(CONFIG_PROFILES.keys()),
        help="Choose the tuning profile to run."
    )
    args = parser.parse_args()

    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    tuning_grid = selected["tuning_grid"]
    output_csv = selected["output_csv"]

    num_params = len(tuning_grid)
    total_trials = 10 if num_params == 1 else max(30, 7 * num_params)
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

    csv_file_path = f"{args.profile}_tuning_time.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time_Seconds"])
        writer.writerow([iteration_time])

if __name__ == "__main__":
    main()
