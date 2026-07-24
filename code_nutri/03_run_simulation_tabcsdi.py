import os
import argparse
import yaml
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from benchmarks.TabCSDI.tabcsdi import TabCSDI

data_info_srs = {
    "cat_vars": [
        "idx", "usborn", "high_chol", "female",
        "bkg_pr", "bkg_o", "hypertension", "R",
        "W"
    ],
    "num_vars": [
        "id", "c_age", "c_bmi", "sbp",
        "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg",
        "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true",
    ],
    "phase2_vars": [
        "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true"
    ],
    "phase1_vars": [
        "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg"
    ]
}

DATA_INFOS = {
    "srs": data_info_srs,
    "rs": {
        "cat_vars": data_info_srs["cat_vars"],
        "num_vars": data_info_srs["num_vars"] + ["rs"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "wrs": {
        "cat_vars": data_info_srs["cat_vars"],
        "num_vars": data_info_srs["num_vars"] + ["wrs"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "sfs": {
        "cat_vars": data_info_srs["cat_vars"],
        "num_vars": data_info_srs["num_vars"] + ["score"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "ods_tail": data_info_srs,
    "neyman_ods": {
        "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
        "num_vars": data_info_srs["num_vars"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "neyman_inf": {
        "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
        "num_vars": data_info_srs["num_vars"] + ["inf"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    }
}


def generate_config_profiles(model_name):
    targets = {
        "srs": "SRS",
        "rs": "RS",
        "wrs": "WRS",
        "sfs": "SFS",
        "ods_tail": "ODS_TAIL",
        "neyman_ods": "Neyman_ODS",
        "neyman_inf": "Neyman_INF"
    }

    profiles = {}
    yaml_path = f"./data/Config/best_config_{model_name}.yaml"

    for key, folder in targets.items():
        profiles[key] = {
            "yaml_file": yaml_path,
            "data_folder": folder,
            "output_dir": folder,
            "info_key": key
        }

    return profiles


CONFIG_PROFILES = generate_config_profiles("tabcsdi")

print(f"Loaded {len(CONFIG_PROFILES)} profiles for the tabcsdi model.")


def setup_env(output_subfolder):
    sim_root = Path("./simulations")
    data_root = Path("./data")

    save_dir = sim_root / output_subfolder / "tabcsdi"
    save_dir.mkdir(parents=True, exist_ok=True)

    return data_root, save_dir


def main():
    parser = argparse.ArgumentParser(description="Run TabCSDI simulation for a selected sampling design.")
    parser.add_argument(
        "--profile",
        type=str,
        default="ods_tail",
        choices=list(CONFIG_PROFILES.keys())
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=500)
    args = parser.parse_args()

    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    data_folder = selected["data_folder"]
    output_dir = selected["output_dir"]
    data_info = DATA_INFOS[selected["info_key"]]

    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    data_root, save_dir = setup_env(output_dir)
    print(f"[{args.profile.upper()}] Save directory ready: {save_dir}")

    start_idx = args.start
    end_idx = args.end + 1

    print(f"Local Mode: processing indices {start_idx} to {end_idx - 1}")

    timing_data = []

    for i in range(start_idx, end_idx):
        start_time = time.time()
        digit = str(i).zfill(4)
        file_path = data_root / "Sample" / data_folder / f"{digit}.csv"
        save_path = save_dir / f"{digit}.parquet"

        if not file_path.exists():
            print(f"Warning: File missing: {file_path}, skipping...")
            continue

        if save_path.exists():
            print(f"Already done: {save_path}, skipping...")
            continue

        print(f"Processing: {file_path} -> {save_path}")

        tabcsdi_mod = TabCSDI(base_config, data_info)
        tabcsdi_mod.fit(str(file_path))
        tabcsdi_mod.impute(save_path=str(save_path))

        end_time = time.time()
        timing_data.append([i, end_time - start_time])

    time_dir = Path("./simulations/time")
    time_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = time_dir / f"tabcsdi_{args.profile}_iteration_times_{start_idx}_{end_idx - 1}.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Time_Seconds"])
        writer.writerows(timing_data)


if __name__ == "__main__":
    main()
