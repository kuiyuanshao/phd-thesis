import os
import socket
import argparse
import yaml
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from benchmarks.GAIN.gain import GAIN

data_info_srs = {
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

DATA_INFOS = {
    "srs": data_info_srs,
    "rs": {
        "weight_var": "W",
        "cat_vars": data_info_srs["cat_vars"],
        "num_vars": data_info_srs["num_vars"] + ["rs"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "wrs": {
        "weight_var": "W",
        "cat_vars": data_info_srs["cat_vars"],
        "num_vars": data_info_srs["num_vars"] + ["wrs"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "sfs": {
        "weight_var": "W",
        "cat_vars": data_info_srs["cat_vars"],
        "num_vars": data_info_srs["num_vars"] + ["score"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "ods_tail": data_info_srs,
    "neyman_ods": {
        "weight_var": "W",
        "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
        "num_vars": data_info_srs["num_vars"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "neyman_inf": {
        "weight_var": "W",
        "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
        "num_vars": data_info_srs["num_vars"] + ["inf"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "neyman_ods_unval": {
        "weight_var": "W",
        "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
        "num_vars": data_info_srs["num_vars"],
        "phase2_vars": data_info_srs["phase2_vars"],
        "phase1_vars": data_info_srs["phase1_vars"]
    },
    "neyman_inf_unval": {
        "weight_var": "W",
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
        "neyman_inf": "Neyman_INF",
        "neyman_ods_unval": "Neyman_ODS_UNVAL",
        "neyman_inf_unval": "Neyman_INF_UNVAL"
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

CONFIG_PROFILES = generate_config_profiles("gain")

print(f"Loaded {len(CONFIG_PROFILES)} profiles for the gain model.")


def setup_env(output_subfolder):
    is_nesi = "nesi.org.nz" in socket.getfqdn() or os.path.exists("/opt/nesi")

    # if is_nesi:
    #     sim_root = Path("~/00_nesi_projects/uoa03789_nobackup/simulations/code_nutri").expanduser()
    # else:
    #     sim_root = Path("./simulations")
    sim_root = Path("./simulations")
    data_root = Path("./data")

    save_dir = sim_root / output_subfolder / "gain"
    save_dir.mkdir(parents=True, exist_ok=True)

    return data_root, save_dir


def main():
    parser = argparse.ArgumentParser(description="Simulation with selected sampling profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="ods_tail",
        choices=list(CONFIG_PROFILES.keys())
    )
    parser.add_argument(
        "--enable_slurm_array",
        action="store_true"
    )
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

    import time
    import csv

    timing_data = []

    start_idx = 1
    end_idx = 501
    task_id = 1

    if args.enable_slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 1))
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

        normalized_id = task_id - task_min
        chunk_size = (500 + task_count - 1) // task_count

        start_idx = 1 + normalized_id * chunk_size
        end_idx = min(start_idx + chunk_size, 501)

        print(f"SLURM Array Mode: Task ID {task_id}, processing indices {start_idx} to {end_idx - 1}")

    for i in range(start_idx, end_idx):
        start_time = time.time()
        digit = str(i).zfill(4)
        file_path = data_root / "Sample" / data_folder / f"{digit}.csv"
        save_path = save_dir / f"{digit}.parquet"

        if not file_path.exists():
            print(f"Warning: File missing: {file_path}, skipping...")
            continue

        print(f"Processing: {file_path} -> {save_path}")

        gain_mod = GAIN(base_config, data_info)
        gain_mod.fit(str(file_path))
        gain_mod.impute(save_path=str(save_path))

        end_time = time.time()
        iteration_time = end_time - start_time
        timing_data.append([i, iteration_time])

    csv_file_path = f"./simulations/gain_{args.profile}_iteration_times.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Time_Seconds"])
        writer.writerows(timing_data)

if __name__ == "__main__":
    main()