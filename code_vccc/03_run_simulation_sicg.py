import os
import socket
import argparse
import yaml
import time
import csv
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from SICG.sicg import SICG

data_info_srs = {
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


CONFIG_PROFILES = generate_config_profiles("sicg")

print(f"Loaded {len(CONFIG_PROFILES)} profiles for the sicg model.")


def setup_env(data_folder):
    is_nesi = "nesi.org.nz" in socket.getfqdn() or os.path.exists("/opt/nesi")
    sim_root = Path("./simulations")
    data_root = Path("./data")
    save_dir = sim_root / data_folder / "sicg"
    save_dir.mkdir(parents=True, exist_ok=True)

    return data_root, save_dir, sim_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        default="srs",
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
    data_info = DATA_INFOS[selected["info_key"]]

    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    data_root, save_dir, sim_root = setup_env(data_folder)
    print(f"[{args.profile.upper()}] Environment ready.")

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

        if not save_path.exists():
            print(f"[{data_folder}] Result not found. Processing: {file_path} -> {save_path}")
            sicg_mod = SICG(base_config, data_info)
            sicg_mod.fit(str(file_path))
            sicg_mod.impute(save_path=str(save_path))
        else:
            print(f"[{data_folder}] Result exists at {save_path}. Skipping.")

        end_time = time.time()
        iteration_time = end_time - start_time
        timing_data.append([i, iteration_time])

    if timing_data:
        csv_file_path = sim_root / f"sicg_{args.profile}_iteration_times_task_{task_id}.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Time_Seconds"])
            writer.writerows(timing_data)
        print(f"Saved timing data to {csv_file_path}")


if __name__ == "__main__":
    main()