import yaml
import os
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from sicg import SICG

import socket
from pathlib import Path

# Between-imputation-variance (B_m) study on the nutri SRS sample; run from SICG/ablation.

def setup_directories():
    is_nesi = "nesi.org.nz" in socket.getfqdn() or os.path.exists("/opt/nesi")
    base_path = Path("./simulations/NUTRI")
    sub_dirs = ["NONE", "BOOTSTRAP", "DROPOUT", "SWAG"]
    for folder in sub_dirs:
        target = base_path / folder
        target.mkdir(parents=True, exist_ok=True)
    return base_path
sim_root = setup_directories()
print(f"Simulation root directory: {sim_root}")

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

CONFIG_PROFILES = {
    "none": {
        "yaml_file": "./bm_configs/config_none.yaml",
        "output_dir": "NONE"
    },
    "bootstrap": {
        "yaml_file": "./bm_configs/config_bootstrap.yaml",
        "output_dir": "BOOTSTRAP"
    },
    "dropout": {
        "yaml_file": "./bm_configs/config_dropout.yaml",
        "output_dir": "DROPOUT"
    },
    "swag": {
        "yaml_file": "./bm_configs/config_swag.yaml",
        "output_dir": "SWAG"
    }
}

def run_profile(profile, n_rep, enable_slurm_array):
    selected = CONFIG_PROFILES[profile]
    yaml_file = selected["yaml_file"]
    output_dir = selected["output_dir"]

    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    start_idx = 1
    end_idx = n_rep + 1

    if enable_slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 1))
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

        normalized_id = task_id - task_min
        chunk_size = (n_rep + task_count - 1) // task_count

        start_idx = 1 + normalized_id * chunk_size
        end_idx = min(start_idx + chunk_size, n_rep + 1)

        print(f"SLURM Array Mode: Task ID {task_id}, processing indices {start_idx} to {end_idx - 1}")

    for i in range(start_idx, end_idx):
        digit = str(i).zfill(4)
        file_path_srs = f"../../code_nutri/data/Sample/SRS/{digit}.csv"
        save_path = sim_root / output_dir / f"{digit}.parquet"

        if save_path.exists():
            print(f"Output already exists, skipping: {save_path}")
            continue
        if not os.path.exists(file_path_srs):
            print(f"Warning: File missing: {file_path_srs}, skipping...")
            continue

        print(f"[{profile.upper()}] Processing: {file_path_srs}")
        sicg_mod = SICG(base_config, data_info_srs)
        sicg_mod.fit(file_path_srs)
        sicg_mod.impute(save_path=save_path)

def main():
    parser = argparse.ArgumentParser(description="Nutri B_m simulation with selected MI-approximation profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=list(CONFIG_PROFILES.keys()) + ["all"],
        help="MI-approximation profile to run ('all' loops over the four)."
    )
    parser.add_argument(
        "--n_rep",
        type=int,
        default=20,
        help="Number of light-weight replicates (SRS samples) per profile."
    )
    parser.add_argument(
        "--enable_slurm_array",
        action="store_true"
    )
    args = parser.parse_args()

    profiles = list(CONFIG_PROFILES.keys()) if args.profile == "all" else [args.profile]
    for profile in profiles:
        print(f"\n=== Nutri B_m profile: {profile} ({args.n_rep} reps) ===")
        run_profile(profile, args.n_rep, args.enable_slurm_array)

if __name__ == "__main__":
    main()
