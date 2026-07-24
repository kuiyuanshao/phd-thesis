import os
import argparse
import time
import csv
import yaml
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from SICG.sicg import SICG

# Released design weight W enters ONLY as a POTENTIAL COVARIATE, encoded categorically
# (W takes one value per joint design stratum, so its levels act as stratum labels).
# weight_var is deliberately NOT set: the internal training-weighting machinery stays off,
# so W is offered to the model like any other released variable, nothing more.
data_info_infsamp = {
    "cat_vars": ["Z3", "R", "W"],
    "num_vars": ["Z1", "Z2", "X", "Xt", "Y"],
    "phase2_vars": ["X"],
    "phase1_vars": ["Xt"],
}
DATA_INFOS = {"infsamp": data_info_infsamp}
DESIGN_FOLDER = {"infsamp": "InfSamp"}


def main():
    parser = argparse.ArgumentParser(description="SICG imputation for the InfSamp design.")
    parser.add_argument("--profile", type=str, default="infsamp", choices=list(DATA_INFOS.keys()))
    parser.add_argument("--enable_slurm_array", action="store_true")
    args = parser.parse_args()

    data_info = DATA_INFOS[args.profile]; folder = DESIGN_FOLDER[args.profile]
    with open("./data/Config/best_config_sicg.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    data_root = Path("./data")
    save_dir = Path("./simulations") / folder / "sicg"
    save_dir.mkdir(parents=True, exist_ok=True)

    start_idx, end_idx, task_id = 1, 501, 1
    if args.enable_slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 1))
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        normalized_id = task_id - task_min
        chunk_size = (500 + task_count - 1) // task_count
        start_idx = 1 + normalized_id * chunk_size
        end_idx = min(start_idx + chunk_size, 501)
        print(f"SLURM Array Mode: Task ID {task_id}, indices {start_idx} to {end_idx - 1}")

    timing = []
    for i in range(start_idx, end_idx):
        t0 = time.time(); digit = str(i).zfill(4)
        fp = data_root / "Sample" / folder / f"{digit}.csv"; sp = save_dir / f"{digit}.parquet"
        if not fp.exists():
            print(f"missing {fp}, skipping"); continue
        if sp.exists():
            continue
        sicg_mod = SICG(base_config, data_info)
        sicg_mod.fit(str(fp))
        sicg_mod.impute(save_path=str(sp))
        timing.append([i, time.time() - t0])

    os.makedirs("./simulations/time", exist_ok=True)
    with open(f"./simulations/time/sicg_{args.profile}_iteration_times_task_{task_id}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Iteration", "Time_Seconds"]); w.writerows(timing)


if __name__ == "__main__":
    main()
