import os, argparse, yaml, time, csv
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from benchmarks.GAIN.gain import GAIN

data_info_infsamp = {
    "cat_vars": ["Z3", "R", "W"],
    "num_vars": ["Z1", "Z2", "X", "Xt", "Y"],
    "phase2_vars": ["X"],
    "phase1_vars": ["Xt"],
}
DATA_INFOS = {"infsamp": data_info_infsamp}
DESIGN_FOLDER = {"infsamp": "InfSamp"}


def main():
    parser = argparse.ArgumentParser(description="GAIN imputation (Design 0).")
    parser.add_argument("--profile", type=str, default="infsamp", choices=list(DATA_INFOS.keys()))
    parser.add_argument("--enable_slurm_array", action="store_true")
    args = parser.parse_args()

    data_info = DATA_INFOS[args.profile]; folder = DESIGN_FOLDER[args.profile]
    with open("./data/Config/best_config_gain.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    data_root = Path("./data"); save_dir = Path("./simulations") / folder / "gain"
    save_dir.mkdir(parents=True, exist_ok=True)

    start_idx, end_idx, task_id = 1, 501, 1
    if args.enable_slurm_array:
        task_id    = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        task_min   = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 1))
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        nid = task_id - task_min; chunk = (500 + task_count - 1) // task_count
        start_idx = 1 + nid * chunk; end_idx = min(start_idx + chunk, 501)

    timing = []
    for i in range(start_idx, end_idx):
        t0 = time.time(); digit = str(i).zfill(4)
        fp = data_root / "Sample" / folder / f"{digit}.csv"; sp = save_dir / f"{digit}.parquet"
        if not fp.exists():
            print(f"missing {fp}, skipping"); continue
        if sp.exists():
            continue
        gain_mod = GAIN(base_config, data_info)
        gain_mod.fit(str(fp))
        gain_mod.impute(save_path=str(sp))
        timing.append([i, time.time() - t0])

    os.makedirs("./simulations/time", exist_ok=True)
    with open(f"./simulations/time/gain_{args.profile}_iteration_times_task_{task_id}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Iteration", "Time_Seconds"]); w.writerows(timing)


if __name__ == "__main__":
    main()
