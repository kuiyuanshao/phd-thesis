import argparse, pandas as pd, sys, os, yaml, time, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../benchmarks/GAIN")))
from tuner import BivariateTuner

data_info = {
    "cat_vars": ["Z3", "R", "W"],
    "num_vars": ["Z1", "Z2", "X", "Xt", "Y"],
    "phase2_vars": ["X"],
    "phase1_vars": ["Xt"],
}

CONFIG_PROFILES = {
    "gain": {
        "yaml_file": "base_config_gain.yaml",
        "output_csv": "gain_tuning_results.csv",
        "tuning_grid": {
            "lr": ("log_float", 1e-4, 1e-1),
            "alpha": ("int", 1, 500),
            "beta": ("int", 1, 100),
            "batch_size": ("int_exp2", 4, 8),
            "hint_rate": ("float", 0.5, 0.95),
        },
    },
    "gain_epoch": {
        "yaml_file": "base_config_gain_epoch.yaml",
        "output_csv": "gain_epoch_tuning_results.csv",
        "tuning_grid": {"epochs": ("int", 5000, 50000)},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="gain", choices=list(CONFIG_PROFILES.keys()))
    args = parser.parse_args()
    sel = CONFIG_PROFILES[args.profile]
    num_params = len(sel["tuning_grid"]); total_trials = 10 if num_params == 1 else max(30, 7 * num_params)
    with open(sel["yaml_file"], "r") as f:
        base_config = yaml.safe_load(f)
    df = pd.read_csv("../../data/Sample/InfSamp/0001.csv").loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]

    t0 = time.time()
    tuner = BivariateTuner(df, base_config, sel["tuning_grid"], data_info, n_splits=3)
    tuner.tune(n_trials=total_trials, output_csv=sel["output_csv"])
    elapsed = time.time() - t0

    with open(f"{args.profile}_tuning_time.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Time_Seconds"]); w.writerow([elapsed])


if __name__ == "__main__":
    main()
