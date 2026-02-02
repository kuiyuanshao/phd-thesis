#!/bin/bash -e
#SBATCH --job-name=rddm_tune
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1

# --- 1. Load Environment ---
source /usr/share/lmod/lmod/init/bash
module use /opt/nesi/lmod/generic
module use /opt/nesi/lmod/zen3
module use /opt/nesi/lmod/mahuika

module purge
module load Python/3.11.3-gimkl-2022a


echo "Job Started. Array Task ID: $SLURM_ARRAY_TASK_ID"

TASKS=("srs" "bal" "ney")
CURRENT_TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "Running Python Tuning for: $CURRENT_TASK"

python tuning_rddm.py --task $CURRENT_TASK

echo "Python finished for $CURRENT_TASK."