#!/bin/bash -e
#SBATCH --job-name=rddm_tune
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1

module purge
module load CUDA/11.0.2
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /nesi/project/uoa03789/phd-thesis/tpvmi_rddm/my_venv/bin/activate

echo "Job Started. Array Task ID: $SLURM_ARRAY_TASK_ID"

TASKS=("srs" "bal" "ney")
CURRENT_TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "Running Python Tuning for: $CURRENT_TASK"

python tuning_rddm.py --task $CURRENT_TASK > /dev/null 2>&1

echo "Python finished for $CURRENT_TASK."