#!/bin/bash -e
#SBATCH --job-name=rddm_tune
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --array=0-1

module purge
module load CUDA/11.0.2
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/ksha712/00_nesi_projects/uoa03789_nobackup/venv/bin/activate

echo "Job Started. Array Task ID: $SLURM_ARRAY_TASK_ID"

PROFILES=("kl" "kl_cfg")
CURRENT_TASK=${PROFILES[$SLURM_ARRAY_TASK_ID]}

echo "Running Python Tuning for: $CURRENT_TASK"
python tuning_rddm.py --profile $CURRENT_TASK
