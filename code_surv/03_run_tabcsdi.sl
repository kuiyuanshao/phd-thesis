#!/bin/bash -e
#SBATCH --job-name=tabcsdi_sim
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-5
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1

module purge
module load CUDA/11.0.2
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/ksha712/00_nesi_projects/uoa03789_nobackup/venv/bin/activate

TYPES=("SampleE" "SampleOE")
PROFILES=("srs" "balance" "neyman")

CURRENT_TYPE=${TYPES[$((SLURM_ARRAY_TASK_ID / 3))]}
CURRENT_PROFILE=${PROFILES[$((SLURM_ARRAY_TASK_ID % 3))]}

echo "Job Started. Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Python Tuning for Type: $CURRENT_TYPE, Profile: $CURRENT_PROFILE"

python 03_run_simulation_tabcsdi.py --type $CURRENT_TYPE --profile $CURRENT_PROFILE