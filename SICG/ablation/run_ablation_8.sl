#!/bin/bash -e
#SBATCH --job-name=sicg_ablation
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-25
#SBATCH --time=48:00:00
#SBATCH --partition=milan
# #SBATCH --partition=gpu
# #SBATCH --gpus-per-node=A100:1

module purge
# module load CUDA/11.0.2
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/ksha712/00_nesi_projects/uoa03789_nobackup/venv/bin/activate

echo "Job Started. Array Task ID: $SLURM_ARRAY_TASK_ID"

# BOOTSTRAP trains m=5 models per replicate (~5x the single-model runs); raise
# --array or --time if a task hits the walltime.
PROFILES=("bootstrap")

for CURRENT_TASK in "${PROFILES[@]}"; do
    echo "Running Python simulation for: $CURRENT_TASK"
    python run_ablation.py --profile $CURRENT_TASK --enable_slurm_array
done
