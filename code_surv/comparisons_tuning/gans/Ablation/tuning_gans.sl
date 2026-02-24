#!/bin/bash -e
#SBATCH --job-name=gans_tune
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1

module purge
module load CUDA/11.0.2
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/ksha712/00_nesi_projects/uoa03789_nobackup/venv/bin/activate

echo "Running Python Tuning"

python tuning_gans.py --task "residual_pac"