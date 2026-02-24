#!/bin/bash -e
#SBATCH --job-name=rddm_tune
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
source /nesi/project/uoa03789/phd-thesis/tpvmi_rddm/my_venv/bin/activate

python tuning_rddm.py --task "ce"
