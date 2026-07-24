#!/bin/bash -e
#SBATCH --job-name=sicg_sim_infsamp
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-20
#SBATCH --time=10:00:00
#SBATCH --partition=milan
# #SBATCH --partition=gpu
# #SBATCH --gpus-per-node=A100:1

module purge
# module load CUDA/11.0.2
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/rkan531/00_nesi_projects/uoa04769_nobackup/venv/bin/activate

# NeSI doubles cpus-per-task at submission. Confine BLAS/OMP to the allocated
# cores so concurrent array tasks do not over-subscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job Started. Array Task ID: $SLURM_ARRAY_TASK_ID (InfSamp)"

python 03_run_simulation_sicg.py --profile infsamp --enable_slurm_array > /dev/null
