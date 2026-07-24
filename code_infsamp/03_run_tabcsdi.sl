#!/bin/bash -e
#SBATCH --job-name=tabcsdi_sim_infsamp
#SBATCH --partition=milan
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=20:00:00
#SBATCH --array=1-100

module purge
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/rkan531/00_nesi_projects/uoa04769_nobackup/venv/bin/activate

# NeSI doubles cpus-per-task at submission (8 -> 16 effective). Confine BLAS/OMP
# to the allocated cores so concurrent array tasks don't over-subscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Array Task ID: $SLURM_ARRAY_TASK_ID (InfSamp)"

# One replicate per array task (array 1-100 => first 100 replicates only), matching the
# nutri/surv/vccc TabCSDI runs.
python 03_run_simulation_tabcsdi.py --profile infsamp --enable_slurm_array
