#!/bin/bash -e
#SBATCH --job-name=tabcsdi_sim
#SBATCH --partition=milan
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=168:00:00
#SBATCH --array=1-100

module purge
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/ksha712/00_nesi_projects/uoa03789_nobackup/venv/bin/activate

# NeSI doubles cpus-per-task at submission (8 -> 16 effective).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Each array task = ONE replicate (= task id), run sequentially across ALL designs.
# 100 reps per design (array 1-100). Existing results are skipped by the python.
REP=$SLURM_ARRAY_TASK_ID
DESIGNS=("srs" "rs" "wrs" "sfs" "ods_tail" "neyman_ods" "neyman_inf")

for D in "${DESIGNS[@]}"; do
    echo "Rep ${REP}, design ${D} on $(hostname), ${OMP_NUM_THREADS} threads"
    python 03_run_simulation_tabcsdi.py --profile "$D" --start "$REP" --end "$REP"
done
