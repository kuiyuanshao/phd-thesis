#!/bin/bash -e
#SBATCH --job-name=tabcsdi_sim
#SBATCH --partition=milan
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=168:00:00
#SBATCH --array=1-100

module purge
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/ksha712/00_nesi_projects/uoa03789_nobackup/venv/bin/activate

# NeSI doubles cpus-per-task at submission (8 -> 16 effective).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Each array task = ONE replicate (= task id), run sequentially across ALL
# (SampleType x design) combos. 100 reps each (array 1-100). Skips existing.
REP=$SLURM_ARRAY_TASK_ID
TYPES=("SampleE" "SampleOE")
PROFILES=("srs" "balance" "neyman")

for T in "${TYPES[@]}"; do
    for P in "${PROFILES[@]}"; do
        echo "Rep ${REP}, type ${T}, profile ${P} on $(hostname), ${OMP_NUM_THREADS} threads"
        python 03_run_simulation_tabcsdi.py --type "$T" --profile "$P" --start "$REP" --end "$REP"
    done
done
