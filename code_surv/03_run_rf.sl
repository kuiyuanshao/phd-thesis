#!/bin/bash -e
#SBATCH --job-name=rf_sim_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --array=1-50
#SBATCH --time=168:00:00
#SBATCH --partition=milan

module purge
module load R/4.3.2-foss-2023a

export SAMP="All"

if [ "$SLURM_ARRAY_TASK_ID" -le 25 ]; then
    TYPE="SampleE"
    CHUNK_ID=$SLURM_ARRAY_TASK_ID
else
    TYPE="SampleOE"
    CHUNK_ID=$((SLURM_ARRAY_TASK_ID - 25))
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Type: $TYPE with Chunk ID: $CHUNK_ID"

Rscript 03_run_simulation_rf.R $CHUNK_ID $TYPE $SAMP