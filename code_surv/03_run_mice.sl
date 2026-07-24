#!/bin/bash -e
#SBATCH --job-name=mice_sim_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-50
#SBATCH --time=96:00:00
#SBATCH --partition=milan

module purge
module load R/4.3.2-foss-2023a

export SAMP="All"
MAX_TASKS=$SLURM_ARRAY_TASK_MAX
HALF_TASKS=$((MAX_TASKS / 2))

if [ "$SLURM_ARRAY_TASK_ID" -le "$HALF_TASKS" ]; then
    TYPE="SampleE"
    CHUNK_ID=$SLURM_ARRAY_TASK_ID
else
    TYPE="SampleOE"
    CHUNK_ID=$((SLURM_ARRAY_TASK_ID - HALF_TASKS))
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Type: $TYPE with Chunk ID: $CHUNK_ID"
echo "Total chunks per type: $HALF_TASKS"

Rscript 03_run_simulation_mice.R $CHUNK_ID $TYPE $SAMP $HALF_TASKS