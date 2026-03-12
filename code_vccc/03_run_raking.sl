#!/bin/bash -e
#SBATCH --job-name=raking_sim_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-25
#SBATCH --time=96:00:00
#SBATCH --partition=milan

module purge
module load R/4.3.2-foss-2023a

export SAMP="All"

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing all sampling designs for this chunk."

Rscript 03_run_simulation_raking.R $SLURM_ARRAY_TASK_ID $SAMP