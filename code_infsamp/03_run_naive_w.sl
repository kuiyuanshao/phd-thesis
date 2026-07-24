#!/bin/bash -e
#SBATCH --job-name=naive_w_sim_infsamp
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=1-20
#SBATCH --time=02:00:00
#SBATCH --partition=milan

module purge
module load R/4.3.2-foss-2023a

echo "Task ID: $SLURM_ARRAY_TASK_ID (InfSamp, single design)"

Rscript 03_run_simulation_naive_w.R $SLURM_ARRAY_TASK_ID
