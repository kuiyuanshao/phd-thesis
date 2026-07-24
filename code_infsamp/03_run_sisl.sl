#!/bin/bash -e
#SBATCH --job-name=sisl_sim_infsamp
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-20
#SBATCH --time=11:00:00
#SBATCH --partition=milan

module purge
module load R/4.3.2-foss-2023a

echo "Task ID: $SLURM_ARRAY_TASK_ID (InfSamp, single design)"

Rscript 03_run_simulation_sisl.R $SLURM_ARRAY_TASK_ID
