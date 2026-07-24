#!/bin/bash -e
#SBATCH --job-name=sisl_varcomp
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --array=1-20
#SBATCH --time=08:00:00
#SBATCH --partition=milan

# Dual-bootstrap vs dual-sandwich SISL variance comparison, 500 replicates in 20 chunks
# of 25. Replicate i always runs under set.seed(sim_seed[i]) from
# ../../code_infsamp/data/params/simulation_seed.RData, so results are identical no matter
# how the array is sliced or resumed. Idempotent: completed replicates are skipped, so a
# re-submit only computes what is missing. Results: ./results/results_dualboot.csv and
# ./results/results_dualsandwich.csv (one row per replicate per file).
# R package requirements beyond base: statmod, numDeriv (Rfast optional).

module purge
module load R/4.3.2-foss-2023a

echo "Task ID: $SLURM_ARRAY_TASK_ID (SISL variance comparison, replicates $(( ($SLURM_ARRAY_TASK_ID-1)*25+1 ))-$(( $SLURM_ARRAY_TASK_ID*25 )))"

Rscript run_comparison.R $SLURM_ARRAY_TASK_ID
