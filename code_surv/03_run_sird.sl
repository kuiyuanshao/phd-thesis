#!/bin/bash -e
#SBATCH --job-name=sird_sim
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-50
#SBATCH --time=96:00:00
#SBATCH --partition=milan
# #SBATCH --partition=gpu
# #SBATCH --gpus-per-node=A100:1

module purge
# module load CUDA/11.0.2
module load Python/3.11.3-gimkl-2022a

export PYTHONNOUSERSITE=1
source /home/ksha712/00_nesi_projects/uoa03789_nobackup/venv/bin/activate

export ORIGINAL_TASK_ID=$SLURM_ARRAY_TASK_ID

if [ "$ORIGINAL_TASK_ID" -le 25 ]; then
    TYPE="SampleE"
    CHUNK_ID=$ORIGINAL_TASK_ID
else
    TYPE="SampleOE"
    CHUNK_ID=$((ORIGINAL_TASK_ID - 25))
fi

# Override SLURM variables so Python's internal chunking math works correctly
export SLURM_ARRAY_TASK_ID=$CHUNK_ID
export SLURM_ARRAY_TASK_MIN=1
export SLURM_ARRAY_TASK_COUNT=25

echo "Job Started. Original Array Task ID: $ORIGINAL_TASK_ID"
echo "Processing Type: $TYPE with Chunk ID: $CHUNK_ID"

PROFILES=("srs" "balance" "neyman")

for CURRENT_PROFILE in "${PROFILES[@]}"; do
    echo "Running Python Tuning for Type: $TYPE, Profile: $CURRENT_PROFILE"
    python 03_run_simulation_sird.py --type $TYPE --profile $CURRENT_PROFILE --enable_slurm_array > /dev/null
done