#!/bin/bash
#SBATCH --job-name=Purif_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # A64FX usually has 48 compute cores
#SBATCH --time=00:20:00
#SBATCH --output=../logs/bench_%a.out
#SBATCH --mem=64G

# 1. Define the config file path
CONFIG_FILE="params.txt"

# 2. Extract the specific arguments for this task ID
# 'sed -n ${ID}p' prints only the p-th line
ARGS=$(sed -n "${SLURM_TASK_ID}" $CONFIG_FILE)

echo "Task ID: $SLURM_TASK_ID"
echo "Running with args: $ARGS"

# 3. Run Julia
# Ensure you bind threads correctly for A64FX
export BLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=1
julia --project=../. ../src/main.jl $ARGS

