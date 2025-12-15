#!/bin/bash
#SBATCH --job-name=scf
#SBATCH --nodes=1                 # Request 1 full node
#SBATCH --exclusive               # Ensure no other users share this node
#SBATCH --time=01:00:00           # Adjust runtime as needed
#SBATCH --output=../logsslurm-%j.out     # Standard output logs
#SBATCH --partition=normal-x86
#SBATCH -A f202409396cpcaa2x   
#SBATCH --mem=128G

# --- 1. Load Modules ---
module load parallel/20240722-GCCcore-13.3.0

# --- 2. Threading Safety (CRITICAL) ---
# Since we are running ~32+ independent Julia processes, we MUST force
# each one to use only 1 thread. If we don't, they will fight for resources
# and the node will freeze.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# --- 3. Determine Capacity ---
# When using --exclusive, Slurm tells us how many cores are on the node
CORES_AVAILABLE=$SLURM_CPUS_ON_NODE

echo "Job running on node: $(hostname)"
echo "Cores available: $CORES_AVAILABLE"
echo "Starting GNU Parallel run..."

# --- 4. Execution ---
# --jobs: How many tasks to run at once (equal to number of cores)
# --colsep ' ': Helps parallel parse the spaces in your args correctly
# {}: This placeholder is replaced by a single line from args.txt
# > logs/job_{#}.log: (Optional) Saves output to separate files (job_1.log, job_2.log)
# remove the "> ..." part if you want everything in the main slurm file.

mkdir -p ../logs
parallel --jobs 17 \
    "julia --project=../. ../src/main.jl {} > ../logs/run_{#}.log 2>&1" :::: args.txt

echo "All jobs finished."