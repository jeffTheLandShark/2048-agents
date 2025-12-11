#!/bin/bash
#SBATCH --job-name=exp2048
#SBATCH --output=exp2048.%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=64          # match num_workers in config
#SBATCH --mem=128G                    # adjust as needed
#SBATCH --partition=teaching         # CPU partition

set -euo pipefail

container="/data/containers/msoe-tensorflow-25.02-tf2-py3.sif"
proj_dir="/home/ad.msoe.edu/milleraa/projects/2048-agents"

# Task ID for array jobs (or 0 if no array)
task_id=${SLURM_ARRAY_TASK_ID:-0}
seed=$((1000 + task_id))

cd "${proj_dir}"

# Create temporary Python script for multiprocessing compatibility
tmp_script=$(mktemp --suffix=.py --tmpdir="${proj_dir}")

cat > "${tmp_script}" << 'PYTHON_SCRIPT'
import json
from pathlib import Path
from experiments.run_experiment import run_experiment

# Load config
config_path = Path('experiments/config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Override seed from SLURM array task ID
config['seed'] = SEED_PLACEHOLDER

# Optionally override experiment name to include seed
# config['experiment_name'] = f"{config.get('experiment_name', 'experiment')}_seed_{SEED_PLACEHOLDER}"

# Optionally override num_workers to match SLURM cpus-per-task
# config['num_workers'] = 64

print(f'Running experiment: {config.get("experiment_name", "unnamed")}')
print(f'Seed: {config.get("seed")}')
print(f'Agent: {config["agent"]["type"]}')
print(f'Games: {config["num_games"]}')
print(f'Workers: {config.get("num_workers", "sequential")}')
print()

run_experiment(config)
PYTHON_SCRIPT

# Replace seed placeholder with actual value
sed -i "s/SEED_PLACEHOLDER/${seed}/g" "${tmp_script}"

singularity exec \
  -B /data:/data \
  -B "${proj_dir}:${proj_dir}" \
  --pwd "${proj_dir}" \
  "${container}" \
  /bin/bash -lc "
    python -m pip install -r requirements.txt && \
    python '${tmp_script}'
  "

# Clean up temporary script
rm -f "${tmp_script}"

