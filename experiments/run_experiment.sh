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

# Create wrapper script for multiprocessing compatibility
wrapper_script="${proj_dir}/experiments/run_slurm_experiment.py"

cat > "${wrapper_script}" << 'WRAPPER_SCRIPT'
import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.run_experiment import run_experiment

if __name__ == '__main__':
    # Load config
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Override seed from SLURM environment variable
    if 'SLURM_SEED' in os.environ:
        config['seed'] = int(os.environ['SLURM_SEED'])

    # Optionally override experiment name to include seed
    # config['experiment_name'] = f"{config.get('experiment_name', 'experiment')}_seed_{config['seed']}"

    # Optionally override num_workers to match SLURM cpus-per-task
    # config['num_workers'] = 64

    print(f'Running experiment: {config.get("experiment_name", "unnamed")}')
    print(f'Seed: {config.get("seed")}')
    print(f'Agent: {config["agent"]["type"]}')
    print(f'Games: {config["num_games"]}')
    print(f'Workers: {config.get("num_workers", "sequential")}')
    print()

    run_experiment(config)
WRAPPER_SCRIPT

singularity exec \
  -B /data:/data \
  -B "${proj_dir}:${proj_dir}" \
  --pwd "${proj_dir}" \
  "${container}" \
  /bin/bash -lc "
    export SLURM_SEED=${seed} && \
    python -m pip install -r requirements.txt && \
    python '${wrapper_script}'
  "

# Clean up wrapper script
rm -f "${wrapper_script}"

