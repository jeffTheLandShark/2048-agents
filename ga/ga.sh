#!/bin/bash
#SBATCH --job-name=ga2048
#SBATCH --output=ga2048.%j.out
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16          # match --num-workers
#SBATCH --mem=16G                   # adjust as needed
#SBATCH --partition=teaching         # CPU partition
#SBATCH --array=0-3                # uncomment for multiple seeds

set -euo pipefail

container="/data/containers/msoe-tensorflow-25.02-tf2-py3.sif"
proj_dir="/home/ad.msoe.edu/milleraa/projects/2048-agents"

# Seed per array task (or single run if no array)
task_id=${SLURM_ARRAY_TASK_ID:-0}
seed=$((1000 + task_id))

cd "${proj_dir}"

singularity exec \
  -B /data:/data \
  -B "${proj_dir}:${proj_dir}" \
  --pwd "${proj_dir}" \
  "${container}" \
  /bin/bash -lc "
    python -m pip install -r requirements.txt && \
    python -m ga.ga_runner \
      --population-size 30 \
      --generations 20 \
      --games-per-genome 3 \
      --depth-limit 3 \
      --time-limit-ms 75 \
      --num-workers 16 \
      --output-dir data/ga_results/run_${task_id} \
      --seed ${seed}
  "