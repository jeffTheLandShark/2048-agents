#!/bin/bash
#SBATCH --job-name=ga2048
#SBATCH --output=ga2048.%A_%a.out
#SBATCH --error=ga2048.%A_%a.err
#SBATCH --time=03:00:00          # pad to 90m if needed
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G                # adjust if needed
#SBATCH --array=0-3              # four parallel runs
#SBATCH --partition=teaching      # adjust to your cluster

seed=$((1000 + SLURM_ARRAY_TASK_ID))

python -m ga.ga_runner \
  --population-size 30 \
  --generations 20 \
  --games-per-genome 3 \
  --depth-limit 3 \
  --time-limit-ms 75 \
  --num-workers 16 \
  --output-dir data/ga_results/run_${SLURM_ARRAY_TASK_ID} \
  --seed ${seed}