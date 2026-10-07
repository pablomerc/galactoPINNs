#!/usr/bin/env bash
# LOO vs chi-stop: one array task per (n, trial) = 7 x 10 = 70 tasks, one GPU each.
# Submit from the galactoPINNs repo root:
#   sbatch --array=0-69%8 clean_experiments/fire_sims_v2/slurm/loo_vs_chi.sbatch
# (%8 = at most 8 GPUs at once; adjust to your allocation). Tasks are independent and
# resumable: a finished (n, trial) shard is skipped, so resubmitting the array is safe.
#
# EDIT for your cluster: partition/account/qos lines, GPU type, time limit.
#SBATCH --job-name=loo_vs_chi
#SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=clean_experiments/fire_sims_v2/logs/loo_vs_chi_%A_%a.log

set -euo pipefail
N_LIST=(20 50 100 150 200 250 500)
TRIALS_PER_N=10
ID=${SLURM_ARRAY_TASK_ID:-0}
N=${N_LIST[$((ID / TRIALS_PER_N))]}
TRIAL=$((ID % TRIALS_PER_N))

# folds per vmap batch: 0 = all n folds at once. Lower (e.g. 128) if n=500 runs out
# of GPU memory; results do not depend on it (gated bit-identical on CPU).
FOLD_CHUNK=${FOLD_CHUNK:-0}

# reproducible GPU reductions (small speed cost); comment out to trade it for speed
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"

echo "host $(hostname)  task $ID -> n=$N trial=$TRIAL  fold_chunk=$FOLD_CHUNK"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
.venv/bin/python clean_experiments/fire_sims_v2/loo_vs_chi.py \
    --n-list "$N" --trials "$TRIAL" --fold-chunk "$FOLD_CHUNK" \
    --outdir clean_experiments/fire_sims_v2/results_loo_vs_chi
