#!/bin/bash
# Single-AMD-GPU run of the noisy_data GPU-parallel driver.
# Fill in the <...> site-specific fields (account, partition, module/container).
# See CLUSTER_AMD.md for the why behind every line.

#SBATCH -J noisy_vmap
#SBATCH -A <project>
#SBATCH -p <gpu-partition>          # Frontier: batch | LUMI: standard-g | Setonix: gpu | (ask your admin)
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                # 1 GPU == 1 GCD == 64 GB on MI250X (portable across sites)
#SBATCH --cpus-per-task=8           # XLA compiles ONE big fused program; give it host cores + RAM
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o noisy_vmap_%j.out

set -euo pipefail

# --- Persistent XLA compile cache (HIGH VALUE) --------------------------------
# The hessian-in-grad + vmap(~18 models) + scan(1500 epochs) graph fuses into one
# large, slow-to-compile program. Compile once, reuse across restarts / array tasks.
# Put it on a SHARED parallel filesystem, not per-task node scratch.
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-$HOME}/jax_cache"

# --- Memory (18 small MLPs fit easily in one GCD; defaults are fine) ----------
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXP_DIR="$SLURM_SUBMIT_DIR"          # submit this script from clean_experiments/noisy_data/
cd "$EXP_DIR"

# =============================================================================
# CHOOSE ONE launch route. Container is preferred on shared clusters.
# =============================================================================

# --- Route A: AMD prebuilt container via Apptainer (recommended) --------------
# apptainer pull docker://rocm/jax:rocm7.2.4-jax0.8.2-py3.12   # do once, outside the job
# SIF=path/to/jax_rocm7.2.4_0.8.2.sif
# srun apptainer exec --rocm \
#     --bind "$EXP_DIR":"$EXP_DIR" --bind "$JAX_COMPILATION_CACHE_DIR":"$JAX_COMPILATION_CACHE_DIR" \
#     "$SIF" python run_experiment_vmap.py --matmul-precision highest

# --- Route B: module + venv (only if system ROCm minor matches the plugin) ----
module load rocm                     # need a ROCm 7.2.x module; 6.x CANNOT run jax 0.8.2
source .venv/bin/activate            # a venv where `pip install "jax[rocm7-local]"` succeeded
srun python run_experiment_vmap.py --matmul-precision highest

# First run on a new machine: verify correctness before the real sweep ---------
#   srun python run_experiment_vmap.py --self-test
