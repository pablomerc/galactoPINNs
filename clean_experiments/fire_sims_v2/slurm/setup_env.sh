#!/usr/bin/env bash
# One-time environment setup for loo_vs_chi.py on an NVIDIA GPU cluster.
# Run from the galactoPINNs repo root on a node with internet access (login node is fine):
#   bash clean_experiments/fire_sims_v2/slurm/setup_env.sh
# Afterwards ALWAYS call .venv/bin/python directly (not `uv run`): a uv sync would
# drop the CUDA plugin installed below, silently falling back to CPU.
set -euo pipefail

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 1) the locked environment (jax/jaxlib 0.8.2, flax 0.12.2, galax from git, ...)
uv sync --frozen

# 2) the CUDA backend for the SAME jax version (pip wheels bundle CUDA 12 + cuDNN;
#    needs an NVIDIA driver >= 525). Swap cuda12 -> cuda13 only if the driver requires it.
uv pip install --python .venv/bin/python "jax[cuda12]==0.8.2"

# 3) report (prints cpu on a login node without a GPU — that is fine)
.venv/bin/python - <<'PY'
import jax, flax, optax
print("jax", jax.__version__, "flax", flax.__version__, "optax", optax.__version__)
print("devices:", jax.devices())
PY
