"""Noise layer for the FIRE noisy-LOS experiment.

Copied verbatim from clean_experiments/noisy_data/noisy_datasets.py (the
noise machinery is truth-agnostic — sigmas live in physical mm/s/yr and the
injection acts on scaled arrays through the fitted a-transformer). Copied
rather than imported because noisy_datasets.py does
`from datasets import ...` against ITS sibling directory, which would
collide with this experiment's own datasets.py on sys.path.

Noise is ABSOLUTE (pulsar-timing errors don't scale with signal), injected
along the sightline: a_noisy = a + eps*nhat, eps ~ N(0, sigma) — the LOS
projection a·nhat gains exactly eps.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# 1 mm/s/yr = 1 km/s/Myr = 1/977.79222 kpc/Myr^2
MMSYR_TO_KPCMYR2 = 1.0 / 977.79222

# Per-pulsar sigma(a_LOS) [mm/s/yr] of the catalog sample defined in
# ../mock_catalog/catalog.py (Donlon+2025 minus its EXCLUDE list), sorted.
# ~10th/50th/75th percentile = 0.52 / 1.68 / 3.88 (the constant levels below predate this).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # clean_experiments/
from mock_catalog.catalog import load_catalog, sigma_pool_mmsyr  # noqa: E402

CATALOG_SIGMAS_MMSYR = tuple(float(s) for s in sigma_pool_mmsyr(load_catalog()))

# Arms of the same family share the sigma-draw seed -> identical noisy data;
# arm comparisons are loss comparisons, never draw luck.
NOISE_FAMILY_SEED = {"s0": 0, "s0.5": 1, "s1.7": 2, "s4.3": 3, "het": 4}


def sigma_scale_factor(cfg):
    """Scaled acceleration units per 1 mm/s/yr, from the fitted a-transformer
    (an isotropic constant factor, so one probe vector measures it exactly)."""
    eps = 1e-3
    a_fac = float(jnp.linalg.norm(
        cfg["a_transformer"].transform(jnp.array([[eps, 0.0, 0.0]])))) / eps
    return MMSYR_TO_KPCMYR2 * a_fac


def draw_sigmas_mmsyr(rng, n, family):
    """'sX' -> constant level X; 'het' -> resampled with replacement from the
    catalog sigmas."""
    if family == "het":
        return rng.choice(np.asarray(CATALOG_SIGMAS_MMSYR), size=n, replace=True)
    return np.full(n, float(family[1:]))


def apply_los_noise(rng, a_n, n_vecs, sig_scaled):
    """a + eps*nhat (scaled units), eps ~ N(0, sig_scaled) per point.
    sig_scaled == 0 returns a_n unchanged, bit-exact."""
    eps = rng.normal(0.0, 1.0, size=len(sig_scaled)) * np.asarray(sig_scaled)
    return a_n + jnp.asarray(eps)[:, None] * n_vecs


def arm_weights(sig_mmsyr, arm):
    """'plain' -> None; 'W' -> w ∝ 1/sigma (whitened L1); 'C' -> w ∝ 1/sigma^2
    (chi^2 with los_loss='sq'). W/C mean-normalized to 1."""
    if arm == "plain":
        return None
    if arm not in ("W", "C"):
        raise ValueError(f"unknown arm '{arm}' (use 'plain', 'W', or 'C').")
    w = 1.0 / np.asarray(sig_mmsyr) ** (1 if arm == "W" else 2)
    return jnp.asarray(w / w.mean())
