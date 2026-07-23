"""Dataset construction for the noisy_data experiment.

Repeats clean_experiments/sun_los_recovery with ONE twist: Gaussian measurement
noise on the observed LOS accelerations. Geometry, truth, and scaling are the
parent experiment's, imported verbatim — this module only adds the noise layer:
per-point sigmas (constant levels or resampled from the real Donlon+2025
catalog), unit conversion mm/s/yr -> scaled, seeded injection along the
sightline, and the per-arm importance weights.

Noise is ABSOLUTE (assuming that pulsar-timing errors don't scale with signal) and injected
as a_noisy = a + eps*nhat, eps ~ N(0, sigma): the LOS loss only ever sees the
projection a·nhat, which receives exactly the intended scalar noise. Doing it this way we avoid
edits in src.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Parent experiment geometry/truth/scaling — reused verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sun_los_recovery"))
from datasets import GC, HALO_RS, OBSERVER, build_data, sample_colloc_scaled

__all__ = [
    "GC", "HALO_RS", "OBSERVER", "build_data", "sample_colloc_scaled",
    "CATALOG_SIGMAS_MMSYR", "MMSYR_TO_KPCMYR2", "NOISE_FAMILY_SEED",
    "sigma_scale_factor", "draw_sigmas_mmsyr", "apply_los_noise", "arm_weights",
]



# 1 mm/s/yr = 1 km/s/Myr = 1/977.79222 kpc/Myr^2 (the LOS-acceleration currency
# of pulsar timing)
MMSYR_TO_KPCMYR2 = 1.0 / 977.79222

# Per-pulsar sigma(a_LOS) [mm/s/yr] of the 51 Donlon+2025 pulsars selected by
# , sorted. Embedded as literals
# (6 significant figures)
# ~10th/50th/75th percentile = 0.52 / 1.69 / 4.26 set the constant levels.
CATALOG_SIGMAS_MMSYR = (
    0.118371, 0.326056, 0.33556, 0.339927, 0.420831, 0.523387,
    0.532496, 0.554591, 0.576837, 0.670016, 0.726771, 0.745114,
    0.763416, 0.770922, 0.851857, 0.915736, 0.920055, 1.01904,
    1.0872, 1.36575, 1.46129, 1.46334, 1.50857, 1.64275,
    1.68204, 1.69031, 1.82323, 1.87895, 1.92658, 2.06964,
    2.41148, 2.77526, 2.79087, 2.81604, 3.15476, 3.16167,
    3.64418, 4.12575, 4.4002, 4.53605, 5.95132, 6.07849,
    6.35578, 6.93263, 7.12839, 7.36009, 8.57286, 13.0403,
    13.5344, 18.2143, 18.2143,
)

# Noise FAMILY = which sigma draw a setting uses. Arms of the same family
# (het / hetW / hetC, or s1.7 / s1.7C) share the seed, hence the SAME noise
# realization — arm comparisons are then loss comparisons, not draw luck.
NOISE_FAMILY_SEED = {"s0": 0, "s0.5": 1, "s1.7": 2, "s4.3": 3, "het": 4}


def sigma_scale_factor(cfg):
    """Scaled acceleration units per 1 mm/s/yr, from the fitted a-transformer.

    The transformer is an isotropic constant factor, so one probe vector
    measures it exactly.
    """

    eps = 1e-3
    a_fac = float(jnp.linalg.norm(
        cfg["a_transformer"].transform(jnp.array([[eps, 0.0, 0.0]])))) / eps # to get the scaler
    return MMSYR_TO_KPCMYR2 * a_fac


def draw_sigmas_mmsyr(rng, n, family):
    """Per-point sigma in mm/s/yr for a noise family.

    'sX' -> constant level X for all n points; 'het' -> resampled with
    replacement from the 51 real catalog sigmas.
    """
    if family == "het":
        return rng.choice(np.asarray(CATALOG_SIGMAS_MMSYR), size=n, replace=True)
    return np.full(n, float(family[1:]))


def apply_los_noise(rng, a_n, n_vecs, sig_scaled):
    """Perturb the 3D targets along the sightline: a + eps*nhat (scaled units).

    eps ~ N(0, sig_scaled) per point. The LOS projection a·nhat gains exactly
    eps; transverse components change too, but no LOS mode ever reads them.
    sig_scaled == 0 returns a_n unchanged, bit-exact.
    """
    eps = rng.normal(0.0, 1.0, size=len(sig_scaled)) * np.asarray(sig_scaled)
    return a_n + jnp.asarray(eps)[:, None] * n_vecs


def arm_weights(sig_mmsyr, arm):
    """Per-point importance weights for a training arm.

    'plain' -> None (unweighted);
    'W' -> w ∝ 1/sigma (whitened L1);
    'C' -> w ∝ 1/sigma^2 (chi^2 when combined with los_loss='sq').
    W and C are mean-normalized to 1 so the anchor/data balance inside the
    shared LOS mean is untouched at init (weights=ones == unweighted,
    verified bit-exact); being scale-free they use mm/s/yr directly.
    """
    if arm == "plain":
        return None
    if arm not in ("W", "C"):
        raise ValueError(f"unknown arm '{arm}' (use 'plain', 'W', or 'C').")
    w = 1.0 / np.asarray(sig_mmsyr) ** (1 if arm == "W" else 2)
    return jnp.asarray(w / w.mean())
