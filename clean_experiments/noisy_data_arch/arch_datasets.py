"""Per-architecture data + model factory for the noisy_data_arch experiment.

Compares three rungs of the PINN ladder (paper Sec. 2.4-2.6) on the SAME
noisy line-of-sight problem as clean_experiments/noisy_data, restricted to:
  - one eval region: sun4 (the 4 kpc Sun ball, = the training ball)
  - one noise family: 'het' (per-point sigma resampled from the 51 real
    Donlon+2025 catalog values)
  - one arm: 'W' (importance weight w ~ 1/sigma, L1 loss)

  PINN III : scale='nfw', include_analytic=False       (NN + radial scaling)
  PINN IV  : + fixed NFW analytic baseline              (NN learns residual)
  PINN V   : + trainable NFW baseline, joint training   ((m, r_s) learned)

include_analytic=True changes the fitted a_star (data.py scale_data uses the
residual u - u_analytic for u_star), so IV/V get a different a_transformer --
hence a different sigma_scale_factor -- than III. The comparison stays valid:
recovery error is relative (scale-free), and the SAME physical noise draw
(mm/s/yr) is converted to each arch's scaled units via its own factor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flax import nnx
import galax.potential as gp

# Sibling experiment modules: geometry/scaling from sun_los_recovery, the
# noise layer from noisy_data. In situ (clean_experiments/noisy_data_arch/)
# these live one level up; fall back to $GALACTOPINNS_CLEAN or the known repo
# path so the module also imports from a scratch dir.
def _find_clean():
    cands = [Path(__file__).resolve().parent.parent,
             Path(os.environ.get("GALACTOPINNS_CLEAN", "")),
             Path("/Users/pablom.perez/Desktop/Linas-group/galactoPINNs/clean_experiments")]
    for c in cands:
        if (c and (c / "sun_los_recovery" / "datasets.py").exists()
                and (c / "noisy_data" / "noisy_datasets.py").exists()):
            return c
    raise ImportError("could not locate clean_experiments (sun_los_recovery "
                      "+ noisy_data siblings)")

_CLEAN = _find_clean()
sys.path.insert(0, str(_CLEAN / "sun_los_recovery"))
sys.path.insert(0, str(_CLEAN / "noisy_data"))

from datasets import (  # noqa: E402
    GC, HALO_RS, OBSERVER, build_data, sample_colloc_scaled,
)
from noisy_datasets import (  # noqa: E402
    NOISE_FAMILY_SEED, apply_los_noise, arm_weights, draw_sigmas_mmsyr,
    sigma_scale_factor,
)
from galactoPINNs.layers import TrainableGalaxPotential  # noqa: E402
from galactoPINNs.models.static_model import StaticModel  # noqa: E402

__all__ = [
    "ARCHES", "BASE_M", "BASE_RS", "GC", "HALO_RS", "OBSERVER",
    "NOISE_FAMILY_SEED", "apply_los_noise", "arm_weights", "build_arch",
    "draw_sigmas_mmsyr", "make_model", "sample_colloc_scaled",
    "sigma_scale_factor", "trainable_mr",
]

ARCHES = ("III", "IV", "V")

# Analytic-baseline init for IV/V. Matches build_data's internal scaling NFW
# (so the scaling envelope and baseline agree at init). Override BASE_M/BASE_RS
# at the driver to test PINN V's "fix a misspecified baseline" claim.
BASE_M, BASE_RS = 5.4e11, HALO_RS


def build_arch(arch, true_potential, pool, n_val, r_train, regions):
    """Build the train pool + val regions + cfg for one architecture.

    Returns a dict with x_pool/a_pool (scaled), x_pool_phys, cfg, val_scaled,
    dist_sun, and fac (= sigma_scale_factor for THIS arch's scaling).
    """
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; use one of {ARCHES}")
    include_analytic = arch != "III"
    (x_pool, a_pool, x_pool_phys, cfg,
     val_scaled, dist_sun, x_obs, a_obs) = build_data(
        true_potential, pool, n_val, r_train, regions,
        include_analytic=include_analytic)

    # build_data returns cfg WITHOUT ab_potential (it only used one internally
    # for the residual-scaling fit). Attach what each arch's forward needs.
    cfg = dict(cfg)
    if arch == "IV":
        cfg["ab_potential"] = gp.NFWPotential(m=BASE_M, r_s=BASE_RS, units="galactic")
        cfg["trainable"] = False
    elif arch == "V":
        cfg["trainable"] = True  # baseline lives in the trainable layer, not cfg

    fac = sigma_scale_factor(cfg)
    return {
        "arch": arch, "x_pool": x_pool, "a_pool": a_pool,
        "x_pool_phys": x_pool_phys, "cfg": cfg, "val_scaled": val_scaled,
        "dist_sun": dist_sun, "fac": fac,
    }


def make_model(arch, cfg, seed, m=BASE_M, r_s=BASE_RS):
    """Construct the StaticModel for an architecture (V gets a trainable layer)."""
    if arch == "V":
        layer = TrainableGalaxPotential(
            PotClass=gp.NFWPotential,
            init_kwargs={"m": m, "r_s": r_s},
            trainable=("m", "r_s"),
        )
        return StaticModel(cfg, rngs=nnx.Rngs(seed), trainable_analytic_layer=layer)
    return StaticModel(cfg, rngs=nnx.Rngs(seed))


def trainable_mr(model):
    """Current (m, r_s) of a PINN V model's trainable baseline (else None)."""
    layer = getattr(model, "trainable_analytic_layer", None)
    if layer is None:
        return None
    p = layer._get_built_params()
    return float(p["m"]), float(p["r_s"])
