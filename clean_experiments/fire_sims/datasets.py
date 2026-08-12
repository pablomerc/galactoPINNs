"""Dataset construction for the fire_los_recovery experiment.

Drop-in replacement for clean_experiments/sun_los_recovery/datasets.py with
the analytic galax truth swapped for the FIRE m12i snapshot truth built by
fire_truth.py (cache/truth_<tag>.npz). Everything downstream of build_data
(driver, losses, eval, plots) consumes only arrays, so the return contract
is reproduced exactly — plus scaled SMOOTHED-reference val labels, used for
the second error metric (PINN error vs unresolvable small-scale power).

Tracer positions are real star particles (age > 1 Gyr by default), so there
is no density function and no rejection sampling here: the sampling already
happened against the simulation's own stellar density in fire_truth.py.

Scaling mirrors the parent experiment: fit ONCE on the training pool via
galactoPINNs.data.scale_data with r_s = HALO_RS = 15.62 kpc (kept identical
to sun_los_recovery so scaled quantities are directly comparable; for m12i
this is a non-dimensionalization choice, not a fitted NFW radius).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import galax.potential as gp

from galactoPINNs.data import sample_angles, scale_data

OBSERVER = np.array([-8.1, 0.0, 0.0])  # Sun, principal-axis frame (kpc)
GC = np.zeros(3)
HALO_RS = 15.62  # kpc — x* and cfg["r_s"], kept from sun_los_recovery
CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_TRUTH = CACHE_DIR / "truth_old1gyr.npz"


def sample_colloc_scaled(key, n_colloc, r_min, r_max, x_transformer):
    """Uniform-in-volume points in the GC-centered shell r in (r_min, r_max)
    kpc, returned in scaled coordinates. Verbatim from sun_los_recovery
    (potential-agnostic — the rho>=0 prior is valid for any galaxy)."""
    key_r, key_ang = jr.split(key)
    u01 = jr.uniform(key_r, (n_colloc,))
    r = (u01 * (r_max**3 - r_min**3) + r_min**3) ** (1.0 / 3.0)
    theta, phi = sample_angles(key_ang, n_colloc)
    xyz_phys = jnp.stack(
        [r * jnp.sin(phi) * jnp.cos(theta),
         r * jnp.sin(phi) * jnp.sin(theta),
         r * jnp.cos(phi)],
        axis=1,
    )  # (N_c, 3) kpc
    return x_transformer.transform(xyz_phys)  # (N_c, 3) scaled


def build_data(truth_path=DEFAULT_TRUTH, n_val=4096, include_analytic=False,
               ab_potential=None, u_star_override=None):
    """FIRE analog of sun_los_recovery.datasets.build_data.

    ab_potential overrides the analytic baseline used for the residual
    u* = max|u - u_baseline| scaling when include_analytic=True (default
    None = the misspecified MW-tuned NFW, unchanged from every earlier run).

    Returns the parent's 8-tuple
        (x_train_scaled, a_train_scaled, x_pool_phys, cfg,
         val_scaled, dist_sun, x_obs, a_obs)
    plus two FIRE extras:
        val_scaled_smooth : {region: a_val_smooth_scaled}
        meta              : the truth-cache metadata dict (provenance,
                            softening, frame corrections, cross-checks)
    """
    z = np.load(truth_path)
    meta = json.loads(str(z["meta"]))
    assert np.allclose(meta["observer_kpc"], OBSERVER), \
        "truth cache was built with a different observer position"
    regions = list(meta["regions"].keys())
    for name in regions:
        have = z[f"x_val|{name}"].shape[0]
        if have < n_val:
            raise ValueError(
                f"truth cache holds only {have} val points for {name!r} "
                f"but n_val={n_val} was requested — rebuild the cache with "
                f"a larger --n-val or lower the driver's --n-val")

    x_pool_phys = np.asarray(z["x_pool"], dtype=np.float64)
    a_pool_phys = np.asarray(z["a_pool"])
    u_pool_phys = np.asarray(z["u_pool"])

    # --- fit the non-dimensionalization ONCE, on the train pool only.
    # scale_data requires x/a/u_val keys; the first region satisfies the
    # signature without influencing the constant-factor fit.
    first = regions[0]
    raw = {
        "x_train": jnp.asarray(x_pool_phys),
        "a_train": jnp.asarray(a_pool_phys),
        "u_train": jnp.asarray(u_pool_phys),
        "x_val": jnp.asarray(z[f"x_val|{first}"][:n_val]),
        "a_val": jnp.asarray(z[f"a_val|{first}"][:n_val]),
        "u_val": jnp.asarray(z[f"u_val|{first}"][:n_val]),
    }
    if ab_potential is None:
        ab_potential = gp.NFWPotential(m=5.4e11, r_s=HALO_RS,
                                       units="galactic")
    scale_config = {"r_s": HALO_RS, "include_analytic": include_analytic,
                    "ab_potential": ab_potential, "u_star_override": u_star_override}
    scaled, tf = scale_data(raw, scale_config)
    cfg = {
        "x_transformer": tf["x"], "a_transformer": tf["a"],
        "u_transformer": tf["u"],
        "r_s": HALO_RS, "include_analytic": include_analytic,
        "scale": "nfw", "depth": 6,
    }

    # --- scale every val region with the pool-fitted constant factors,
    # truth and smoothed-reference labels alike
    val_scaled, val_scaled_smooth, dist_sun = {}, {}, {}
    for name in regions:
        xv = np.asarray(z[f"x_val|{name}"][:n_val], dtype=np.float64)
        val_scaled[name] = (tf["x"].transform(jnp.asarray(xv)),
                            tf["a"].transform(jnp.asarray(z[f"a_val|{name}"][:n_val])))
        val_scaled_smooth[name] = tf["a"].transform(
            jnp.asarray(z[f"a_val_smooth|{name}"][:n_val]))
        dist_sun[name] = np.linalg.norm(xv - OBSERVER, axis=1)

    # --- Sun anchor (truth labels; frame-corrected in fire_truth.py)
    x_obs = tf["x"].transform(jnp.asarray(z["x_obs"]))   # (1, 3)
    a_obs = tf["a"].transform(jnp.asarray(z["a_obs"]))   # (1, 3)

    return (scaled["x_train"], scaled["a_train"], x_pool_phys, cfg,
            val_scaled, dist_sun, x_obs, a_obs, val_scaled_smooth, meta)


if __name__ == "__main__":
    (x_pool, a_pool, x_pool_phys, cfg, val_scaled, dist_sun,
     x_obs, a_obs, val_sm, meta) = build_data()
    print(f"x_pool {x_pool.shape}, a_pool {a_pool.shape}, finite: "
          f"{bool(jnp.isfinite(x_pool).all() & jnp.isfinite(a_pool).all())}")
    print(f"|x_pool| scaled max = "
          f"{float(jnp.linalg.norm(x_pool, axis=1).max()):.4f} "
          f"(phys max r_GC = {np.linalg.norm(x_pool_phys, axis=1).max():.2f} kpc)")
    print(f"x_obs (scaled) = {np.asarray(x_obs).round(4)}")
    print(f"|a_obs| (scaled) = {float(jnp.linalg.norm(a_obs)):.4f}")
    for name, (xv, av) in val_scaled.items():
        d = dist_sun[name]
        sm_gap = float(jnp.median(jnp.linalg.norm(av - val_sm[name], axis=1)
                                  / jnp.linalg.norm(av, axis=1)))
        print(f"val {name:6s}: {xv.shape[0]} pts, dist-from-Sun "
              f"[{d.min():.2f}, {d.max():.2f}] kpc, "
              f"median |a_true - a_smooth|/|a_true| = {sm_gap:.3f}")
    colloc = sample_colloc_scaled(jr.PRNGKey(1000), 512, 0.5, 25.0,
                                  cfg["x_transformer"])
    r_c = np.linalg.norm(np.asarray(
        cfg["x_transformer"].inverse_transform(colloc)), axis=1)
    print(f"colloc: {colloc.shape}, phys r in [{r_c.min():.2f}, "
          f"{r_c.max():.2f}] kpc")
    print(f"a* implied by u*: a_transformer(1 kpc/Myr^2) = "
          f"{float(cfg['a_transformer'].transform(jnp.ones((1, 3)))[0, 0]):.4f}")
