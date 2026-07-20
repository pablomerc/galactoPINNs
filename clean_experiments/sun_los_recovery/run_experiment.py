"""Driver for the sun_los_recovery experiment.

Config, five training modes, (n x trial x mode x region) loop, CSV + NPZ.
Saves predicted vectors so LOS/transverse and cylindrical decompositions
can be derived offline (and summaries go in the CSV).
"""

from __future__ import annotations

import argparse
import csv
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx
import galax.potential as gp

from galactoPINNs.evaluate import (
    cylindrical_error_decomposition,
    error_decomposition,
)
from galactoPINNs.models.static_model import StaticModel
from galactoPINNs.train import train_model_static

from datasets import (
    OBSERVER,
    GC,
    build_data,
    sample_colloc_scaled,
)

MODES = ("3D", "LOS", "LOS+Sun", "LOS+rho", "LOS+Sun+rho")
REGIONS = ("sun4", "sun15", "gc15")


def rho_neg_fraction(model, x_scaled):
    """Fraction of points with predicted rho < 0 (rho has the sign of the scaled lap)."""
    lap = model.compute_laplacian(x_scaled)
    return float(jnp.mean(lap < 0.0))


def train_one(cfg, x_n, a_n, epochs, seed, **loss_kwargs):
    model = StaticModel(cfg, rngs=nnx.Rngs(seed))
    train_model_static(
        model, optax.adam(1e-3), x_n, a_n, epochs, log_every=0, **loss_kwargs,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", default="50,100,200,500,1000")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=30000)
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--r-train", type=float, default=4.0)
    ap.add_argument("--r-eval-sun", type=float, default=15.0)
    ap.add_argument("--r-eval-gc", type=float, default=15.0)
    ap.add_argument("--lambda-sun", type=float, default=1.0)
    ap.add_argument("--lambda-rho", type=float, default=1.0)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--r-min-colloc", type=float, default=0.5)
    ap.add_argument("--r-colloc-max", type=float, default=25.0)
    ap.add_argument("--n-val-lap", type=int, default=512)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    n_grid = [int(s) for s in args.n_grid.split(",")]
    regions = {
        "sun4": (OBSERVER, args.r_train),
        "sun15": (OBSERVER, args.r_eval_sun),
        "gc15": (GC, args.r_eval_gc),
    }

    true_potential = gp.MilkyWayPotential()
    (x_pool, a_pool, x_pool_phys, cfg,
     val_scaled, dist_sun, x_obs, a_obs) = build_data(
        true_potential, args.pool, args.n_val, args.r_train,
        regions, include_analytic=False,
    )

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")
    npz_path = os.path.join(args.outdir, "per_point_errors.npz")
    perpoint = {f"dist|{name}": dist_sun[name] for name in REGIONS}

    # Per-region truth + geometry (saved once). Sightlines are always
    # Sun-anchored — the decomp question is relative to what the observer sees.
    x_tf = cfg["x_transformer"]
    val_geom = {}
    for name in REGIONS:
        x_val, a_val = val_scaled[name]
        xv_phys = np.asarray(x_tf.inverse_transform(x_val))
        dxv = xv_phys - OBSERVER
        nhat = jnp.asarray(dxv / np.linalg.norm(dxv, axis=1, keepdims=True))
        val_geom[name] = (xv_phys, nhat)
        perpoint[f"x_val_phys|{name}"] = xv_phys.astype(np.float32)
        perpoint[f"a_val|{name}"] = np.asarray(a_val, dtype=np.float32)
        perpoint[f"nhat|{name}"] = np.asarray(nhat, dtype=np.float32)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "trial", "mode", "region", "rel_err", "neg_frac",
                    "rel_los", "rel_trv", "rel_R", "rel_z", "rel_phi"])

        for n in n_grid:
            for trial in range(args.trials):
                # subsample n training points from the pool (per-trial RNG)
                rng = np.random.default_rng(trial)
                idx = rng.choice(x_pool.shape[0], size=n, replace=False)
                x_n, a_n = x_pool[idx], a_pool[idx]

                # sightlines: unit Sun->star directions, in PHYSICAL space
                dx = x_pool_phys[idx] - OBSERVER
                n_vecs = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))

                # fresh collocation set per trial
                x_colloc = sample_colloc_scaled(
                    jr.PRNGKey(1000 + trial), args.n_colloc,
                    args.r_min_colloc, args.r_colloc_max, cfg["x_transformer"],
                )

                models = {
                    "3D": train_one(cfg, x_n, a_n, args.epochs, seed=trial),
                    "LOS": train_one(cfg, x_n, a_n, args.epochs, seed=trial,
                                     n_vecs=n_vecs, line_of_sight=True),
                    "LOS+Sun": train_one(cfg, x_n, a_n, args.epochs, seed=trial,
                                         n_vecs=n_vecs, line_of_sight=True,
                                         anchor_x=x_obs, anchor_a=a_obs,
                                         lambda_anchor=args.lambda_sun),
                    "LOS+rho": train_one(cfg, x_n, a_n, args.epochs, seed=trial,
                                         n_vecs=n_vecs, line_of_sight=True,
                                         colloc_x=x_colloc, lambda_rho=args.lambda_rho),
                    "LOS+Sun+rho": train_one(cfg, x_n, a_n, args.epochs, seed=trial,
                                             n_vecs=n_vecs, line_of_sight=True,
                                             anchor_x=x_obs, anchor_a=a_obs,
                                             lambda_anchor=args.lambda_sun,
                                             colloc_x=x_colloc, lambda_rho=args.lambda_rho),
                }

                for region in REGIONS:
                    x_val, a_val = val_scaled[region]
                    xv_phys, nhat = val_geom[region]
                    x_val_lap = x_val[: min(args.n_val_lap, x_val.shape[0])]
                    for mode in MODES:
                        a_pred = models[mode](x_val)["acceleration"]
                        d = error_decomposition(a_val, a_pred, nhat)
                        c = cylindrical_error_decomposition(
                            a_val, a_pred, jnp.asarray(xv_phys),
                        )
                        err = float(jnp.mean(d["rel_error_magnitude"]))
                        neg = rho_neg_fraction(models[mode], x_val_lap)
                        key = f"{region}|{n}|{trial}|{mode}"
                        perpoint[f"err|{key}"] = np.asarray(
                            d["rel_error_magnitude"], np.float32,
                        )
                        perpoint[f"apred|{key}"] = np.asarray(a_pred, np.float32)
                        w.writerow([n, trial, mode, region, err, neg,
                                    float(jnp.mean(d["rel_los_error"])),
                                    float(jnp.mean(d["rel_transverse_error"])),
                                    float(jnp.mean(c["rel_R_error"])),
                                    float(jnp.mean(c["rel_z_error"])),
                                    float(jnp.mean(c["rel_phi_error"]))])
                        print(f"n={n:5d} trial={trial} {region:6s} {mode:12s} "
                              f"rel_err={err:.4f} neg_frac={neg:.3f} "
                              f"los={float(jnp.mean(d['rel_los_error'])):.4f} "
                              f"trv={float(jnp.mean(d['rel_transverse_error'])):.4f}",
                              flush=True)

    np.savez_compressed(npz_path, **perpoint)
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
