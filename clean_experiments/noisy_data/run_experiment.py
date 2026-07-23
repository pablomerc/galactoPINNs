"""Driver for the noisy_data experiment.

Repeats clean_experiments/sun_los_recovery (same geometry, truth, scaling,
subsampling seeds, hyperparameters), but the observed LOS accelerations carry
Gaussian measurement noise. The new axis is the SETTING = noise level x
training arm:

  s0 / s0.5 / s1.7 / s4.3 — constant sigma [mm/s/yr] (catalog markers:
                            zero / ~10th pct / median / 75th pct), plain L1
  het                     — per-point sigma resampled from the 51 real
                            catalog values, plain L1
  hetW                    — same noise draws, importance_weight ∝ 1/sigma
                            (whitened L1)
  hetC                    — same noise draws, importance_weight ∝ 1/sigma^2
                            with los_loss='sq' (chi^2-style)
  s1.7W                   — sigma=1.7 with w ∝ 1/sigma (whitened L1;
                            same noise realization as s1.7 / s1.7C)
  s1.7C                   — sigma=1.7 with the sq loss (pure L1-vs-L2
                            comparison, same noise realization as s1.7)

Modes per setting: LOS and LOS+Sun+rho; the 3D reference is trained at s0
only (a noisy-3D experiment needs a different, per-component noise model).
The Sun anchor is noiseless: its real uncertainty (~0.13 mm/s/yr from
Sgr A* VLBI) is subdominant to every pulsar sigma.

Freeze baseline: the s0 rows must reproduce the parent's results.csv
bit-exactly. Requires the GUIDE_0 src edits (importance_weight pass-through
and los_loss on train_model_static / train_step_static).
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

from noisy_datasets import (
    GC,
    NOISE_FAMILY_SEED,
    OBSERVER,
    apply_los_noise,
    arm_weights,
    build_data,
    draw_sigmas_mmsyr,
    sample_colloc_scaled,
    sigma_scale_factor,
)

# label -> (noise family, arm, los_loss). Arms sharing a family share the
# noise realization (same eps seed), so arm differences are loss differences.
SETTINGS = {
    "s0":    ("s0",   "plain", "l1"),
    "s0.5":  ("s0.5", "plain", "l1"),
    "s1.7":  ("s1.7", "plain", "l1"),
    "s4.3":  ("s4.3", "plain", "l1"),
    "het":   ("het",  "plain", "l1"),
    "hetW":  ("het",  "W",     "l1"),
    "hetC":  ("het",  "C",     "sq"),
    "s1.7W": ("s1.7", "W",     "l1"),
    "s1.7C": ("s1.7", "C",     "sq"),
}
MODES = ("LOS", "LOS+Sun+rho")
REGIONS = ("sun4", "sun15", "gc15")


def train_one(cfg, x_n, a_n, epochs, seed, weights=None, los_loss="l1",
              **loss_kwargs):
    model = StaticModel(cfg, rngs=nnx.Rngs(seed))
    train_model_static(
        model, optax.adam(1e-3), x_n, a_n, epochs, log_every=0,
        importance_weight=weights, los_loss=los_loss, **loss_kwargs,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", default="50,100,200,500")
    ap.add_argument("--settings", default=",".join(SETTINGS))
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
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    n_grid = [int(s) for s in args.n_grid.split(",")]
    settings = [s for s in args.settings.split(",") if s]
    unknown = [s for s in settings if s not in SETTINGS]
    if unknown:
        raise SystemExit(f"unknown settings {unknown}; known: {list(SETTINGS)}")

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

    fac = sigma_scale_factor(cfg)  # scaled units per mm/s/yr
    print(f"1 mm/s/yr = {fac:.5f} scaled units")

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
        w.writerow(["n_samples", "trial", "setting", "sigma_mmsyr", "arm",
                    "mode", "region", "rel_err",
                    "rel_los", "rel_trv", "rel_R", "rel_z", "rel_phi",
                    "train_chi", "train_err_true_mmsyr"])

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

                a_los_true = jnp.sum(a_n * n_vecs, axis=1)  # scaled

                los_kw = dict(n_vecs=n_vecs, line_of_sight=True)
                lsr_kw = dict(n_vecs=n_vecs, line_of_sight=True,
                              anchor_x=x_obs, anchor_a=a_obs,
                              lambda_anchor=args.lambda_sun,
                              colloc_x=x_colloc, lambda_rho=args.lambda_rho)

                for label in settings:
                    family, arm, los_loss = SETTINGS[label]
                    # sigma draw + noise realization: seeded per (trial, n,
                    # FAMILY), so all arms of one family train on identical
                    # noisy data.
                    sig_i = draw_sigmas_mmsyr(
                        np.random.default_rng(9000 + 137 * trial + n), n, family)
                    eps_rng = np.random.default_rng(
                        7000 + 1009 * trial + 31 * n + NOISE_FAMILY_SEED[family])
                    a_noisy = apply_los_noise(eps_rng, a_n, n_vecs, sig_i * fac)
                    weights = arm_weights(sig_i, arm)
                    sig_scaled = jnp.asarray(sig_i) * fac

                    modes = {"LOS": (a_noisy, los_kw),
                             "LOS+Sun+rho": (a_noisy, lsr_kw)}
                    if label == "s0":
                        modes["3D"] = (a_n, {})

                    for mode, (a_tgt, kw) in modes.items():
                        model = train_one(cfg, x_n, a_tgt, args.epochs,
                                          seed=trial, weights=weights,
                                          los_loss=los_loss, **kw)

                        # train diagnostics on the n sightlines:
                        # chi = <|r|/sigma> vs observations (0.798 = matched
                        # to the noise; well below = fitting noise), and the
                        # error vs the TRUE projection in mm/s/yr.
                        a_pred_tr = model(x_n)["acceleration"]
                        los_pred = jnp.sum(a_pred_tr * n_vecs, axis=1)
                        los_obs = jnp.sum(a_tgt * n_vecs, axis=1)
                        if sig_i.max() > 0:
                            chi = float(jnp.mean(
                                jnp.abs(los_pred - los_obs) / sig_scaled))
                        else:
                            chi = float("nan")
                        err_true = float(jnp.mean(
                            jnp.abs(los_pred - a_los_true))) / fac

                        region_err = {}
                        for region in REGIONS:
                            x_val, a_val = val_scaled[region]
                            xv_phys, nhat = val_geom[region]
                            a_pred = model(x_val)["acceleration"]
                            d = error_decomposition(a_val, a_pred, nhat)
                            c = cylindrical_error_decomposition(
                                a_val, a_pred, jnp.asarray(xv_phys),
                            )
                            err = float(jnp.mean(d["rel_error_magnitude"]))
                            region_err[region] = err
                            key = f"{region}|{n}|{trial}|{label}|{mode}"
                            perpoint[f"err|{key}"] = np.asarray(
                                d["rel_error_magnitude"], np.float32)
                            perpoint[f"apred|{key}"] = np.asarray(a_pred, np.float32)
                            w.writerow([n, trial, label,
                                        f"{np.median(sig_i):.2f}", arm, mode,
                                        region, err,
                                        float(jnp.mean(d["rel_los_error"])),
                                        float(jnp.mean(d["rel_transverse_error"])),
                                        float(jnp.mean(c["rel_R_error"])),
                                        float(jnp.mean(c["rel_z_error"])),
                                        float(jnp.mean(c["rel_phi_error"])),
                                        f"{chi:.3f}", f"{err_true:.3f}"])
                        f.flush()
                        print(f"n={n:5d} trial={trial} {label:6s} {mode:12s} "
                              f"chi={chi:5.2f} errTrue={err_true:5.2f}mm/s/yr "
                              f"rel_err(sun4)={region_err['sun4']:.4f}", flush=True)

    np.savez_compressed(npz_path, **perpoint)
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
