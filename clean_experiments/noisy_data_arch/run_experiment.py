"""Driver: PINN III vs IV vs V on noisy LOS accelerations near the Sun.

Sequential (one model at a time) -- the scope is small (|arches| x |modes| x
trials per n) and it makes checkpointing trivial. Fixed choices vs the parent
noisy_data experiment:
  region = sun4 only          noise family = het (catalog sigma)
  arm    = W (w ~ 1/sigma, L1) modes = LOS, LOS+rho (rho>=0 hinge, NO Sun anchor)

The subsample indices and the physical noise draw are seeded by (trial, n)
ONLY, so all three architectures see the identical physical data; each converts
sigma to its own scaled units via its own sigma_scale_factor.
"""

from __future__ import annotations

import argparse
import csv
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import galax.potential as gp

from galactoPINNs.evaluate import (
    cylindrical_error_decomposition,
    error_decomposition,
)
from galactoPINNs.train import train_model_static

import arch_datasets as ad
from arch_datasets import (
    NOISE_FAMILY_SEED, OBSERVER, apply_los_noise, arm_weights, build_arch,
    draw_sigmas_mmsyr, make_model, sample_colloc_scaled, trainable_mr,
)
from checkpoints import save_model

REGION = "sun4"
FAMILY = "het"
ARM = "W"
MODES = ("LOS", "LOS+rho")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", default="50,100,200,500")
    ap.add_argument("--arches", default=",".join(ad.ARCHES))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=30000)
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--r-train", type=float, default=4.0)
    ap.add_argument("--lambda-rho", type=float, default=1.0)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--r-min-colloc", type=float, default=0.5)
    ap.add_argument("--r-colloc-max", type=float, default=25.0)
    ap.add_argument("--v-misspec", type=float, default=1.0,
                    help="multiply PINN V baseline init mass by this (test the "
                         "'fix a wrong baseline' claim; 1.0 = paper value)")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--no-checkpoints", action="store_true")
    args = ap.parse_args()

    n_grid = [int(s) for s in args.n_grid.split(",")]
    arches = [a for a in args.arches.split(",") if a]
    modes = [m for m in args.modes.split(",") if m]
    regions = {REGION: (OBSERVER, args.r_train)}
    true_potential = gp.MilkyWayPotential()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")
    ckpt_root = os.path.join(args.outdir, "checkpoints")
    perpoint = {}
    manifest = []

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "trial", "arch", "mode", "region",
                    "sigma_mmsyr", "rel_err", "rel_los", "rel_trv",
                    "rel_R", "rel_z", "rel_phi", "train_chi",
                    "train_err_true_mmsyr", "learned_m", "learned_rs"])

        for arch in arches:
            # data + scaling are per-architecture (include_analytic changes
            # a_star); built ONCE and reused across n / trial / mode.
            m_init = ad.BASE_M * (args.v_misspec if arch == "V" else 1.0)
            D = build_arch(arch, true_potential, args.pool, args.n_val,
                           args.r_train, regions)
            cfg, fac = D["cfg"], D["fac"]
            x_val, a_val = D["val_scaled"][REGION]
            xv_phys = np.asarray(cfg["x_transformer"].inverse_transform(x_val))
            dxv = xv_phys - OBSERVER
            nhat_v = jnp.asarray(dxv / np.linalg.norm(dxv, axis=1, keepdims=True))
            print(f"[PINN {arch}] 1 mm/s/yr = {fac:.5f} scaled units", flush=True)

            for n in n_grid:
                for trial in range(args.trials):
                    # shared physical subsample + sightlines (seed = trial)
                    rng = np.random.default_rng(trial)
                    idx = rng.choice(D["x_pool"].shape[0], size=n, replace=False)
                    x_n, a_n = D["x_pool"][idx], D["a_pool"][idx]
                    dx = D["x_pool_phys"][idx] - OBSERVER
                    nv = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))
                    a_los_true = jnp.sum(a_n * nv, axis=1)

                    # shared physical noise (het), seeded by (trial, n)
                    sig_i = draw_sigmas_mmsyr(
                        np.random.default_rng(9000 + 137 * trial + n), n, FAMILY)
                    eps_rng = np.random.default_rng(
                        7000 + 1009 * trial + 31 * n + NOISE_FAMILY_SEED[FAMILY])
                    a_noisy = apply_los_noise(eps_rng, a_n, nv, sig_i * fac)
                    w_ip = arm_weights(sig_i, ARM)
                    sig_scaled = jnp.asarray(sig_i) * fac

                    colloc = sample_colloc_scaled(
                        jr.PRNGKey(1000 + trial), args.n_colloc,
                        args.r_min_colloc, args.r_colloc_max, cfg["x_transformer"])

                    for mode in modes:
                        kw = dict(n_vecs=nv, line_of_sight=True)
                        if mode == "LOS+rho":                   # NO Sun anchor
                            kw.update(colloc_x=colloc, lambda_rho=args.lambda_rho)

                        model = make_model(arch, cfg, seed=trial, m=m_init,
                                           r_s=ad.BASE_RS)
                        train_model_static(
                            model, optax.adam(1e-3), x_n, a_noisy, args.epochs,
                            log_every=0, importance_weight=w_ip, los_loss="l1",
                            **kw)

                        # train diagnostics on the n sightlines
                        a_tr = model(x_n)["acceleration"]
                        los_pred = jnp.sum(a_tr * nv, axis=1)
                        los_obs = jnp.sum(a_noisy * nv, axis=1)
                        chi = float(jnp.mean(jnp.abs(los_pred - los_obs) / sig_scaled))
                        err_true = float(jnp.mean(jnp.abs(los_pred - a_los_true))) / fac

                        # eval on sun4
                        a_pred = model(x_val)["acceleration"]
                        d = error_decomposition(a_val, a_pred, nhat_v)
                        c = cylindrical_error_decomposition(
                            a_val, a_pred, jnp.asarray(xv_phys))
                        err = float(jnp.mean(d["rel_error_magnitude"]))
                        mr = trainable_mr(model)
                        lm, lrs = (mr if mr is not None else ("", ""))

                        key = f"{n}|{trial}|{arch}|{mode}"
                        perpoint[f"err|{key}"] = np.asarray(
                            d["rel_error_magnitude"], np.float32)
                        perpoint[f"apred|{key}"] = np.asarray(a_pred, np.float32)

                        w.writerow([
                            n, trial, arch, mode, REGION,
                            f"{np.median(sig_i):.2f}", err,
                            float(jnp.mean(d["rel_los_error"])),
                            float(jnp.mean(d["rel_transverse_error"])),
                            float(jnp.mean(c["rel_R_error"])),
                            float(jnp.mean(c["rel_z_error"])),
                            float(jnp.mean(c["rel_phi_error"])),
                            f"{chi:.3f}", f"{err_true:.3f}",
                            f"{lm:.4e}" if lm != "" else "",
                            f"{lrs:.4f}" if lrs != "" else "",
                        ])
                        f.flush()

                        if not args.no_checkpoints:
                            ckdir = os.path.join(
                                ckpt_root, f"PINN{arch}_{mode}_n{n}_t{trial}")
                            save_model(ckdir, arch, cfg, model,
                                       meta={"n": n, "trial": trial, "arch": arch,
                                             "mode": mode, "fac": fac})
                            manifest.append(
                                [arch, mode, n, trial,
                                 os.path.relpath(ckdir, args.outdir)])

                        print(f"[PINN {arch}] n={n:4d} t={trial} {mode:8s} "
                              f"chi={chi:5.2f} errTrue={err_true:5.2f}mm/s/yr "
                              f"rel_err={err:.4f}"
                              + (f"  m={lm:.3e} rs={lrs:.3f}" if lm != "" else ""),
                              flush=True)

    np.savez_compressed(os.path.join(args.outdir, "per_point_errors.npz"), **perpoint)
    if manifest and not args.no_checkpoints:
        with open(os.path.join(ckpt_root, "manifest.csv"), "w", newline="") as mf:
            mw = csv.writer(mf)
            mw.writerow(["arch", "mode", "n", "trial", "path"])
            mw.writerows(manifest)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
