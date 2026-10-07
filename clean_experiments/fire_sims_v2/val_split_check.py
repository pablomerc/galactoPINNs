"""Large-n cross-check: chi-matched stopping vs validation-split stopping.

Question. The BFE-baseline ladder showed the fixed 1500-epoch budget
overfits in the residual scaling (train chi far below 0.798). The proposed
fix is chi-matched stopping (Morozov discrepancy principle: stop when the
whitened train misfit reaches the known noise level) because at n~50 a
validation split is unaffordable. This script checks, at large n where a
split IS affordable, that the two criteria pick statistically
indistinguishable stopping points — the evidence needed to trust
chi-matching where it can't be cross-checked.

Design (LOS mode, hetW noise — identical draws to the arch ladders):
  for arch in {III (plain scaling), IV-BFE (residual scaling)},
      n in {200, 500}, trial in 0..5:
    * FULL trajectory : train on all n points, record per epoch
        chi_fit(all n), rel_err(sun4 sub), rel_err(sun15 sub)
    * SPLIT trajectory: train on 85% of the same points, record per epoch
        chi_fit(85%), chi_val(15% holdout), rel_err(sun4), rel_err(sun15)
    stopping rules evaluated:
        chi-full   : first epoch chi_fit(full) <= 0.798        [no holdout]
        val-split  : argmin_e chi_val on the split trajectory
        val-refit  : apply the val-split epoch to the FULL trajectory
        oracle     : argmin_e rel_err_sun4 on the full trajectory
        e1500      : the ladder's current fixed budget (full trajectory end)

Outputs: results_val_split_check/{trajectories.npz, summary.csv} +
val_split_check.png (example trajectory + agreement scatter).

Run: uv run python experiments/fire_los_recovery/val_split_check.py
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx

from galactoPINNs.train import los_residual

from datasets import DEFAULT_TRUTH, HALO_RS, sample_colloc_scaled
from noise import arm_weights
from run_arch_bfe_vmap import (
    arch_cfg_bfe,
    build_contexts_bfe,
    make_model_bfe,
)
from run_arch_vmap import CTX_OF_ARCH
from bfe_baseline import DEFAULT_COEFFS, load_bfe

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # clean_experiments/
from mock_catalog.draw import draw_trial, load_pool
DEFAULT_POOL = Path(__file__).resolve().parents[1] / "mock_catalog" / "cache" / "pool_count.npz"

CHI_STAR = 0.7978845608  # E|N(0,1)| = sqrt(2/pi)
FAMILY, ARM = "het", "W"
ARCHES = ("III", "IV")
HOLDOUT_FRAC = 0.15


def make_traj_fn(graphdef, rest, tx, epochs, n_eval,
                 use_anchor=False, use_rho=False,
                 lam_anchor=1.0, lam_rho=1.0):
    """Jitted full-batch trainer that records per-epoch diagnostics AND the
    params at the chi-crossing (Morozov stop), carried through the scan.

    Static over (epochs, array shapes, mode); reused across trials of the
    same (arch, n, split-size). x_fit/heldout arrays carry the split; for
    the full-data run the holdout arrays are size-1 dummies (mask=0).
    Anchor/rho terms mirror run_noise_vmap.make_ensemble_fns exactly
    (L1 anchor, denom n+3; mean relu(-laplacian) on colloc).
    """

    @jax.jit
    def run(params, x_f, a_f, nv_f, w_f, invsig_f,
            x_h, a_h, nv_h, invsig_h, h_mask,
            x_ev, a_ev, x_obs, a_obs, colloc):
        opt = tx.init(params)

        def losfn(m, x, a, nv):
            # Sun-relative observable: (a(x) - a(x_obs)).n  vs  (a - a_obs).n
            return los_residual(m, x, a - a_obs, nv, x_obs=x_obs)

        def loss_fn(p):
            m = nnx.merge(graphdef, p, rest)
            r = losfn(m, x_f, a_f, nv_f)
            perlos = w_f * jnp.abs(r)
            if use_anchor:
                ar = (m(x_obs)["acceleration"] - a_obs).reshape(-1)
                anch = jnp.abs(ar)
                loss = ((jnp.sum(perlos) + lam_anchor * jnp.sum(anch))
                        / (x_f.shape[0] + anch.size))
            else:
                loss = jnp.mean(perlos)
            if use_rho:
                lap = m.compute_laplacian(colloc)
                loss = loss + lam_rho * jnp.mean(jax.nn.relu(-lap))
            return loss

        def step(carry, _):
            p, o, snap, crossed = carry
            _, grads = jax.value_and_grad(loss_fn)(p)
            updates, o = tx.update(grads, o, p)
            p = optax.apply_updates(p, updates)

            m = nnx.merge(graphdef, p, rest)
            chi_fit = jnp.mean(jnp.abs(losfn(m, x_f, a_f, nv_f)) * invsig_f)
            chi_val = (jnp.sum(jnp.abs(losfn(m, x_h, a_h, nv_h))
                               * invsig_h * h_mask)
                       / jnp.maximum(jnp.sum(h_mask), 1.0))
            newly = jnp.logical_and(jnp.logical_not(crossed),
                                    chi_fit <= CHI_STAR)
            snap = jax.tree.map(lambda s, q: jnp.where(newly, q, s), snap, p)
            crossed = jnp.logical_or(crossed, chi_fit <= CHI_STAR)
            a_pred = m(x_ev)["acceleration"]
            rel = (jnp.linalg.norm(a_pred - a_ev, axis=1)
                   / jnp.linalg.norm(a_ev, axis=1))
            rec4 = jnp.mean(rel[:n_eval])
            rec15 = jnp.mean(rel[n_eval:])
            return (p, o, snap, crossed), jnp.stack(
                [chi_fit, chi_val, rec4, rec15])

        (p_end, _, snap, crossed), hist = jax.lax.scan(
            step, (params, opt, params, jnp.asarray(False)), None,
            length=epochs)
        return hist, snap, p_end, crossed  # snap = params at chi-crossing

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-cache", default=str(DEFAULT_TRUTH))
    ap.add_argument("--coeffs", default=str(DEFAULT_COEFFS))
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--n-list", default="200,500")
    ap.add_argument("--baseline", default="bfe", choices=["bfe", "nfw"],
                    help="analytic baseline for the residual arm (IV): the "
                         "sim-fitted BFE or the misspecified MW-tuned NFW")
    ap.add_argument("--u-star-plain", type=float, default=None,
                    help="a-priori u* for the plain scaling (III); "
                         "None = legacy truth-derived max|u|")
    ap.add_argument("--u-star-res", type=float, default=None,
                    help="a-priori u* for the residual scaling (IV); "
                         "None = legacy max|u - u_baseline|")
    ap.add_argument("--arches", default="III,IV")
    ap.add_argument("--mode", default="LOS",
                    choices=["LOS", "LOS+rho", "LOS+Sun+rho"])
    ap.add_argument("--lambda-sun", type=float, default=1.0)
    ap.add_argument("--lambda-rho", type=float, default=1.0)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--n-eval-sub", type=int, default=512)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results_val_split_check"))
    ap.add_argument("--matmul-precision", default="highest",
                    choices=["highest", "high", "default"])
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print(f"jax backend: {jax.default_backend()}")

    if args.baseline == "bfe":
        base_pot, BFECls, _ = load_bfe(args.coeffs)
        m_fit = float(np.asarray(base_pot.m))
    else:
        import galax.potential as gp
        base_pot = gp.NFWPotential(m=5.4e11, r_s=HALO_RS, units="galactic")
        BFECls, m_fit = None, 5.4e11   # V not used here; IV ignores these

    ctxs, _ = build_contexts_bfe(args.truth_cache, 4096, base_pot,
                                 args.u_star_plain, args.u_star_res)
    tx = optax.adam(1e-3)
    pool = load_pool(args.pool)

    os.makedirs(args.outdir, exist_ok=True)
    npz_payload, rows = {}, []
    t_start = time.time()

    for arch in [a for a in args.arches.split(',') if a]:
        key = CTX_OF_ARCH[arch]
        c = ctxs[key]
        cfg = arch_cfg_bfe(arch, c["cfg"], base_pot)
        fac = c["fac"]

        # eval subsets: first n_eval_sub points of sun4 and sun15, stacked
        ev4 = c["val_scaled"]["sun4"]
        ev15 = c["val_scaled"]["sun15"]
        x_ev = jnp.concatenate([ev4[0][:args.n_eval_sub],
                                ev15[0][:args.n_eval_sub]])
        a_ev = jnp.concatenate([ev4[1][:args.n_eval_sub],
                                ev15[1][:args.n_eval_sub]])

        use_anchor = "Sun" in args.mode
        use_rho = "rho" in args.mode
        x_obs, a_obs = c["x_obs"], c["a_obs"]
        ckpt_dir = os.path.join(args.outdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        for n in [int(s) for s in args.n_list.split(",")]:
            n_hold = int(round(HOLDOUT_FRAC * n))
            traj_full = traj_split = None  # jit per (arch, n) shapes
            for trial in range(args.trials):
                # ---- data: identical seeds to the arch ladders ----
                d = draw_trial(pool, trial, n, FAMILY)      # physical units
                x_n = cfg["x_transformer"].transform(jnp.asarray(d["x"]))
                a_n = cfg["a_transformer"].transform(jnp.asarray(d["a"]))
                nv = jnp.asarray(d["n_hat"])
                sig = d["sigma_mmsyr"]
                a_noisy = a_n + jnp.asarray(d["eps_std"] * sig * fac)[:, None] * nv
                invsig = jnp.asarray(1.0 / (sig * fac))  # scaled units

                # ---- split (seeded independently of the draws) ----
                hold = np.sort(np.random.default_rng(500 + trial)
                               .choice(n, size=n_hold, replace=False))
                fit = np.setdiff1d(np.arange(n), hold)

                params0 = nnx.split(
                    make_model_bfe(arch, cfg, trial, BFECls, m_fit, HALO_RS),
                    nnx.Param, ...)[1]
                colloc = (sample_colloc_scaled(
                              jr.PRNGKey(1000 + trial), args.n_colloc,
                              0.5, 25.0, cfg["x_transformer"])
                          if use_rho else jnp.zeros((1, 3)))
                if traj_full is None:
                    template = make_model_bfe(arch, cfg, 0, BFECls, m_fit,
                                              HALO_RS)
                    graphdef, _, rest = nnx.split(template, nnx.Param, ...)
                    traj_full = make_traj_fn(graphdef, rest, tx, args.epochs,
                                             args.n_eval_sub,
                                             use_anchor=use_anchor,
                                             use_rho=use_rho,
                                             lam_anchor=args.lambda_sun,
                                             lam_rho=args.lambda_rho)
                    traj_split = traj_full  # same jitted fn, shapes differ

                def wgt(sig_sub):
                    return arm_weights(sig_sub, ARM)

                t0 = time.time()
                # FULL trajectory (dummy 1-point holdout, masked out)
                hist_f, snap, p_end, crossed_j = traj_full(
                    params0, x_n, a_noisy, nv, wgt(sig), invsig,
                    x_n[:1], a_noisy[:1], nv[:1], invsig[:1],
                    jnp.zeros(1), x_ev, a_ev, x_obs, a_obs, colloc)
                hist_f = np.asarray(hist_f)
                if not bool(crossed_j):
                    snap = p_end   # never crossed: fall back to last epoch
                # SPLIT trajectory
                hist_s, _, _, _ = traj_split(
                    params0, x_n[fit], a_noisy[fit], nv[fit],
                    wgt(sig[fit]), invsig[fit],
                    x_n[hold], a_noisy[hold], nv[hold], invsig[hold],
                    jnp.ones(n_hold), x_ev, a_ev, x_obs, a_obs, colloc)
                hist_s = np.asarray(hist_s)
                dt = time.time() - t0
                ckpt = os.path.join(
                    ckpt_dir, f"{arch}_{args.baseline}_"
                    f"{args.mode.replace('+', '')}_n{n}_t{trial}.pkl")
                with open(ckpt, "wb") as fh:
                    pickle.dump(dict(
                        state=jax.tree.map(np.asarray, snap),
                        arch=arch, baseline=args.baseline, mode=args.mode,
                        n=n, trial=trial,
                        crossed=bool(crossed_j)), fh)

                # ---- stopping rules (epochs are 1-indexed in reports) ----
                def first_crossing(chi):
                    below = np.flatnonzero(chi <= CHI_STAR)
                    return int(below[0]) if below.size else len(chi) - 1

                e_chi_full = first_crossing(hist_f[:, 0])
                e_chi_split = first_crossing(hist_s[:, 0])
                e_val = int(np.argmin(hist_s[:, 1]))
                e_orc = int(np.argmin(hist_f[:, 2]))

                crossed = bool(np.any(hist_f[:, 0] <= CHI_STAR))
                row = dict(
                    arch=arch, n=n, trial=trial,
                    e_chi_full=e_chi_full, e_chi_split=e_chi_split,
                    e_val=e_val, e_oracle=e_orc, chi_crossed=crossed,
                    rec4_chi_full=hist_f[e_chi_full, 2],
                    rec15_chi_full=hist_f[e_chi_full, 3],
                    rec4_val_split=hist_s[e_val, 2],
                    rec15_val_split=hist_s[e_val, 3],
                    rec4_val_refit=hist_f[e_val, 2],
                    rec15_val_refit=hist_f[e_val, 3],
                    rec4_oracle=hist_f[e_orc, 2],
                    rec4_e1500=hist_f[-1, 2],
                    rec15_e1500=hist_f[-1, 3],
                    chi_end_full=hist_f[-1, 0],
                    chi_val_min=hist_s[e_val, 1],
                )
                rows.append(row)
                npz_payload[f"full|{arch}|{n}|{trial}"] = hist_f.astype(
                    np.float32)
                npz_payload[f"split|{arch}|{n}|{trial}"] = hist_s.astype(
                    np.float32)
                print(f"{arch:3s} n={n:4d} t={trial}  "
                      f"e_chi={e_chi_full:4d}{'' if crossed else '*'} "
                      f"e_val={e_val:4d} e_orc={e_orc:4d} | "
                      f"rec4: chi {row['rec4_chi_full']:.3f} "
                      f"val {row['rec4_val_split']:.3f} "
                      f"refit {row['rec4_val_refit']:.3f} "
                      f"orc {row['rec4_oracle']:.3f} "
                      f"e1500 {row['rec4_e1500']:.3f}  ({dt:.0f}s)",
                      flush=True)

    with open(os.path.join(args.outdir, "summary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(args.outdir, "trajectories.npz"),
                        **npz_payload)
    print(f"total {time.time() - t_start:.0f}s; wrote {args.outdir}/")


if __name__ == "__main__":
    main()
