"""Leave-one-out vs chi-matched early stopping, as a function of n.

Question. chi-stop (Morozov discrepancy principle) ends training at the first
epoch where the whitened train misfit mean|r|/sigma reaches E|N(0,1)| =
sqrt(2/pi). It needs the noise level to be right. Leave-one-out (LOO)
cross-validation picks the epoch from the data alone and stays affordable at
pulsar-catalog n (n fits of n-1 points). Which recovers the field better,
and how does that change with n?

Setup = results_nfw_LOSSunrho (val_split_check.py --baseline nfw --mode
LOS+Sun+rho): PINN IV with the MW-tuned NFW baseline (m = 5.4e11 Msun,
r_s = 15.62 kpc), Sun-relative LOS loss + Sun anchor + rho>=0 hinge on 512
colloc points, hetW catalog noise, the 47-pulsar S(r)S(delta) mock pool,
1500 full-batch Adam epochs. Trials 0-5 reproduce that run's chi-stop rows.

Per (n, trial):
  FOLDS  n models; fold k trains on every pulsar but k. Same init, colloc
         points and Sun anchor in every fold; the held-out pulsar gets weight
         0 and the rest the arm weights of the (n-1)-point subset, so a fold
         is the sliced fit written with fixed shapes, and all folds train in
         one vmap. Recorded every epoch: z_k(e) = r_k(e) / sigma_k, the signed
         whitened residual of fold k on its held-out pulsar.
  FULL   one model on all n points, recording every epoch chi_fit, the legacy
         rec4/rec15 and the truth metrics: acceleration error in the 2-kpc Sun
         bubble (absolute and Sun-relative, 2048 uniform-in-volume points),
         the Gauss-law flux through the 2-kpc Sun sphere (-> <rho_tot>), and
         the midplane density dex within 2 kpc. Params are snapshotted at the
         chi crossing and at the LOO-family epochs.

Stopping rules — all read the FULL trajectory, so every rule is the same
model at a different epoch (a paired comparison):
  chi       first e with chi_fit(e) <= sqrt(2/pi); the last epoch if never
  loo       argmin_e CV(e),  CV(e) = mean_k |z_k(e)|
  loo1se    earliest e with CV(e) <= CV(e*) + SE(e*)  (one-standard-error rule)
  iwloo     argmin_e CV_w(e) = sum_k w_k |z_k(e)| / sum_k w_k with
            w_k = 1[r_k <= 2 kpc] / p_pool(x_k): importance weights that map the
            pulsar distribution onto the uniform-in-volume 2-kpc bubble the
            accuracy metric averages over (covariate-shift CV)
  iwloo1se  the one-standard-error rule on CV_w
p_pool is a kNN density estimate (k = 32) on the trial's own candidate pool,
the set draw_trial samples from. The shards keep every z_k(e), so other
statistics (flattened weights, chi^2, ...) can be recomputed offline.

Outputs (an existing shard is skipped, so a killed job just reruns):
  <outdir>/shards/n{n:04d}_t{trial:02d}.npz          curves, per-pulsar arrays, rule epochs
  <outdir>/checkpoints/<rule>/n{n:04d}_t{trial:02d}.pkl    params at that rule's epoch
analyze_loo_vs_chi.py turns the shards into tables and figures.

Run (repo root):
  uv run python clean_experiments/fire_sims_v2/loo_vs_chi.py --n-list 20,50 --trials 0-1
  (cluster: clean_experiments/fire_sims_v2/slurm/README.md)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx
from scipy.spatial import cKDTree

import astropy.units as au
from astropy.constants import G as _G_SI
import galax.potential as gp

from galactoPINNs.train import los_residual

from datasets import HALO_RS, OBSERVER, sample_colloc_scaled
from noise import arm_weights
from run_arch_bfe_vmap import arch_cfg_bfe, build_contexts_bfe, make_model_bfe
from val_split_check import CHI_STAR

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))                                # clean_experiments/
from mock_catalog.draw import draw_trial, load_pool, trial_pool    # noqa: E402

# Inputs are bundled in loo_inputs/ (see AGENT_loo_vs_chi.md for provenance):
# the truth cache, the 10-trial mock pool, the 2-kpc bubble truth points and the
# midplane density truth grid — exact copies / key subsets of the local caches.
INPUTS = HERE / "loo_inputs"
DEFAULT_TRUTH_IN = INPUTS / "truth_old1gyr.npz"         # == datasets.DEFAULT_TRUTH
DEFAULT_POOL = INPUTS / "pool_count_t10.npz"
DEFAULT_BUBBLES = INPUTS / "bubbles_r2.npz"
DEFAULT_GRIDS = INPUTS / "midplane_rho_xy.npz"

FAMILY, ARM = "het", "W"          # catalog noise, whitened L1
NFW_M = 5.4e11                    # MW-tuned NFW of val_split_check.py --baseline nfw
R_EVAL = 2.0                      # kpc: bubble / sphere / midplane-disc radius
N_EVAL = 512                      # legacy rec4/rec15 subsets
N_SPHERE = 2048                   # Gauss-law surface points
LIM, NG = 6.0, 120                # nb05 midplane grid (Sun-centred, kpc)
RULES = ("chi", "loo", "loo1se", "iwloo", "iwloo1se")

G = float(_G_SI.to(au.kpc**3 / (au.Msun * au.Myr**2)).value)
FOUR_PI_G = 4.0 * np.pi * G
TO_MSUN_PC3 = 1e-9

# per-epoch columns of the FULL trajectory
FULL_COLS = ("loss", "chi_fit", "rec4", "rec15", "acc_b2", "gacc_b2",
             "flux_b2", "dex_b2", "negfrac_b2", "sun_off")


# ---------------------------------------------------------------------------
# geometry for the per-epoch truth metrics
# ---------------------------------------------------------------------------

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph),
                     np.cos(ph)], axis=1)


def eval_geometry(cfg, bubbles_path, grids_path):
    """Scaled evaluation sets: 2-kpc bubble (x, a), the 2-kpc Sun sphere and
    the midplane pixels within 2 kpc with their truth density. Same points
    and recipes as the fire_sims_v2 evaluator (metrics_block/gauge_block)."""
    x_tf, a_tf, u_tf = cfg["x_transformer"], cfg["a_transformer"], cfg["u_transformer"]
    zb = np.load(bubbles_path)
    xb = np.asarray(zb[f"x_bubble|{R_EVAL}"], dtype=np.float64)
    ab = np.asarray(zb[f"a_bubble|{R_EVAL}"], dtype=np.float64)

    nhat = fibonacci_sphere(N_SPHERE)
    surf = OBSERVER + R_EVAL * nhat

    zg = np.load(grids_path)
    g1 = np.linspace(-LIM, LIM, NG)
    GA, GB = np.meshgrid(g1, g1, indexing="xy")
    rho_t = zg["rho2d|gas|xy"] + zg["rho2d|star|xy"] + zg["rho2d|dark|xy"]
    inb = np.sqrt(GA**2 + GB**2) <= R_EVAL
    pix = np.stack([GA[inb] + OBSERVER[0], GB[inb], np.zeros(inb.sum())], -1)
    pix_rho = np.asarray(rho_t[inb], dtype=np.float64)

    x_fac = float(x_tf.transform(jnp.ones((1, 3)))[0, 0])
    a_fac = float(a_tf.transform(jnp.ones((1, 3)))[0, 0])
    u_fac = float(u_tf.transform(jnp.asarray([1.0]))[0])
    return dict(
        bub_x=x_tf.transform(jnp.asarray(xb)), bub_a=a_tf.transform(jnp.asarray(ab)),
        surf_x=x_tf.transform(jnp.asarray(surf)), surf_n=jnp.asarray(nhat),
        pix_x=x_tf.transform(jnp.asarray(pix)),
        pix_rho_t=jnp.asarray(np.where(pix_rho > 1e-6, pix_rho, 1.0)),
        pix_ok=jnp.asarray(pix_rho > 1e-6),
        # scaled laplacian -> rho [Msun/pc^3]; scaled flux -> <rho> [Msun/pc^3]
        lap_to_rho=x_fac**2 / u_fac / FOUR_PI_G * TO_MSUN_PC3,
        flux_to_rho=-3.0 / (FOUR_PI_G * R_EVAL) / a_fac * TO_MSUN_PC3,
        n_bubble=xb.shape[0], n_pix=int(inb.sum()))


# ---------------------------------------------------------------------------
# trainers
# ---------------------------------------------------------------------------

def make_full_fn(graphdef, rest, tx, epochs, n_snap, lam_anchor, lam_rho):
    """One model on all n points. Training ops are val_split_check.make_traj_fn's
    (LOS+Sun+rho branch) verbatim; the extra outputs are read after each update
    and never feed back. Snapshots: the chi crossing (latched) and the epochs
    in snap_at (the LOO-family rules, known before this runs)."""

    @jax.jit
    def run(params, x_f, a_f, nv_f, w_f, invsig_f, x_obs, a_obs, colloc,
            x_ev, a_ev, bub_x, bub_a, surf_x, surf_n, pix_x, pix_rho_t, pix_ok,
            lap_to_rho, snap_at):
        opt = tx.init(params)

        def losfn(m, x, a, nv):
            return los_residual(m, x, a - a_obs, nv, x_obs=x_obs)

        def loss_fn(p):
            m = nnx.merge(graphdef, p, rest)
            r = losfn(m, x_f, a_f, nv_f)
            perlos = w_f * jnp.abs(r)
            ar = (m(x_obs)["acceleration"] - a_obs).reshape(-1)
            anch = jnp.abs(ar)
            loss = ((jnp.sum(perlos) + lam_anchor * jnp.sum(anch))
                    / (x_f.shape[0] + anch.size))
            lap = m.compute_laplacian(colloc)
            return loss + lam_rho * jnp.mean(jax.nn.relu(-lap))

        def step(carry, e):
            p, o, snap, crossed, snaps = carry
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, o = tx.update(grads, o, p)
            p = optax.apply_updates(p, updates)

            m = nnx.merge(graphdef, p, rest)
            chi_fit = jnp.mean(jnp.abs(losfn(m, x_f, a_f, nv_f)) * invsig_f)
            newly = jnp.logical_and(jnp.logical_not(crossed), chi_fit <= CHI_STAR)
            snap = jax.tree.map(lambda s, q: jnp.where(newly, q, s), snap, p)
            crossed = jnp.logical_or(crossed, chi_fit <= CHI_STAR)
            snaps = tuple(jax.tree.map(lambda s, q, j=j: jnp.where(e == snap_at[j], q, s), sj, p)
                          for j, sj in enumerate(snaps))

            # legacy 512-pt sun4 / sun15 subsets (the val_split_check gate)
            a_pred = m(x_ev)["acceleration"]
            rel = jnp.linalg.norm(a_pred - a_ev, axis=1) / jnp.linalg.norm(a_ev, axis=1)
            n_ev = x_ev.shape[0] // 2
            rec4, rec15 = jnp.mean(rel[:n_ev]), jnp.mean(rel[n_ev:])

            # 2-kpc bubble: absolute and Sun-relative (gauge-free) errors
            a_b = m(bub_x)["acceleration"]
            a_s = m(x_obs)["acceleration"]
            nb = jnp.linalg.norm(bub_a, axis=1)
            acc_b2 = jnp.mean(jnp.linalg.norm(a_b - bub_a, axis=1) / nb)
            gacc_b2 = jnp.mean(jnp.linalg.norm((a_b - a_s) - (bub_a - a_obs), axis=1) / nb)
            sun_off = jnp.linalg.norm(a_s - a_obs)

            # Gauss law through the 2-kpc Sun sphere (scaled flux; -> rho offline)
            flux = jnp.mean(jnp.sum(m(surf_x)["acceleration"] * surf_n, axis=1))

            # midplane density within 2 kpc: median |log10 rho_pred/rho_true|
            # over pixels with rho_pred > 0 (nb05/nb06 recipe)
            rho_p = m.compute_laplacian(pix_x) * lap_to_rho
            ok = jnp.logical_and(pix_ok, rho_p > 0)
            ldex = jnp.abs(jnp.log10(jnp.where(ok, rho_p, 1.0) / pix_rho_t))
            dex = jnp.nanmedian(jnp.where(ok, ldex, jnp.nan))
            negfrac = jnp.mean(rho_p <= 0)

            row = jnp.stack([loss, chi_fit, rec4, rec15, acc_b2, gacc_b2,
                             flux, dex, negfrac, sun_off])
            return (p, o, snap, crossed, snaps), row

        snaps0 = tuple(params for _ in range(n_snap))
        (p_end, _, snap, crossed, snaps), hist = jax.lax.scan(
            step, (params, opt, params, jnp.asarray(False), snaps0),
            jnp.arange(epochs))
        return hist, snap, p_end, crossed, snaps

    return run


def make_fold_fn(graphdef, rest, tx, epochs, lam_anchor, lam_rho):
    """F leave-one-out folds in one vmap. Fold k's weight row is 0 at k and
    arm_weights(sigma[others]) elsewhere; the denominator is the (n-1)-point
    fit's n-1+3. Records the signed whitened held-out residual every epoch."""

    def loss_fn(p, w, x, a, nv, x_obs, a_obs, colloc):
        m = nnx.merge(graphdef, p, rest)
        r = los_residual(m, x, a - a_obs, nv, x_obs=x_obs)
        perlos = w * jnp.abs(r)
        ar = (m(x_obs)["acceleration"] - a_obs).reshape(-1)
        anch = jnp.abs(ar)
        loss = ((jnp.sum(perlos) + lam_anchor * jnp.sum(anch))
                / (x.shape[0] - 1 + anch.size))
        lap = m.compute_laplacian(colloc)
        return loss + lam_rho * jnp.mean(jax.nn.relu(-lap))

    def held_out(p, xk, ak, nvk, invsigk, x_obs, a_obs):
        m = nnx.merge(graphdef, p, rest)
        r = los_residual(m, xk[None, :], ak[None, :] - a_obs, nvk[None, :], x_obs=x_obs)
        return r[0] * invsigk

    grad_all = jax.vmap(jax.grad(loss_fn), in_axes=(0, 0, None, None, None, None, None, None))
    held_all = jax.vmap(held_out, in_axes=(0, 0, 0, 0, 0, None, None))

    @jax.jit
    def run(params0, W, xk, ak, nvk, invsigk, x, a, nv, x_obs, a_obs, colloc):
        F = W.shape[0]
        P = jax.tree.map(lambda v: jnp.broadcast_to(v, (F,) + v.shape), params0)
        opt = tx.init(P)

        def step(carry, _):
            p, o = carry
            grads = grad_all(p, W, x, a, nv, x_obs, a_obs, colloc)
            updates, o = tx.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return (p, o), held_all(p, xk, ak, nvk, invsigk, x_obs, a_obs)

        _, Z = jax.lax.scan(step, (P, opt), None, length=epochs)
        return Z                                                # (epochs, F)

    return run


# ---------------------------------------------------------------------------
# stopping rules
# ---------------------------------------------------------------------------

def first_crossing(chi):
    below = np.flatnonzero(chi <= CHI_STAR)
    return (int(below[0]), True) if below.size else (len(chi) - 1, False)


def cv_curves(Z, omega=None):
    """CV(e) and its standard error from the (epochs, n) held-out residuals.
    omega=None: plain mean over folds; else the self-normalized weighted mean
    with the SE of a weighted mean, sqrt(sum w^2 (|z| - CV)^2)."""
    A = np.abs(np.asarray(Z, dtype=np.float64))
    if omega is None:
        return A.mean(1), A.std(1, ddof=1) / np.sqrt(A.shape[1])
    w = np.asarray(omega, dtype=np.float64)
    w = w / w.sum()
    cv = A @ w
    se = np.sqrt(((A - cv[:, None]) ** 2) @ (w ** 2))
    return cv, se


def argmin_and_1se(cv, se):
    e_min = int(np.argmin(cv))
    e_1se = int(np.flatnonzero(cv <= cv[e_min] + se[e_min])[0])
    return e_min, e_1se


def iw_weights(pool, trial, d, knn, r_iw):
    """w_k = 1[r_k <= r_iw] / p_pool(x_k), p_pool = kNN density on the trial's
    candidate pool (x_k is itself a pool member: its own zero distance is dropped)."""
    xp = np.asarray(trial_pool(pool, trial)["x"], dtype=np.float64)
    dist, _ = cKDTree(xp).query(np.asarray(d["x"], dtype=np.float64), k=knn + 1)
    r_k = dist[:, knn]
    p_hat = knn / (xp.shape[0] * 4.0 / 3.0 * np.pi * r_k**3)
    inside = np.asarray(d["r"]) <= r_iw
    omega = np.where(inside, 1.0 / p_hat, 0.0)
    return omega, p_hat


# ---------------------------------------------------------------------------
# reference check (CPU vs GPU sanity)
# ---------------------------------------------------------------------------

def ref_entry(n, trial, epochs_rule, hist, cv, cvw):
    probe = sorted({min(e, len(hist) - 1) for e in (0, 10, 100, 500, len(hist) - 1)})
    return dict(n=n, trial=trial, epochs=epochs_rule,
                probe_epochs=probe,
                chi_fit=[float(hist[e, FULL_COLS.index("chi_fit")]) for e in probe],
                gacc_b2=[float(hist[e, FULL_COLS.index("gacc_b2")]) for e in probe],
                flux_b2=[float(hist[e, FULL_COLS.index("flux_b2")]) for e in probe],
                cv=[float(cv[e]) for e in probe],
                cvw=[float(cvw[e]) for e in probe])


def compare_ref(entry, ref):
    def rd(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
    lines = [f"[ref] n={entry['n']} trial={entry['trial']} vs reference "
             f"({ref.get('backend', '?')}):"]
    for rule, e in entry["epochs"].items():
        e0 = ref["epochs"].get(rule)
        lines.append(f"    {rule:9s} epoch {e:5d}  ref {e0:5d}  "
                     f"{'same' if e == e0 else f'diff {e - e0:+d}'}")
    for key in ("chi_fit", "gacc_b2", "flux_b2", "cv", "cvw"):
        lines.append(f"    {key:9s} max rel diff over probe epochs "
                     f"{ref['probe_epochs']}: {rd(entry[key], ref[key]):.1e}")
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_trials(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth-cache", default=str(DEFAULT_TRUTH_IN))
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--bubbles", default=str(DEFAULT_BUBBLES),
                    help="npz with the 2-kpc bubble truth points (x_bubble|2.0, a_bubble|2.0)")
    ap.add_argument("--grids", default=str(DEFAULT_GRIDS),
                    help="nb05 truth grids (midplane density truth for the dex metric)")
    ap.add_argument("--n-list", default="20,50,100,150,200,250,500")
    ap.add_argument("--trials", default="0-9", help="e.g. 0-9 or 0,2,5")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lambda-sun", type=float, default=1.0)
    ap.add_argument("--lambda-rho", type=float, default=1.0)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--knn", type=int, default=32, help="k of the pool density estimate (IW-LOO)")
    ap.add_argument("--r-iw", type=float, default=R_EVAL,
                    help="IW-LOO target: uniform in the Sun ball of this radius [kpc]")
    ap.add_argument("--fold-chunk", type=int, default=0,
                    help="folds per vmap batch (0 = all n at once); lower it if the GPU runs out of memory")
    ap.add_argument("--full-only", action="store_true",
                    help="skip the folds (chi-stop only); for the reproduction gate")
    ap.add_argument("--no-ckpts", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write-ref", default=None, help="write a reference JSON of key numbers")
    ap.add_argument("--check-ref", default=None, help="compare against a reference JSON")
    ap.add_argument("--outdir", default=str(HERE / "results_loo_vs_chi"))
    ap.add_argument("--matmul-precision", default="highest",
                    choices=["highest", "high", "default"])
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    dev = jax.devices()[0]
    print(f"jax {jax.__version__} backend: {jax.default_backend()}  device: "
          f"{getattr(dev, 'device_kind', dev)}", flush=True)

    inputs = {k: getattr(args, k.replace("-", "_"))
              for k in ("truth-cache", "pool", "bubbles", "grids")}
    hashes = {k: sha256(v) for k, v in inputs.items()}
    for k, v in inputs.items():
        print(f"input {k:12s} sha256 {hashes[k][:16]}  {v}", flush=True)

    # ---- model context: IV + MW-tuned NFW, residual scaling (as val_split_check) ----
    nfw = gp.NFWPotential(m=NFW_M, r_s=HALO_RS, units="galactic")
    ctxs, _ = build_contexts_bfe(args.truth_cache, 4096, nfw)
    c = ctxs["residual"]
    cfg = arch_cfg_bfe("IV", c["cfg"], nfw)
    fac = c["fac"]
    x_obs, a_obs = c["x_obs"], c["a_obs"]
    ev4, ev15 = c["val_scaled"]["sun4"], c["val_scaled"]["sun15"]
    x_ev = jnp.concatenate([ev4[0][:N_EVAL], ev15[0][:N_EVAL]])
    a_ev = jnp.concatenate([ev4[1][:N_EVAL], ev15[1][:N_EVAL]])
    geo = eval_geometry(cfg, args.bubbles, args.grids)
    print(f"eval: {geo['n_bubble']} bubble points, {N_SPHERE} sphere points, "
          f"{geo['n_pix']} midplane pixels (R <= {R_EVAL} kpc)", flush=True)

    template = make_model_bfe("IV", cfg, 0, None, NFW_M, HALO_RS)
    graphdef, _, rest = nnx.split(template, nnx.Param, ...)
    tx = optax.adam(1e-3)
    full_fn = make_full_fn(graphdef, rest, tx, args.epochs, 4,
                           args.lambda_sun, args.lambda_rho)
    fold_fn = make_fold_fn(graphdef, rest, tx, args.epochs,
                           args.lambda_sun, args.lambda_rho)

    pool = load_pool(args.pool)
    n_pools = int(pool["meta"]["n_trials"])
    trials = parse_trials(args.trials)
    if max(trials) >= n_pools:
        raise SystemExit(f"{args.pool} holds {n_pools} trial pools; trials {trials} requested")

    out = Path(args.outdir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    ref = json.loads(Path(args.check_ref).read_text()) if args.check_ref else None
    ref_out = []
    config = dict(vars(args), input_sha256=hashes, jax=jax.__version__,
                  backend=jax.default_backend(),
                  device=str(getattr(dev, "device_kind", dev)),
                  chi_star=CHI_STAR, family=FAMILY, arm=ARM, nfw_m=NFW_M,
                  halo_rs=HALO_RS, r_eval=R_EVAL, full_cols=FULL_COLS,
                  flux_to_rho=geo["flux_to_rho"])
    t_start = time.time()

    for n in [int(s) for s in args.n_list.split(",") if s]:
        for trial in trials:
            shard = out / "shards" / f"n{n:04d}_t{trial:02d}.npz"
            if shard.exists() and not args.overwrite:
                print(f"n={n:4d} t={trial}: shard exists, skipping", flush=True)
                continue

            # ---- data: identical seeds to val_split_check / the arch ladders ----
            d = draw_trial(pool, trial, n, FAMILY)              # physical units
            x_n = cfg["x_transformer"].transform(jnp.asarray(d["x"]))
            a_n = cfg["a_transformer"].transform(jnp.asarray(d["a"]))
            nv = jnp.asarray(d["n_hat"])
            sig = d["sigma_mmsyr"]
            a_noisy = a_n + jnp.asarray(d["eps_std"] * sig * fac)[:, None] * nv
            invsig = jnp.asarray(1.0 / (sig * fac))           # scaled units
            params0 = nnx.split(make_model_bfe("IV", cfg, trial, None, NFW_M, HALO_RS),
                                nnx.Param, ...)[1]
            colloc = sample_colloc_scaled(jr.PRNGKey(1000 + trial), args.n_colloc,
                                          0.5, 25.0, cfg["x_transformer"])
            omega, p_hat = iw_weights(pool, trial, d, args.knn, args.r_iw)

            # ---- FOLDS ----
            t0 = time.time()
            if args.full_only:
                Z = np.full((args.epochs, n), np.nan, dtype=np.float32)
            else:
                W = np.zeros((n, n), dtype=np.float32)
                for k in range(n):
                    fit = np.delete(np.arange(n), k)
                    W[k, fit] = np.asarray(arm_weights(sig[fit], ARM))
                chunk = n if args.fold_chunk <= 0 else min(args.fold_chunk, n)
                ks = np.arange(n)
                pad = (-n) % chunk
                ks_pad = np.concatenate([ks, np.full(pad, ks[-1])])
                Zs = []
                for s in range(0, ks_pad.size, chunk):
                    kk = ks_pad[s:s + chunk]
                    Zs.append(np.asarray(fold_fn(
                        params0, jnp.asarray(W[kk]), x_n[kk], a_noisy[kk], nv[kk],
                        invsig[kk], x_n, a_noisy, nv, x_obs, a_obs, colloc)))
                Z = np.concatenate(Zs, axis=1)[:, :n].astype(np.float32)
            t_folds = time.time() - t0

            # ---- rules that need only the folds ----
            if args.full_only:
                e_loo = e_loo1se = e_iw = e_iw1se = args.epochs - 1
                cv = cvw = np.full(args.epochs, np.nan)
            else:
                cv, se = cv_curves(Z)
                e_loo, e_loo1se = argmin_and_1se(cv, se)
                if omega.sum() > 0:
                    cvw, sew = cv_curves(Z, omega)
                    e_iw, e_iw1se = argmin_and_1se(cvw, sew)
                else:                     # no pulsar inside r_iw: IW undefined
                    cvw = np.full(args.epochs, np.nan)
                    e_iw, e_iw1se = e_loo, e_loo1se

            # ---- FULL trajectory ----
            t0 = time.time()
            snap_at = jnp.asarray([e_loo, e_loo1se, e_iw, e_iw1se], dtype=jnp.int32)
            hist, snap_chi, p_end, crossed_j, snaps = full_fn(
                params0, x_n, a_noisy, nv, arm_weights(sig, ARM), invsig, x_obs, a_obs,
                colloc, x_ev, a_ev, geo["bub_x"], geo["bub_a"], geo["surf_x"],
                geo["surf_n"], geo["pix_x"], geo["pix_rho_t"], geo["pix_ok"],
                geo["lap_to_rho"], snap_at)
            hist = np.asarray(hist)
            t_full = time.time() - t0
            e_chi, crossed = first_crossing(hist[:, FULL_COLS.index("chi_fit")])
            assert crossed == bool(crossed_j)
            if not crossed:
                snap_chi = p_end                      # never crossed: last epoch

            epochs_rule = dict(chi=e_chi, loo=e_loo, loo1se=e_loo1se,
                               iwloo=e_iw, iwloo1se=e_iw1se)
            np.savez_compressed(
                shard, Z=Z, full=hist.astype(np.float32), full_cols=np.array(FULL_COLS),
                sigma_mmsyr=sig, eps_std=d["eps_std"], r_kpc=d["r"], l_deg=d["l"],
                b_deg=d["b"], pool_idx=d["idx"], omega=omega, p_hat=p_hat,
                chi_crossed=crossed, seconds_folds=t_folds, seconds_full=t_full,
                **{f"e_{k}": v for k, v in epochs_rule.items()},
                config=json.dumps(config, default=str))

            if not args.no_ckpts:
                states = dict(chi=snap_chi, loo=snaps[0], loo1se=snaps[1],
                              iwloo=snaps[2], iwloo1se=snaps[3])
                for rule, st in states.items():
                    if args.full_only and rule != "chi":
                        continue
                    ck = out / "checkpoints" / rule / f"n{n:04d}_t{trial:02d}.pkl"
                    ck.parent.mkdir(parents=True, exist_ok=True)
                    with open(ck, "wb") as fh:
                        pickle.dump(dict(state=jax.tree.map(np.asarray, st), rule=rule,
                                         epoch=epochs_rule[rule], n=n, trial=trial,
                                         crossed=crossed, baseline="nfw",
                                         mode="LOS+Sun+rho"), fh)

            g = hist[:, FULL_COLS.index("gacc_b2")]
            if args.full_only:
                loo_txt, loo_g = "e_loo=   - e_iw=   -", ""
            else:
                loo_txt = f"e_loo={e_loo:4d} e_iw={e_iw:4d}"
                loo_g = f"loo {g[e_loo]:.3f} iw {g[e_iw]:.3f} "
            print(f"n={n:4d} t={trial}  e_chi={e_chi:4d}{'' if crossed else '*'} "
                  f"{loo_txt} e_orc={int(np.argmin(g)):4d} | "
                  f"gacc_b2: chi {g[e_chi]:.3f} {loo_g}orc {g.min():.3f} | "
                  f"n_in(r<={args.r_iw:g})={int((omega > 0).sum())} "
                  f"| folds {t_folds:.0f}s full {t_full:.0f}s", flush=True)

            entry = ref_entry(n, trial, epochs_rule, hist, cv, cvw)
            ref_out.append(entry)
            if ref is not None:
                match = [r for r in ref["entries"] if r["n"] == n and r["trial"] == trial]
                if match:
                    compare_ref(entry, dict(match[0], backend=ref.get("backend")))

    if args.write_ref:
        Path(args.write_ref).write_text(json.dumps(
            dict(backend=jax.default_backend(), jax=jax.__version__,
                 input_sha256=hashes, entries=ref_out), indent=1))
        print(f"wrote reference {args.write_ref}")
    print(f"total {time.time() - t_start:.0f}s; wrote {out}/", flush=True)


if __name__ == "__main__":
    main()
