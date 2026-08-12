"""Architecture ladder with a simulation-fitted BFE analytic baseline.

Same experiment as run_arch_vmap.py (PINN III/IV/V under catalog noise +
whitened L1, modes LOS and LOS+rho, vmap-parallel trials) with ONE change:
the analytic baseline of PINN IV/V is no longer the deliberately
misspecified MW-tuned NFW but a Hernquist-Ostriker basis-function expansion
with all orders n,l <= 4 fit to the m12i particle table itself
(fit_bfe.py -> cache/bfe_nmax4_lmax4.npz):

  III : scale="nfw" x NN — no baseline (bit-identical to run_arch_vmap's
        III; kept as the shared control)
  IV  : III + include_analytic with the FIXED BFE baseline (well-specified:
        ~12% median accel error vs labels, vs ~68-86% for the NFW)
  V   : IV but the BFE's overall (log10 m, r_s) are trained jointly with
        the MLP — m rescales the whole expansion, r_s dilates its radial
        scale; the 75 shape coefficients stay frozen

Faithfulness notes carried over from run_arch_vmap.py:
- III uses the include_analytic=False scaling (u* = max|u|); IV/V share the
  residual scaling u* = max|u - u_BFE| (a ~9x smaller u* than the NFW
  residual — the same physical noise is LARGER in scaled units).
  Noise is drawn once in PHYSICAL units per (trial, n) and converted
  per-context, so all architectures train on identical physical noisy data.
- cfg["ab_potential"] must be set explicitly for IV, cfg["trainable"]=True
  for V; each arch gets its own vmap group.

Outputs: results_arch_bfe_vmap/{results.csv, per_point_errors.npz}, schema
identical to results_arch_vmap (learned_m/learned_rs now describe the BFE).
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx

from galactoPINNs.evaluate import (
    cylindrical_error_decomposition,
    error_decomposition,
)
from galactoPINNs.layers import TrainableGalaxPotential
from galactoPINNs.models.static_model import StaticModel

from datasets import (
    OBSERVER,
    DEFAULT_TRUTH,
    HALO_RS,
    build_data,
    sample_colloc_scaled,
)
from noise import (
    NOISE_FAMILY_SEED,
    apply_los_noise,
    arm_weights,
    draw_sigmas_mmsyr,
    sigma_scale_factor,
)
from run_noise_vmap import make_ensemble_fns, stack_states, take_member
from run_arch_vmap import switch_row_arch, learned_mr, CTX_OF_ARCH
from bfe_baseline import DEFAULT_COEFFS, load_bfe

ARCHES = ("III", "IV", "V")
MODES = ("LOS", "LOS+rho")
REGIONS = ("sun4", "sun15", "gc15")
FAMILY, ARM, LOS_LOSS = "het", "W", "l1"   # catalog noise, whitened L1


def make_model_bfe(arch, cfg, seed, PotCls, m, r_s):
    """PINN V wraps the BFE class; III/IV are plain StaticModels."""
    if arch == "V":
        layer = TrainableGalaxPotential(
            PotClass=PotCls,
            init_kwargs={"m": m, "r_s": r_s},
            trainable=("m", "r_s"),
        )
        return StaticModel(cfg, rngs=nnx.Rngs(seed),
                           trainable_analytic_layer=layer)
    return StaticModel(cfg, rngs=nnx.Rngs(seed))


def arch_cfg_bfe(arch, base_cfg, bfe_pot):
    """Specialize the build_data cfg per architecture, BFE flavor."""
    cfg = dict(base_cfg)
    if arch == "IV":
        cfg["ab_potential"] = bfe_pot
        cfg["trainable"] = False
    elif arch == "V":
        cfg["trainable"] = True
    return cfg


def build_contexts_bfe(truth_cache, n_val, bfe_pot,
                       u_star_plain=None, u_star_res=None):
    """Data contexts: plain (III) and residual-vs-BFE (IV/V)."""
    ctxs = {}
    for key, inc in (("plain", False), ("residual", True)):
        (x_pool, a_pool, x_pool_phys, cfg, val_scaled, dist_sun,
         x_obs, a_obs, val_sm, meta) = build_data(
            truth_cache, n_val=n_val, include_analytic=inc,
            ab_potential=bfe_pot if inc else None,
            u_star_override=u_star_res if inc else u_star_plain)
        z = np.load(truth_cache)
        a_pool_sm = cfg["a_transformer"].transform(
            jnp.asarray(z["a_pool_smooth"]))
        ctxs[key] = dict(x_pool=x_pool, a_pool=a_pool, a_pool_sm=a_pool_sm,
                         x_pool_phys=x_pool_phys, cfg=cfg,
                         val_scaled=val_scaled, dist_sun=dist_sun,
                         x_obs=x_obs, a_obs=a_obs, val_sm=val_sm,
                         fac=sigma_scale_factor(cfg))
    return ctxs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-cache", default=str(DEFAULT_TRUTH))
    ap.add_argument("--coeffs", default=str(DEFAULT_COEFFS))
    ap.add_argument("--n-grid", default="50,100,200,500")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--lambda-rho", type=float, default=1.0)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--v-misspec", type=float, default=1.0,
                    help="factor on PINN V's initial m")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_arch_bfe_vmap"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--matmul-precision", default="highest",
                    choices=["highest", "high", "default"])
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    bfe_pot, BFECls, bfe_meta = load_bfe(args.coeffs)
    m_fit = float(np.asarray(bfe_pot.m))
    print(f"BFE baseline: nmax={bfe_meta['nmax']} lmax={bfe_meta['lmax']} "
          f"r_s={bfe_meta['r_s_kpc']} kpc, m={m_fit:.4e} Msun "
          f"(fit to {bfe_meta['n_particles_fit']:,} of "
          f"{bfe_meta['n_particles_available']:,} particles "
          f"< {bfe_meta['r_max_fit_kpc']} kpc)")

    if args.self_test:
        raise SystemExit(self_test(args, bfe_pot, BFECls, m_fit))

    n_grid = [int(s) for s in args.n_grid.split(",")]
    ctxs, meta = build_contexts_bfe(args.truth_cache, args.n_val, bfe_pot)
    for k, c in ctxs.items():
        print(f"ctx {k:8s}: 1 mm/s/yr = {c['fac']:.5f} scaled units")

    tx = optax.adam(1e-3)
    arch_state = {}
    for arch in ARCHES:
        c = ctxs[CTX_OF_ARCH[arch]]
        cfg = arch_cfg_bfe(arch, c["cfg"], bfe_pot)
        m_init = m_fit * (args.v_misspec if arch == "V" else 1.0)
        template = make_model_bfe(arch, cfg, 0, BFECls, m_init, HALO_RS)
        graphdef, _, rest = nnx.split(template, nnx.Param, ...)
        params_by_trial = [
            nnx.split(make_model_bfe(arch, cfg, t, BFECls, m_init, HALO_RS),
                      nnx.Param, ...)[1]
            for t in range(args.trials)]
        train_norho, train_rho, eval_shared, eval_permodel = make_ensemble_fns(
            graphdef, rest, c["x_obs"], c["a_obs"], tx, args.epochs)
        colloc_by_trial = [
            sample_colloc_scaled(jr.PRNGKey(1000 + t), args.n_colloc,
                                 0.5, 25.0, cfg["x_transformer"])
            for t in range(args.trials)]
        arch_state[arch] = dict(cfg=cfg, graphdef=graphdef, rest=rest,
                                params_by_trial=params_by_trial,
                                train_norho=train_norho, train_rho=train_rho,
                                eval_shared=eval_shared,
                                eval_permodel=eval_permodel,
                                colloc_by_trial=colloc_by_trial)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")
    npz_path = os.path.join(args.outdir, "per_point_errors.npz")
    perpoint = {}
    for name in REGIONS:
        c = ctxs["plain"]
        perpoint[f"dist|{name}"] = c["dist_sun"][name]
    t_start = time.time()

    val_geom = {}
    for key, c in ctxs.items():
        x_tf = c["cfg"]["x_transformer"]
        for name in REGIONS:
            x_val, _ = c["val_scaled"][name]
            xv_phys = np.asarray(x_tf.inverse_transform(x_val))
            dxv = xv_phys - OBSERVER
            nhat = jnp.asarray(dxv / np.linalg.norm(dxv, axis=1,
                                                    keepdims=True))
            val_geom[(key, name)] = (xv_phys, nhat)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "trial", "arch", "mode", "region",
                    "sigma_mmsyr", "rel_err", "rel_los", "rel_trv",
                    "rel_R", "rel_z", "rel_phi",
                    "rel_err_sm", "rel_los_sm", "rel_trv_sm",
                    "train_chi", "train_err_true_mmsyr",
                    "train_err_smooth_mmsyr", "learned_m", "learned_rs"])

        for n in n_grid:
            idx_by_trial, sig_by_trial = {}, {}
            for trial in range(args.trials):
                rng = np.random.default_rng(trial)
                idx_by_trial[trial] = rng.choice(
                    ctxs["plain"]["x_pool"].shape[0], size=n, replace=False)
                sig_by_trial[trial] = draw_sigmas_mmsyr(
                    np.random.default_rng(9000 + 137 * trial + n), n, FAMILY)

            geom = {}
            for key, c in ctxs.items():
                for trial in range(args.trials):
                    idx = idx_by_trial[trial]
                    x_n = c["x_pool"][idx]
                    a_n = c["a_pool"][idx]
                    dx = c["x_pool_phys"][idx] - OBSERVER
                    nv = jnp.asarray(dx / np.linalg.norm(dx, axis=1,
                                                         keepdims=True))
                    sig = sig_by_trial[trial]
                    eps_rng = np.random.default_rng(
                        7000 + 1009 * trial + 31 * n
                        + NOISE_FAMILY_SEED[FAMILY])
                    a_noisy = apply_los_noise(eps_rng, a_n, nv,
                                              sig * c["fac"])
                    wgt = arm_weights(sig, ARM)
                    geom[(key, trial)] = dict(
                        x_n=x_n, nv=nv, a_noisy=a_noisy, w_ip=wgt,
                        a_los_true=jnp.sum(a_n * nv, axis=1),
                        a_los_sm=jnp.sum(c["a_pool_sm"][idx] * nv, axis=1))

            results = {}
            for arch in ARCHES:
                st = arch_state[arch]
                key = CTX_OF_ARCH[arch]
                for mode in MODES:
                    needs_rho = mode == "LOS+rho"
                    trainer = st["train_rho"] if needs_rho else st["train_norho"]
                    X = jnp.stack([geom[(key, t)]["x_n"]
                                   for t in range(args.trials)])
                    A = jnp.stack([geom[(key, t)]["a_noisy"]
                                   for t in range(args.trials)])
                    NV = jnp.stack([geom[(key, t)]["nv"]
                                    for t in range(args.trials)])
                    WIP = jnp.stack([geom[(key, t)]["w_ip"]
                                     for t in range(args.trials)])
                    SW = jnp.asarray(
                        [switch_row_arch(mode, args.lambda_rho)]
                        * args.trials, dtype=jnp.float32)
                    params = stack_states(st["params_by_trial"])
                    opt = tx.init(params)
                    arrays = (X, A, NV, WIP, SW) if not needs_rho else (
                        X, A, NV, WIP, SW, jnp.stack(st["colloc_by_trial"]))
                    t0 = time.time()
                    trained, _ = trainer(params, opt, *arrays)
                    jax.block_until_ready(trained)
                    a_tr = st["eval_permodel"](trained, X)
                    results[(arch, mode)] = (trained, a_tr)
                    print(f"n={n:5d} PINN-{arch:3s} {mode:8s}: "
                          f"{args.trials} trials in {time.time()-t0:.1f}s",
                          flush=True)

            for arch in ARCHES:
                st = arch_state[arch]
                key = CTX_OF_ARCH[arch]
                c = ctxs[key]
                for mode in MODES:
                    trained, a_tr = results[(arch, mode)]
                    for trial in range(args.trials):
                        g = geom[(key, trial)]
                        sig = sig_by_trial[trial]
                        sig_scaled = jnp.asarray(sig) * c["fac"]
                        los_pred = jnp.sum(a_tr[trial] * g["nv"], axis=1)
                        los_obs = jnp.sum(g["a_noisy"] * g["nv"], axis=1)
                        chi = float(jnp.mean(jnp.abs(los_pred - los_obs)
                                             / sig_scaled))
                        err_true = float(jnp.mean(jnp.abs(
                            los_pred - g["a_los_true"]))) / c["fac"]
                        err_sm = float(jnp.mean(jnp.abs(
                            los_pred - g["a_los_sm"]))) / c["fac"]
                        if arch == "V":
                            member = nnx.merge(
                                st["graphdef"], take_member(trained, trial),
                                st["rest"])
                            lm, lrs = learned_mr(member)
                            lm_s, lrs_s = f"{lm:.4e}", f"{lrs:.4f}"
                        else:
                            lm_s = lrs_s = ""
                        for region in REGIONS:
                            x_val, a_val = c["val_scaled"][region]
                            a_val_sm = c["val_sm"][region]
                            xv_phys, nhat = val_geom[(key, region)]
                            a_pred = st["eval_shared"](trained, x_val)[trial]
                            d = error_decomposition(a_val, a_pred, nhat)
                            dsm = error_decomposition(a_val_sm, a_pred, nhat)
                            cyl = cylindrical_error_decomposition(
                                a_val, a_pred, jnp.asarray(xv_phys))
                            err = float(jnp.mean(d["rel_error_magnitude"]))
                            kk = f"{region}|{n}|{trial}|{arch}|{mode}"
                            perpoint[f"err|{kk}"] = np.asarray(
                                d["rel_error_magnitude"], np.float32)
                            perpoint[f"err_smooth|{kk}"] = np.asarray(
                                dsm["rel_error_magnitude"], np.float32)
                            w.writerow([n, trial, arch, mode, region,
                                        f"{np.median(sig):.2f}", err,
                                        float(jnp.mean(d["rel_los_error"])),
                                        float(jnp.mean(d["rel_transverse_error"])),
                                        float(jnp.mean(cyl["rel_R_error"])),
                                        float(jnp.mean(cyl["rel_z_error"])),
                                        float(jnp.mean(cyl["rel_phi_error"])),
                                        float(jnp.mean(dsm["rel_error_magnitude"])),
                                        float(jnp.mean(dsm["rel_los_error"])),
                                        float(jnp.mean(dsm["rel_transverse_error"])),
                                        f"{chi:.3f}", f"{err_true:.3f}",
                                        f"{err_sm:.3f}", lm_s, lrs_s])
                        f.flush()
                        msg = (f"n={n:5d} t={trial} PINN-{arch:3s} {mode:8s} "
                               f"chi={chi:5.2f} errTrue={err_true:5.2f}")
                        if arch == "V":
                            msg += f" m={lm_s} rs={lrs_s}"
                        print(msg, flush=True)

    np.savez_compressed(npz_path, **perpoint)
    print(f"total wall time: {time.time() - t_start:.1f}s")
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


# ---------------------------------------------------------------------------
# Self-test: per-arch vmap groups vs train_model_static, BFE flavor.
# ---------------------------------------------------------------------------

def self_test(args, bfe_pot, BFECls, m_fit):
    from galactoPINNs.train import train_model_static

    print("self-test: BFE arch vmap groups vs train_model_static (FIRE, hetW)")
    ctxs, _ = build_contexts_bfe(args.truth_cache, 256, bfe_pot)
    n, epochs, trial = 40, 40, 0
    tx = optax.adam(1e-3)
    ok = True

    for arch in ARCHES:
        c = ctxs[CTX_OF_ARCH[arch]]
        cfg = arch_cfg_bfe(arch, c["cfg"], bfe_pot)
        rng = np.random.default_rng(trial)
        idx = rng.choice(c["x_pool"].shape[0], size=n, replace=False)
        x_n, a_n = c["x_pool"][idx], c["a_pool"][idx]
        dx = c["x_pool_phys"][idx] - OBSERVER
        nv = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))
        sig = draw_sigmas_mmsyr(np.random.default_rng(9000 + 137 * trial + n),
                                n, FAMILY)
        eps_rng = np.random.default_rng(
            7000 + 1009 * trial + 31 * n + NOISE_FAMILY_SEED[FAMILY])
        a_noisy = apply_los_noise(eps_rng, a_n, nv, sig * c["fac"])
        w_ip = arm_weights(sig, ARM)
        colloc = sample_colloc_scaled(jr.PRNGKey(1000 + trial), 64, 0.5, 25.0,
                                      cfg["x_transformer"])

        template = make_model_bfe(arch, cfg, 0, BFECls, m_fit, HALO_RS)
        graphdef, params0, rest = nnx.split(template, nnx.Param, ...)
        tn, tr, _, _ = make_ensemble_fns(graphdef, rest, c["x_obs"],
                                         c["a_obs"], tx, epochs)
        for mode in MODES:
            needs_rho = mode == "LOS+rho"
            SW = jnp.asarray([switch_row_arch(mode, 1.0)], dtype=jnp.float32)
            params = stack_states([params0])
            opt = tx.init(params)
            arrays = (jnp.stack([x_n]), jnp.stack([a_noisy]),
                      jnp.stack([nv]), jnp.stack([w_ip]), SW)
            if needs_rho:
                arrays = arrays + (jnp.stack([colloc]),)
            trained, _ = (tr if needs_rho else tn)(params, opt, *arrays)

            model = make_model_bfe(arch, cfg, 0, BFECls, m_fit, HALO_RS)
            kw = dict(n_vecs=nv, line_of_sight=True, los_loss=LOS_LOSS,
                      importance_weight=w_ip)
            if needs_rho:
                kw.update(colloc_x=colloc, lambda_rho=1.0)
            train_model_static(model, optax.adam(1e-3), x_n, a_noisy, epochs,
                               log_every=0, **kw)
            _, p_seq, _ = nnx.split(model, nnx.Param, ...)
            p_vmap = take_member(trained, 0)
            num = sum(float(jnp.sum((a - b) ** 2)) for a, b in
                      zip(jax.tree.leaves(p_seq), jax.tree.leaves(p_vmap)))
            den = sum(float(jnp.sum(a ** 2)) for a in jax.tree.leaves(p_seq))
            rel = (num / den) ** 0.5
            status = "ok" if rel < 1e-4 else "FAIL"
            ok &= rel < 1e-4
            print(f"  PINN-{arch:3s} {mode:8s} param reldiff: {rel:.2e}  "
                  f"{status}")
    print("self-test PASSED" if ok else "self-test FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
