"""GPU-parallel driver for the FIRE noisy-LOS experiment.

FIRE adaptation of clean_experiments/noisy_data/run_experiment_vmap.py — the
Donlon+2025 catalog-noise machinery (settings table, family seeds, W/C arms,
per-model importance weights + l1/sq switch) is identical; see the parent's
docstring for mechanics. What changed:

- Truth from the FIRE m12i cache (``--truth-cache``) instead of galax's
  MilkyWayPotential; geometry is baked into the cache.
- Each (n, trial, setting, mode, region) row is also evaluated against the
  0.5 kpc-smoothed reference field (``rel_err_sm``/``rel_los_sm``/
  ``rel_trv_sm`` columns), and the train-point diagnostic gains
  ``train_err_smooth_mmsyr`` — mean |LOS error| against the SMOOTHED truth,
  arguably the fairer "did the sigma-aware arm help" number when the clumpy
  truth itself carries ~2% graininess.

Outputs: results.csv + per_point_errors.npz in --outdir
(default results_noise_vmap/), keys as the parent plus the smooth variants.
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
from galactoPINNs.models.static_model import StaticModel

from datasets import (
    OBSERVER,
    GC,
    DEFAULT_TRUTH,
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

# label -> (noise family, arm, los_loss). Same table as noisy_data.
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

NORHO_MODES = ("LOS", "3D")        # 3D reference (s0 only) joins the no-rho group
RHO_MODES = ("LOS+Sun+rho",)

LAMBDA_REL = 1.0
EPS = 1e-10


def switch_row(mode, los_loss, lambda_sun, lambda_rho):
    """Per-model loss switches (w3d, wlos, anch_on, lam_anchor, lam_rho, sq_on)."""
    sq = 1.0 if los_loss == "sq" else 0.0
    return {
        "3D":          (1.0, 0.0, 0.0, 0.0,        0.0,        0.0),
        "LOS":         (0.0, 1.0, 0.0, 0.0,        0.0,        sq),
        "LOS+Sun+rho": (0.0, 1.0, 1.0, lambda_sun, lambda_rho, sq),
    }[mode]


# ---------------------------------------------------------------------------
# Batched loss / train / eval — verbatim from the noisy_data vmap driver.
# ---------------------------------------------------------------------------

def make_ensemble_fns(graphdef, rest, x_obs, a_obs, tx, epochs):

    def data_terms(m, x, a, nv, w_ip, sw):
        w3d, wlos, anch_on, lam_anchor, _, sq_on = sw
        a_pred = m(x)["acceleration"]                       # (n, 3)

        dn = jnp.linalg.norm(a_pred - a, axis=1)
        tn = jnp.linalg.norm(a, axis=1)
        per3d = w_ip * (dn + LAMBDA_REL * dn / (tn + EPS))
        loss3d = jnp.mean(per3d)

        r = jnp.sum(a_pred * nv, axis=1) - jnp.sum(a * nv, axis=1)
        perlos = w_ip * ((1.0 - sq_on) * jnp.abs(r) + sq_on * r ** 2)
        ar = (m(x_obs)["acceleration"] - a_obs).reshape(-1)
        anch = (1.0 - sq_on) * jnp.abs(ar) + sq_on * ar ** 2
        denom = x.shape[0] + anch_on * anch.size
        loss_los = (jnp.sum(perlos) + anch_on * lam_anchor * jnp.sum(anch)) / denom

        return w3d * loss3d + wlos * loss_los

    def loss_norho(params, x, a, nv, w_ip, sw):
        m = nnx.merge(graphdef, params, rest)
        return data_terms(m, x, a, nv, w_ip, sw)

    def loss_rho(params, x, a, nv, w_ip, sw, colloc):
        m = nnx.merge(graphdef, params, rest)
        lap = m.compute_laplacian(colloc)
        rho_pen = jnp.mean(jax.nn.relu(-lap))
        return data_terms(m, x, a, nv, w_ip, sw) + sw[4] * rho_pen

    def make_trainer(loss_fn):
        @jax.jit
        def train_group(params, opt, *arrays):
            def stepf(carry, _):
                p, o = carry
                losses, grads = jax.vmap(jax.value_and_grad(loss_fn))(p, *arrays)
                updates, o = tx.update(grads, o, p)
                return (optax.apply_updates(p, updates), o), losses
            (p, _), hist = jax.lax.scan(stepf, (params, opt), None, length=epochs)
            return p, hist[-1]
        return train_group

    @jax.jit
    def eval_shared(params, x_val):
        def one(p, xv):
            return nnx.merge(graphdef, p, rest)(xv)["acceleration"]
        return jax.vmap(one, in_axes=(0, None))(params, x_val)

    @jax.jit
    def eval_permodel(params, X):
        def one(p, xi):
            return nnx.merge(graphdef, p, rest)(xi)["acceleration"]
        return jax.vmap(one, in_axes=(0, 0))(params, X)

    return make_trainer(loss_norho), make_trainer(loss_rho), eval_shared, eval_permodel


def stack_states(states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def take_member(stacked, i):
    return jax.tree.map(lambda x: x[i], stacked)


# ---------------------------------------------------------------------------
# Self-test: vmap ensemble vs train_model_static, across arms, FIRE arrays.
# ---------------------------------------------------------------------------

def self_test(truth_cache):
    from galactoPINNs.train import train_model_static

    print("self-test: vmap ensemble vs train_model_static across arms (FIRE)")
    (x_pool, a_pool, x_pool_phys, cfg, val_scaled, _d,
     x_obs, a_obs, _val_sm, _meta) = build_data(truth_cache, n_val=256)
    fac = sigma_scale_factor(cfg)
    n, epochs, lam, trial = 40, 40, 1.0, 0

    rng = np.random.default_rng(trial)
    idx = rng.choice(x_pool.shape[0], size=n, replace=False)
    x_n, a_n = x_pool[idx], a_pool[idx]
    dx = x_pool_phys[idx] - OBSERVER
    nv = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))
    colloc = sample_colloc_scaled(jr.PRNGKey(1000 + trial), 64, 0.5, 25.0,
                                  cfg["x_transformer"])

    check = [("s1.7", "LOS"), ("s1.7", "LOS+Sun+rho"),
             ("s1.7W", "LOS"), ("hetC", "LOS+Sun+rho"), ("s0", "3D")]

    def spec_data(label):
        family, arm, los_loss = SETTINGS[label]
        sig_i = draw_sigmas_mmsyr(np.random.default_rng(9000 + 137 * trial + n),
                                  n, family)
        eps_rng = np.random.default_rng(
            7000 + 1009 * trial + 31 * n + NOISE_FAMILY_SEED[family])
        a_noisy = apply_los_noise(eps_rng, a_n, nv, sig_i * fac)
        w = arm_weights(sig_i, arm)
        w_ip = jnp.ones(n) if w is None else w
        return los_loss, a_noisy, w_ip

    template = StaticModel(cfg, rngs=nnx.Rngs(0))
    graphdef, params0, rest = nnx.split(template, nnx.Param, ...)
    tx = optax.adam(1e-3)
    tn, tr, _, _ = make_ensemble_fns(graphdef, rest, x_obs, a_obs, tx, epochs)

    results = {}
    for modes, trainer, needs_rho in ((NORHO_MODES, tn, False), (RHO_MODES, tr, True)):
        specs = [(l, mo) for (l, mo) in check if mo in modes]
        if not specs:
            continue
        A, NV, WIP, SW = [], [], [], []
        for label, mode in specs:
            los_loss, a_noisy, w_ip = spec_data(label)
            A.append(a_noisy if mode != "3D" else a_n)
            NV.append(nv); WIP.append(w_ip)
            SW.append(switch_row(mode, los_loss, lam, lam))
        X = jnp.stack([x_n] * len(specs))
        A = jnp.stack(A); NV = jnp.stack(NV); WIP = jnp.stack(WIP)
        SW = jnp.asarray(SW, dtype=jnp.float32)
        params = stack_states([params0] * len(specs))
        opt = tx.init(params)
        arrays = (X, A, NV, WIP, SW) if not needs_rho else (
            X, A, NV, WIP, SW, jnp.stack([colloc] * len(specs)))
        trained, _ = trainer(params, opt, *arrays)
        jax.block_until_ready(trained)
        for i, spec in enumerate(specs):
            results[spec] = (trained, i)

    ok = True
    for label, mode in check:
        los_loss, a_noisy, w_ip = spec_data(label)
        weights = None if SETTINGS[label][1] == "plain" else w_ip
        kw = {
            "3D": dict(),
            "LOS": dict(n_vecs=nv, line_of_sight=True, los_loss=los_loss,
                        importance_weight=weights),
            "LOS+Sun+rho": dict(n_vecs=nv, line_of_sight=True, los_loss=los_loss,
                                importance_weight=weights, anchor_x=x_obs,
                                anchor_a=a_obs, lambda_anchor=lam,
                                colloc_x=colloc, lambda_rho=lam),
        }[mode]
        a_tgt = a_n if mode == "3D" else a_noisy
        model = StaticModel(cfg, rngs=nnx.Rngs(0))
        train_model_static(model, optax.adam(1e-3), x_n, a_tgt, epochs,
                           log_every=0, **kw)
        _, p_seq, _ = nnx.split(model, nnx.Param, ...)
        stacked, i = results[(label, mode)]
        p_vmap = take_member(stacked, i)
        num = sum(float(jnp.sum((a - b) ** 2)) for a, b in
                  zip(jax.tree.leaves(p_seq), jax.tree.leaves(p_vmap)))
        den = sum(float(jnp.sum(a ** 2)) for a in jax.tree.leaves(p_seq))
        rel = (num / den) ** 0.5
        status = "ok" if rel < 1e-4 else "FAIL"
        ok &= rel < 1e-4
        print(f"  {label:6s} {mode:12s} param reldiff: {rel:.2e}  {status}")
    print("self-test PASSED" if ok else "self-test FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-cache", default=str(DEFAULT_TRUTH))
    ap.add_argument("--n-grid", default="50,100,200,500")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--lambda-sun", type=float, default=1.0)
    ap.add_argument("--lambda-rho", type=float, default=1.0)
    ap.add_argument("--n-colloc", type=int, default=512)
    ap.add_argument("--r-min-colloc", type=float, default=0.5)
    ap.add_argument("--r-colloc-max", type=float, default=25.0)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_noise_vmap"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--matmul-precision", default="highest",
                    choices=["highest", "high", "default"])
    ap.add_argument("--jax-cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR"))
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    if args.jax_cache:
        jax.config.update("jax_compilation_cache_dir", args.jax_cache)
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    if args.self_test:
        raise SystemExit(self_test(args.truth_cache))

    n_grid = [int(s) for s in args.n_grid.split(",")]
    settings = [s for s in args.settings.split(",") if s]
    unknown = [s for s in settings if s not in SETTINGS]
    if unknown:
        raise SystemExit(f"unknown settings {unknown}; known: {list(SETTINGS)}")

    (x_pool, a_pool, x_pool_phys, cfg, val_scaled, dist_sun,
     x_obs, a_obs, val_sm, meta) = build_data(args.truth_cache, args.n_val)
    assert set(meta["regions"]) == set(REGIONS), meta["regions"]
    fac = sigma_scale_factor(cfg)
    print(f"truth cache: {args.truth_cache}")
    print(f"1 mm/s/yr = {fac:.5f} scaled units")

    # smoothed-reference labels for the train pool (per-point diagnostics);
    # scaled with the same pool-fitted transformer
    z = np.load(args.truth_cache)
    a_pool_sm = cfg["a_transformer"].transform(jnp.asarray(z["a_pool_smooth"]))

    template = StaticModel(cfg, rngs=nnx.Rngs(0))
    graphdef, _, rest = nnx.split(template, nnx.Param, ...)
    params_by_trial = [
        nnx.split(StaticModel(cfg, rngs=nnx.Rngs(t)), nnx.Param, ...)[1]
        for t in range(args.trials)]
    colloc_by_trial = [
        sample_colloc_scaled(jr.PRNGKey(1000 + t), args.n_colloc,
                             args.r_min_colloc, args.r_colloc_max,
                             cfg["x_transformer"])
        for t in range(args.trials)]

    tx = optax.adam(1e-3)
    train_norho, train_rho, eval_shared, eval_permodel = make_ensemble_fns(
        graphdef, rest, x_obs, a_obs, tx, args.epochs)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")
    npz_path = os.path.join(args.outdir, "per_point_errors.npz")
    perpoint = {f"dist|{name}": dist_sun[name] for name in REGIONS}
    t_start = time.time()

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
        perpoint[f"a_val_smooth|{name}"] = np.asarray(val_sm[name],
                                                      dtype=np.float32)
        perpoint[f"nhat|{name}"] = np.asarray(nhat, dtype=np.float32)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "trial", "setting", "sigma_mmsyr", "arm",
                    "mode", "region", "rel_err",
                    "rel_los", "rel_trv", "rel_R", "rel_z", "rel_phi",
                    "rel_err_sm", "rel_los_sm", "rel_trv_sm",
                    "train_chi", "train_err_true_mmsyr",
                    "train_err_smooth_mmsyr"])

        for n in n_grid:
            # ---- per-(trial) geometry, shared across settings ----
            geom = {}
            for trial in range(args.trials):
                rng = np.random.default_rng(trial)
                idx = rng.choice(x_pool.shape[0], size=n, replace=False)
                x_n, a_n = x_pool[idx], a_pool[idx]
                dx = x_pool_phys[idx] - OBSERVER
                nv = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))
                geom[trial] = (x_n, a_n, nv,
                               jnp.sum(a_n * nv, axis=1),
                               jnp.sum(a_pool_sm[idx] * nv, axis=1))

            # ---- per-(trial,label) noise draw + weights ----
            noise = {}
            for trial in range(args.trials):
                x_n, a_n, nv, _, _ = geom[trial]
                for label in settings:
                    family, arm, los_loss = SETTINGS[label]
                    sig_i = draw_sigmas_mmsyr(
                        np.random.default_rng(9000 + 137 * trial + n), n, family)
                    eps_rng = np.random.default_rng(
                        7000 + 1009 * trial + 31 * n + NOISE_FAMILY_SEED[family])
                    a_noisy = apply_los_noise(eps_rng, a_n, nv, sig_i * fac)
                    wgt = arm_weights(sig_i, arm)
                    w_ip = jnp.ones(n) if wgt is None else wgt
                    noise[(trial, label)] = (sig_i, a_noisy, w_ip, los_loss)

            # ---- build training specs and the two vmap groups ----
            def make_specs(modes):
                out = []
                for trial in range(args.trials):
                    for label in settings:
                        for mode in MODES:
                            if mode in modes:
                                out.append((trial, label, mode))
                        if "3D" in modes and label == "s0":
                            out.append((trial, "s0", "3D"))
                return out

            results = {}
            timings = {}
            for modes, trainer, needs_rho in (
                (NORHO_MODES, train_norho, False),
                (RHO_MODES, train_rho, True),
            ):
                specs = make_specs(modes)
                if not specs:
                    continue
                X, A, NV, WIP, SW, COL, P = [], [], [], [], [], [], []
                for (trial, label, mode) in specs:
                    x_n, a_n, nv, _, _ = geom[trial]
                    sig_i, a_noisy, w_ip, los_loss = noise[(trial, label)]
                    X.append(x_n)
                    A.append(a_n if mode == "3D" else a_noisy)
                    NV.append(nv)
                    WIP.append(jnp.ones(n) if mode == "3D" else w_ip)
                    SW.append(switch_row(mode, los_loss if mode != "3D" else "l1",
                                         args.lambda_sun, args.lambda_rho))
                    COL.append(colloc_by_trial[trial])
                    P.append(params_by_trial[trial])
                X = jnp.stack(X); A = jnp.stack(A); NV = jnp.stack(NV)
                WIP = jnp.stack(WIP); SW = jnp.asarray(SW, dtype=jnp.float32)
                params = stack_states(P); opt = tx.init(params)
                arrays = (X, A, NV, WIP, SW) if not needs_rho else (
                    X, A, NV, WIP, SW, jnp.stack(COL))

                t0 = time.time()
                trained, final_loss = trainer(params, opt, *arrays)
                jax.block_until_ready(trained)
                timings[modes] = time.time() - t0

                a_tr = eval_permodel(trained, X)
                for i, spec in enumerate(specs):
                    results[spec] = (trained, i, X, a_tr[i])
                print(f"n={n:5d} trained {len(specs):2d} models "
                      f"[{'+'.join(modes)}] in {timings[modes]:.1f}s", flush=True)

            # ---- evaluate on the 3 regions (shared val sets per group) ----
            groups = {}
            for spec, (stacked, i, _, _) in results.items():
                groups.setdefault(id(stacked), (stacked, []))[1].append((spec, i))
            region_pred = {}
            for stacked, members in groups.values():
                for region in REGIONS:
                    x_val, _ = val_scaled[region]
                    apred = eval_shared(stacked, x_val)
                    for spec, i in members:
                        region_pred[(spec, region)] = apred[i]

            # ---- emit rows ----
            for trial in range(args.trials):
                x_n, a_n, nv, a_los_true, a_los_sm = geom[trial]
                for label in settings:
                    sig_i, a_noisy, w_ip, los_loss = noise[(trial, label)]
                    sig_scaled = jnp.asarray(sig_i) * fac
                    _, arm, _ = SETTINGS[label]
                    modes = list(MODES) + (["3D"] if label == "s0" else [])
                    for mode in modes:
                        spec = (trial, label, mode)
                        _, _, _, a_pred_tr = results[spec]
                        a_tgt = a_n if mode == "3D" else a_noisy
                        los_pred = jnp.sum(a_pred_tr * nv, axis=1)
                        los_obs = jnp.sum(a_tgt * nv, axis=1)
                        if sig_i.max() > 0:
                            chi = float(jnp.mean(jnp.abs(los_pred - los_obs) / sig_scaled))
                        else:
                            chi = float("nan")
                        err_true = float(jnp.mean(jnp.abs(los_pred - a_los_true))) / fac
                        err_sm = float(jnp.mean(jnp.abs(los_pred - a_los_sm))) / fac

                        for region in REGIONS:
                            x_val, a_val = val_scaled[region]
                            a_val_sm = val_sm[region]
                            xv_phys, nhat = val_geom[region]
                            a_pred = region_pred[(spec, region)]
                            d = error_decomposition(a_val, a_pred, nhat)
                            dsm = error_decomposition(a_val_sm, a_pred, nhat)
                            c = cylindrical_error_decomposition(
                                a_val, a_pred, jnp.asarray(xv_phys))
                            err = float(jnp.mean(d["rel_error_magnitude"]))
                            key = f"{region}|{n}|{trial}|{label}|{mode}"
                            perpoint[f"err|{key}"] = np.asarray(
                                d["rel_error_magnitude"], np.float32)
                            perpoint[f"err_smooth|{key}"] = np.asarray(
                                dsm["rel_error_magnitude"], np.float32)
                            perpoint[f"apred|{key}"] = np.asarray(a_pred, np.float32)
                            w.writerow([n, trial, label,
                                        f"{np.median(sig_i):.2f}", arm, mode,
                                        region, err,
                                        float(jnp.mean(d["rel_los_error"])),
                                        float(jnp.mean(d["rel_transverse_error"])),
                                        float(jnp.mean(c["rel_R_error"])),
                                        float(jnp.mean(c["rel_z_error"])),
                                        float(jnp.mean(c["rel_phi_error"])),
                                        float(jnp.mean(dsm["rel_error_magnitude"])),
                                        float(jnp.mean(dsm["rel_los_error"])),
                                        float(jnp.mean(dsm["rel_transverse_error"])),
                                        f"{chi:.3f}", f"{err_true:.3f}",
                                        f"{err_sm:.3f}"])
                        f.flush()
                        print(f"n={n:5d} trial={trial} {label:6s} {mode:12s} "
                              f"chi={chi:5.2f} errTrue={err_true:5.2f} "
                              f"errSm={err_sm:5.2f} mm/s/yr", flush=True)

    np.savez_compressed(npz_path, **perpoint)
    print(f"total wall time: {time.time() - t_start:.1f}s")
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
