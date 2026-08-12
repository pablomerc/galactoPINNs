"""GPU-parallel driver for the sun_los_recovery experiment.

Same experiment as ``run_experiment.py`` (same config, modes, outputs), but all
(trial x mode) trainings for a given n run SIMULTANEOUSLY as one batched
ensemble instead of one at a time:

- ``jax.vmap`` over a stacked axis of model parameters trains many independent
  models in one device program. ``optax.adam`` on the stacked parameter pytree
  is exactly per-model adam (its state is elementwise), so each ensemble member
  follows the same trajectory it would have alone.
- The five loss variants are expressed as one loss with per-model switch
  weights (w_3d, w_los, anchor_on, lambda_anchor, lambda_rho), so every member
  shares a single computation graph — the requirement for vmap.
- Models are grouped by whether they need the rho>=0 Laplacian hinge; the
  Hessian term dominates the step cost, so modes without it never pay for it.
- ``jax.lax.scan`` over epochs compiles the whole training loop into one XLA
  call: 1500 epochs = 1 dispatch instead of 1500.

Outputs are identical in format to ``run_experiment.py`` (results.csv +
per_point_errors.npz, same keys/ordering). Values match to float32 trajectory
noise, not bitwise: vmap reorders float ops, and adam slowly amplifies ~1e-8
differences. Statistically the two drivers are the same experiment.

Backends
--------
- NVIDIA cluster (the target): install ``jax[cuda12]`` and run as-is; jax picks
  up the GPU automatically. This is where the ensemble pays off — a single
  width-128 MLP training leaves a datacenter GPU ~idle, while the stacked
  ensemble fills it. More trials are nearly free: raise ``--trials`` for
  tighter error bars at little wall-clock cost.
- Multi-GPU / many nodes: split the work at the CLI level, e.g. one SLURM array
  task per ``--n-grid`` value (results.csv files concatenate cleanly).
- Apple Silicon: runs on CPU. Gains are modest (XLA already multithreads the
  big matmuls; ~1.1x at n=1000, more at small n where dispatch overhead
  dominates). ``jax-metal`` is NOT supported: it is pinned to jax 0.4.x and
  lacks the higher-order autodiff this model needs (Hessian-of-MLP inside a
  gradient), so it cannot run this stack.
- Set ``--jax-cache DIR`` (or env JAX_COMPILATION_CACHE_DIR) to persist XLA
  compilations across runs; saves ~0.5-1 min of startup on repeat runs.

Verification
------------
``python run_experiment_vmap.py --self-test`` trains all five modes both ways
(this driver vs galactoPINNs.train.train_model_static) on a tiny problem and
asserts the final parameters agree. Run it once on any new machine/backend.
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
import galax.potential as gp

from galactoPINNs.evaluate import (
    cylindrical_error_decomposition,
    error_decomposition,
)
from galactoPINNs.models.static_model import StaticModel

from datasets import (
    OBSERVER,
    GC,
    build_data,
    sample_colloc_scaled,
)

MODES = ("3D", "LOS", "LOS+Sun", "LOS+rho", "LOS+Sun+rho")
REGIONS = ("sun4", "sun15", "gc15")

# Modes that carry the rho>=0 collocation hinge (the Hessian term). They train
# in their own vmap group so the other modes never compute Hessians.
NORHO_MODES = ("3D", "LOS", "LOS+Sun")
RHO_MODES = ("LOS+rho", "LOS+Sun+rho")

LAMBDA_REL = 1.0  # relative-error weight in the 3D loss (train.py default)
EPS = 1e-10


def mode_weight_row(mode, lambda_sun, lambda_rho):
    """Per-model loss switches: (w_3d, w_los, anchor_on, lambda_anchor, lambda_rho).

    Zero switches contribute exactly zero gradient (0 * finite = 0 in IEEE),
    so each ensemble member optimizes precisely its own objective.
    """
    return {
        "3D":          (1.0, 0.0, 0.0, 0.0, 0.0),
        "LOS":         (0.0, 1.0, 0.0, 0.0, 0.0),
        "LOS+Sun":     (0.0, 1.0, 1.0, lambda_sun, 0.0),
        "LOS+rho":     (0.0, 1.0, 0.0, 0.0, lambda_rho),
        "LOS+Sun+rho": (0.0, 1.0, 1.0, lambda_sun, lambda_rho),
    }[mode]


# ---------------------------------------------------------------------------
# Batched loss / train / eval builders.
# The math mirrors galactoPINNs.train.train_step_static term by term
# (_acc_loss and _acc_los_loss); --self-test checks the equivalence.
# ---------------------------------------------------------------------------

def make_ensemble_fns(graphdef, rest, x_obs, a_obs, tx, epochs):
    """Build the jitted group trainers and the vmapped evaluator.

    graphdef/rest come from ``nnx.split(model, nnx.Param, ...)`` of a template
    model; they are identical for every member, only the Param state differs
    and carries the stacked leading axis.
    """

    def data_terms(m, x, a, nv, w):
        """3D + LOS(+anchor) data losses, combined via the switch weights."""
        w3d, wlos, anch_on, lam_anchor, _ = w
        a_pred = m(x)["acceleration"]                       # (n, 3)

        # 3D loss: mean(|da| + LAMBDA_REL * |da|/|a|)   [train.py _acc_loss]
        diff_norm = jnp.linalg.norm(a_pred - a, axis=1)
        true_norm = jnp.linalg.norm(a, axis=1)
        per3d = diff_norm + LAMBDA_REL * diff_norm / (true_norm + EPS)
        loss3d = jnp.mean(per3d)

        # LOS loss with the Sun anchor folded into the shared mean
        # [train.py _acc_los_loss]: (sum|dlos| + lam*sum|danchor|) / (n + 3M)
        alos_pred = jnp.sum(a_pred * nv, axis=1)
        alos_true = jnp.sum(a * nv, axis=1)
        perlos = jnp.abs(alos_pred - alos_true)
        anchor_abs = jnp.abs(m(x_obs)["acceleration"] - a_obs).reshape(-1)
        denom = x.shape[0] + anch_on * anchor_abs.size
        loss_los = (jnp.sum(perlos)
                    + anch_on * lam_anchor * jnp.sum(anchor_abs)) / denom

        return w3d * loss3d + wlos * loss_los

    def loss_norho(params, x, a, nv, w):
        m = nnx.merge(graphdef, params, rest)
        return data_terms(m, x, a, nv, w)

    def loss_rho(params, x, a, nv, colloc, w):
        m = nnx.merge(graphdef, params, rest)
        lap = m.compute_laplacian(colloc)                   # (K,) ~ rho
        rho_pen = jnp.mean(jax.nn.relu(-lap))               # hinge iff rho < 0
        return data_terms(m, x, a, nv, w) + w[4] * rho_pen

    def make_trainer(loss_fn):
        @jax.jit
        def train_group(params, opt, *arrays):
            def stepf(carry, _):
                p, o = carry
                losses, grads = jax.vmap(jax.value_and_grad(loss_fn))(p, *arrays)
                updates, o = tx.update(grads, o, p)
                return (optax.apply_updates(p, updates), o), losses

            (p, _), hist = jax.lax.scan(stepf, (params, opt), None, length=epochs)
            return p, hist[-1]  # trained stacked params, final per-model loss

        return train_group

    @jax.jit
    def eval_group(params, x_val, x_lap):
        """Predicted accelerations + rho<0 fraction for every ensemble member.

        Error decompositions happen on host afterwards, through the same
        galactoPINNs.evaluate functions the sequential driver uses.
        """
        def one(p, xv, xl):
            m = nnx.merge(graphdef, p, rest)
            a_pred = m(xv)["acceleration"]
            neg = jnp.mean(m.compute_laplacian(xl) < 0.0)
            return a_pred, neg

        return jax.vmap(one, in_axes=(0, None, None))(params, x_val, x_lap)

    return make_trainer(loss_norho), make_trainer(loss_rho), eval_group


def stack_states(states):
    """Stack a list of identically-shaped nnx.State pytrees along a new axis 0."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def take_member(stacked, i):
    """Slice ensemble member i out of a stacked pytree."""
    return jax.tree.map(lambda x: x[i], stacked)


def train_one_n(x_by_trial, a_by_trial, nv_by_trial, colloc_by_trial,
                params_by_trial, trainers, tx, lambda_sun, lambda_rho):
    """Train every (trial, mode) model for one n; return {(trial, mode): ...}.

    Two vmap groups: NORHO_MODES and RHO_MODES, each stacked over
    trials x modes. Returns per-spec trained params (as slices of the stacked
    state), the final loss, and per-group wall time.
    """
    train_norho, train_rho = trainers
    trials = len(x_by_trial)
    out, timings = {}, {}

    for modes, needs_rho, trainer in (
        (NORHO_MODES, False, train_norho),
        (RHO_MODES, True, train_rho),
    ):
        specs = [(t, mode) for t in range(trials) for mode in modes]
        X = jnp.stack([x_by_trial[t] for t, _ in specs])
        A = jnp.stack([a_by_trial[t] for t, _ in specs])
        NV = jnp.stack([nv_by_trial[t] for t, _ in specs])
        W = jnp.asarray([mode_weight_row(m, lambda_sun, lambda_rho)
                         for _, m in specs], dtype=jnp.float32)
        params = stack_states([params_by_trial[t] for t, _ in specs])
        opt = tx.init(params)

        arrays = (X, A, NV) if not needs_rho else (
            X, A, NV, jnp.stack([colloc_by_trial[t] for t, _ in specs]))

        t0 = time.time()
        trained, final_loss = trainer(params, opt, *arrays, W)
        jax.block_until_ready(trained)
        timings[modes] = time.time() - t0

        for i, spec in enumerate(specs):
            out[spec] = (trained, i, float(final_loss[i]))

    return out, timings


# ---------------------------------------------------------------------------
# Self-test: same models trained via this driver and via train_model_static
# must land on the same parameters (up to float32 op-reordering noise).
# ---------------------------------------------------------------------------

def self_test():
    from galactoPINNs.train import train_model_static

    print("self-test: tiny problem, 5 modes, vmap ensemble vs train_model_static")
    true_potential = gp.MilkyWayPotential()
    regions = {"sun4": (OBSERVER, 4.0)}
    (x_pool, a_pool, x_pool_phys, cfg, val_scaled, _dist, x_obs, a_obs) = build_data(
        true_potential, 2000, 256, 4.0, regions, include_analytic=False,
    )
    n, epochs, lam = 40, 30, 1.0
    idx = np.random.default_rng(0).choice(x_pool.shape[0], size=n, replace=False)
    x_n, a_n = x_pool[idx], a_pool[idx]
    dx = x_pool_phys[idx] - OBSERVER
    nv = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))
    colloc = sample_colloc_scaled(jr.PRNGKey(1000), 64, 0.5, 25.0,
                                  cfg["x_transformer"])

    template = StaticModel(cfg, rngs=nnx.Rngs(0))
    graphdef, params0, rest = nnx.split(template, nnx.Param, ...)
    tx = optax.adam(1e-3)
    trainers = make_ensemble_fns(graphdef, rest, x_obs, a_obs, tx, epochs)[:2]
    results, _ = train_one_n([x_n], [a_n], [nv], [colloc], [params0],
                             trainers, tx, lam, lam)

    seq_kwargs = {
        "3D": dict(),
        "LOS": dict(n_vecs=nv, line_of_sight=True),
        "LOS+Sun": dict(n_vecs=nv, line_of_sight=True,
                        anchor_x=x_obs, anchor_a=a_obs, lambda_anchor=lam),
        "LOS+rho": dict(n_vecs=nv, line_of_sight=True,
                        colloc_x=colloc, lambda_rho=lam),
        "LOS+Sun+rho": dict(n_vecs=nv, line_of_sight=True,
                            anchor_x=x_obs, anchor_a=a_obs, lambda_anchor=lam,
                            colloc_x=colloc, lambda_rho=lam),
    }
    ok = True
    for mode in MODES:
        model = StaticModel(cfg, rngs=nnx.Rngs(0))
        train_model_static(model, optax.adam(1e-3), x_n, a_n, epochs,
                           log_every=0, **seq_kwargs[mode])
        _, p_seq, _ = nnx.split(model, nnx.Param, ...)
        stacked, i, _ = results[(0, mode)]
        p_vmap = take_member(stacked, i)
        num = sum(float(jnp.sum((a - b) ** 2)) for a, b in
                  zip(jax.tree.leaves(p_seq), jax.tree.leaves(p_vmap)))
        den = sum(float(jnp.sum(a ** 2)) for a in jax.tree.leaves(p_seq))
        rel = (num / den) ** 0.5
        status = "ok" if rel < 1e-4 else "FAIL"
        ok &= rel < 1e-4
        print(f"  {mode:12s} param reldiff vs sequential: {rel:.2e}  {status}")
    print("self-test PASSED" if ok else "self-test FAILED")
    return 0 if ok else 1


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
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_vmap"))
    ap.add_argument("--self-test", action="store_true",
                    help="verify equivalence with train_model_static and exit")
    ap.add_argument("--matmul-precision", default="highest",
                    choices=["highest", "high", "default"],
                    help="'highest' keeps f32 matmuls exact on Ampere+ GPUs "
                         "(TF32 off); 'default' is faster but loosens the "
                         "Hessian-based rho term")
    ap.add_argument("--jax-cache", default=os.environ.get(
        "JAX_COMPILATION_CACHE_DIR"), help="persistent XLA compilation cache dir")
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    if args.jax_cache:
        jax.config.update("jax_compilation_cache_dir", args.jax_cache)
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    if args.self_test:
        raise SystemExit(self_test())

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

    # Per-trial state, shared across all n exactly as in run_experiment.py:
    # init seed = trial (same init for every mode and every n of a trial),
    # collocation set drawn once per trial from PRNGKey(1000 + trial).
    template = StaticModel(cfg, rngs=nnx.Rngs(0))
    graphdef, _, rest = nnx.split(template, nnx.Param, ...)
    params_by_trial = [
        nnx.split(StaticModel(cfg, rngs=nnx.Rngs(t)), nnx.Param, ...)[1]
        for t in range(args.trials)
    ]
    colloc_by_trial = [
        sample_colloc_scaled(jr.PRNGKey(1000 + t), args.n_colloc,
                             args.r_min_colloc, args.r_colloc_max,
                             cfg["x_transformer"])
        for t in range(args.trials)
    ]

    tx = optax.adam(1e-3)
    train_norho, train_rho, eval_group = make_ensemble_fns(
        graphdef, rest, x_obs, a_obs, tx, args.epochs)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")
    npz_path = os.path.join(args.outdir, "per_point_errors.npz")
    perpoint = {f"dist|{name}": dist_sun[name] for name in REGIONS}
    t_start = time.time()

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
            # per-trial subsample + sightlines, same RNG stream as the
            # sequential driver: fresh default_rng(trial) for each (n, trial)
            x_by_trial, a_by_trial, nv_by_trial = [], [], []
            for trial in range(args.trials):
                rng = np.random.default_rng(trial)
                idx = rng.choice(x_pool.shape[0], size=n, replace=False)
                x_by_trial.append(x_pool[idx])
                a_by_trial.append(a_pool[idx])
                dx = x_pool_phys[idx] - OBSERVER
                nv_by_trial.append(
                    jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True)))

            results, timings = train_one_n(
                x_by_trial, a_by_trial, nv_by_trial, colloc_by_trial,
                params_by_trial, (train_norho, train_rho), tx,
                args.lambda_sun, args.lambda_rho)
            for modes, dt in timings.items():
                m_count = len(modes) * args.trials
                print(f"n={n:5d} trained {m_count:2d} models "
                      f"[{'+'.join(modes)}] in {dt:.1f}s", flush=True)

            # evaluate each vmap group once per region (params stay stacked)
            group_ids = {}
            for spec, (stacked, i, _) in results.items():
                group_ids.setdefault(id(stacked), (stacked, []))[1].append(
                    (spec, i))
            evals = {}  # (trial, mode, region) -> (a_pred, neg_frac)
            for stacked, members in group_ids.values():
                for region in REGIONS:
                    x_val, _ = val_scaled[region]
                    x_val_lap = x_val[: min(args.n_val_lap, x_val.shape[0])]
                    apred, neg = eval_group(stacked, x_val, x_val_lap)
                    for (trial, mode), i in members:
                        evals[(trial, mode, region)] = (apred[i], float(neg[i]))

            # emit rows in the sequential driver's order: trial, region, mode
            for trial in range(args.trials):
                for region in REGIONS:
                    x_val, a_val = val_scaled[region]
                    xv_phys, nhat = val_geom[region]
                    for mode in MODES:
                        a_pred, neg = evals[(trial, mode, region)]
                        d = error_decomposition(a_val, a_pred, nhat)
                        c = cylindrical_error_decomposition(
                            a_val, a_pred, jnp.asarray(xv_phys),
                        )
                        err = float(jnp.mean(d["rel_error_magnitude"]))
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
    print(f"total wall time: {time.time() - t_start:.1f}s")
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
