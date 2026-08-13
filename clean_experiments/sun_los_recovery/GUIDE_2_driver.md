# GUIDE 2 — `run_experiment.py`

Goal: the driver — config, the five training modes, the (n × trial × mode × region)
loop, and results written to CSV + NPZ. Prereq: `datasets.py` finished and its
self-test matching (GUIDE 1).

## The payoff of the src work

The old (private) version of this experiment needed **three custom `@nnx.jit` train
steps plus three wrapper loops (~120 lines)** because the anchor and the positivity
hinge weren't part of the package. Your `train.py` parameters collapse all of it into
one function — every mode is now *just kwargs*:

```python
def train_one(cfg, x_n, a_n, epochs, seed, **loss_kwargs):
    model = StaticModel(cfg, rngs=nnx.Rngs(seed))
    train_model_static(
        model, optax.adam(1e-3), x_n, a_n, epochs, log_every=0, **loss_kwargs,
    )
    return model
```

and the five modes are:

```python
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
```

Same `seed=trial` for every mode → **identical network initializations**, so mode
differences are loss differences, not init luck.

## Skeleton

```
imports (argparse, csv, os, numpy, jax, jnp, jr, optax, nnx,
         galax.potential as gp, StaticModel, train_model_static,
         and from datasets: OBSERVER, GC, build_data, sample_colloc_scaled)

MODES = ("3D", "LOS", "LOS+Sun", "LOS+rho", "LOS+Sun+rho")
REGIONS = ("sun4", "sun15", "gc15")

metrics: rel_accel_errors_per_point, rho_neg_fraction
train_one (above)
main(): argparse -> build_data -> loop -> CSV/NPZ
```

## Metrics

The headline metric is the **relative error of the full 3D acceleration field** —
even the LOS-trained models are judged on the complete vector field, that's the whole
point. The secondary diagnostic is the fraction of eval points with negative implied
density (does the hinge actually fix the pathology it targets?).

```python
def rel_accel_errors_per_point(model, x, a_true):
    """Per-point relative error of the predicted FULL acceleration field."""
    a_pred = model(x)["acceleration"]
    num = jnp.linalg.norm(a_pred - a_true, axis=1)
    den = jnp.linalg.norm(a_true, axis=1) + 1e-10
    return np.asarray(num / den)


def rho_neg_fraction(model, x_scaled):
    """Fraction of points with predicted rho < 0 (rho has the sign of the scaled lap)."""
    lap = model.compute_laplacian(x_scaled)
    return float(jnp.mean(lap < 0.0))
```

`rho_neg_fraction` costs a Hessian per point — cap the points it sees
(`--n-val-lap 512`), don't run it on the full val set.

## Argparse (defaults = the calibrated values)

```python
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
```

Notes: pool ≫ max(n_grid) so per-trial subsamples don't overlap much; λ values sit in
the calibrated plateaus (λ_rho ∈ [1,3]; larger risks the a_z bias); the collocation
shell [0.5, 25] kpc covers all three eval regions (sun15 reaches 23.1 kpc from GC).

## Main loop

```python
def main():
    args = ...                      # argparse block above
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

    csv_path = os.path.join(args.outdir, "results.csv")
    npz_path = os.path.join(args.outdir, "per_point_errors.npz")
    perpoint = {f"dist|{name}": dist_sun[name] for name in REGIONS}

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "trial", "mode", "region", "rel_err", "neg_frac"])

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

                models = { ... }     # the five train_one calls (top of guide)

                for region in REGIONS:
                    x_val, a_val = val_scaled[region]
                    x_val_lap = x_val[: min(args.n_val_lap, x_val.shape[0])]
                    for mode in MODES:
                        errs = rel_accel_errors_per_point(models[mode], x_val, a_val)
                        err = float(np.mean(errs))
                        neg = rho_neg_fraction(models[mode], x_val_lap)
                        perpoint[f"err|{region}|{n}|{trial}|{mode}"] = errs.astype(np.float32)
                        w.writerow([n, trial, mode, region, err, neg])
                        print(f"n={n:5d} trial={trial} {region:6s} {mode:12s} "
                              f"rel_err={err:.4f} neg_frac={neg:.3f}", flush=True)

    np.savez_compressed(npz_path, **perpoint)
```

Design points worth internalizing:

- **Pool-then-subsample**: the pool (and the scaling!) is built once; each (n, trial)
  draws a subset. Models across n and trials therefore live in identical scaled
  units — comparisons are honest.
- **`n_vecs` from physical positions**: unit directions are invariant under the
  isotropic scaling, so they multiply scaled accelerations legitimately.
- **Per-point errors saved to NPZ** keyed `err|region|n|trial|mode`, with
  `dist|region` for the distance-binned plots (GUIDE 3). NPZ is gitignored — it's
  regenerable output, only results.csv and PNGs are tracked.

## Running

Wiring check first (~minutes):

```bash
uv run python clean_experiments/sun_los_recovery/run_experiment.py \
    --n-grid 50,100 --trials 1 --epochs 30 --pool 2000 --n-val 512
```

Sanity on the quick run: rel_err large everywhere (30 epochs is nothing) but *finite*,
all 5 modes × 3 regions × 2 n's × 1 trial = 30 CSV rows, no NaNs. Then the real run
(hours — the `+rho` modes pay a Hessian per collocation point per step):

```bash
uv run python clean_experiments/sun_los_recovery/run_experiment.py
```

## What to expect (from the old private runs, your answer key)

- `3D` best in-region (`sun4`); LOS ~4× its error at the same n.
- `+Sun` helps LOS at low n (≲ a few hundred), fades by n≈1000.
- `+rho` is the big lever at low n and **especially outside the training ball**
  (`sun15`, `gc15`), where extrapolation otherwise plateaus at ~20–30% error;
  `neg_frac` should drop toward ~0 for the `+rho` modes.
- `LOS+Sun+rho` ≈ best LOS-only variant at pulsar-like n.

If your clean rerun reproduces these qualitative rankings, the reimplementation is
validated end-to-end.
