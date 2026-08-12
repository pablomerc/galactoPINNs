# clean_experiments

Hand-written, shareable experiments (unlike `experiments/`, which is gitignored and
stays local). Everything here is typed and understood line-by-line; the physics and
design rationale live in the `GUIDE_*.md` files next to the code.

> The guides are working documents written while building the experiment. They double
> as documentation — keep or prune them before pushing, as you prefer.

## sun_los_recovery — the experiment

**Question.** With a realistic *pulsar-like* observing geometry — a few dozen to a few
hundred line-of-sight (LOS) accelerations measured from inside a small volume around
the Sun — how well can a PINN recover the Galactic acceleration field, and how much do
two physics add-ons buy us?

**Geometry.** Training points are density-weighted samples (stars live where mass is)
in a ball of radius 4 kpc around the Sun at (-8.1, 0, 0) kpc galactocentric. Truth is
galax's `MilkyWayPotential`.

**Eval regions** (one training geometry, three evaluation volumes — the same trained
model is evaluated on all three, so cross-region comparisons use identical models):

| region  | volume                      | probes                          |
|---------|-----------------------------|---------------------------------|
| `sun4`  | r < 4 kpc of the Sun        | interpolation (training region) |
| `sun15` | r < 15 kpc of the Sun       | local extrapolation             |
| `gc15`  | r < 15 kpc of the GC        | extrapolation across the inner galaxy (up to ~23 kpc from the Sun) |

**Modes** (all five per (n, trial), identical inits per seed):

| mode          | loss                                                       |
|---------------|------------------------------------------------------------|
| `3D`          | full 3-vector accelerations (upper bound / reference)      |
| `LOS`         | scalar projections a·n̂ only (observer = Sun)              |
| `LOS+Sun`     | LOS + the Sun's full 3D acceleration as one anchor point   |
| `LOS+rho`     | LOS + mass-positivity hinge mean(relu(-∇²Φ)) on collocation points |
| `LOS+Sun+rho` | both add-ons                                               |

**Why the collocation points are GC-centered** (uniform in volume, shell 0.5–25 kpc,
*not* density-weighted): the positivity prior is valid everywhere, violations live in
the low-density outskirts, and the interesting question is whether the prior
disciplines the field in the large unconstrained volume *outside* the training ball.
25 kpc covers all three eval regions.

## What already lives in `src/` (done, verified 2026-07-17)

The loss mechanisms were added to `galactoPINNs.train.train_step_static` /
`train_model_static` as optional, default-off parameters — so this experiment needs
**no custom training code**, just kwargs:

| parameter | meaning | design lesson |
|---|---|---|
| `anchor_x`, `anchor_a` `(M,3)` | points with known full 3D acceleration (the Sun) | same units as the LOS residuals → folded into **one shared mean** with the n sightline terms; a separate `mean()` would weight the single anchor ~n× per point |
| `lambda_anchor` | datapoint-equivalents per anchor component | default 1.0 |
| `colloc_x` `(K,3)` | collocation points for the ρ≥0 prior | different units from the data term → **separate weighted term**, `+ lambda_rho * mean(relu(-lap))` |
| `lambda_rho` | hinge weight vs the data loss | plateau ≈ [1, 3]; too large biases ∂²Φ/∂z² near the disk → a_z bias |

ρ≥0 as a PINN loss is prior art: cite Green & Ting, arXiv:2205.02244.

## Files

- `sun_los_recovery/datasets.py` — samplers + truth + scaling (build with `GUIDE_1_datasets.md`)
- `sun_los_recovery/run_experiment.py` — config, 5 modes × n-grid × trials, CSV (build with `GUIDE_2_driver.md`)
- `sun_los_recovery/plots.py` — figures from results.csv / per_point_errors.npz (`GUIDE_3_plots.md`)

## Status checklist

- [x] `train.py`: anchor mechanism (verified bit-exact vs baselines)
- [x] `train.py`: positivity mechanism (verified bit-exact vs baselines)
- [x] Laplacian notebook (`notebooks/laplacian.ipynb`)
- [ ] `datasets.py` piece A — density fn + ball sampler
- [ ] `datasets.py` piece B — targets, scaling, colloc, anchor
- [ ] `run_experiment.py`
- [ ] `plots.py`
- [ ] full run + results

Note: the repo gitignore excludes `*.npy`, `*.npz`, `data/` — datasets are always
regenerated from seeds; only code, CSV and PNGs are tracked.
