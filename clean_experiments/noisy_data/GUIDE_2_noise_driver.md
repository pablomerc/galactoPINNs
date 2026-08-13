# GUIDE 2 — `run_experiment.py` (driver)

Goal: the driver — parent geometry and hyperparameters, a new SETTINGS axis
(noise level × training arm), the (n × trial × setting × mode × region) loop,
CSV + NPZ. Prereqs: GUIDE 0 applied and verified; GUIDE 1's `noisy_datasets.py`
ready (`draw_sigmas_mmsyr`, `apply_los_noise`, `arm_weights`, …).

## The settings axis

One experiment, **nine** settings (LLH deferred). Each answers a specific
question:

| label | family | arm | `los_loss` | question |
|---|---|---|---|---|
| `s0` | σ=0 | plain | l1 | freeze baseline ≡ parent experiment |
| `s0.5` | σ=0.5 | plain | l1 | dose-response: 10th-pct noise |
| `s1.7` | σ=1.7 | plain | l1 | dose-response: median catalog noise |
| `s4.3` | σ=4.3 | plain | l1 | dose-response: 75th-pct noise |
| `het` | catalog | plain | l1 | realistic noise, σ ignored |
| `hetW` | catalog | w ∝ 1/σ, mean-norm | l1 | does whitened L1 rescue it? |
| `hetC` | catalog | w ∝ 1/σ², mean-norm | sq | does the proper χ² likelihood? |
| `s1.7W` | σ=1.7 | w ∝ 1/σ, mean-norm | l1 | whitened L1 at constant σ |
| `s1.7C` | σ=1.7 | w ∝ 1/σ², mean-norm | sq | pure L1-vs-L2, same noise draw |

(`s1.7W` / `s1.7C`'s constant weights mean-normalize to exactly 1 — verified
a no-op — so they isolate the loss *functional form* against plain `s1.7`,
on the same noise draw.)

Design decisions worth internalizing:

- **Arms of one family share the noise realization.** The σ draw is seeded by
  `(trial, n)` and the ε draw by `(trial, n, family)` — so `het`/`hetW`/`hetC`
  train on *identical* noisy data, and `s1.7`/`s1.7W`/`s1.7C` likewise. Arm
  differences are loss differences, never draw luck.
- **Modes are cut to `LOS` and `LOS+Sun+rho`** — the bracketing pair of the
  parent's LOS family. The `3D` reference is trained at `s0` only: our
  injection puts noise *along the sightline*, which is the right model for an
  LOS observable but not for a 3-vector measurement (that would need
  per-component σ from proper motions — a different experiment).
- **The Sun anchor stays noiseless** (σ_sun ≈ 0.13 mm/s/yr ≪ every pulsar σ).
- **Everything the parent seeds, we seed identically** (subsample
  `default_rng(trial)`, collocation `PRNGKey(1000 + trial)`, init
  `seed=trial`) — that's what makes the `s0` freeze check meaningful.

## New metrics: is the fit calibrated to the noise?

Recovery metrics are the parent's (full-vector `rel_err`, LOS/transverse +
cylindrical decompositions). We skip `neg_frac` here — density positivity is
not a question this experiment asks. Two *training-point* diagnostics are new,
computed per trained model on its n sightlines:

- `train_chi` = ⟨|a_los_pred − a_los_obs| / σ⟩. For a fit that matches the
  noise level this is E|N(0,1)| ≈ **0.798**; values well below mean the
  network is memorizing noise. (`nan` at σ=0.)
- `train_err_true_mmsyr` = ⟨|a_los_pred − a_los_TRUE|⟩ in mm/s/yr — how far
  the fit sits from the noiseless truth it never saw. This is the cleanest
  single number for "did the σ-aware arm actually help".

## The code

Full file, in pieces. Docstring + imports + the settings table:

```python
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
```

`train_one` grows the two new kwargs — everything else about training is
untouched:

```python
def train_one(cfg, x_n, a_n, epochs, seed, weights=None, los_loss="l1",
              **loss_kwargs):
    model = StaticModel(cfg, rngs=nnx.Rngs(seed))
    train_model_static(
        model, optax.adam(1e-3), x_n, a_n, epochs, log_every=0,
        importance_weight=weights, los_loss=los_loss, **loss_kwargs,
    )
    return model
```

Argparse = the parent's plus `--settings`, with the default n-grid trimmed to
`50,100,200,500` (the noisy questions live at low n; extend it if you want
the n=1000 point):

```python
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
```

Setup (build_data, the mm/s/yr conversion, region geometry saved once to the
NPZ) is the parent's with one addition:

```python
    fac = sigma_scale_factor(cfg)  # scaled units per mm/s/yr
    print(f"1 mm/s/yr = {fac:.5f} scaled units")
```

The CSV header:

```python
        w.writerow(["n_samples", "trial", "setting", "sigma_mmsyr", "arm",
                    "mode", "region", "rel_err",
                    "rel_los", "rel_trv", "rel_R", "rel_z", "rel_phi",
                    "train_chi", "train_err_true_mmsyr"])
```

The heart — inside the parent's `for n / for trial` loop (subsample,
sightlines, collocation all identical), the per-setting block:

```python
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
```

The region-evaluation block is the parent's, with three deltas: NPZ keys gain
the setting (`err|{region}|{n}|{trial}|{label}|{mode}`, same for `apred|`),
the CSV row gains `label, median(sig_i), arm, chi, err_true`, and a
`region_err` dict feeds the progress print (so it reports sun4, not whichever
region the loop ended on):

```python
                        region_err = {}
                        for region in REGIONS:
                            # ... parent's eval: error_decomposition,
                            # cylindrical_error_decomposition
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
```

(`f.flush()` after every model: a noisy grid is long, and a killed run should
leave a usable partial CSV.)

## Running

Wiring check (~2 min):

```bash
uv run python clean_experiments/noisy_data/run_experiment.py \
    --n-grid 50 --trials 1 --epochs 30 --pool 2000 --n-val 512 --outdir /tmp/noisy_smoke
```

Expect **57 data rows** (9 settings × 2 modes × 3 regions + 3 for the 3D
reference at s0), no NaNs in the metric columns, `train_chi = nan` only for
`s0`, and the header line `1 mm/s/yr = 0.07914 scaled units`.

## Freeze check — the σ=0 arm IS the parent

After a real run (or a targeted `--settings s0` run at the defaults), the s0
rows must match the parent's `results.csv` **bit-exactly** for every
overlapping `(n, trial, mode, region)`:

```bash
uv run python - <<'EOF'
import csv
parent = {(r['n_samples'], r['trial'], r['mode'], r['region']): r['rel_err']
          for r in csv.DictReader(open('clean_experiments/sun_los_recovery/results.csv'))}
new = [r for r in csv.DictReader(open('clean_experiments/noisy_data/results.csv'))
       if r['setting'] == 's0']
match = sum(abs(float(parent[k]) - float(r['rel_err'])) < 1e-15
            for r in new if (k := (r['n_samples'], r['trial'], r['mode'], r['region'])) in parent)
total = sum((r['n_samples'], r['trial'], r['mode'], r['region']) in parent for r in new)
print(f"{match}/{total} overlapping s0 rows bit-exact")
EOF
```

Anything less than 100% means a seed or code-path drifted — stop and diff.

## The real run

```bash
uv run python clean_experiments/noisy_data/run_experiment.py
```

4 n-values × 3 trials × (9 settings × 2 modes + 1) = **228 models** at 1500
epochs — expect several hours on CPU (the `+rho` mode pays a Hessian per
collocation point per step, exactly like the parent).

## What to expect (answer key)

**Verified 2026-07-20** with the full 1500-epoch config at
`--n-grid 50,100 --trials 1` (pool 30000, n_val 4096), originally including
`hetLLH` (dropped here). Re-running that exact command must reproduce every
number below exactly for the overlapping settings (all seeds fixed), and the
freeze check above must report **18/18 overlapping s0 rows bit-exact** — it
did.

Trial-0 numbers (`rel_err` on sun4 / sun15; `chi`, `errTrue` from the sun4
rows — they're train diagnostics, identical across regions):

| setting | mode | n=50: sun4 / sun15, chi, errTrue | n=100: sun4 / sun15, chi, errTrue |
|---|---|---|---|
| s0 | 3D | 0.0405 / 0.5342, –, 0.063 | 0.0240 / 0.2884, –, 0.074 |
| s0 | LOS | 0.0873 / 0.3320, –, 0.080 | 0.0653 / 0.2835, –, 0.064 |
| s0 | LOS+Sun+rho | 0.0881 / 0.3447, –, 0.096 | 0.0547 / 0.3084, –, 0.052 |
| s0.5 | LOS | 0.1602 / 1.5123, 0.458, 0.395 | 0.1333 / 0.9587, 0.632, 0.332 |
| s0.5 | LOS+Sun+rho | 0.1373 / 0.8860, 0.546, 0.353 | 0.1264 / 0.9421, 0.628, 0.279 |
| s1.7 | LOS | 0.5761 / 5.1805, 0.279, 1.014 | 0.4511 / 2.8093, 0.488, 1.021 |
| s1.7 | LOS+Sun+rho | 0.5820 / 5.4701, 0.211, 1.056 | 0.4184 / 3.7300, 0.485, 0.978 |
| s4.3 | LOS | 1.6220 / 2.0655, 0.264, 3.125 | 2.3340 / 27.6215, 0.290, 3.517 |
| s4.3 | LOS+Sun+rho | 0.8244 / 8.0017, 0.342, 2.728 | 1.2800 / 14.7069, 0.482, 2.905 |
| het | LOS | 1.1704 / 4.5163, 0.307, 1.365 | 0.6175 / 3.6500, 0.644, 0.872 |
| het | LOS+Sun+rho | 1.0271 / 8.7922, 0.615, 1.355 | 0.2195 / 2.1518, 0.721, 0.669 |
| hetW | LOS | 0.3036 / 2.7763, 0.381, 0.890 | 0.1991 / 0.4831, 0.670, 0.525 |
| hetW | LOS+Sun+rho | 0.2851 / 2.2275, 0.446, 0.867 | 0.1667 / 0.4458, 0.684, 0.464 |
| hetC | LOS | 0.3730 / 3.4039, 0.349, 0.982 | 0.1725 / 0.7239, 0.677, 0.371 |
| hetC | LOS+Sun+rho | 0.1856 / 0.9162, 0.863, 0.446 | 0.1473 / 0.5365, 0.686, 0.341 |
| s1.7C | LOS | 0.7109 / 4.1180, 0.094, 1.116 | 0.7663 / 5.2229, 0.392, 1.174 |
| s1.7C | LOS+Sun+rho | 0.6415 / 6.8026, 0.169, 1.116 | 0.3269 / 2.4889, 0.643, 0.872 |

For the full 3-trial run, the private quick-scan medians (n = 50/100/200,
sun4, same seeds for the overlapping settings) are the qualitative answer key:

- **Dose-response** (LOS, plain): 0.082/0.056/0.050 at σ=0 →
  0.16/0.13/0.09 at σ=0.5 → 0.58/0.43/0.31 at σ=1.7 → ~1.3/1.5/0.93 at σ=4.3.
  Median catalog noise costs roughly **6× in recovery error**; 75th-pct noise
  destroys the fit outright.
- **The σ-aware arms rescue the catalog regime**: het 0.94/0.83/0.43 →
  hetW 0.32/0.21/0.13 — a **~3× error reduction**, recovering to within ~2.5×
  of the noiseless baseline by n=200.
- **hetC (χ²)** is competitive with hetW and tends to win with the anchor+rho
  add-ons at n≥100 (see the trial-0 table: best noisy sun4 numbers in the
  column) — but read the prior-upweight caveat below before crediting the
  Gaussian likelihood. At n=50 the *unweighted* sq loss (s1.7C, chi≈0.09)
  badly overfits noise — L1's robustness is real, and the 0.798 calibration
  line makes it visible.
- **Everything still underfits its σ** (chi < 0.798 across the board): 1500
  fixed epochs memorize some noise at every level. That's a deliberately open
  question (early stopping / epochs-under-noise), noted for a follow-up.
- **Interpretation caveat for the weighted `+Sun+rho` arms** (caught by the
  quick-scan's adversarial review): mean-normalizing the weights preserves
  the data-vs-prior balance only at *initialization*. Once the fit converges
  to the noise level (|r_i| ∝ σ_i), the whitened data term is smaller than
  the unweighted one by ≈ mean(σ)·mean(1/σ) = **3.4×** for hetW and
  ≈ mean(σ²)·mean(1/σ²) = **80×** for hetC — so λ_anchor/λ_rho are
  effectively upweighted by those factors. Part of the weighted `+Sun+rho`
  wins (including hetC's calibrated chi = 0.863 at n=50) is *stronger
  priors*, not the likelihood alone. The plain-`LOS` columns are the clean
  weighting comparison.
