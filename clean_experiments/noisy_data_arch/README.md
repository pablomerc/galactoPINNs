# noisy_data_arch — PINN III vs IV vs V on noisy LOS accelerations

A focused **architecture** follow-up to [`../noisy_data`](../noisy_data). Same
noisy line-of-sight problem near the Sun, but instead of sweeping noise levels
and training arms, we hold the noise/arm/region fixed and compare **three rungs
of the PINN ladder** (paper §2.4–2.6):

| variant | what it adds | `StaticModel` config | the NN learns… |
|---|---|---|---|
| **PINN III** | (II) + spatial scaling | `scale="nfw"`, `include_analytic=False` | the whole scaled potential |
| **PINN IV** | (III) + **fixed** analytic baseline | `include_analytic=True`, fixed `NFWPotential`, `trainable=False` | the **residual** Φ − Φ_NFW |
| **PINN V** | (IV) + **trainable** baseline | `trainable=True`, `TrainableGalaxPotential(NFW, ("m","r_s"))` | residual **and** the baseline's `(m, r_s)`, jointly |

**Scientific question.** Given only ~50–500 noisy line-of-sight accelerations
in the 4 kpc ball around the Sun, does adding (and then *learning*) an analytic
NFW baseline actually improve recovery, or does the NN + radial scaling already
do the job? PINN V additionally reports the halo `(m, r_s)` it infers.

## Fixed choices (the axes v1 swept, pinned here)

- **Region:** `sun4` only — the 4 kpc Sun ball (also the training ball).
- **Noise family:** `het` — per-point σ resampled with replacement from the 51
  real Donlon+2025 catalog σ values (mm/s/yr), injected as absolute Gaussian
  noise along each sightline, `a → a + ε n̂, ε ~ N(0, σ)`.
- **Arm:** `W` — importance weight `w ∝ 1/σ` (mean-normalized), L1 loss.
  ("Whitened L1" / Laplace-LLH: downweights noisy pulsars.)

## The two modes

- **`LOS`** — line-of-sight loss only: `w·|a_pred·n̂ − a_obs·n̂|`. Mimics pulsar
  timing (only the sightline projection is observed).
- **`LOS+rho`** — LOS loss **+ a ρ≥0 hinge** `λ_ρ·mean(relu(−∇²Φ))` at
  collocation points (uniform-in-volume shell, 0.5–25 kpc, GC-centered).
  Physics prior: density `ρ ∝ ∇²Φ` can't be negative. **No Sun anchor** (unlike
  `noisy_data`'s `LOS+Sun+rho`), so this isolates the positivity prior.

## Two design points worth knowing

1. **Scaling differs per architecture.** `include_analytic=True` makes
   `scale_data` fit `u_star` from the *residual* `u − u_NFW`, which changes
   `a_star` — so **IV/V get a different `a_transformer` (and `fac`) than III**
   (≈ 0.079 vs ≈ 0.249 scaled-units-per-mm/s/yr). The comparison stays clean
   because (a) recovery error is *relative* (scale-free) and (b) the driver
   draws the noise **once in physical units** (shared `n̂`, σ, RNG across all
   three arches) and converts to each arch's scaled units via its own
   `sigma_scale_factor`. The underlying noisy physical data is identical across
   arches; only the internal units differ.

2. **The ρ≥0 hinge sees the baseline too (IV/V).** With `include_analytic=True`,
   `compute_laplacian` traces **NN + baseline**. The NFW baseline is ρ≥0 by
   construction, so IV/V start far "cleaner" than III even under plain `LOS`
   (empirically ~4–8% negative-density collocation points vs III's ~70%). The
   hinge therefore acts mostly on the NN part for IV/V — a subtly different
   constraint than in III. Not a bug; just interpret accordingly.

## PINN V training (fully joint)

V's `(m, r_s)` are `nnx.Param`s inside the `TrainableGalaxPotential` layer, so
the default `train_model_static` optimizer (`wrt=nnx.Param`) trains them
**jointly** with the NN weights — one optimizer, one loss, no staged schedule.
(The paper's alternate/staged scheme is better for *interpretable* parameter
inference; joint is the right call for a field-recovery comparison.) Use
`--v-misspec F` to multiply V's initial baseline mass by `F` and test its
ability to correct a wrong prior.

## How to run

From this directory, with the repo venv:

```bash
# full defaults: n = 50,100,200,500 · 3 trials · 1500 epochs · pool 30000
.venv/bin/python run_experiment.py

# a quick look first (tiny; ~30 s on CPU)
.venv/bin/python run_experiment.py --n-grid 50,100 --trials 1 --epochs 200 \
    --pool 5000 --n-val 512 --outdir ./quick

# figures
.venv/bin/python plots.py                    # reads ./results.csv by default
```

Useful flags: `--arches III,IV,V`, `--modes LOS,LOS+rho`, `--lambda-rho`,
`--n-colloc`, `--v-misspec`, `--no-checkpoints`, `--outdir`.
Full-default runtime is roughly **15–40 min wall on a laptop CPU** (IV/V cost
more than III: galax baseline eval + Hessian); much faster on a GPU.

## Outputs

- **`results.csv`** — one row per `(n, trial, arch, mode)` on `sun4`:
  `rel_err`, LOS/transverse decomposition (`rel_los`, `rel_trv`), cylindrical
  decomposition (`rel_R`, `rel_z`, `rel_phi`), `train_chi` (⟨|resid|/σ⟩ on the
  training sightlines; ≈ 0.80 = matched to noise, ≪ = fitting noise),
  `train_err_true_mmsyr`, and `learned_m` / `learned_rs` (PINN V only).
- **`per_point_errors.npz`** — per-val-point `err|<key>` and predicted
  accelerations `apred|<key>`, keyed `"{n}|{trial}|{arch}|{mode}"`.
- **`checkpoints/PINN{arch}_{mode}_n{n}_t{trial}/bundle.pkl`** — a self-contained
  reload bundle per model, plus `checkpoints/manifest.csv`.
- **`recovery_vs_n.png`**, **`chi_vs_n.png`** — from `plots.py`.

## Evaluating a saved model later

```python
from checkpoints import load_model
from galactoPINNs.model_potential import make_galax_potential

model, cfg, meta = load_model("checkpoints/PINNV_LOS+rho_n500_t0/bundle.pkl")
pot = make_galax_potential(model)      # a galax AbstractPotential
# pot.acceleration(...) / pot.potential(...) in physical units, integrate orbits, etc.
```

Reload reproduces the trained model's predictions bit-exactly (verified). The
bundle stores the arch tag, the scaling `cfg` (transformers), the trained Param
state, and — for IV/V — the baseline scalars needed to rebuild the galax NFW
(the galax object itself is stripped, not pickled, then reconstructed on load;
for V it is rebuilt at the *learned* `(m, r_s)`).

## Files

| file | role |
|---|---|
| `arch_datasets.py` | per-arch `build_arch` + `make_model`; re-exports the noise machinery |
| `run_experiment.py` | sequential driver (loops arch → n → trial → mode) |
| `checkpoints.py` | `save_model` / `load_model` reload bundles |
| `plots.py` | `recovery_vs_n.png`, `chi_vs_n.png` (csv + numpy, no pandas) |

Reuses [`../sun_los_recovery/datasets.py`](../sun_los_recovery/datasets.py)
(`build_data`, geometry, scaling, collocation) and
[`../noisy_data/noisy_datasets.py`](../noisy_data/noisy_datasets.py) (catalog σ,
noise injection, arm weights, `sigma_scale_factor`) verbatim — the sibling paths
are located automatically by `arch_datasets._find_clean()`.
