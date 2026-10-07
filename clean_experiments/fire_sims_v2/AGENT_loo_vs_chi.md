# LOO vs χ-stop on the GPU cluster — handoff for an agent

You are running a prepared experiment, not designing one. Everything you need
is in this commit: code, inputs (`loo_inputs/`), a SLURM array script and
setup script (`slurm/`), and the analysis script. **Do not edit `src/`, the
seeds, the loss, or the inputs.** If something fails, stop and report rather
than "fixing" the physics.

Owner: Pablo Mercader-Perez. All paths below are relative to the galactoPINNs
repo root.

## 1. The question

When training the PINN on n noisy pulsar line-of-sight accelerations, when
should training stop?

- **χ-stop** (Morozov discrepancy principle, current default). Stop at the
  first epoch where the whitened training misfit `mean_i |r_i|/σ_i` reaches
  `E|N(0,1)| = sqrt(2/π) = 0.798`. It needs the noise level σ to be right.
  Known weakness on FIRE: small-scale texture acts like extra noise, so χ-stop
  can overshoot. With the NFW baseline at n=500, χ often never crosses within
  1500 epochs and falls back to the last epoch.
- **LOO** (leave-one-out cross-validation). Pick the epoch from the data alone,
  using n fits of n−1 points.
- **IW-LOO** (importance-weighted LOO). LOO measures error where the pulsars
  are, and they cluster near the Sun (r75 ≈ 1.7 kpc). IW-LOO reweights each
  held-out pulsar by `1[r ≤ 2 kpc] / p_pool(x)`, so the CV targets the
  uniform-in-volume 2-kpc bubble that the accuracy metric averages over. This
  is covariate-shift CV.

The deliverable is the final error of each rule on held-out truth, as a
function of n ∈ {20, 50, 100, 150, 200, 250, 500}, over 10 trials (mean ± std).

## 2. Setup (frozen; do not change)

| Setting | Value |
|---|---|
| Model | PINN IV: MLP residual on a fixed MW-tuned NFW baseline, m = 5.4e11 M☉, r_s = 15.62 kpc |
| Loss | Sun-relative LOS L1 with 1/σ weights (hetW catalog noise) + exact Sun-acceleration anchor (λ=1) + ρ≥0 hinge on 512 collocation points (λ=1) |
| Data | 47-pulsar-catalog-like S(r)S(δ) mock pool from FIRE m12i (10 trial pools) |
| Training | full-batch Adam 1e-3, 1500 epochs, `jax_default_matmul_precision=highest` |
| Identical to | `val_split_check.py --baseline nfw --mode LOS+Sun+rho` (results_nfw_LOSSunrho), trials 0–5 |

Per (n, trial), `loo_vs_chi.py`:

1. **FOLDS:** trains n models. Fold k drops pulsar k by giving it weight 0,
   with the other weights renormalized as for the (n−1)-subset. All folds
   share the init, collocation points and Sun anchor. They train in one `vmap`
   (`--fold-chunk` splits the vmap if GPU memory is short; the result is
   unchanged). Recorded every epoch: `z_k(e) = r_k/σ_k`, the signed whitened
   residual on the held-out pulsar.
2. **Rules from the folds:** `loo = argmin_e mean_k|z_k|`; `loo1se` = the
   earliest epoch within one standard error of that minimum; `iwloo` and
   `iwloo1se` are the same with the importance weights.
3. **FULL:** trains one model on all n points, recording every epoch:
   - χ_fit
   - the 2-kpc-bubble acceleration error
   - the Gauss-law ⟨ρ_tot⟩ flux through the 2-kpc Sun sphere
   - the midplane density dex within 2 kpc
   - legacy rec4/rec15

   Every rule reads this same trajectory at its own epoch, so the comparison
   is paired.
4. **Writes:**
   - `results_loo_vs_chi/shards/n{n:04d}_t{trial:02d}.npz` (all curves)
   - `results_loo_vs_chi/checkpoints/<rule>/n{n:04d}_t{trial:02d}.pkl` (params at
     the χ, loo, loo1se, iwloo and iwloo1se epochs)

   An existing shard is skipped, so reruns resume.

Cost is dominated by the folds: Σn = 1,270 fold models per trial, each with a
Hessian over 512 collocation points every step. On CPU (M4 Max) a single
full-trajectory model takes about 40 s per 1500 epochs, so CPU is not viable
for the whole grid. That's why this runs on the GPU.

## 3. Inputs (`loo_inputs/`, force-added; `*.npz` is gitignored repo-wide)

| File | What it is | sha256 (first 16) |
|---|---|---|
| `truth_old1gyr.npz` | FIRE m12i truth cache (training-pool labels, val regions, Sun) = `experiments/fire_los_recovery/cache/truth_old1gyr.npz` | `75c883a227c6cc5e` |
| `pool_count_t10.npz` | 10-trial mock pool, `build_pool.py --n-trials 10`; trials 0–5 bit-identical to the frozen 6-trial `pool_count.npz` (verified) | `e9de5c5d3ec8b388` |
| `bubbles_r2.npz` | 2048 uniform-in-volume truth points in the 2-kpc Sun bubble (keys of `pool_rejection.npz`) | `b923e301034446ee` |
| `midplane_rho_xy.npz` | midplane truth density grid, Sun-centred ±6 kpc, 120² (keys of `fire_sims/cache/nb05_truth_grids.npz`) | `8f9b94524c5215fb` |
| `density_truth_bubbles.json` | particle truth ⟨ρ⟩ (total/baryons/dark) in Sun spheres; used only by the analysis | `390d895769d08e59` |
| `ref_cpu_n20-50_t0-1.json` | CPU reference numbers for the GPU check in step 2 (may arrive in a follow-up commit) | — |

The driver prints every input's sha256 at start-up. They must match this
table.

## 4. Steps

**Step 1: environment** (once, from the repo root, on a node with internet):

```bash
bash clean_experiments/fire_sims_v2/slurm/setup_env.sh
```

This runs `uv sync --frozen`, then installs `jax[cuda12]==0.8.2` into
`.venv`. From then on use **`.venv/bin/python`**, never `uv run`, which
re-syncs and drops the CUDA plugin.

**Step 2: GPU check against the CPU reference** (interactive GPU session or a
short job, about 10–20 min):

```bash
.venv/bin/python -c "import jax; print(jax.devices())"     # must list a CUDA device
.venv/bin/python clean_experiments/fire_sims_v2/loo_vs_chi.py \
    --n-list 20,50 --trials 0-1 \
    --check-ref clean_experiments/fire_sims_v2/loo_inputs/ref_cpu_n20-50_t0-1.json \
    --outdir clean_experiments/fire_sims_v2/results_loo_vs_chi_gpucheck
```

Pass criteria. GPU float32 reductions differ from CPU, so this check is not
bit-exact, and late epochs are expected to separate.

The reason was measured on CPU. A masked LOO fold and the equivalent sliced
fit, which differ only in float summation order:
- agree to 1e-5 through epoch about 750;
- then one discrete jump at a single epoch (an Adam / ρ≥0-hinge "kick") puts
  them about 5–10% apart, after which both wander at the same level.

Late training is sensitive to rounding. That's a property of the training,
not a bug.

So:
- **Must hold:** relative diffs ≲1e-4 at probe epochs 0, 10 and 100, for
  χ_fit, gacc_b2, flux_b2, cv and cvw.
- **Informational only:** epochs 500 and 1499. Differences there of a few
  percent or more are expected.
- **Rule epochs:** an early rule epoch (≲300) should match within a few
  epochs. A late one may differ.

Report what you see. Stop and report only if the early probes disagree
(>1e-3), because that would mean a real environment problem: wrong jax
version, TF32 matmuls, or wrong inputs.

χ-stop on the full trajectory should also reproduce the frozen run (`e_chi`,
rec4 at e_chi):
- The early-crossing rows (all of n=50; n=200 t1–t5; n=500 t2) should match
  e_chi within ±1 and rec4 to ≲1e-3.
- The late rows (n=200 t0 at 1339, and the n=500 rows that never cross) may
  differ for the reason above.

| n | trial: e_chi (rec4) |
|---|---|
| 50 | t0: 94 (0.1623), t1: 16 (0.4792), t2: 96 (0.3060), t3: 21 (0.1978), t4: 28 (0.2110), t5: 23 (0.1846) |
| 200 | t0: 1339 (0.1184), t1: 205 (0.1510), t2: 189 (0.1389), t3: 91 (0.1142), t4: 260 (0.1381), t5: 345 (0.1407) |
| 500 | t0: never (0.1690), t1: never (0.1654), t2: 66 (0.1227), t3: never (0.1563), t4: never (0.1165), t5: 669 (0.1230) |

To check rec4 at e_chi in a shard:

```python
import numpy as np
z = np.load(shard)
cols = list(z["full_cols"])
e = int(z["e_chi"])
print(e, z["full"][e, cols.index("rec4")])
```

Every row printed by the run also shows the per-shard wall-clock
(`folds Xs full Ys`). **Report these timings.** They set the time limit for
step 4.

**Step 3: memory and timing at the largest n** (one task):

```bash
FOLD_CHUNK=0 sbatch --array=60 clean_experiments/fire_sims_v2/slurm/loo_vs_chi.sbatch   # n=500, trial 0
```

- **OOM:** resubmit with `FOLD_CHUNK=128` (then 64).
- **Time limit:** take that task's wall-clock, multiply by about 1.2, and set
  it as `--time` in the sbatch file for the whole array. n=500 is the slowest
  task; smaller n are cheaper.

**Step 4: the full grid.** Array task id → n = `N_LIST[id // 10]`,
trial = `id % 10`, with `N_LIST = (20 50 100 150 200 250 500)`.

```bash
# first edit --partition / --account / --time in slurm/loo_vs_chi.sbatch
sbatch --array=0-69%8 clean_experiments/fire_sims_v2/slurm/loo_vs_chi.sbatch
```

Logs go to `clean_experiments/fire_sims_v2/logs/loo_vs_chi_<job>_<task>.log`.
After the array finishes, check that 70 shards exist:

```bash
ls clean_experiments/fire_sims_v2/results_loo_vs_chi/shards | wc -l   # expect 70
```

Resubmit the same `--array` range for any missing ones; finished shards are
skipped.

**Step 5: analysis** (CPU is fine):

```bash
.venv/bin/python clean_experiments/fire_sims_v2/analyze_loo_vs_chi.py
```

It writes these files to `results_loo_vs_chi/analysis/`:
- `table.md` and `paired.md`
- `rows.csv`
- `fig_error_vs_n.png` (the headline)
- `fig_paired_ratio.png`
- `fig_stop_epochs.png`
- `fig_curves_n{20,50,500}_t0.png`

It also re-derives every rule epoch from the stored curves and asserts that
they match the run.

**Step 6: report back.** Send:
1. The step-2 check output, plus the GPU model and timings.
2. `table.md` and `paired.md`.
3. The four figure types.

Copy results to the laptop with:

```bash
rsync -av <cluster>:<repo>/clean_experiments/fire_sims_v2/results_loo_vs_chi/ \
    clean_experiments/fire_sims_v2/results_loo_vs_chi/
```

Shards plus checkpoints come to about 0.3 GB.

## 5. Reading the results

- **Metrics:** all on the 2-kpc Sun bubble, against FIRE particle truth never
  used in training.
  - `acc`: mean relative acceleration error over 2048 uniform-in-volume
    points, Sun-relative (gauge-free: what the pulsars constrain). The
    absolute version, `acc_abs` in `rows.csv`, also carries the model's Sun
    offset. That offset is large at early stops with the NFW baseline
    (3–4 mm/s/yr at n=20), so don't headline `acc_abs`.
  - `rho`: % error of ⟨ρ_tot⟩ from Gauss's law. The figure shows |error|;
    the table shows the signed error, plus the ρ_DM error after subtracting
    the known baryons, which is about 3× larger.
  - `dex`: median |log10 ρ_pred/ρ_true| over midplane pixels within 2 kpc.
- **`oracle`:** the per-metric argmin of the truth curve. It's a floor
  showing what perfect stopping would give, not a usable rule.
- **`last`:** epoch 1499 (the fixed budget).
- **Prior expectations, to flag if violated:**
  - χ-stop is early at n=50 (epochs 16–96).
  - With the NFW baseline χ rarely crosses at n=500.
  - LOO CV curves are noisy at n=20, so the 1-SE variants exist for this.
  - IW-LOO has a small effective sample (`n_eff` in `table.md`), so expect it
    to be noisier than plain LOO.
- **Possible outcome:** LOO's argmin can land at the last epoch, if the
  held-out misfit is still falling at epoch 1500. That's a result, not a bug.
  Report how often it happens.

## 6. Do not

- Do not edit `src/galactoPINNs`, `datasets.py`, `noise.py`, `mock_catalog/`
  or the inputs.
- Do not change `--epochs`, the λ's or the precision flag.
- Do not delete shards to "clean up". Resubmission relies on them.
- Do not add any co-author or attribution lines to commits.
