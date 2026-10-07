# mock_catalog: catalog-like mock pulsars from FIRE m12i

This folder turns FIRE m12i star particles into mock pulsar catalogs whose positions, noise
and observable follow the real Donlon+2025 catalog. It is the single source of truth for
the mock population: the training experiments draw from pools built here, and the
selection-function figures of the paper are made here. Nothing in `src/galactoPINNs`
depends on it.

| file | what it does |
|---|---|
| `donlon2025_catalog.csv` | copy of `Linas-group/data/pulsars/data.csv` (53 entries; header + units row) |
| `catalog.py` | the catalog **sample**: exclusions, channel choice, r_s, sigma pool, program per pulsar |
| `selection.py` | S(x) = S_r(r_sun) S_delta(delta): radial and on-sky selection, sky statistics |
| `candidates.py` | old star particles near the observer, minus the frozen validation particles |
| `build_pool.py` | per-trial rejection sampling + pytreegrav truth labels -> `cache/pool_<sky>.npz` |
| `draw.py` | one trial's n mock pulsars: Sun-relative LOS accelerations with catalog noise |
| `make_figures.py` | the paper's selection-function figures + `figs/selection_stats.json` |
| `rs_gamma_fit.ipynb` | derivation of r_s: Gamma(3, r_s) fit, alpha and truncation checks, m12i-density check |

## 1. The catalog sample (`catalog.py`)

One definition feeds every mock ingredient, so r_s, the noise and S_delta cannot drift apart.

- **Duplicate** (the `EXCLUDE` dict): `J0737-3039B`, the second pulsar of the double
  pulsar. Its row is identical to J0737-3039A (same binary, same acceleration), so it is
  one measurement, not two.
- **Fractional-distance-error cut** (`MAX_FRAC_DIST_ERR = 1.0`): drops pulsars with
  `DIST_ERR / DIST > 1`. `DIST = 1/parallax` for every entry, so the ratio is the inverse
  parallax significance, and the cut removes the pulsars whose parallax is below 1 sigma,
  i.e. whose distance is not measured:

  | pulsar | distance [kpc] | DIST_ERR / DIST |
  |---|---|---|
  | J1946+3417 | 2.17 ± 7.09 | 3.26 |
  | J2302+4442 | 2.50 ± 5.62 | 2.25 |
  | J1721-2457 | 10.0 ± 20.0 | 2.00 |
  | J2322-2650 | 0.77 ± 1.18 | 1.54 |
  | J2229+2643 | 5.00 ± 7.50 | 1.50 |

  J0931-1902 sits at exactly 1.00 and is kept. This is a cut on distance *quality*, not
  distance: Donlon+2025 cut at 3 kpc for their fits ("the 48 out of 53 pulsars ...
  within 3 kpc"), a validity range for their parametric models, which keeps J1946+3417
  and J2302+4442 and drops well-measured distant pulsars such as B1913+16 (4.2 kpc).
- `LEGACY_SAMPLE` is the 51-pulsar sample of the 2026-09 pipelines (J1721-2457 dropped by
  name, no ratio cut), kept for regression checks.
- **Channel**: binary orbital decay `ALOS_PB` where present, else spin-down `ALOS_PS`.
- **Program** (for S_delta): the ATNF reference tag of the acceleration channel,
  `../paper_figures/sky_coverage/donlon52_refs.csv` mapped by `programs.py` there.

Sample: **47 pulsars** (26 binary, 21 spin-down), r_s = mean distance / 3 =
**0.474 ± 0.040 kpc** (see `rs_gamma_fit.ipynb`: alpha = 3 is consistent with a free fit,
p = 0.3; the m12i old-star density instead of the r² assumption gives 0.52 ± 0.05 kpc), sigma(a_LOS) median 1.51 mm/s/yr (p10/p75 0.48 / 3.16), program counts
NANOGrav 15 / EPTA 13 / PPTA 10 / Other 9.

`uv run python clean_experiments/mock_catalog/catalog.py` prints the sample and checks
the loader against the earlier pipelines: the legacy sample's r_s equals the S(r) pool's
value to all digits, and the sigma pool equals the old hard-coded list (frozen in
`experiments/fire_los_recovery/noise.py`) plus J0125-2327 and J1400-1431 (the 3-sigma
outliers it had dropped) minus exactly the pulsars this sample excludes.

## 2. The selection function (`selection.py`)

Target distribution of mock pulsar positions:

    p(x) ∝ n_*(x) · S_r(r_sun) · S_delta(delta)

- **n_*(x)**: star particles older than 1 Gyr (MSPs are old), within 5 kpc of the observer
  at (-8.1, 0, 0) kpc (`candidates.py`, ~575k particles). The 4,534 frozen validation
  particles (sun4 / sun15 / gc15 of the truth cache) are removed, so a training pulsar is
  never an evaluation point.
- **S_r(r) = exp(-r / r_s)**: for a locally uniform tracer density the distances follow
  r² exp(-r / r_s), a Gamma(3, r_s) law with mean 3 r_s, hence r_s = mean(d) / 3.
- **S_delta(delta) ∝ Σ_k w_k 1_k(delta)**, normalised to max 1: a mixture of the
  declination bands the timing programs can reach (Arecibo -1..+38, Green Bank > -46
  outside the Arecibo strip, EPTA > -39, PPTA < +27, Other = all sky), with **count
  weights** w_k = N_k / N_tot (the default; `--sky rate` gives w_k = N_k / Omega_k).
  With the 47-pulsar sample (N_k = Arecibo 8, Green Bank 7, EPTA 13, PPTA 10, Other 9),
  S_delta per band (-90, -46, -39, -1, 27, 38, 90) is 0.48 / 0.65 / 0.98 / 1.00 / 0.75 / 0.73.
- **Mock geometry**: a star's (l, b) is its direction from the observer in the simulation
  frame (x toward the centre, z along the disk axis), and delta follows from (l, b) by the
  J2000 rotation, as if m12i were the Milky Way seen from Earth. The frame fixes the disk
  plane but not the sense of rotation or which side is north; `--flip {l,b,lb}` mirrors
  the mock sky to test that choice.

`uv run python clean_experiments/mock_catalog/selection.py` prints the S_delta table and
checks that, with the same 52 pulsars, it equals
`../paper_figures/sky_selection/selection.py` to machine precision for both weightings.

## 3. Building a pool (`build_pool.py`)

One pool per trial t = 0..5: candidate i is kept iff `u_i < S_r(r_i) S_delta(delta_i)`,
with `u = default_rng(t).uniform(size=N_candidates)`. Both factors peak at 1, so the kept
set is an exact unweighted sample of the target. Because u is shared, the joint pool of a
trial is nested inside its S_r-only pool.

Truth labels are a pytreegrav tree summation over all 147M particles with the frozen
softenings and theta, then the frozen frame corrections of the truth cache
(`a -> a - a_frame`, `phi -> phi + a_frame·x - u0`). Before writing, a gate re-sums the
first 256 frozen pool particles and the observer and requires agreement to 1e-6.

```bash
cd /path/to/galactoPINNs
# 1. selection only (seconds): per-trial pool sizes and sky statistics vs the catalog
uv run python clean_experiments/mock_catalog/build_pool.py --dry-run
# 2. the pool for the experiments: S_r S_delta, count weights -> cache/pool_count.npz
uv run python clean_experiments/mock_catalog/build_pool.py
# 3. check a draw
uv run python clean_experiments/mock_catalog/draw.py clean_experiments/mock_catalog/cache/pool_count.npz
```

Step 2 takes about 2.5 minutes on the laptop, almost all of it the tree summation. It
reads `experiments/fire_los_recovery/cache/{particles.npz, truth_old1gyr.npz}`
(override with `--particles`, `--truth`) and needs `pytreegrav==1.1.4`; other versions
change the tree field at ~1e-3 and the gate refuses to write. `cache/` is gitignored
(`*.npz`), so pools are rebuilt from these commands, not committed.

Pool file keys:

| key | content |
|---|---|
| `x_union`, `a_union`, `u_union` | position [kpc], acceleration [kpc/Myr²], potential [kpc²/Myr²] of every particle in any trial's pool |
| `r_union`, `l_union`, `b_union`, `dec_union`, `p_union` | heliocentric distance, sky position, acceptance probability |
| `cand_index_union` | index into the star block of `particles.npz` |
| `idx\|t` | trial t's pool, as rows of the union arrays |
| `x_obs`, `a_obs`, `u_obs` | the observer (Sun) and its true field |
| `sigma_pool_mmsyr` | the catalog sigma(a_LOS) values the noise resamples |
| `catalog_names`, `meta` | the catalog sample; JSON with settings, exclusions, N_k, S_delta, per-trial statistics, gate |

Other options: `--sky {count,rate,none}`, `--flip`, `--n-trials`, `--r-cand`,
`--age-min`, `--out`, `--check-against OLD.npz` (require identical indices and labels),
`--legacy-sample` (the 51-pulsar sample, for that check).

With the 47-pulsar sample each trial's pool holds ~3,100-3,200 stars (union of the six:
14,681), with median / p75 distance 1.14 / 1.67 kpc (catalog 1.17 / 1.67). Every old star
particle within 5 kpc is already a candidate, so this is the ceiling for one realization:
an n = 2000 catalog uses ~60% of it. Realizations overlap through the particles nearest the
observer (accepted with p close to 1): two pools share ~12% of their stars, two n = 2000
catalogs ~8% of their pulsars (<~2% for n <= 500). Larger catalogs would need several tracers
per particle (e.g. positions jittered within the particle's smoothing scale, with new
truth labels).

**Verified** (2026-09-28): `--sky none --legacy-sample --check-against
experiments/repeating_with_rejectancesampling/cache/pool_rejection.npz` reproduces the
earlier S(r)-only pool exactly (same per-trial indices, same positions, acceleration
difference 0.0), and the gate re-sums the frozen labels with difference 0.0.

## 4. Drawing a training set (`draw.py`)

```python
from mock_catalog.draw import load_pool, draw_trial
pool = load_pool("clean_experiments/mock_catalog/cache/pool_count.npz")
d = draw_trial(pool, trial=0, n=50, family="het")
```

A draw is n positions from trial t's pool with the **Sun-relative** line-of-sight
acceleration a pulsar actually measures, plus catalog noise:

    n_hat_i = (x_i - x_obs)/|x_i - x_obs|,  y_i = (a_i - a_obs)·n_hat_i + sigma_i eps_i

`family="het"` resamples sigma from the catalog (`sigma_pool_mmsyr`); `"s0"`, `"s0.5"`,
`"s1.7"`, `"s4.3"` are constant levels. Seeds follow the earlier FIRE harnesses:
positions `default_rng(trial)`, sigmas `default_rng(9000 + 137 trial + n)`, eps
`default_rng(7000 + 1009 trial + 31 n + family seed)`. On an S_r-only pool the positions
and eps are identical to the rejection rerun's, and `y_true + a_obs·n_hat` equals its
absolute LOS target (checked).

Everything is in physical units (kpc, kpc/Myr²). To train with the relative loss
(`galactoPINNs.train`, `x_obs=`), scale with the experiment's transformers and pass the
scalar through the 3-vector target slot as `y * n_hat` (the loss dots it with n_hat):

```python
x_n     = x_tf.transform(d["x"])
x_obs_n = x_tf.transform(d["x_obs"][None, :])
a_n     = a_tf.transform(d["y_los"][:, None] * d["n_hat"])    # relative, noisy
w       = (1 / d["sigma"]) / np.mean(1 / d["sigma"])          # whitened L1 weights
train_model_static(model, tx, x_n, a_n, epochs, n_vecs=d["n_hat"], line_of_sight=True,
                   x_obs=x_obs_n, importance_weight=w,
                   anchor_x=x_obs_n, anchor_a=a_tf.transform(d["a_obs"][None, :]))
```

For a catalog-like n = 50 draw the median |y_true| is ~0.8 mm/s/yr against a median
sigma of ~1.4: most single mock pulsars have S/N < 1.

## 5. Figures (`make_figures.py`)

```bash
uv run python clean_experiments/mock_catalog/make_figures.py      # figs/ + PDFs into accelerations-paper/figures/
```

- `selection_function`: S_r(r_sun) and S_delta(delta) (count weights; rate dashed for
  reference), with the catalog distances and declinations as rugs.
- `selection_sampling_6panel`: p(r_sun), p(l), p(b); column A catalog vs a uniform draw of
  the candidates, column B catalog vs the S_r S_delta sample (S_r-only dashed). The
  longitude panel of column B adds the uniform draw and the S_delta-only sample (n_* S_delta),
  and vertical lines mark each distribution's peak: uniform -2.5 deg, S_r only -13 deg,
  S_delta only +3 deg, joint +11.5 deg, catalog +30.5 deg.
- `selection_sky_2d`: sky density p(l, b) of the catalog, the uniform draw n_*, the S_r
  sample and the S_r S_delta sample: highest-density regions holding 40% / 90% of each
  sample, all smoothed with the catalog's Scott's-rule kernel (44 deg in l, 16 deg in b), so
  the ~3,200-star mocks are compared with the 47-pulsar catalog at the same resolution.
- `selection_sky_2d_radec` (not in the paper): the same in equatorial coordinates (RA increasing leftward),
  where S_delta is a set of horizontal bands (telescope limits dotted, Galactic plane and
  centre marked).
- `figs/selection_stats.json`: the numbers for the text (catalog statistics, N_k, S_delta
  per band for both weightings, quadrant fractions and |l| < 90 fraction as mean ± std over
  the 6 trials).

The figures read the catalog sample from `catalog.py`, so rerun this after changing
`EXCLUDE` or `MAX_FRAC_DIST_ERR`. Colours are the paper palette (catalog blue, uniform
draw orange, S_r green, S_r S_delta purple); since blue/purple and orange/green are hard
to tell apart for colour-blind readers, the references also differ in line style
(uniform dotted, S_r dashed).

## 6. Not in this folder

- **Evaluation sets.** The frozen sun4 / sun15 / gc15 sets are in `truth_old1gyr.npz`; the
  uniform-in-volume Sun bubbles (R = 0.5, 2, 4 kpc) and radial profile are in
  `experiments/repeating_with_rejectancesampling/cache/pool_rejection.npz`.
- **Distance errors.** Mock positions are exact. The catalog sigma already contains the
  Shklovskii-propagated part of the distance error; the position error itself
  (errors-in-variables) is not simulated here.
