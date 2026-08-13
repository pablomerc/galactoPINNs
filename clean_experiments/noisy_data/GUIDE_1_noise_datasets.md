# GUIDE 1 — `datasets.py` (the noise layer)

Goal: `noisy_data/datasets.py` — the parent experiment's geometry/truth/scaling
reused verbatim, plus the one new thing this experiment adds: a seeded,
physically calibrated measurement-noise layer for the LOS accelerations.

Prereq: GUIDE 0 applied and verified (the driver in GUIDE 2 needs both src
edits; this module itself runs on unpatched src).

## Physics first: what "realistic noise" means here

From the real catalog (Donlon+2025, the 51-pulsar PB-else-PS selection used by
`experiments/gp_audit/catalog.py`):

- per-pulsar σ(a_LOS) spans **0.12 → 18.2 mm/s/yr** (heavy right tail),
  percentiles 10/25/50/75/90 = 0.52 / 0.77 / **1.69** / 4.26 / 7.36;
- the informative (Sun-relative) LOS signal in a 4-kpc ball has median
  ≈ 2 mm/s/yr — **the typical real pulsar is a ≲1σ measurement**;
- the *raw* projection a·n̂ our mock trains on has RMS ≈ 5 mm/s/yr, so the
  median noise level is ≈ 30% of the raw RMS but ≈ 100% of the informative
  signal. Quote S/N in the relative currency, never the raw one — raw flatters.

Design rules that follow:

1. **Absolute noise, not fractional** — pulsar-timing errors don't scale with
   the signal. Constant levels 0 / 0.5 / 1.7 / 4.3 mm/s/yr (≈ zero / 10th pct /
   median / 75th pct of the catalog) give the dose-response curve; a
   heteroscedastic mode resamples the 51 real sigmas for the realistic point.
2. **Inject along the sightline**: `a_noisy = a + ε·n̂`, ε ~ N(0, σ). The LOS
   loss only ever reads the projection `a·n̂`, which receives exactly ε — and
   σ=0 stays bit-exact with the parent (freeze baseline). No src changes for
   data.
3. **The Sun anchor stays noiseless**: its real uncertainty (Sgr A* VLBI,
   σ ≈ 0.13 mm/s/yr) is subdominant to every pulsar σ.
4. Noise is drawn in the **driver**, per (n, trial, family) with fixed seeds —
   the pool itself stays clean, and arms that share a noise family train on
   identical noisy data (GUIDE 2).

## Piece A — imports, parent reuse, constants

One gotcha decides the import style: this file is *also* named `datasets.py`.
A plain `from datasets import ...` inside it would resolve to **itself**
mid-import (the module is already in `sys.modules`, half-initialized) and fail
the moment the driver does `import datasets`. Loading the parent **by path
under its own module name** sidesteps the shadowing:

```python
"""Dataset construction for the noisy_data experiment.

Repeats clean_experiments/sun_los_recovery with ONE twist: Gaussian measurement
noise on the observed LOS accelerations. Geometry, truth, and scaling are the
parent experiment's, imported verbatim — this module only adds the noise layer:
per-point sigmas (constant levels or resampled from the real Donlon+2025
catalog), unit conversion mm/s/yr -> scaled, seeded injection along the
sightline, and the per-arm importance weights.

Noise is ABSOLUTE (pulsar-timing errors don't scale with signal) and injected
as a_noisy = a + eps*nhat, eps ~ N(0, sigma): the LOS loss only ever sees the
projection a·nhat, which receives exactly the intended scalar noise, and the
sigma=0 arm stays bit-exact with the parent experiment.
"""

import importlib.util
import os

import jax.numpy as jnp
import numpy as np

# The parent experiment, reused verbatim. Loaded by path under its OWN module
# name ("parent_datasets") — this file is also called datasets.py, so a plain
# `from datasets import ...` would resolve to ourselves mid-import and fail.
_PARENT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "sun_los_recovery", "datasets.py")
_spec = importlib.util.spec_from_file_location("parent_datasets", _PARENT)
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)

GC = _parent.GC
HALO_RS = _parent.HALO_RS
OBSERVER = _parent.OBSERVER
build_data = _parent.build_data
sample_colloc_scaled = _parent.sample_colloc_scaled

__all__ = [
    "GC", "HALO_RS", "OBSERVER", "build_data", "sample_colloc_scaled",
    "CATALOG_SIGMAS_MMSYR", "MMSYR_TO_KPCMYR2", "NOISE_FAMILY_SEED",
    "sigma_scale_factor", "draw_sigmas_mmsyr", "apply_los_noise", "arm_weights",
]
```

The constants. The catalog sigmas are embedded as literals (6 significant
figures) because the repo gitignores `data/` and `*.npy` — this keeps the
clean experiment self-contained and tracked:

```python
# 1 mm/s/yr = 1 km/s/Myr = 1/977.79222 kpc/Myr^2 (the LOS-acceleration currency
# of pulsar timing; same constant as experiments/gp_audit/audit_config.py).
MMSYR_TO_KPCMYR2 = 1.0 / 977.79222

# Per-pulsar sigma(a_LOS) [mm/s/yr] of the 51 Donlon+2025 pulsars selected by
# experiments/gp_audit/catalog.py (PB-else-PS), sorted. Embedded as literals
# (6 significant figures) because data/ and *.npy are gitignored; markers
# ~10th/50th/75th percentile = 0.52 / 1.69 / 4.26 set the constant levels.
CATALOG_SIGMAS_MMSYR = (
    0.118371, 0.326056, 0.33556, 0.339927, 0.420831, 0.523387,
    0.532496, 0.554591, 0.576837, 0.670016, 0.726771, 0.745114,
    0.763416, 0.770922, 0.851857, 0.915736, 0.920055, 1.01904,
    1.0872, 1.36575, 1.46129, 1.46334, 1.50857, 1.64275,
    1.68204, 1.69031, 1.82323, 1.87895, 1.92658, 2.06964,
    2.41148, 2.77526, 2.79087, 2.81604, 3.15476, 3.16167,
    3.64418, 4.12575, 4.4002, 4.53605, 5.95132, 6.07849,
    6.35578, 6.93263, 7.12839, 7.36009, 8.57286, 13.0403,
    13.5344, 18.2143, 18.2143,
)

# Noise FAMILY = which sigma draw a setting uses. Arms of the same family
# (het / hetW / hetC, or s1.7 / s1.7C) share the seed, hence the SAME noise
# realization — arm comparisons are then loss comparisons, not draw luck.
NOISE_FAMILY_SEED = {"s0": 0, "s0.5": 1, "s1.7": 2, "s4.3": 3, "het": 4}
```

## Piece B — the four functions

**Unit conversion.** σ lives in mm/s/yr everywhere humans read it (configs,
CSV, plots) and gets converted to scaled units only at the injection point.
The a-transformer is an isotropic constant factor, so one probe vector
measures it exactly:

```python
def sigma_scale_factor(cfg):
    """Scaled acceleration units per 1 mm/s/yr, from the fitted a-transformer.

    The transformer is an isotropic constant factor, so one probe vector
    measures it exactly.
    """
    a_fac = float(jnp.linalg.norm(
        cfg["a_transformer"].transform(jnp.array([[1e-3, 0.0, 0.0]])))) / 1e-3
    return MMSYR_TO_KPCMYR2 * a_fac
```

(Probe = 1e-3 kpc/Myr² on the x-axis; divide it back out, multiply by the
kpc/Myr² per mm/s/yr. With the full 30 000-point pool this comes out to
1 mm/s/yr = 0.07902 scaled; with the 2 000-point self-test pool, 0.07914 —
the scaling is fit on the pool max, so it moves slightly with pool size.)

**Sigma draws.** A "family" is either a constant level (`"s1.7"` → 1.7 for
everyone) or `"het"` (resample the 51 real values with replacement):

```python
def draw_sigmas_mmsyr(rng, n, family):
    """Per-point sigma in mm/s/yr for a noise family.

    'sX' -> constant level X for all n points; 'het' -> resampled with
    replacement from the 51 real catalog sigmas.
    """
    if family == "het":
        return rng.choice(np.asarray(CATALOG_SIGMAS_MMSYR), size=n, replace=True)
    return np.full(n, float(family[1:]))
```

**Injection.** The trick from the header: perturb the 3D target *along the
sightline*. The LOS projection gains exactly ε; the transverse components
change too, but no LOS mode ever reads them (and the 3D reference only trains
at σ=0):

```python
def apply_los_noise(rng, a_n, n_vecs, sig_scaled):
    """Perturb the 3D targets along the sightline: a + eps*nhat (scaled units).

    eps ~ N(0, sig_scaled) per point. The LOS projection a·nhat gains exactly
    eps; transverse components change too, but no LOS mode ever reads them.
    sig_scaled == 0 returns a_n unchanged, bit-exact.
    """
    eps = rng.normal(0.0, 1.0, size=len(sig_scaled)) * np.asarray(sig_scaled)
    return a_n + jnp.asarray(eps)[:, None] * n_vecs
```

**Arm weights.** The whitening rule from GUIDE 0 — match the weight to the
residual power (`1/σ` for `|r|`, `1/σ²` for `r²`), mean-normalize so
`weights=ones ≡ unweighted` and the anchor/data balance is untouched. Being
scale-free, the weights can be computed straight from mm/s/yr:

```python
def arm_weights(sig_mmsyr, arm, fac=None):
    """Per-point importance weights for a training arm.

    'plain' -> None (unweighted); 'W' -> w ∝ 1/sigma (whitened L1);
    'C' -> w ∝ 1/sigma^2 (chi^2 when combined with los_loss='sq').
    W and C are mean-normalized to 1 so the anchor/data balance inside the
    shared LOS mean is untouched at init (weights=ones == unweighted,
    verified bit-exact); being scale-free they use mm/s/yr directly.

    'LLH' -> UNnormalized w_i = 1/sigma_i in SCALED units (requires fac):
    the Laplace-likelihood arm. With lambda_anchor = 1/sigma_sun(scaled) the
    src loss (sum(w|r|) + lambda*sum(|r_a|)) / (N+3M) IS the Laplace NLL up
    to a constant factor — physics-set weights, zero tuned lambdas, and no
    convergence-time rebalancing confound.
    """
    if arm == "plain":
        return None
    if arm == "LLH":
        if fac is None:
            raise ValueError("arm 'LLH' needs fac (scaled units per mm/s/yr).")
        return jnp.asarray(1.0 / (np.asarray(sig_mmsyr) * fac))
    w = 1.0 / np.asarray(sig_mmsyr) ** (1 if arm == "W" else 2)
    return jnp.asarray(w / w.mean())
```

Worth staring at: with the real catalog draws, the W weights span a **~114×**
dynamic range but the C weights span **~13 000×** — the "correct" Gaussian
likelihood effectively throws away the noisy half of the catalog and rides the
few sub-0.5 mm/s/yr pulsars. Whether that wins or loses against the robust L1
arms is exactly what the experiment measures.

Why `LLH` skips the mean-normalization the other arms use: normalization
keeps the data-vs-anchor balance fixed *at initialization*, but once the fit
converges to the noise level the whitened data term shrinks by
mean(σ)·mean(1/σ) ≈ 3.4× (W) or mean(σ²)·mean(1/σ²) ≈ 80× (C), silently
upweighting λ_anchor/λ_rho — the confound caught in the quick-scan's
adversarial review. The likelihood convention makes weights *physics-set*
instead: every term is a residual divided by its own σ (the Sun anchor priced
at the real σ_sun = 0.13 mm/s/yr via `lambda_anchor = 1/σ_sun`), so there is
nothing left to tune and nothing to rebalance. This is the measured-best mode
of the 2026-07-20 L1-vs-L2 follow-up in `experiments/los_noise/`.

## Piece C — self-test

`__main__` block: catalog stats, a pulsar-like n=50 draw subsampled *exactly*
as the driver does (trial 0), injection checks at s0 and s1.7 with the
driver's seed scheme, weight checks, and a 2-panel `noise_preview.png`
(catalog histogram with the constant levels; true vs observed a·n̂ for the
σ=1.7 and het realizations).

```python
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    import galax.potential as gp

    cat = np.asarray(CATALOG_SIGMAS_MMSYR)
    q = np.percentile(cat, [10, 25, 50, 75, 90])
    print(f"catalog: n={len(cat)}, min={cat.min():.3f}, max={cat.max():.3f}")
    print("pct 10/25/50/75/90 = " + "/".join(f"{v:.3f}" for v in q))

    true_potential = gp.MilkyWayPotential()
    regions = {"sun4": (OBSERVER, 4.0), "sun15": (OBSERVER, 15.0),
               "gc15": (GC, 15.0)}
    x_pool, a_pool, x_pool_phys, cfg, val_scaled, dist_sun, x_obs, a_obs = build_data(
        true_potential, pool=2000, n_val=256, r_train=4.0,
        regions=regions, include_analytic=False,
    )
    fac = sigma_scale_factor(cfg)
    print(f"1 mm/s/yr = {fac:.5f} scaled units (pool=2000 fit)")

    # A pulsar-like n=50 draw, subsampled exactly as the driver does (trial 0)
    n, trial = 50, 0
    rng = np.random.default_rng(trial)
    idx = rng.choice(x_pool.shape[0], size=n, replace=False)
    x_n, a_n = x_pool[idx], a_pool[idx]
    dx = x_pool_phys[idx] - OBSERVER
    n_vecs = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))
    a_los_true = jnp.sum(a_n * n_vecs, axis=1)
    print(f"raw LOS projection: RMS = {float(jnp.sqrt(jnp.mean(a_los_true**2)))/fac:.2f} "
          f"mm/s/yr (n={n} draw)")

    # het sigma draw (driver seed scheme: 9000 + 137*trial + n)
    sig_het = draw_sigmas_mmsyr(np.random.default_rng(9000 + 137 * trial + n),
                                n, "het")
    print(f"het draw: median={np.median(sig_het):.3f}, mean={sig_het.mean():.3f} mm/s/yr")

    # noise injection at the median level (family seed scheme:
    # 7000 + 1009*trial + 31*n + NOISE_FAMILY_SEED[family])
    checks = []
    for family in ("s0", "s1.7"):
        sig = draw_sigmas_mmsyr(np.random.default_rng(1), n, family)
        eps_rng = np.random.default_rng(
            7000 + 1009 * trial + 31 * n + NOISE_FAMILY_SEED[family])
        a_noisy = apply_los_noise(eps_rng, a_n, n_vecs, sig * fac)
        delta = a_noisy - a_n
        d_los = jnp.sum(delta * n_vecs, axis=1)
        trv = delta - d_los[:, None] * n_vecs
        checks.append((family, a_noisy))
        print(f"{family:5s}: injected std = {float(jnp.std(d_los))/fac:.3f} mm/s/yr, "
              f"max transverse leak = {float(jnp.abs(trv).max()):.1e}, "
              f"bit-exact = {bool((a_noisy == a_n).all())}")

    # weights
    for arm in ("W", "C"):
        w = arm_weights(sig_het, arm)
        print(f"arm {arm}: mean = {float(w.mean()):.6f}, "
              f"dynamic range = {float(w.max()/w.min()):.0f}x")
    w = arm_weights(sig_het, "LLH", fac)
    print(f"arm LLH: w in [{float(w.min()):.2f}, {float(w.max()):.2f}] "
          f"(unnormalized 1/sigma_scaled); "
          f"lambda_sun = {1.0/(0.13*fac):.1f}")

    # preview figure: catalog + what the noise does to the observable
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].hist(cat, bins=np.geomspace(0.1, 20, 18), color="#2980b9", alpha=0.8)
    for lev in (0.5, 1.7, 4.3):
        axes[0].axvline(lev, ls="--", c="#c0392b", lw=1.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\sigma(a_{\rm LOS})$ [mm/s/yr]")
    axes[0].set_ylabel("pulsars")
    axes[0].set_title("Donlon+2025 catalog sigmas\n(dashed: constant levels)")

    los_true = np.asarray(a_los_true) / fac
    _, a_noisy17 = checks[1]
    los_n17 = np.asarray(jnp.sum(a_noisy17 * n_vecs, axis=1)) / fac
    eps_rng = np.random.default_rng(7000 + 1009 * trial + 31 * n
                                    + NOISE_FAMILY_SEED["het"])
    a_noisyhet = apply_los_noise(eps_rng, a_n, n_vecs, sig_het * fac)
    los_het = np.asarray(jnp.sum(a_noisyhet * n_vecs, axis=1)) / fac
    lim = np.abs(los_true).max() * 1.15
    axes[1].plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
    axes[1].scatter(los_true, los_n17, s=18, c="#c0392b", alpha=0.7,
                    label=r"$\sigma=1.7$")
    axes[1].scatter(los_true, los_het, s=18, c="#27ae60", alpha=0.7,
                    label="het (catalog)", marker="s")
    axes[1].set_xlabel(r"true $a\cdot\hat n$ [mm/s/yr]")
    axes[1].set_ylabel(r"observed $a\cdot\hat n$ [mm/s/yr]")
    axes[1].set_title(f"n={n} mock observation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = Path(__file__).with_name("noise_preview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
```

## Run it

```bash
uv run python clean_experiments/noisy_data/datasets.py
```

Expected output — **exact** (everything is seeded):

```
catalog: n=51, min=0.118, max=18.214
pct 10/25/50/75/90 = 0.523/0.767/1.690/4.263/7.360
1 mm/s/yr = 0.07914 scaled units (pool=2000 fit)
raw LOS projection: RMS = 5.07 mm/s/yr (n=50 draw)
het draw: median=1.461, mean=2.478 mm/s/yr
s0   : injected std = 0.000 mm/s/yr, max transverse leak = 0.0e+00, bit-exact = True
s1.7 : injected std = 1.470 mm/s/yr, max transverse leak = 3.0e-08, bit-exact = False
arm W: mean = 1.000000, dynamic range = 114x
arm C: mean = 1.000000, dynamic range = 13073x
arm LLH: w in [0.93, 106.75] (unnormalized 1/sigma_scaled); lambda_sun = 97.2
wrote .../clean_experiments/noisy_data/noise_preview.png
```

Reading the checks:

- **s0 bit-exact = True** is the heart of the freeze baseline: adding
  0·n̂ literally returns the same array, so the σ=0 arm of the driver *is*
  the parent experiment.
- **injected std = 1.470 at σ=1.7**: the sample std of 50 Gaussian draws has
  ~10% scatter — one realization coming out at 1.47 is a draw, not a bug.
- **transverse leak 3.0e-08 (≠ 0)**: the scaled arrays are float32; recomputing
  the projection of `ε·n̂` costs one rounding at ~1e-8. Float-epsilon-level,
  by construction not readable by any LOS loss.
- In the right preview panel the green (het) squares hug the y=x line where a
  precise pulsar was drawn and fly off it where an 18 mm/s/yr one landed —
  the visual argument for the weighted arms.
