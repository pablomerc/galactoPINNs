# GUIDE 3 — `plots.py`

Goal: figures from `results.csv` + `per_point_errors.npz`. Deliberately the lightest
guide — the plotting is standard matplotlib; the choices below are what matters.

Keep plotting in its own script (rerunnable without retraining):
`uv run python clean_experiments/sun_los_recovery/plots.py`

## Figure 1 — recovery vs n (the headline)

One panel per **region** (sun4 / sun15 / gc15), x = n_samples (log), y = mean rel_err
across trials (log), one line per **mode** with per-trial scatter or error bars.

- Read `results.csv` with `csv.DictReader` or pandas; aggregate over `trial`.
- log-log axes; `ScalarFormatter` on x so ticks read 50, 100, 500 rather than 10^k.
- Keep mode colors consistent across ALL figures (one shared `COLORS = {mode: ...}`
  dict at the top).

What it should show: the `3D`→`LOS` gap (~4× in-region), `+Sun` closing part of the
gap at low n, `+rho` dominating outside the training ball.

## Figure 2 — error vs distance from the Sun

The sun-sphere experiment's signature plot: where does the field break down?

- From the NPZ: `dist|<region>` (per-point distances) and
  `err|<region>|<n>|<trial>|<mode>` (per-point relative errors).
- For each region and each n: bin points by distance (e.g. 10–15 bins), plot median
  error per bin, one line per mode, trials pooled. One panel per n (or a small grid).
- Vertical line at r_train = 4 kpc: everything right of it is extrapolation.

Expected shape: errors flat-ish inside the training ball, rising past 4 kpc; the
`+rho` modes rise much more slowly (that's the prior disciplining the unconstrained
volume).

## Figure 3 — negative-density fraction vs n

Per region: y = mean `neg_frac` across trials, one line per mode. This is the "does
the hinge do its literal job" diagnostic — `+rho` modes should sit near zero, plain
`LOS` visibly above it, `3D` in between.

## Bookkeeping

- `matplotlib.use("Agg")` before pyplot (headless-safe).
- Save PNGs next to the CSV (`recovery_vs_nsamples.png`, `err_vs_distance_<region>.png`,
  `neg_fraction_vs_nsamples.png`). PNGs are tracked by git — commit the finals, and
  these are the figures to drop in Discord for Nathaniel.
- The old private experiment's `plot_from_outputs.py` is your answer key for the
  binning details — consult it, but write your own.
