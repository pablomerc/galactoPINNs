# GUIDE 3 — `plots.py`

Goal: four figures from `results.csv` alone (no retraining, no NPZ needed).
Prereq: a driver run — even the 2-n verification run renders everything.

## The four figures and what each answers

1. **`noise_dose_response.png`** — rel_err vs σ (the four constant levels),
   3 region panels, lines = mode × n (lowest/highest n, dashed=LOS,
   solid=LOS+Sun+rho). *How fast does recovery degrade with noise, and does
   the physics-prior mode degrade slower?*
2. **`recovery_vs_n_noise.png`** — rel_err vs n, one line per constant-σ
   setting; panel rows = modes, columns = regions. *Can more pulsars buy back
   what noise took away?* (the "quantity vs quality" forecast question).
3. **`recovery_vs_n_arms.png`** — same axes, settings = s0 reference +
   het/hetW/hetC + the s1.7/s1.7C pair. *Do σ-aware losses rescue the
   realistic catalog regime?* Because arms share noise realizations within a
   family, gaps between these lines are pure loss effects.
4. **`calibration_chi.png`** — the train-point diagnostics vs n:
   ⟨|r|/σ⟩ against the 0.798 matched-to-noise line, and the error vs the
   *true* LOS in mm/s/yr. *Which arms fit the field and which memorize
   noise?*

Design notes:

- Color encodes the **setting** here (mode is a panel), unlike the parent
  where color encoded mode. The two χ² arms are dotted; every setting has a
  unique color+marker so the busy arms figure stays readable.
- `hetLLH` has no `LOS+Sun+rho` rows (its anchored mode is `LOS+Sun`,
  physics-set, no ρ hinge) — `setting_mode()` maps it into the anchored panel
  row, and its legend label says so explicitly.
- `agg` means ± std over trials, exactly like the parent (`std=0` bars on a
  1-trial verification run are fine).
- chi/err_true are per-model train diagnostics repeated across the three
  region rows of the CSV — the calibration plot restricts to `region=="sun4"`
  so each model counts once.
- Everything degrades gracefully when settings are missing from the CSV
  (subset runs via `--settings` still plot).

## The code

```python
"""Figures from the noisy_data results.csv (no retraining).

Regenerates:
  - noise_dose_response.png     (3 region panels: rel_err vs sigma, constant-
                                 sigma settings, lines = mode x n)
  - recovery_vs_n_noise.png     (2 modes x 3 regions: rel_err vs n, one line
                                 per constant-sigma setting)
  - recovery_vs_n_arms.png      (2 modes x 3 regions: rel_err vs n, arms at
                                 catalog noise: het / hetW / hetC, plus the
                                 s1.7 vs s1.7C L1-vs-chi2 pair and s0)
  - calibration_chi.png         (2 x 2: train chi = <|r|/sigma> and train
                                 error vs truth [mm/s/yr], per setting vs n)

Usage:
  uv run python clean_experiments/noisy_data/plots.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

MODES = ("LOS", "LOS+Sun+rho")
REGIONS = ("sun4", "sun15", "gc15")
REGION_LABELS = {
    "sun4": "eval: r < 4 kpc of Sun (training region)",
    "sun15": "eval: r < 15 kpc of Sun",
    "gc15": "eval: r < 15 kpc of GC",
}

HOMO_SETTINGS = ("s0", "s0.5", "s1.7", "s4.3")
HOMO_SIGMA = {"s0": 0.0, "s0.5": 0.5, "s1.7": 1.7, "s4.3": 4.3}

# CVD-safe palette; color encodes the SETTING here (mode is a panel).
SETTING_COLORS = {
    "s0": "#7f8c8d",
    "s0.5": "#4a90d9",
    "s1.7": "#d97706",
    "s4.3": "#8e2431",
    "het": "#dc143c",
    "hetW": "#0f9e8a",
    "hetC": "#8e44ad",
    "s1.7C": "#c2185b",
    "hetLLH": "#1a237e",
}
SETTING_MARKERS = {
    "s0": "o", "s0.5": "s", "s1.7": "^", "s4.3": "v",
    "het": "<", "hetW": "D", "hetC": "P", "s1.7C": "X", "hetLLH": "*",
}
SETTING_LABELS = {
    "s0": "sigma = 0 (parent baseline)",
    "s0.5": "sigma = 0.5 mm/s/yr (10th pct)",
    "s1.7": "sigma = 1.7 mm/s/yr (median)",
    "s4.3": "sigma = 4.3 mm/s/yr (75th pct)",
    "het": "catalog sigmas, unweighted L1",
    "hetW": "catalog sigmas, w ~ 1/sigma (L1)",
    "hetC": "catalog sigmas, chi^2 (sq, w ~ 1/sigma^2)",
    "s1.7C": "sigma = 1.7, chi^2 loss",
    "hetLLH": "Laplace LLH, physics-set (w=1/sigma, lam_sun=1/sigma_sun, no rho)",
}
CHI_BENCHMARK = math.sqrt(2.0 / math.pi)  # E|N(0,1)| ~= 0.798

MODE_STYLE = {"LOS": "--", "LOS+Sun+rho": "-"}


def setting_mode(setting, mode):
    """hetLLH's anchored mode is LOS+Sun (physics-set, no rho hinge) — it
    stands in for LOS+Sun+rho in the anchored panel row."""
    if setting == "hetLLH" and mode == "LOS+Sun+rho":
        return "LOS+Sun"
    return mode


def load_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            chi = float(r["train_chi"]) if r["train_chi"] not in ("", "nan") else np.nan
            rows.append(dict(
                n=int(r["n_samples"]), trial=int(r["trial"]),
                setting=r["setting"], arm=r["arm"], mode=r["mode"],
                region=r["region"], rel_err=float(r["rel_err"]),
                chi=chi,
                err_true=float(r["train_err_true_mmsyr"]),
            ))
    return rows


def agg(rows, key, **sel):
    vals = [r[key] for r in rows
            if all(r[k] == v for k, v in sel.items()) and np.isfinite(r[key])]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def style_kw(setting):
    return dict(color=SETTING_COLORS[setting], marker=SETTING_MARKERS[setting],
                capsize=3, lw=1.9, ms=5, elinewidth=1.0, alpha=0.9)


def plot_dose_response(rows, outdir):
    """rel_err vs sigma (constant-sigma settings), lines = mode x n."""
    n_grid = sorted({r["n"] for r in rows})
    n_lo, n_hi = n_grid[0], n_grid[-1]
    sigmas = [HOMO_SIGMA[s] for s in HOMO_SETTINGS]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=False)
    for ax, region in zip(axes, REGIONS):
        for mode in MODES:
            for n, alpha in ((n_lo, 0.55), (n_hi, 1.0)):
                mean = np.array([agg(rows, "rel_err", region=region, n=n,
                                     mode=mode, setting=s)[0]
                                 for s in HOMO_SETTINGS]) * 100.0
                std = np.array([agg(rows, "rel_err", region=region, n=n,
                                    mode=mode, setting=s)[1]
                                for s in HOMO_SETTINGS]) * 100.0
                ax.errorbar(sigmas, mean, yerr=std, fmt="o" + MODE_STYLE[mode],
                            color="#dc143c" if mode == "LOS" else "#d97706",
                            capsize=3, lw=1.9, ms=5, elinewidth=1.0, alpha=alpha,
                            label=f"{mode}, n={n}")
        ax.set_xlabel(r"$\sigma(a_{\rm LOS})$ [mm/s/yr]")
        ax.set_title(REGION_LABELS[region], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("validation accel recovery error [%]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Noise dose-response (constant sigma, plain L1 training)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, "noise_dose_response.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def plot_vs_n_by_setting(rows, settings, fname, suptitle, outdir,
                         ylims=None):
    """rel_err vs n; rows of panels = modes, columns = regions."""
    n_grid = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(len(MODES), 3, figsize=(15.5, 9.2),
                             sharex=True, sharey=False)
    axes = np.atleast_2d(axes)
    for i, mode in enumerate(MODES):
        for j, region in enumerate(REGIONS):
            ax = axes[i, j]
            for s in settings:
                m = setting_mode(s, mode)
                mean = np.array([agg(rows, "rel_err", region=region, n=n,
                                     mode=m, setting=s)[0]
                                 for n in n_grid]) * 100.0
                std = np.array([agg(rows, "rel_err", region=region, n=n,
                                    mode=m, setting=s)[1]
                                for n in n_grid]) * 100.0
                ls = ":" if s in ("s1.7C", "hetC") else "-"
                ax.errorbar(n_grid, mean, yerr=std,
                            fmt=SETTING_MARKERS[s] + ls,
                            label=SETTING_LABELS[s], **{k: v for k, v in
                            style_kw(s).items() if k != "marker"})
            ax.set_xscale("log")
            ax.set_xticks(n_grid)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
            ax.grid(True, alpha=0.3)
            if ylims is not None and region in ylims:
                ax.set_ylim(*ylims[region])
            else:
                ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(REGION_LABELS[region], fontsize=10)
            if i == len(MODES) - 1:
                ax.set_xlabel("n_samples (LOS training measurements)")
            if j == 0:
                ax.set_ylabel(f"{mode}\nrecovery error [%]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def plot_calibration(rows, outdir):
    """Train-point diagnostics vs n per setting: chi and error vs truth."""
    n_grid = sorted({r["n"] for r in rows})
    settings = [s for s in SETTING_LABELS
                if s != "s0" and any(r["setting"] == s for r in rows)]
    # chi/err_true are train diagnostics repeated across the 3 region rows —
    # restrict to one region so each model counts once.
    sub = [r for r in rows if r["region"] == "sun4"]
    fig, axes = plt.subplots(2, len(MODES), figsize=(11.5, 8.6),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for j, mode in enumerate(MODES):
        for s in settings:
            m = setting_mode(s, mode)
            chi_m = np.array([agg(sub, "chi", n=n, mode=m, setting=s)[0]
                              for n in n_grid])
            chi_s = np.array([agg(sub, "chi", n=n, mode=m, setting=s)[1]
                              for n in n_grid])
            et_m = np.array([agg(sub, "err_true", n=n, mode=m, setting=s)[0]
                             for n in n_grid])
            et_s = np.array([agg(sub, "err_true", n=n, mode=m, setting=s)[1]
                             for n in n_grid])
            ls = ":" if s in ("s1.7C", "hetC") else "-"
            kw = {k: v for k, v in style_kw(s).items() if k != "marker"}
            axes[0, j].errorbar(n_grid, chi_m, yerr=chi_s,
                                fmt=SETTING_MARKERS[s] + ls,
                                label=SETTING_LABELS[s], **kw)
            axes[1, j].errorbar(n_grid, et_m, yerr=et_s,
                                fmt=SETTING_MARKERS[s] + ls,
                                label=SETTING_LABELS[s], **kw)
        axes[0, j].axhline(CHI_BENCHMARK, color="k", ls="--", lw=1.1)
        axes[0, j].set_title(mode, fontsize=10)
        axes[1, j].set_xlabel("n_samples")
        for i in (0, 1):
            axes[i, j].set_xscale("log")
            axes[i, j].set_xticks(n_grid)
            axes[i, j].xaxis.set_major_formatter(ScalarFormatter())
            axes[i, j].minorticks_off()
            axes[i, j].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel(r"train $\langle |r|/\sigma \rangle$"
                          "\n(0.798 = matched to noise)")
    axes[1, 0].set_ylabel("train error vs TRUE LOS\n[mm/s/yr]")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fit-to-noise calibration on the training sightlines",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, "calibration_chi.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=here)
    ap.add_argument("--csv", default=os.path.join(here, "results.csv"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.csv)

    plot_dose_response(rows, args.outdir)

    plot_vs_n_by_setting(
        rows, [s for s in HOMO_SETTINGS
               if any(r["setting"] == s for r in rows)],
        fname="recovery_vs_n_noise.png",
        suptitle="Recovery vs n at each constant noise level (plain L1)",
        outdir=args.outdir,
    )

    arm_settings = [s for s in ("s0", "het", "hetW", "hetC", "hetLLH",
                                "s1.7", "s1.7C")
                    if any(r["setting"] == s for r in rows)]
    plot_vs_n_by_setting(
        rows, arm_settings,
        fname="recovery_vs_n_arms.png",
        suptitle=("Do sigma-aware losses rescue the catalog-noise regime? "
                  "(shared noise realizations within each family)"),
        outdir=args.outdir,
    )

    plot_calibration(rows, args.outdir)


if __name__ == "__main__":
    main()
```

## Run it

```bash
uv run python clean_experiments/noisy_data/plots.py
```

All four figures were verified 2026-07-20 against the full-epoch
`--n-grid 50,100 --trials 1` verification CSV.

Reading the verified figures:

- In the **arms** figure the ordering at catalog noise is
  `het ≫ hetW ≈ hetC ≫ s0` in-region, and the σ-aware arms collapse most of
  the extrapolation blow-up in sun15/gc15 — visibly rescuing the realistic
  regime.
- In **calibration**, everything sits *below* the 0.798 line (all arms
  memorize some noise at fixed 1500 epochs), the unweighted sq arm (`s1.7C`)
  sits lowest at n=50 — L2's noise-chasing made visible — and `hetC` with
  anchor+rho is the only one that touches the line.

## Housekeeping once everything runs

Add a `noisy_data` section to `clean_experiments/README.md` (question, the
settings table, pointer to the guides), extend its status checklist, and
remember the repo tracking rules: code + `results.csv` + PNGs are tracked;
the NPZ is regenerable output and stays gitignored.
