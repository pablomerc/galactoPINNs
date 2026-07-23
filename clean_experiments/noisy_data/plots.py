"""Figures from the noisy_data results.csv (no retraining).

Regenerates:
  - noise_dose_response.png     (3 region panels: rel_err vs sigma, constant-
                                 sigma settings, lines = mode x n)
  - recovery_vs_n_noise.png     (2 modes x 3 regions: rel_err vs n, one line
                                 per constant-sigma setting)
  - recovery_vs_n_arms.png      (2 modes x 3 regions: rel_err vs n, arms at
                                 catalog noise: het / hetW / hetC, plus the
                                 s1.7 / s1.7W / s1.7C trio and s0)
  - calibration_chi.png         (2 x 2: train chi = <|r|/sigma> and train
                                 error vs truth [mm/s/yr], per setting vs n)

Usage:
  uv run python clean_experiments/noisy_data/plots.py
  uv run python clean_experiments/noisy_data/plots.py \
      --csv /tmp/noisy_smoke/results.csv --outdir /tmp/noisy_smoke
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
    "s1.7W": "#2e86ab",
    "s1.7C": "#c2185b",
}
SETTING_MARKERS = {
    "s0": "o", "s0.5": "s", "s1.7": "^", "s4.3": "v",
    "het": "<", "hetW": "D", "hetC": "P", "s1.7W": ">", "s1.7C": "X",
}
SETTING_LABELS = {
    "s0": "sigma = 0 (parent baseline)",
    "s0.5": "sigma = 0.5 mm/s/yr (10th pct)",
    "s1.7": "sigma = 1.7 mm/s/yr (median)",
    "s4.3": "sigma = 4.3 mm/s/yr (75th pct)",
    "het": "catalog sigmas, unweighted L1",
    "hetW": "catalog sigmas, w ~ 1/sigma (L1)",
    "hetC": "catalog sigmas, chi^2 (sq, w ~ 1/sigma^2)",
    "s1.7W": "sigma = 1.7, whitened L1 (w ~ 1/sigma; no-op vs s1.7)",
    "s1.7C": "sigma = 1.7, chi^2 loss",
}
CHI_BENCHMARK = math.sqrt(2.0 / math.pi)  # E|N(0,1)| ~= 0.798

MODE_STYLE = {"LOS": "--", "LOS+Sun+rho": "-"}
CHI2_SETTINGS = ("s1.7C", "hetC")


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
                mean = np.array([agg(rows, "rel_err", region=region, n=n,
                                     mode=mode, setting=s)[0]
                                 for n in n_grid]) * 100.0
                std = np.array([agg(rows, "rel_err", region=region, n=n,
                                    mode=mode, setting=s)[1]
                                for n in n_grid]) * 100.0
                ls = ":" if s in CHI2_SETTINGS else "-"
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
            chi_m = np.array([agg(sub, "chi", n=n, mode=mode, setting=s)[0]
                              for n in n_grid])
            chi_s = np.array([agg(sub, "chi", n=n, mode=mode, setting=s)[1]
                              for n in n_grid])
            et_m = np.array([agg(sub, "err_true", n=n, mode=mode, setting=s)[0]
                             for n in n_grid])
            et_s = np.array([agg(sub, "err_true", n=n, mode=mode, setting=s)[1]
                             for n in n_grid])
            ls = ":" if s in CHI2_SETTINGS else "-"
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

    arm_settings = [s for s in ("s0", "het", "hetW", "hetC",
                                "s1.7", "s1.7W", "s1.7C")
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
