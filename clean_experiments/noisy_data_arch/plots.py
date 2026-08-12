"""Plots for noisy_data_arch: PINN III/IV/V recovery on sun4.

Reads results.csv and makes:
  recovery_vs_n.png  -- median relative error vs n, one line per arch x mode,
                        IQR band over trials (the headline comparison).
  chi_vs_n.png       -- train chi vs n (0.8 ~ matched to noise; << means the
                        model is fitting noise), same grouping.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCH_COLOR = {"III": "#c0392b", "IV": "#2980b9", "V": "#27ae60"}
MODE_STYLE = {"LOS": dict(ls="--", marker="o"),
              "LOS+rho": dict(ls="-", marker="s")}


def _load(path):
    """results.csv (sun4 rows) -> {(arch, mode, n): {col: [values over trials]}}."""
    out = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["region"] != "sun4":
                continue
            key = (r["arch"], r["mode"], int(r["n_samples"]))
            for col in ("rel_err", "train_chi"):
                out[key][col].append(float(r[col]))
    return out


def _panel(ax, data, ycol, ylabel, logy=True, scale=1.0):
    ns_all = sorted({k[2] for k in data})
    arches = [a for a in ("III", "IV", "V") if any(k[0] == a for k in data)]
    modes = [m for m in ("LOS", "LOS+rho") if any(k[1] == m for k in data)]
    for arch in arches:
        for mode in modes:
            ns = [n for n in ns_all if (arch, mode, n) in data]
            if not ns:
                continue
            med = [scale * np.median(data[(arch, mode, n)][ycol]) for n in ns]
            lo = [scale * np.quantile(data[(arch, mode, n)][ycol], 0.25) for n in ns]
            hi = [scale * np.quantile(data[(arch, mode, n)][ycol], 0.75) for n in ns]
            ax.plot(ns, med, color=ARCH_COLOR[arch], lw=1.8,
                    label=f"PINN {arch} · {mode}", **MODE_STYLE[mode])
            ax.fill_between(ns, lo, hi, color=ARCH_COLOR[arch], alpha=0.12, lw=0)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("n training sightlines")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    data = _load(os.path.join(args.indir, "results.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))
    _panel(ax, data, "rel_err", "median relative acceleration error (sun4) [%]",
           logy=False, scale=100.0)          # percent, linear y-axis
    ax.set_ylim(bottom=0)
    ax.set_title("Recovery vs n — PINN III/IV/V, LOS vs LOS+ρ")
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    p1 = os.path.join(args.indir, "recovery_vs_n.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"wrote {p1}")

    fig, ax = plt.subplots(figsize=(7, 5))
    _panel(ax, data, "train_chi", "train χ = ⟨|resid|/σ⟩", logy=False)
    ax.axhline(0.798, color="0.4", ls=":", lw=1, label="L1 matched-to-noise (0.80)")
    ax.set_title("Noise calibration vs n")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(args.indir, "chi_vs_n.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
