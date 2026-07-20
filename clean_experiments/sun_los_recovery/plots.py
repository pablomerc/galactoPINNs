"""Figures from results.csv + per_point_errors.npz (no retraining).

Regenerates:
  - recovery_vs_nsamples.png           (3 panels: sun4 / sun15 / gc15)
  - recovery_decomp_vs_nsamples.png    (2 regions × LOS/transverse; needs
                                        rel_los / rel_trv columns in the CSV)
  - recovery_cyl_vs_nsamples.png       (2 regions × R/z/φ; needs rel_R /
                                        rel_z / rel_phi columns)
  - neg_fraction_vs_nsamples.png       (3 panels)
  - err_vs_distance_<region>.png       (binned median error vs distance from
                                        the Sun, one panel per n, trials pooled)

Usage:
  uv run python clean_experiments/sun_los_recovery/plots.py
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

MODES = ("3D", "LOS", "LOS+Sun", "LOS+rho", "LOS+Sun+rho")
REGIONS = ("sun4", "sun15", "gc15")
REGION_LABELS = {
    "sun4": "eval: r < 4 kpc of Sun (training region)",
    "sun15": "eval: r < 15 kpc of Sun",
    "gc15": "eval: r < 15 kpc of GC",
}

# CVD-validated palette (shared across every figure).
COLORS = {
    "3D": "#4a90d9",
    "LOS": "#dc143c",
    "LOS+Sun": "#0f9e8a",
    "LOS+rho": "#8e44ad",
    "LOS+Sun+rho": "#d97706",
}
MARKERS = {"3D": "o", "LOS": "s", "LOS+Sun": "^", "LOS+rho": "D", "LOS+Sun+rho": "v"}
MODE_LABELS = {
    "3D": "3D full-vector accelerations",
    "LOS": "LOS accelerations (Sun vantage)",
    "LOS+Sun": "LOS + a(Sun) 3-vector anchor",
    "LOS+rho": "LOS + rho>=0 positivity penalty",
    "LOS+Sun+rho": "LOS + a(Sun) anchor + rho>=0",
}

BIN_WIDTH = {"sun4": 0.5, "sun15": 1.0, "gc15": 1.0}
MIN_BIN_COUNT = 30
R_TRAIN = 4.0  # kpc, Sun-centered training ball


DECOMP_KEYS = ("rel_los", "rel_trv", "rel_R", "rel_z", "rel_phi")


def _optional_float(raw: str | None):
    text = (raw or "").strip()
    return float(text) if text else None


def load_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            row = dict(
                n=int(r["n_samples"]),
                trial=int(r["trial"]),
                mode=r["mode"],
                region=r["region"],
                rel_err=float(r["rel_err"]),
                neg_frac=_optional_float(r.get("neg_frac")),
            )
            for key in DECOMP_KEYS:
                row[key] = _optional_float(r.get(key))
            rows.append(row)
    return rows


def has_decomp(rows: list[dict], keys: tuple[str, ...]) -> bool:
    return any(r[k] is not None for r in rows for k in keys)


def agg(rows, region, n, mode, key):
    vals = [
        r[key]
        for r in rows
        if r["region"] == region
        and r["n"] == n
        and r["mode"] == mode
        and r[key] is not None
    ]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def plot_vs_n(
    rows, key, ylabel, fname, outdir, *, logy: bool, scale: float = 1.0,
    sharey: bool | str = True, ylims: dict[str, tuple[float, float]] | None = None,
):
    n_grid = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=sharey)
    for ax, region in zip(axes, REGIONS):
        for mode in MODES:
            mean = np.array([agg(rows, region, n, mode, key)[0] for n in n_grid]) * scale
            std = np.array([agg(rows, region, n, mode, key)[1] for n in n_grid]) * scale
            ax.errorbar(
                n_grid, mean, yerr=std,
                fmt=MARKERS[mode] + "-", color=COLORS[mode],
                capsize=3, lw=1.9, ms=5, elinewidth=1.0, alpha=0.9,
                label=MODE_LABELS[mode],
            )
        ax.set_xscale("log")
        ax.set_xticks(n_grid)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("n_samples (LOS training measurements)")
        ax.set_title(REGION_LABELS[region], fontsize=10)
        ax.grid(True, alpha=0.3)
        if ylims is not None and region in ylims:
            ax.set_ylim(*ylims[region])
        elif not logy:
            ax.set_ylim(bottom=0)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Training: density-weighted points in a 4 kpc ball around the Sun",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def plot_decomp_vs_n(
    rows, components, fname, outdir, *,
    regions: tuple[str, ...] = ("sun4", "sun15"),
    ylims: dict[str, tuple[float, float]] | None = None,
    suptitle: str,
):
    """Rows = eval regions, columns = error components; shared y per region row."""
    n_grid = sorted({r["n"] for r in rows})
    if ylims is None:
        ylims = {}
        for region in regions:
            tops = [
                (m + s) * 100.0
                for key, _ in components
                for mode in MODES
                for n in n_grid
                for m, s in [agg(rows, region, n, mode, key)]
                if np.isfinite(m)
            ]
            if tops:
                ylims[region] = (0.0, 1.08 * max(tops))
    nrows, ncols = len(regions), len(components)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols + 0.5, 4.2 * nrows),
        sharex=True, sharey=False,
    )
    axes = np.atleast_2d(axes)
    for i, region in enumerate(regions):
        for j, (key, col_title) in enumerate(components):
            ax = axes[i, j]
            for mode in MODES:
                mean = np.array(
                    [agg(rows, region, n, mode, key)[0] for n in n_grid]
                ) * 100.0
                std = np.array(
                    [agg(rows, region, n, mode, key)[1] for n in n_grid]
                ) * 100.0
                ax.errorbar(
                    n_grid, mean, yerr=std,
                    fmt=MARKERS[mode] + "-", color=COLORS[mode],
                    capsize=3, lw=1.9, ms=5, elinewidth=1.0, alpha=0.9,
                    label=MODE_LABELS[mode],
                )
            ax.set_xscale("log")
            ax.set_xticks(n_grid)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
            ax.grid(True, alpha=0.3)
            if ylims is not None and region in ylims:
                ax.set_ylim(*ylims[region])
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlim(n_grid[0] * 0.8, n_grid[-1] * 1.25)
            if i == 0:
                ax.set_title(col_title, fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel("n_samples (LOS training measurements)")
            if j == 0:
                ax.set_ylabel(f"{REGION_LABELS[region]}\nerror [%]")
                if i == 0:
                    ax.legend(fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def plot_err_vs_distance(perpoint, n_grid, trials, outdir):
    for region in REGIONS:
        dist_key = f"dist|{region}"
        if dist_key not in perpoint:
            print(f"skip err_vs_distance_{region}: missing {dist_key}")
            continue
        dist = np.asarray(perpoint[dist_key])
        width = BIN_WIDTH[region]
        edges = np.arange(0.0, dist.max() + width, width)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dist_pooled = np.tile(dist, len(trials))

        ncols, nrows = 3, int(np.ceil(len(n_grid) / 3))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(15.5, 4.6 * nrows), sharex=True, sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for k, n in enumerate(n_grid):
            ax = axes[k]
            for mode in MODES:
                keys = [f"err|{region}|{n}|{t}|{mode}" for t in trials]
                if not all(key in perpoint for key in keys):
                    continue
                errs = np.concatenate([perpoint[key] for key in keys])
                which = np.digitize(dist_pooled, edges) - 1
                med = np.full(len(centers), np.nan)
                for b in range(len(centers)):
                    sel = errs[which == b]
                    if sel.size >= MIN_BIN_COUNT:
                        med[b] = np.median(sel)
                ok = np.isfinite(med)
                ax.plot(
                    centers[ok], med[ok] * 100.0,
                    marker=MARKERS[mode], ms=4, lw=1.7,
                    color=COLORS[mode], alpha=0.9, label=MODE_LABELS[mode],
                )
            ax.axvline(R_TRAIN, color="gray", ls=":", lw=1.3)
            ax.set_title(f"n = {n}", fontsize=10)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)
        for k in range(len(n_grid), len(axes)):
            axes[k].set_visible(False)
        axes[0].legend(fontsize=8)
        axes[0].text(
            R_TRAIN, axes[0].get_ylim()[1], " training-ball edge",
            fontsize=7, color="gray", va="top", ha="left",
        )
        fig.supxlabel("distance from the Sun [kpc]")
        fig.supylabel("median accel recovery error [%] (trials pooled)")
        fig.suptitle(
            f"{REGION_LABELS[region]} — trained on r < {R_TRAIN:g} kpc ball around the Sun",
            fontsize=11,
        )
        fig.tight_layout()
        out = os.path.join(outdir, f"err_vs_distance_{region}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("saved", out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=here)
    ap.add_argument("--csv", default=os.path.join(here, "results.csv"))
    ap.add_argument("--npz", default=os.path.join(here, "per_point_errors.npz"))
    args = ap.parse_args()

    rows = load_rows(args.csv)
    n_grid = sorted({r["n"] for r in rows})
    trials = sorted({r["trial"] for r in rows})
    perpoint = dict(np.load(args.npz))

    # Figure 1 — recovery vs n; CSV stores fractions, plot as percent.
    # sun4 errors are ~4× smaller than the extrapolation panels → own y-scale.
    plot_vs_n(
        rows, "rel_err",
        ylabel="validation accel recovery error [%]",
        fname="recovery_vs_nsamples.png",
        outdir=args.outdir,
        logy=False,
        scale=100.0,
        sharey=False,
        ylims={"sun4": (0, 12), "sun15": (0, 55), "gc15": (0, 55)},
    )

    # Decomposition vs n (needs columns written by the GUIDE-2 decomp edit).
    # Per-region y-limits are data-driven inside plot_decomp_vs_n.
    if has_decomp(rows, ("rel_los", "rel_trv")):
        plot_decomp_vs_n(
            rows,
            components=(
                ("rel_los", "LOS error [%]"),
                ("rel_trv", "transverse error [%]"),
            ),
            fname="recovery_decomp_vs_nsamples.png",
            outdir=args.outdir,
            regions=("sun4", "sun15"),
            suptitle=(
                "Where the recovery error lives: LOS vs transverse "
                "(Sun-anchored sightlines)"
            ),
        )
    else:
        print(
            "skip recovery_decomp_vs_nsamples.png: CSV lacks rel_los/rel_trv "
            "(re-run run_experiment.py after the decomposition edit)"
        )

    if has_decomp(rows, ("rel_R", "rel_z", "rel_phi")):
        plot_decomp_vs_n(
            rows,
            components=(
                ("rel_R", "R (radial) error [%]"),
                ("rel_z", "z (vertical) error [%]"),
                ("rel_phi", "φ (azimuthal) error [%]"),
            ),
            fname="recovery_cyl_vs_nsamples.png",
            outdir=args.outdir,
            regions=("sun4", "sun15"),
            suptitle=(
                "Where the recovery error lives: cylindrical R / z / φ"
            ),
        )
    else:
        print(
            "skip recovery_cyl_vs_nsamples.png: CSV lacks rel_R/rel_z/rel_phi "
            "(re-run run_experiment.py after the decomposition edit)"
        )

    # Figure 3 — negative-density fraction vs n
    plot_vs_n(
        rows, "neg_frac",
        ylabel="fraction of val points with predicted rho < 0",
        fname="neg_fraction_vs_nsamples.png",
        outdir=args.outdir,
        logy=False,
    )

    # Figure 2 — error vs distance from the Sun
    # Prefer n/trial lists implied by the NPZ err keys when present.
    npz_ns, npz_trials = set(), set()
    for key in perpoint:
        if key.startswith("err|"):
            parts = key.split("|")
            if len(parts) == 5:
                npz_ns.add(int(parts[2]))
                npz_trials.add(int(parts[3]))
    if npz_ns:
        n_grid = sorted(npz_ns)
        trials = sorted(npz_trials)
    plot_err_vs_distance(perpoint, n_grid, trials, args.outdir)


if __name__ == "__main__":
    main()
