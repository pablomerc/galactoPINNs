#!/usr/bin/env python
"""Paper figures of the mock-catalog selection function.

Reads the catalog sample (catalog.py), the candidates (candidates.py) and the
selection (selection.py, count weights), redraws trial 0 exactly as build_pool.py
does (u = default_rng(0).uniform; S_r-only and joint samples share u, so the joint
sample is nested in the S_r one), and writes three figures:

    selection_function        S_r(r_sun) and S_delta(delta), with the catalog as rugs
    selection_sampling_6panel p(r_sun), p(l), p(b): catalog vs uniform draw (A) and vs
                              the S_r S_delta sample (B), S_r-only for reference
    selection_sky_2d          sky density p(l, b) of catalog, uniform, S_r, S_r S_delta

plus selection_stats.json with the numbers quoted in the paper text. Output goes to
figs/ here and, unless --no-paper, a PDF copy to accelerations-paper/figures/.

    uv run python clean_experiments/mock_catalog/make_figures.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mock_catalog.candidates import OBSERVER, load_candidates  # noqa: E402
from mock_catalog.catalog import load_catalog, radial_scale  # noqa: E402
from mock_catalog.selection import (BAND_EDGES, DEC_ARECIBO_HI, DEC_ARECIBO_LO, DEC_GBT,  # noqa: E402
                                    DEC_NANCAY, DEC_PARKES, SkySelection, acceptance,
                                    declination, radial_selection, sky_stats)

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:  # noqa: BLE001
    pass
plt.rcParams["figure.dpi"] = 150

PAPER_DIR = HERE.parents[2] / "accelerations-paper" / "figures"
# the paper palette (sampling_suite.PAPER_PALETTE): catalog blue, uniform draw orange,
# S_r green, S_r S_delta purple. Blue/purple and orange/green are close for colour-blind
# readers (blue/purple even for normal vision), so the references also differ in line
# style (uniform dotted, S_r dashed) and every panel is labelled.
C_CAT, C_UNI, C_JOINT, C_SR = "#638ccc", "#c57c3c", "#ab62c0", "#72a555"
C_INK, C_REF = "#1a1a1a", "#7f7f7f"
LS_UNI, LS_SR = ":", "--"


# ----------------------------------------------------------------------------- helpers
def wrap180(l_deg):
    return (np.asarray(l_deg, float) + 180.0) % 360.0 - 180.0


def kde_1d(sample, grid, period=None):
    """Gaussian KDE (Scott); periodic coordinates are tiled by +-period."""
    if period is None:
        return gaussian_kde(sample)(grid)
    h = gaussian_kde(sample).factor * sample.std(ddof=1)
    tiled = np.concatenate([sample - period, sample, sample + period])
    return 3.0 * gaussian_kde(tiled, bw_method=h / tiled.std(ddof=1))(grid)


CELL = 0.5
L_EDGES, B_EDGES = np.arange(-180, 180 + CELL / 2, CELL), np.arange(-90, 90 + CELL / 2, CELL)
LGRID, BGRID = np.meshgrid(0.5 * (L_EDGES[1:] + L_EDGES[:-1]), 0.5 * (B_EDGES[1:] + B_EDGES[:-1]))


def kde_lb(l, b):
    """Binned 2D KDE [deg^-2], periodic in l, reflected at the poles, Scott width per axis."""
    H, _, _ = np.histogram2d(l, b, bins=[L_EDGES, B_EDGES])
    h = len(l) ** (-1 / 6) * np.array([np.std(l, ddof=1), np.std(b, ddof=1)])
    p = gaussian_filter(H, sigma=h / CELL, mode=("wrap", "reflect"))
    return p.T / (p.sum() * CELL**2)


def mass_levels(p, fractions=(0.9, 0.5)):
    """Density thresholds whose super-level sets enclose the given mass fractions."""
    z = np.sort(p.ravel())[::-1]
    cum = np.cumsum(z) * CELL**2
    return [z[np.searchsorted(cum, f)] for f in fractions]


def save(fig, name, out_dir, paper_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    if paper_dir is not None:
        shutil.copy(out_dir / f"{name}.pdf", paper_dir / f"{name}.pdf")
    print(f"[fig] {name}")


# ----------------------------------------------------------------------------- figures
def fig_selection_function(cat, r_s, sels, out_dir, paper_dir):
    fig, (axR, axD) = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw=dict(wspace=0.22))

    rg = np.linspace(0, 5.2, 400)
    axR.plot(rg, radial_selection(rg, r_s), color=C_INK, lw=1.5,
             label=rf"$S_\odot(r_\odot)=e^{{-r_\odot/r_s}}$, $r_s={r_s:.2f}$ kpc")
    axR.plot(cat["d_kpc"], np.full(cat["d_kpc"].size, 0.02), "|", color=C_CAT, ms=9, mew=1.0,
             label=f"catalog distances ({len(cat['name'])})")
    axR.set_xlim(0, 5.2); axR.set_ylim(0, 1.05)
    axR.set_xlabel(r"heliocentric distance $r_\odot$ [kpc]"); axR.set_ylabel("acceptance probability")
    axR.legend(frameon=False, fontsize=7, loc="upper right")
    axR.set_title("(a) distance", fontsize=9, loc="left")

    dg = np.linspace(-90, 90, 3601)
    axD.plot(dg, sels["count"].S(dg), color=C_INK, lw=1.5, label=r"$S(\delta)$, $w_k=N_k/N_{\rm tot}$ (used)")
    axD.plot(dg, sels["rate"].S(dg), color=C_REF, lw=1.0, ls="--", label=r"$S(\delta)$, $w_k=N_k/\Omega_k$")
    dec_cat = declination(cat["l_deg"], cat["b_deg"])
    axD.plot(dec_cat, np.full(dec_cat.size, 0.02), "|", color=C_CAT, ms=9, mew=1.0, label="catalog declinations")
    for d, lab in [(DEC_GBT, "GBT"), (DEC_NANCAY, "Nançay"), (DEC_ARECIBO_LO, "Arecibo"),
                   (DEC_PARKES, "Parkes"), (DEC_ARECIBO_HI, "Arecibo")]:
        axD.axvline(d, color=C_REF, lw=0.5, ls=":")
        axD.text(d + 1.0, 0.08, lab, rotation=90, fontsize=6, color=C_INK, va="bottom", ha="left")
    axD.set_xlim(-90, 90); axD.set_ylim(0, 1.05); axD.set_xticks([-90, -45, 0, 45, 90])
    axD.set_xlabel(r"declination $\delta$ [deg]")
    axD.legend(frameon=False, fontsize=7, loc="upper left")
    axD.set_title("(b) declination", fontsize=9, loc="left")
    save(fig, "selection_function", out_dir, paper_dir)


def fig_six_panel(cat, s, r_s, r_cand, draw_idx, out_dir, paper_dir):
    """s: dict of samples, each (r, l, b): 'uniform', 'sr', 'joint'."""
    rg, lg, bg = np.linspace(0, r_cand + 0.2, 400), np.linspace(-180, 180, 721), np.linspace(-90, 90, 361)
    ref = [np.where(rg <= r_cand, 3 * rg**2 / r_cand**3, np.nan),           # uniform in the ball
           np.full_like(lg, 1 / 360), np.radians(1) * np.cos(np.radians(bg)) / 2]
    HIST = dict(density=True, histtype="stepfilled", alpha=0.35, lw=1.0)
    cat_rlb = (cat["d_kpc"], wrap180(cat["l_deg"]), cat["b_deg"])
    rows = [  # xlabel, ylabel, grid, bins, period, xticks, headroom
        (r"$r_\odot$ [kpc]", r"$p(r_\odot)$ [kpc$^{-1}$]", rg, np.linspace(0, r_cand, 18), None, None, 1.35),
        (r"Galactic longitude $\ell$ [deg]  (0 = Galactic centre)", r"$p(\ell)$ [deg$^{-1}$]", lg,
         np.linspace(-180, 180, 19), 360.0, [-180, -90, 0, 90, 180], 1.6),
        (r"Galactic latitude $b$ [deg]", r"$p(b)$ [deg$^{-1}$]", bg, np.linspace(-90, 90, 13), None,
         [-90, -45, 0, 45, 90], 1.25),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.0), sharey="row")
    n_cat = len(cat["name"])
    for k, ((xl, yl, grid, bins, period, xt, head), (axA, axB)) in enumerate(zip(rows, axes)):
        c, u, sr, j = cat_rlb[k], s["uniform"][k], s["sr"][k], s["joint"][k]
        p_c, p_u, p_sr, p_j = (kde_1d(v, grid, period) for v in (c, u, sr, j))
        axA.hist(c, bins=bins, color=C_CAT, edgecolor=C_CAT, **HIST)
        axA.hist(u, bins=bins, color=C_UNI, edgecolor=C_UNI, **HIST)
        axA.plot(grid, p_c, color=C_CAT, lw=1.6, label=f"pulsar catalog ({n_cat})")
        axA.plot(grid, p_u, color=C_UNI, lw=1.4, ls=LS_UNI, label="m12i old stars, uniform draw")
        axA.plot(grid, ref[k], color=C_REF, lw=0.9, ls="--", label=f"isotropic, uniform in {r_cand:g} kpc")
        axB.hist(c, bins=bins, color=C_CAT, edgecolor=C_CAT, **HIST)
        axB.hist(j, bins=bins, color=C_JOINT, edgecolor=C_JOINT, **HIST)
        axB.plot(grid, p_c, color=C_CAT, lw=1.6, label=f"pulsar catalog ({n_cat})")
        axB.plot(grid, p_j, color=C_JOINT, lw=1.6, label=r"m12i, $S_\odot(r_\odot)\,S(\delta)$")
        axB.plot(grid, p_sr, color=C_SR, lw=1.3, ls=LS_SR, label=r"m12i, $S_\odot(r_\odot)$ only")
        for ax in (axA, axB):
            ax.set_xlim(grid[0], grid[-1]); ax.set_xlabel(xl)
            if xt is not None:
                ax.set_xticks(xt)
        axA.set_ylabel(yl)
        axA.set_ylim(0, head * np.nanmax([p_c.max(), p_u.max(), p_sr.max(), p_j.max()]))
    axes[0, 1].plot(s["joint"][0][draw_idx], np.zeros(len(draw_idx)), "|", color=C_JOINT, ms=8, mew=1.0,
                    label=rf"one $n={len(draw_idx)}$ draw")
    far = cat["d_kpc"] > rg[-1]
    if far.any():
        axes[0, 0].text(0.98, 0.55, f"{far.sum()} catalog pulsars beyond {rg[-1]:.1f} kpc",
                        transform=axes[0, 0].transAxes, ha="right", fontsize=7, color=C_INK)
    # quadrant fractions, the statistic the on-sky selection is meant to move
    st = {k: sky_stats(v[1], v[2]) for k, v in s.items()}
    st_c = sky_stats(cat_rlb[1], cat_rlb[2])
    axes[1, 1].text(0.03, 0.96,
                    "fraction in $0<\\ell<90^\\circ$ / $-90<\\ell<0^\\circ$\n"
                    f"catalog  {st_c['frac_Q1']:.2f} / {st_c['frac_Q4']:.2f}\n"
                    f"$S_\\odot S(\\delta)$  {st['joint']['frac_Q1']:.2f} / {st['joint']['frac_Q4']:.2f}\n"
                    f"$S_\\odot$ only  {st['sr']['frac_Q1']:.2f} / {st['sr']['frac_Q4']:.2f}",
                    transform=axes[1, 1].transAxes, ha="left", va="top", fontsize=6.5, color=C_INK,
                    linespacing=1.35)
    axes[0, 0].set_title("(A)  m12i old stars, uniform draw", fontsize=9)
    axes[0, 1].set_title(r"(B)  after rejection sampling with $S_\odot(r_\odot)\,S(\delta)$", fontsize=9)
    axes[0, 0].legend(frameon=True, fontsize=7, loc="upper right")
    axes[0, 1].legend(frameon=True, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "selection_sampling_6panel", out_dir, paper_dir)


def fig_sky_2d(cat, s, out_dir, paper_dir):
    cat_l, cat_b = wrap180(cat["l_deg"]), cat["b_deg"]
    samples = [  # label, l, b, colour, overlay line style, filled 50% region in the overlay
        (f"pulsar catalog ({len(cat['name'])})", cat_l, cat_b, C_CAT, "-", True),
        ("m12i uniform draw", s["uniform"][1], s["uniform"][2], C_UNI, LS_UNI, False),
        (r"m12i, $S_\odot(r_\odot)$", s["sr"][1], s["sr"][2], C_SR, LS_SR, False),
        (r"m12i, $S_\odot(r_\odot)\,S(\delta)$", s["joint"][1], s["joint"][2], C_JOINT, "-", True),
    ]
    fig = plt.figure(figsize=(8.4, 6.2))
    gs = fig.add_gridspec(2, len(samples), height_ratios=[2.5, 1.0], hspace=0.32, wspace=0.18)
    axT = fig.add_subplot(gs[0, :])
    axB = [fig.add_subplot(gs[1, j]) for j in range(len(samples))]
    for f in (0.5, 0.9):                                  # isotropic reference latitude bands
        for sgn in (1, -1):
            axT.axhline(sgn * np.degrees(np.arcsin(f)), color=C_REF, ls="--", lw=0.8)
    for ax, (label, l, b, col, ls, fill) in zip(axB, samples):
        p = kde_lb(l, b)
        lv90, lv50 = mass_levels(p)
        ax.imshow(p, origin="lower", extent=(-180, 180, -90, 90), aspect="auto", vmin=0,
                  cmap=LinearSegmentedColormap.from_list("w2c", ["white", col]), interpolation="bilinear")
        ax.contour(LGRID, BGRID, p, levels=[lv90, lv50], colors=C_INK, linewidths=[0.5, 0.9])
        ax.plot(cat_l, cat_b, ".", color=C_INK, ms=2.0)
        ax.set_title(label, fontsize=7, pad=3)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_xlabel(r"$\ell$ [deg]")
        if fill:
            axT.contourf(LGRID, BGRID, p, levels=[lv50, p.max()], colors=[col], alpha=0.22)
        axT.contour(LGRID, BGRID, p, levels=[lv50], colors=[col], linewidths=1.6,
                    linestyles=[ls])
        axT.contour(LGRID, BGRID, p, levels=[lv90], colors=[col], linewidths=0.8,
                    linestyles=[ls])
    axT.plot(cat_l, cat_b, "o", color=C_CAT, ms=3.5, mec="w", mew=0.5, zorder=5)
    handles = [Line2D([], [], color=c, ls=ls, lw=1.6) for _, _, _, c, ls, fill in samples]
    handles.append(Line2D([], [], color=C_REF, ls="--", lw=0.8))
    labels = [x[0] for x in samples] + [r"isotropic: $|b|<30^\circ$ (50%), $64^\circ$ (90%)"]
    axT.legend(handles, labels, frameon=False, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, 1.01), ncol=3)
    axT.text(0.02, 0.96, "thick: 50% of the sample\nthin: 90%", transform=axT.transAxes,
             ha="left", va="top", fontsize=7, linespacing=1.4, color=C_INK)
    axT.set_xlabel(r"Galactic longitude $\ell$ [deg]  (0 = Galactic centre)")
    axT.set_ylabel(r"Galactic latitude $b$ [deg]")
    axT.set_xlim(-180, 180); axT.set_ylim(-90, 90)
    axT.set_xticks([-180, -90, 0, 90, 180]); axT.set_yticks([-90, -45, 0, 45, 90])
    axB[0].set_ylabel(r"$b$ [deg]")
    for ax in axB[1:]:
        ax.tick_params(labelleft=False)
    save(fig, "selection_sky_2d", out_dir, paper_dir)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r-cand", type=float, default=5.0)
    ap.add_argument("--age-min", type=float, default=1.0)
    ap.add_argument("--n-uniform", type=int, default=30000, help="size of the uniform draw in column A")
    ap.add_argument("--n-draw", type=int, default=50, help="example draw marked on the r panel")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figs")
    ap.add_argument("--no-paper", action="store_true", help="do not copy PDFs to accelerations-paper/figures")
    args = ap.parse_args()
    paper_dir = None if args.no_paper or not PAPER_DIR.exists() else PAPER_DIR

    cat = load_catalog()
    r_s = radial_scale(cat)
    sels = {k: SkySelection.from_catalog(cat, k) for k in SkySelection.KINDS}
    c = load_candidates(r_cand=args.r_cand, age_min=args.age_min)
    acc_r = acceptance(c["x"], OBSERVER, r_s, None)
    acc_j = acceptance(c["x"], OBSERVER, r_s, sels["count"])

    # trial 0, drawn as build_pool.py draws it; the stats below average all 6 trials
    def trial(t):
        u = np.random.default_rng(t).uniform(size=c["x"].shape[0])
        return u < acc_r["p"], u < acc_j["p"]
    keep_r, keep_j = trial(0)
    uni = np.random.default_rng(0).choice(c["x"].shape[0], size=args.n_uniform, replace=False)
    rlb = lambda a, m: (a["r"][m], a["l"][m], a["b"][m])  # noqa: E731
    s = dict(uniform=rlb(acc_r, uni), sr=rlb(acc_r, keep_r), joint=rlb(acc_j, keep_j))
    draw_idx = np.random.default_rng(0).choice(keep_j.sum(), size=args.n_draw, replace=False)

    fig_selection_function(cat, r_s, sels, args.out_dir, paper_dir)
    fig_six_panel(cat, s, r_s, args.r_cand, draw_idx, args.out_dir, paper_dir)
    fig_sky_2d(cat, s, args.out_dir, paper_dir)

    # numbers for the text: catalog, and mean +- std over 6 trials for each sample
    def summarise(rows):
        keys = ("frac_Q1", "frac_Q4", "frac_abs_l_lt_90", "median_abs_b", "n")
        return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in keys}
    per = {"sr": [], "joint": []}
    for t in range(6):
        kr, kj = trial(t)
        per["sr"].append(sky_stats(acc_r["l"][kr], acc_r["b"][kr]))
        per["joint"].append(sky_stats(acc_j["l"][kj], acc_j["b"][kj]))
    stats = dict(
        catalog=dict(excluded=cat["excluded"], r_s_kpc=r_s,
                     median_abs_a_los_mmsyr=float(np.median(np.abs(cat["a_los_mmsyr"]))),
                     **sky_stats(cat["l_deg"], cat["b_deg"])),
        N_k=sels["count"].N_k, band_edges_deg=BAND_EDGES.tolist(),
        S_band=dict(count=sels["count"].band_S().tolist(), rate=sels["rate"].band_S().tolist()),
        weights=dict(count=sels["count"].weights(), rate=sels["rate"].weights()),
        n_candidates=int(c["x"].shape[0]),
        uniform=sky_stats(s["uniform"][1], s["uniform"][2]),
        sr_6trials=summarise(per["sr"]), joint_6trials=summarise(per["joint"]),
    )
    (args.out_dir / "selection_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[stats] {args.out_dir / 'selection_stats.json'}")
    if paper_dir is not None:
        print(f"[paper] PDFs copied to {paper_dir}")


if __name__ == "__main__":
    main()
