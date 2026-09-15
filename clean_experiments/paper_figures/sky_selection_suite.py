#!/usr/bin/env python
"""Joint distance + on-sky selection of FIRE mock pulsars, one simulation at a time.

Extends sampling_suite.py with the declination selection S(delta) of sky_selection.py:

    candidates  old stars (age > --age-min) within --r-cand of the observer (same cache as
                sampling_suite.py: cache/<sim>_oldstars_r<r-cand>_age<age-min>.npz)
    S(r)        keep star i with probability exp(-r_i / r_s)              (seed --seed)
    S(delta)    keep star i with probability S(delta_i), for each weighting in --weights,
                with ONE shared uniform draw (seed --seed + 1) so the samples are nested;
                delta_i is the J2000 declination of the star's (l, b) seen from the observer,
                i.e. the simulated galaxy is treated as the Milky Way observed from Earth

Outputs (per simulation)
    figs/<sim>_sky2d_<weights>.{pdf,png}     sky density p(l, b): catalog, uniform pool, S(r),
                                             S(r) S(delta); 50% / 90% mass contours
    results/<sim>_sky_selection_stats.json   acceptance, quadrant fractions, |l| < 90, median |b|,
                                             share per declination band, for every sample
    results/<sim>_sky_samples.npz            (l, b) of the mock samples, for sky_selection_summary.py

The simulation frame fixes |z| and the disk plane but not the sense of rotation or which side
is north; S(delta) is symmetric in neither l nor b, so --flip {l,b,lb} mirrors the mock before
computing delta to test that dependence (filenames get a _flip-<x> suffix).

Example
    python sky_selection_suite.py --sim m12f --fire-dir /data/fire/m12f_res7100 --catalog /data/pulsars/data.csv
    python sky_selection_suite.py --sim m12i --catalog /data/pulsars/data.csv        # cache already built
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:  # noqa: BLE001
    pass
plt.rcParams["figure.dpi"] = 150

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sampling_suite import OBSERVER, PAPER_PALETTE, candidates, heliocentric_lb, load_catalog  # noqa: E402
from sky_selection import BAND_EDGES, REFS_CSV, SkySelection, arecibo_strip, declination  # noqa: E402

C_CAT, C_SIM, C_SKY, C_SEL, C_CNT, C_REF = (PAPER_PALETTE[0], PAPER_PALETTE[1], PAPER_PALETTE[2],
                                            PAPER_PALETTE[3], PAPER_PALETTE[4], "0.45")
JOINT_COLOR = {"rate": C_SKY, "count": C_CNT}
JOINT_LABEL = {"rate": r"$S(r_\odot)\,S(\delta)$, $w_k = N_k/\Omega_k$",
               "count": r"$S(r_\odot)\,S(\delta)$, $w_k = N_k/N_{\rm tot}$"}

# ----------------------------------------------------------------------------- sky density
CELL = 0.5                                                   # deg, grid cell of the binned KDE
L_EDGES, B_EDGES = np.arange(-180, 180 + CELL / 2, CELL), np.arange(-90, 90 + CELL / 2, CELL)
LGRID, BGRID = np.meshgrid(0.5 * (L_EDGES[1:] + L_EDGES[:-1]), 0.5 * (B_EDGES[1:] + B_EDGES[:-1]))


def kde_lb(l, b):
    """Binned 2D KDE of an (l, b) sample [deg^-2], periodic in l, mirrored at the poles;
    Gaussian of Scott's-rule width per axis.  Returns (density on (BGRID, LGRID), bandwidths)."""
    H, _, _ = np.histogram2d(l, b, bins=[L_EDGES, B_EDGES])
    h = len(l) ** (-1 / 6) * np.array([np.std(l, ddof=1), np.std(b, ddof=1)])
    p = gaussian_filter(H, sigma=h / CELL, mode=("wrap", "reflect"))
    return p.T / (p.sum() * CELL**2), h


def mass_levels(p, fractions=(0.9, 0.5)):
    """Density thresholds whose super-level sets enclose the given mass fractions."""
    z = np.sort(p.ravel())[::-1]
    cum = np.cumsum(z) * CELL**2
    return [z[np.searchsorted(cum, f)] for f in fractions]


def draw_density_panel(ax, l, b, color, cat_lb=None, title=None):
    """One filled sky-density panel with black 50% / 90% outlines and the catalog as dots."""
    p, _ = kde_lb(l, b)
    lv90, lv50 = mass_levels(p)
    ax.imshow(p, origin="lower", extent=(-180, 180, -90, 90), aspect="auto", vmin=0,
              cmap=LinearSegmentedColormap.from_list("w2c", ["white", color]), interpolation="bilinear")
    ax.contour(LGRID, BGRID, p, levels=[lv90, lv50], colors="k", linewidths=[0.5, 0.9])
    if cat_lb is not None:
        ax.plot(cat_lb[0], cat_lb[1], ".", color="k", ms=2.0)
    if title:
        ax.set_title(title, fontsize=7, pad=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-90, -45, 0, 45, 90])
    return p, (lv90, lv50)


def sky_density_figure(samples, cat_lb, title, out):
    """The paper's p(l, b) figure: overlay of 50% (filled, thick) and 90% (thin) contours on top,
    one panel per sample below.  `samples` = [(label, l, b, color), ...]; the catalog goes first."""
    n = len(samples)
    fig = plt.figure(figsize=(8.4, 6.2))
    gs = fig.add_gridspec(2, n, height_ratios=[2.5, 1.0], hspace=0.32, wspace=0.18)
    axT = fig.add_subplot(gs[0, :])
    axB = [fig.add_subplot(gs[1, j]) for j in range(n)]
    for f in (0.5, 0.9):                                     # isotropic reference bands
        for sgn in (1, -1):
            axT.axhline(sgn * np.degrees(np.arcsin(f)), color=C_REF, ls="--", lw=1.0)
    for ax, (label, l, b, c) in zip(axB, samples):
        p, (lv90, lv50) = draw_density_panel(ax, l, b, c, cat_lb, label)
        axT.contourf(LGRID, BGRID, p, levels=[lv50, p.max()], colors=[c], alpha=0.25)
        axT.contour(LGRID, BGRID, p, levels=[lv50], colors=[c], linewidths=1.8)
        axT.contour(LGRID, BGRID, p, levels=[lv90], colors=[c], linewidths=0.9)
        ax.set_xlabel(r"$\ell$ [deg]")
    axT.plot(cat_lb[0], cat_lb[1], "o", color=C_CAT, ms=3.5, mec="w", mew=0.5, zorder=5)
    handles = [Line2D([], [], color=c, lw=1.8) for *_, c in samples] + [Line2D([], [], color=C_REF, ls="--", lw=1.0)]
    labels = [s[0] for s in samples] + [r"isotropic: $|b| < 30^\circ$ (50%), $64^\circ$ (90%)"]
    axT.legend(handles, labels, frameon=True, fontsize=7, loc="upper right")
    axT.text(0.02, 0.96, "filled / thick: 50% of the sample\nthin: 90%", transform=axT.transAxes,
             ha="left", va="top", fontsize=7, linespacing=1.4)
    axT.set_title(title, fontsize=9)
    axT.set_xlabel(r"Galactic longitude $\ell$ [deg]  (0 = Galactic centre)")
    axT.set_ylabel(r"Galactic latitude $b$ [deg]")
    axT.set_xlim(-180, 180); axT.set_ylim(-90, 90)
    axT.set_xticks([-180, -90, 0, 90, 180]); axT.set_yticks([-90, -45, 0, 45, 90])
    axB[0].set_ylabel(r"$b$ [deg]")
    for ax in axB[1:]:
        ax.tick_params(labelleft=False)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- statistics
def sky_stats(l, b):
    """Quadrant fractions, |l| < 90 fraction, median |b| and the share in each declination band."""
    dec = declination(l, b)
    q1, q4 = np.mean((l > 0) & (l < 90)), np.mean((l > -90) & (l < 0))
    return dict(n=int(len(l)), frac_Q1=float(q1), frac_Q4=float(q4), Q1_over_Q4=float(q1 / q4) if q4 > 0 else None,
                frac_abs_l_lt_90=float(np.mean(np.abs(l) < 90)), median_abs_b=float(np.median(np.abs(b))),
                frac_arecibo_strip=float(np.mean(arecibo_strip(dec))),
                band_share=[float(np.mean((dec >= lo) & (dec < hi))) for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])])


def flip_lb(l, b, flip):
    if "l" in flip:
        l = -l
    if "b" in flip:
        b = -b
    return l, b


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="label, e.g. m12i")
    ap.add_argument("--catalog", required=True, type=Path, help="Donlon+2025 data.csv (distances)")
    ap.add_argument("--refs", type=Path, default=REFS_CSV, help="per-pulsar program table (sky_coverage/donlon52_refs.csv)")
    ap.add_argument("--fire-dir", type=Path, help="simulation directory (read with gizmo_analysis); not needed if the cache exists")
    ap.add_argument("--particles", type=Path, help="a fire_truth.py 'prepare' table instead of a snapshot")
    ap.add_argument("--snapshot", type=int, default=600)
    ap.add_argument("--age-min", type=float, default=1.0, help="tracer age cut [Gyr]")
    ap.add_argument("--r-cand", type=float, default=5.0, help="candidate radius for the rejection [kpc]")
    ap.add_argument("--r-ball", type=float, default=4.0, help="radius of the uniform pool [kpc]")
    ap.add_argument("--n-pool", type=int, default=30000)
    ap.add_argument("--rs", type=float, default=None, help="scale of S(r) [kpc]; default: closed form mean(r_cat[r<10])/3")
    ap.add_argument("--rs-mle", action="store_true", help="use instead the per-simulation MLE of r_s (as sampling_suite.py 'mle')")
    ap.add_argument("--weights", nargs="+", default=["rate", "count"], choices=SkySelection.WEIGHTS)
    ap.add_argument("--flip", default="none", choices=["none", "l", "b", "lb"], help="mirror the mock in l and/or b before S(delta)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    suffix = "" if args.flip == "none" else f"_flip-{args.flip}"

    cat = load_catalog(args.catalog)
    sel = SkySelection.from_refs(args.refs)
    x_c = candidates(args)
    r_c = np.linalg.norm(x_c - OBSERVER, axis=1)

    # uniform pool in the ball (same rule and seed as sampling_suite.py)
    in_ball = np.where(r_c < args.r_ball)[0]
    pool_idx = np.random.default_rng(args.seed).choice(in_ball, size=min(args.n_pool, in_ball.size), replace=False)
    l_p, b_p = flip_lb(*heliocentric_lb(x_c[pool_idx]), args.flip)

    # S(r): closed-form r_s by default, per-simulation MLE on request
    if args.rs is not None:
        rs, rs_tag = args.rs, "given"
    elif args.rs_mle:
        r_fit = cat["r"][cat["r"] < args.r_cand]
        nll = lambda s: np.sum(r_fit / s) + len(r_fit) * np.log(np.sum(np.exp(-r_c / s)))  # noqa: E731
        rs, rs_tag = float(minimize_scalar(nll, bounds=(0.1, 3.0), method="bounded").x), "mle"
    else:
        rs, rs_tag = float(cat["r"][cat["r"] < 10].mean() / 3), "fixed"
    keep_r = np.random.default_rng(args.seed).uniform(size=r_c.size) < np.exp(-r_c / rs)
    l_s, b_s = flip_lb(*heliocentric_lb(x_c[keep_r]), args.flip)
    dec_s = declination(l_s, b_s)
    print(f"[{args.sim}] {r_c.size:,} candidates within {args.r_cand:g} kpc; S(r) with r_s = {rs:.3f} kpc ({rs_tag}) "
          f"keeps {keep_r.sum():,}; flip = {args.flip}")

    # S(delta): one shared draw for every weighting -> nested samples
    u_d = np.random.default_rng(args.seed + 1).uniform(size=dec_s.size)
    out = dict(sim=args.sim, r_cand=args.r_cand, r_ball=args.r_ball, age_min=args.age_min, flip=args.flip,
               n_candidates=int(r_c.size), rs=dict(value=rs, kind=rs_tag), seed=args.seed,
               N_k=sel.N_k, Omega_k=sel.Omega_k,
               catalog=sky_stats(cat["l"], cat["b"]), pool=sky_stats(l_p, b_p),
               S_r=dict(accepted=int(keep_r.sum()), **sky_stats(l_s, b_s)), joint={})
    samples = {"pool_l": l_p, "pool_b": b_p, "S_r_l": l_s, "S_r_b": b_s}
    for w in args.weights:
        keep_d = u_d < sel.S(dec_s, w)
        l_j, b_j = l_s[keep_d], b_s[keep_d]
        out["joint"][w] = dict(accepted=int(keep_d.sum()), acceptance_factor=float(1 / keep_d.mean()),
                               weights=sel.weights(w, normalised=True), **sky_stats(l_j, b_j))
        samples[f"joint_{w}_l"], samples[f"joint_{w}_b"] = l_j, b_j
        sky_density_figure(
            [(f"Pulsar catalog ({cat['n']})", cat["l"], cat["b"], C_CAT),
             (f"{args.sim} pool, uniform draw", l_p, b_p, C_SIM),
             (rf"{args.sim} stars, rejection $S(r_\odot)$", l_s, b_s, C_SEL),
             (f"{args.sim} stars, rejection " + JOINT_LABEL[w], l_j, b_j, JOINT_COLOR[w])],
            (cat["l"], cat["b"]),
            f"{args.sim}: joint selection with {w} weights" + (f", mock mirrored in {args.flip}" if suffix else ""),
            HERE / "figs" / f"{args.sim}_sky2d_{w}{suffix}")
        s = out["joint"][w]
        print(f"[{args.sim}] {w:5s}: keeps {keep_d.sum():,} of {keep_r.sum():,} (x{1/keep_d.mean():.2f}); "
              f"Q1 {s['frac_Q1']:.2f} Q4 {s['frac_Q4']:.2f} (S(r) {out['S_r']['frac_Q1']:.2f}/{out['S_r']['frac_Q4']:.2f}, "
              f"catalog {out['catalog']['frac_Q1']:.2f}/{out['catalog']['frac_Q4']:.2f}); "
              f"Arecibo strip {s['frac_arecibo_strip']:.2f} (catalog {out['catalog']['frac_arecibo_strip']:.2f})")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.sim}_sky_selection_stats{suffix}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(HERE / "results" / f"{args.sim}_sky_samples{suffix}.npz",
                        **{k: v.astype(np.float32) for k, v in samples.items()}, cat_l=cat["l"], cat_b=cat["b"])
    print(f"[{args.sim}] wrote figs/{args.sim}_sky2d_*{suffix}.pdf, results/{args.sim}_sky_selection_stats{suffix}.json, "
          f"results/{args.sim}_sky_samples{suffix}.npz")


if __name__ == "__main__":
    main()
