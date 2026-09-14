#!/usr/bin/env python
"""Catalog-like sampling of FIRE mock pulsars, for one simulation at a time.

Reproduces the 6-panel sampling figure of the paper (column A: catalog vs the uniform draw
of old stars in the 4 kpc ball; column B: catalog vs the rejection-sampled stars) for any
FIRE-2 m12 galaxy, with two acceptance scales for S(r) = exp(-r / r_s):

  fixed : r_s = mean(r_cat[r < 10 kpc]) / 3      closed form for stars uniform in volume
                                                 (Gamma(3) law); 0.50 kpc for Donlon+2025
  mle   : r_s that maximises the catalog likelihood under n_sim(r) S(r), where n_sim is this
          simulation's own old-star density (candidates within --r-cand of the observer)

Input, one of
  --fire-dir  <sim dir>   read snapshot --snapshot (default 600, z = 0) with gizmo_analysis;
                          stars only, principal-axis frame ('host.distance.principal') as in
                          fire_sims/fire_truth.py
  --particles <npz>       a fire_truth.py 'prepare' table (pos, star_age, star_offset)
Old stars (age > --age-min) within --r-cand are cached to cache/<sim>_oldstars.npz.

Outputs (per simulation)
  figs/<sim>_sampling6_rs-fixed.{pdf,png}, figs/<sim>_sampling6_rs-mle.{pdf,png}
  results/<sim>_sampling_stats.json        r_s values, acceptance counts, summary statistics

Example
  python sampling_suite.py --sim m12f --fire-dir /data/fire/m12f_res7100 --catalog /data/pulsars/data.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:  # noqa: BLE001
    pass
plt.rcParams["figure.dpi"] = 150

HERE = Path(__file__).resolve().parent
OBSERVER = np.array([-8.1, 0.0, 0.0])          # Sun, principal-axis frame, kpc (fire_truth.py)
PAPER_PALETTE = ["#638ccc", "#c57c3c", "#ab62c0", "#72a555", "#ca5670"]
C_CAT, C_SIM, C_SEL, C_REF = PAPER_PALETTE[0], PAPER_PALETTE[1], PAPER_PALETTE[3], "0.45"


# ----------------------------------------------------------------------------- geometry
def heliocentric_lb(x, observer=OBSERVER):
    """Galactocentric Cartesian -> Galactic (l, b) [deg] seen from `observer`; l in (-180, 180]."""
    d = x - observer
    l = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    b = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
    return l, b


def galactocentric(l_deg, b_deg, d, observer=OBSERVER):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.stack([observer[0] + d * np.cos(b) * np.cos(l),
                     d * np.cos(b) * np.sin(l), d * np.sin(b)], axis=1)


def wrap180(l_deg):
    return (l_deg + 180.0) % 360.0 - 180.0


def periodic_kde(sample, grid, period=360.0):
    """KDE on a periodic coordinate: tile by +-period, Scott bandwidth from the untiled sample."""
    h = gaussian_kde(sample).factor * sample.std(ddof=1)
    tiled = np.concatenate([sample - period, sample, sample + period])
    return 3.0 * gaussian_kde(tiled, bw_method=h / tiled.std(ddof=1))(grid)


def kde(sample, grid, period=None):
    return gaussian_kde(sample)(grid) if period is None else periodic_kde(sample, grid, period)


# ----------------------------------------------------------------------------- inputs
def load_catalog(path):
    import pandas as pd
    df = pd.read_csv(path, skiprows=[1])                       # row 1 = units
    df = df[df["NAME"] != "J0737-3039B"].reset_index(drop=True)  # same binary as A
    r = df["DIST"].to_numpy(float)
    l, b = wrap180(df["GL"].to_numpy(float)), df["GB"].to_numpy(float)
    z = galactocentric(l, b, r)[:, 2]
    return dict(r=r, l=l, b=b, z=z, n=len(df))


def old_stars_from_particles(path, age_min):
    tab = np.load(path)
    s0, s1 = tab["star_offset"]
    x, age = tab["pos"][s0:s1], tab["star_age"]
    return x[age > age_min].astype(np.float64)


def old_stars_from_snapshot(fire_dir, snapshot, age_min):
    import gizmo_analysis as gizmo
    part = gizmo.io.Read.read_snapshots(
        species=["star"], snapshot_value_kind="index", snapshot_values=snapshot,
        simulation_directory=str(fire_dir),
        properties=["position", "velocity", "mass", "form.scalefactor"],
        assign_hosts=True, assign_hosts_rotation=True)
    x = part["star"].prop("host.distance.principal")            # (N, 3) kpc, disk in x-y
    age = part["star"].prop("age")                              # Gyr
    return np.asarray(x[age > age_min], dtype=np.float64)


def candidates(args):
    """Old stars within r_cand of the observer, cached per simulation."""
    cache = HERE / "cache" / f"{args.sim}_oldstars_r{args.r_cand:g}_age{args.age_min:g}.npz"
    if cache.exists():
        return np.load(cache)["x"].astype(np.float64)
    t0 = time.time()
    if args.particles:
        x = old_stars_from_particles(args.particles, args.age_min)
    elif args.fire_dir:
        x = old_stars_from_snapshot(args.fire_dir, args.snapshot, args.age_min)
    else:
        raise SystemExit("give --fire-dir or --particles")
    x = x[np.linalg.norm(x - OBSERVER, axis=1) < args.r_cand]
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, x=x.astype(np.float32))
    print(f"[{args.sim}] {len(x):,} old stars within {args.r_cand:g} kpc "
          f"({time.time() - t0:.0f} s) -> {cache.name}")
    return x


# ----------------------------------------------------------------------------- figure
def six_panel(cat, pool, sel, rs, r_ball, sim, tag, idx_draw, out):
    r_sel, l_sel, b_sel = sel
    rg, lg, bg = np.linspace(0, 4.5, 400), np.linspace(-180, 180, 721), np.linspace(-90, 90, 361)
    p_uni = np.where(rg <= r_ball, 3 * rg**2 / r_ball**3, np.nan)   # uniform in the ball
    p_l_uni = np.full_like(lg, 1 / 360)
    p_b_uni = np.radians(1) * np.cos(np.radians(bg)) / 2
    HIST = dict(density=True, histtype="stepfilled", alpha=0.35, lw=1.2)
    rows = [  # xlabel, ylabel, catalog, pool, rejection, reference, grid, bins, period, xticks, headroom
        (r"$r_\odot$ [kpc]", r"$p(r_\odot)$ [kpc$^{-1}$]", cat["r"], pool["r"], r_sel, p_uni,
         rg, np.linspace(0, 4.5, 16), None, None, 1.5),
        (r"Galactic longitude $\ell$ [deg]  (0 = Galactic centre)", r"$p(\ell)$ [deg$^{-1}$]",
         cat["l"], pool["l"], l_sel, p_l_uni, lg, np.linspace(-180, 180, 19), 360.0, [-180, -90, 0, 90, 180], 1.2),
        (r"Galactic latitude $b$ [deg]", r"$p(b)$ [deg$^{-1}$]", cat["b"], pool["b"], b_sel, p_b_uni,
         bg, np.linspace(-90, 90, 13), None, [-90, -45, 0, 45, 90], 1.2),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.0), sharey="row")
    for (xl, yl, c, u, s, ref, grid, bn, period, xt, head), (axA, axB) in zip(rows, axes):
        p_c, p_u, p_s = kde(c, grid, period), kde(u, grid, period), kde(s, grid, period)
        axA.hist(c, bins=bn, color=C_CAT, edgecolor=C_CAT, **HIST)
        axA.hist(u, bins=bn, color=C_SIM, edgecolor=C_SIM, **HIST)
        axA.plot(grid, p_c, color=C_CAT, lw=1.8, label=f"Pulsar catalog ({cat['n']})")
        axA.plot(grid, p_u, color=C_SIM, lw=1.8, label=f"{sim} pool, uniform draw")
        axA.plot(grid, ref, color=C_REF, ls="--", lw=1.2, label=f"uniform in {r_ball:g} kpc ball")
        axB.hist(c, bins=bn, color=C_CAT, edgecolor=C_CAT, **HIST)
        axB.hist(s, bins=bn, color=C_SEL, edgecolor=C_SEL, **HIST)
        axB.plot(grid, p_c, color=C_CAT, lw=1.8, label=f"Pulsar catalog ({cat['n']})")
        axB.plot(grid, p_u, color=C_SIM, lw=1.0, label=f"{sim} pool, uniform draw")
        axB.plot(grid, p_s, color=C_SEL, lw=1.8, label=rf"{sim} stars, rejection $S(r_\odot)$")
        for ax in (axA, axB):
            ax.set_xlim(grid[0], grid[-1]); ax.set_xlabel(xl)
            if xt is not None: ax.set_xticks(xt)
        axA.set_ylabel(yl)
        axA.set_ylim(0, head * max(p_c.max(), p_u.max(), p_s.max()))
    far = cat["r"] > 4.5
    if far.any():
        axes[0, 0].annotate(f"{far.sum()} pulsars at\n{cat['r'][far].min():.1f}--{cat['r'][far].max():.0f} kpc "
                            r"$\rightarrow$", xy=(4.45, 0.28), ha="right", va="bottom", fontsize=7, color="k")
    S = lambda r: np.exp(-r / rs)  # noqa: E731
    axS = axes[0, 1].twinx()
    axS.plot(rg, S(rg), color=C_REF, ls="--", lw=1.2, label=r"$S(r_\odot) = e^{-r_\odot/r_s}$")
    axS.set_ylim(0, 1.05)
    axS.set_ylabel(rf"$S(r_\odot)$,  $r_s = {rs:.2f}$ kpc ({tag})", color=C_REF)
    axS.tick_params(axis="y", colors=C_REF)
    q = kde(pool["r"], rg) * S(rg)
    axes[0, 1].plot(rg, q / np.trapezoid(q, rg), color="k", ls=":", lw=0.8, label=r"pool $\times$ $S(r_\odot)$, normalised")
    axes[0, 1].plot(r_sel[idx_draw], np.zeros(len(idx_draw)), "|", color=C_SEL, ms=7, mew=1.0,
                    label=rf"one $n = {len(idx_draw)}$ draw")
    axes[0, 0].set_title(f"(A)  {sim} pool: uniform draw of old stars", fontsize=9)
    axes[0, 1].set_title(rf"(B)  after rejection sampling with $S(r_\odot)$, $r_s$ {tag}", fontsize=9)
    axes[0, 0].legend(frameon=True, fontsize=7, loc="upper right")
    hB, lB = axes[0, 1].get_legend_handles_labels(); hS, lS = axS.get_legend_handles_labels()
    axes[0, 1].legend(hB + hS, lB + lS, frameon=True, fontsize=7, loc="upper center")
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def stats(r, l, b, z):
    return dict(median_r=float(np.median(r)), frac_abs_l_lt_90=float(np.mean(np.abs(l) < 90)),
                median_abs_b=float(np.median(np.abs(b))), median_abs_z=float(np.median(np.abs(z))),
                p84_abs_z=float(np.percentile(np.abs(z), 84)), n=int(len(r)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="label, e.g. m12i")
    ap.add_argument("--catalog", required=True, type=Path, help="Donlon+2025 data.csv")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fire-dir", type=Path); src.add_argument("--particles", type=Path)
    ap.add_argument("--snapshot", type=int, default=600)
    ap.add_argument("--age-min", type=float, default=1.0, help="tracer age cut [Gyr]")
    ap.add_argument("--r-cand", type=float, default=5.0, help="candidate radius for the MLE and rejection [kpc]")
    ap.add_argument("--r-ball", type=float, default=4.0, help="radius of the uniform pool [kpc]")
    ap.add_argument("--n-pool", type=int, default=30000)
    ap.add_argument("--n-draw", type=int, default=50, help="size of the example draw shown on the r panel")
    ap.add_argument("--rs-fixed", type=float, default=None, help="override the closed-form r_s [kpc]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cat = load_catalog(args.catalog)
    x_c = candidates(args)
    r_c = np.linalg.norm(x_c - OBSERVER, axis=1)

    # uniform pool: N_POOL old stars in the ball, seed 0, same rule as fire_truth.py build
    in_ball = np.where(r_c < args.r_ball)[0]
    pool_idx = np.random.default_rng(args.seed).choice(in_ball, size=min(args.n_pool, in_ball.size), replace=False)
    x_p = x_c[pool_idx]
    l_p, b_p = heliocentric_lb(x_p)
    pool = dict(r=r_c[pool_idx], l=l_p, b=b_p, z=x_p[:, 2])

    # the two acceptance scales
    rs_fixed = args.rs_fixed if args.rs_fixed is not None else cat["r"][cat["r"] < 10].mean() / 3
    r_fit = cat["r"][cat["r"] < args.r_cand]
    nll = lambda rs: np.sum(r_fit / rs) + len(r_fit) * np.log(np.sum(np.exp(-r_c / rs)))  # noqa: E731
    rs_mle = float(minimize_scalar(nll, bounds=(0.1, 3.0), method="bounded").x)
    print(f"[{args.sim}] r_s fixed (closed form) = {rs_fixed:.3f} kpc | direct MLE on this sim = {rs_mle:.3f} kpc")

    # rejection sampling with shared uniforms -> nested samples; one seed = one trial
    rng = np.random.default_rng(args.seed)
    u = rng.uniform(size=r_c.size)
    out = dict(sim=args.sim, r_cand=args.r_cand, r_ball=args.r_ball, age_min=args.age_min, n_candidates=int(r_c.size),
               rs=dict(fixed=float(rs_fixed), mle=rs_mle), catalog=stats(cat["r"], cat["l"], cat["b"], cat["z"]),
               pool=stats(**pool), rejection={})
    for tag, rs in (("fixed", rs_fixed), ("mle", rs_mle)):
        keep = u < np.exp(-r_c / rs)
        x_s = x_c[keep]; r_s_ = r_c[keep]; l_s, b_s = heliocentric_lb(x_s)
        idx = rng.choice(len(r_s_), size=min(args.n_draw, len(r_s_)), replace=False)
        six_panel(cat, pool, (r_s_, l_s, b_s), rs, args.r_ball, args.sim, tag, idx,
                  HERE / "figs" / f"{args.sim}_sampling6_rs-{tag}")
        out["rejection"][tag] = dict(rs=float(rs), accepted=int(keep.sum()), **stats(r_s_, l_s, b_s, x_s[:, 2]))
        print(f"[{args.sim}] {tag:5s} r_s={rs:.3f}: accepted {keep.sum():,} of {r_c.size:,}; "
              f"median r {np.median(r_s_):.2f} (catalog {np.median(cat['r']):.2f}), "
              f"frac|l|<90 {np.mean(np.abs(l_s) < 90):.2f} ({np.mean(np.abs(cat['l']) < 90):.2f}), "
              f"median|b| {np.median(np.abs(b_s)):.1f} ({np.median(np.abs(cat['b'])):.1f})")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.sim}_sampling_stats.json").write_text(json.dumps(out, indent=2))
    print(f"[{args.sim}] wrote figs/{args.sim}_sampling6_rs-*.pdf and results/{args.sim}_sampling_stats.json")


if __name__ == "__main__":
    main()
