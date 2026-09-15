#!/usr/bin/env python
"""Suite overview of the sky selection: one row per simulation, one column per mock sample.

Reads results/<sim>_sky_samples.npz and results/<sim>_sky_selection_stats.json written by
suite.py and draws the sky density p(l, b) of the S(r) sample and of the joint
S(r) S(delta) samples (rate and/or count weights), with the 52 catalog pulsars as dots and
black 50% / 90% mass contours.  Also prints the sky statistics of every sample against the
catalog.

    python summary.py                       # every simulation with results
    python summary.py --sims m12i m12f --weights rate

Output: figs/suite_sky2d[_<suffix>].{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
sys.path.insert(0, str(HERE))
from suite import C_CAT, C_SEL, JOINT_COLOR, draw_density_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", nargs="*", default=None, help="default: every results/<sim>_sky_samples.npz")
    ap.add_argument("--weights", nargs="+", default=["rate", "count"], choices=["rate", "count"])
    ap.add_argument("--suffix", default="", help="e.g. _flip-l to summarise a mirrored run")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = HERE / "results"
    sims = args.sims or sorted(p.name.split("_sky_samples")[0] for p in res.glob(f"*_sky_samples{args.suffix}.npz"))
    if not sims:
        raise SystemExit("no results/<sim>_sky_samples.npz found; run suite.py first")
    cols = [("S_r", r"$S(r_\odot)$", C_SEL)] + [(f"joint_{w}", rf"$S(r_\odot)\,S(\delta)$, {w}", JOINT_COLOR[w]) for w in args.weights]

    fig, axes = plt.subplots(len(sims), len(cols), figsize=(2.9 * len(cols), 1.75 * len(sims) + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    header = f"{'sim':6s}{'sample':14s}{'n':>7s}{'Q1':>6s}{'Q4':>6s}{'Q1/Q4':>7s}{'|l|<90':>7s}{'med|b|':>7s}{'Arecibo':>8s}"
    print(header)
    for i, sim in enumerate(sims):
        d = np.load(res / f"{sim}_sky_samples{args.suffix}.npz")
        st = json.loads((res / f"{sim}_sky_selection_stats{args.suffix}.json").read_text())
        cat_lb = (d["cat_l"], d["cat_b"])
        if i == 0:
            c = st["catalog"]
            print(f"{'':6s}{'catalog':14s}{c['n']:7d}{c['frac_Q1']:6.2f}{c['frac_Q4']:6.2f}{c['Q1_over_Q4']:7.2f}"
                  f"{c['frac_abs_l_lt_90']:7.2f}{c['median_abs_b']:7.1f}{c['frac_arecibo_strip']:8.2f}")
        for j, (key, label, color) in enumerate(cols):
            if f"{key}_l" not in d:
                axes[i, j].set_axis_off(); continue
            s = st["S_r"] if key == "S_r" else st["joint"][key.split("_", 1)[1]]
            draw_density_panel(axes[i, j], d[f"{key}_l"], d[f"{key}_b"], color, cat_lb,
                               title=f"{label}   Q1/Q4 = {s['Q1_over_Q4']:.2f}")
            print(f"{sim if j == 0 else '':6s}{key:14s}{s['n']:7d}{s['frac_Q1']:6.2f}{s['frac_Q4']:6.2f}{s['Q1_over_Q4']:7.2f}"
                  f"{s['frac_abs_l_lt_90']:7.2f}{s['median_abs_b']:7.1f}{s['frac_arecibo_strip']:8.2f}")
        axes[i, 0].set_ylabel(f"{sim}\n" + r"$b$ [deg]")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\ell$ [deg]  (0 = GC)")
    fig.suptitle(f"catalog Q1/Q4 = {st['catalog']['Q1_over_Q4']:.2f}; dots: the {st['catalog']['n']} catalog pulsars; "
                 "black: 50% / 90% of each sample", fontsize=8, y=1.0)
    fig.tight_layout()
    out = args.out or HERE / "figs" / f"suite_sky2d{args.suffix}"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {out}.pdf/.png for {sims}")


if __name__ == "__main__":
    main()
