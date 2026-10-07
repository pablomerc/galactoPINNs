"""Paper figure: a fixed training budget overfits noisy data (chi-stop demo).

NFW baseline, full loss (LOS + observer anchor + rho >= 0), catalog-like mock
draws (fire_sims_v2, S(r)S(delta) pool, relative LOS), trial 0, trained on all
points. Paper figure: n = 50 alone; `--n 50 2000` overlays n = 2000 (dashed).
Trajectory columns per epoch: chi_fit, chi_val (dummy), rel_err sun4, rel_err sun15.
Writes accelerations-paper/figures/chi_stop_overfit.{pdf,png}.
Run from the repo root: uv run python clean_experiments/paper_figures/chi_stop_overfit.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:  # noqa: BLE001
    pass
plt.rcParams["figure.dpi"] = 150

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / "clean_experiments/fire_sims_v2"
RUNS = {50: (V2 / "results_nfw_LOSSunrho/trajectories.npz", "-"),
        2000: (V2 / "results_nfw_LOSSunrho_large/trajectories.npz", "--")}
OUT = ROOT.parent / "accelerations-paper/figures/chi_stop_overfit"
PAPER_PALETTE = ["#638ccc", "#c57c3c", "#ab62c0", "#72a555", "#ca5670"]
C_CHI, C_ERR = PAPER_PALETTE[0], PAPER_PALETTE[1]
C_INK, C_REF = "#1a1a1a", "#7f7f7f"
TRIAL, ARCH, N_EP = 0, "IV", 1500
CHI_STAR = np.sqrt(2 / np.pi)

import argparse  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[50], help="catalog sizes to overlay (paper figure: 50)")
ap.add_argument("--trial", type=int, default=TRIAL)
ap.add_argument("--preview", default=None, help="write only a PNG with this name next to the script")
args = ap.parse_args()
TRIAL = args.trial
RUNS = {n: (V2 / ("results_nfw_LOSSunrho_large" if n >= 1000 else "results_nfw_LOSSunrho") / "trajectories.npz",
            "--" if len(args.n) > 1 and n == max(args.n) else "-") for n in args.n}


def smooth(y, k=25):
    return np.convolve(y, np.ones(k) / k, mode="valid")


fig, ax = plt.subplots(figsize=(4.4, 3.2))
handles = []
for n, (path, ls) in RUNS.items():
    h = np.load(path)[f"full|{ARCH}|{n}|{TRIAL}"][:N_EP]
    ep = np.arange(1, h.shape[0] + 1)
    below = np.flatnonzero(h[:, 0] <= CHI_STAR)
    e_chi = int(below[0]) + 1 if below.size else None
    e_min = int(np.argmin(h[:, 2])) + 1
    for col, color, lab in ((0, C_CHI, "Training loss"), (2, C_ERR, "Validation loss")):
        ys = smooth(h[:, col])
        ax.plot(ep[12:12 + len(ys)], ys, color=color, ls=ls, lw=1.5)
        handles.append(Line2D([], [], color=color, ls=ls, lw=1.5,
                               label=rf"{lab}, $n = {n}$" if len(args.n) > 1 else lab))
    if e_chi is not None:
        ax.axvline(e_chi, color=C_INK, ls="-", lw=0.7)
        ax.text(e_chi + 25, 1.0, rf"$\chi$-stop (epoch {e_chi})", fontsize=7, color=C_INK, va="center")
    else:
        ax.text(1480, h[-1, 0] + 0.02, r"no crossing", fontsize=6.5, color=C_CHI,
                ha="right", va="bottom")
    print(f"n={n}: chi-stop {e_chi}, err-min epoch {e_min} ({h[e_min-1,2]:.3f}), "
          f"err@stop {h[e_chi-1,2] if e_chi else float('nan'):.3f}, err@{N_EP} {h[-1,2]:.3f}, chi@{N_EP} {h[-1,0]:.3f}")
ax.set_xlim(0, N_EP); ax.set_ylim(0, 1.1)
ax.grid(True, color="0.85", lw=0.5, zorder=0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("NFW baseline" + (rf", $n = {args.n[0]}$" if len(args.n) == 1 else ""), fontsize=9, loc="left")
ax.legend(handles=handles, frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.16, 0.5))
if args.preview:
    fig.savefig(HERE / f"{args.preview}.png", dpi=170, bbox_inches="tight")
    print("saved preview", HERE / f"{args.preview}.png")
else:
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(HERE / "chi_stop_overfit.png", dpi=170, bbox_inches="tight")  # preview next to the script
    print("saved", OUT.with_suffix(".pdf"))
