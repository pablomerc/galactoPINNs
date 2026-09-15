"""Accessible sky of each timing program behind the Donlon+2025 catalog, one panel per program.

A timing program points at individual pulsars, so it has no survey footprint; what it has is the
sky its telescopes can reach, i.e. a declination band set by each telescope's latitude.  Each
panel shades that band (union over the program's telescopes) and overlays the program's own
pulsars; the rest of the sample is shown in gray for reference.  Galactic Aitoff projection with
l increasing to the left and the Galactic centre in the middle, as in coverage_map.py.
Writes accessible_sky.png here and figures/sky_accessible_by_program.pdf in the paper repo.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from astropy.coordinates import SkyCoord
import astropy.units as u
from programs import program, PROGRAM_ORDER

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[3] / "accelerations-paper"

# declination bands reachable by each program's telescopes (deg).  Limits are the usual
# operational ones: GBT > -46 (lat +38.4), Arecibo -1..+38 (lat +18.3, zenith-angle limit 20 deg),
# Nancay > -39 (sets the EPTA union; Effelsberg/Lovell/Westerbork stop at -30..-35, Sardinia can
# reach -45 but contributed little to DR2), Parkes < +27 (lat -33, zenith-angle limit 60 deg).
ACCESS = {
    "NANOGrav": [("Green Bank", -46, 90, 0.16), ("Arecibo", -1, 38, 0.34)],
    "EPTA":     [("Nançay / Effelsberg / Lovell / Westerbork", -39, 90, 0.16)],
    "PPTA":     [("Parkes", -90, 27, 0.16)],
    "Other":    [],
}
LIMIT_TEXT = {"NANOGrav": r"light: Green Bank $\delta>-46°$;  dark: also Arecibo $-1°<\delta<+38°$ (more sensitive)",
              "EPTA": r"$\delta>-39°$ (Nançay; Effelsberg, Lovell, WSRT $\gtrsim-30°$)",
              "PPTA": r"Parkes $\delta<+27°$",
              "Other": "individual studies, various telescopes (not shaded)"}
PAL = {"NANOGrav": "#0072B2", "EPTA": "#009E73", "PPTA": "#D55E00", "Other": "#E69F00"}
INK, MUTED, GRID, SURF, CTX = "#3b3934", "#6b6860", "#d8d5ce", "#fcfcfb", "#b9b6ae"

df = pd.read_csv(HERE / "donlon52_refs.csv")
df["program"] = df.tag.map(program)
df["x"] = -np.radians((df.GL + 180) % 360 - 180)            # x = -l: l increases to the left
df["y"] = np.radians(df.GB)

xg, yg = np.linspace(-np.pi, np.pi, 721), np.linspace(-np.pi / 2, np.pi / 2, 361)
X, Y = np.meshgrid(xg, yg)
DEC = SkyCoord(l=-np.degrees(X) * u.deg, b=np.degrees(Y) * u.deg, frame="galactic").icrs.dec.deg

def dec_curve(ax, dec0, **kw):
    g = SkyCoord(ra=np.linspace(0, 360, 1441) * u.deg, dec=dec0 * u.deg, frame="icrs").galactic
    x, y = -np.radians((g.l.deg + 180) % 360 - 180), np.radians(g.b.deg)
    jump = np.where(np.abs(np.diff(x)) > np.pi / 2)[0] + 1
    ax.plot(np.insert(x, jump, np.nan), np.insert(y, jump, np.nan), **kw)

fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), subplot_kw=dict(projection="aitoff"))
for ax, prog in zip(axes.flat, PROGRAM_ORDER):
    col = PAL[prog]
    ax.set_facecolor(SURF)
    for name, lo, hi, alpha in ACCESS[prog]:
        ax.contourf(X, Y, ((DEC > lo) & (DEC < hi)).astype(int), levels=[0.5, 1.5],
                    colors=[to_rgba(col, alpha)], zorder=0)
        for lim in (lo, hi):
            if -90 < lim < 90: dec_curve(ax, lim, color=col, lw=0.9, ls="--", alpha=0.8, zorder=1)
    others = df.program != prog
    ax.scatter(df.x[others], df.y[others], s=14, color=CTX, edgecolor="none", zorder=2)
    mine = ~others
    ax.scatter(df.x[mine], df.y[mine], s=42, color=col, edgecolor=SURF, linewidth=1.0, zorder=3)
    n, n_q1 = int(mine.sum()), int((mine & (df.GL < 90)).sum())
    n_q4 = int((mine & (df.GL >= 270)).sum())
    ax.set_title(f"{prog}  ({n} pulsars; {n_q1} in Q1, {n_q4} in Q4)", fontsize=9.5, color=INK, pad=18)
    ax.text(0.5, 1.055, LIMIT_TEXT[prog], transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color=MUTED)
    ax.text(0, np.radians(-7), "GC", ha="center", va="top", fontsize=7, color=MUTED, zorder=4)
    ax.set_xticks(np.radians(np.arange(-120, 121, 60)))
    ax.set_xticklabels([f"{-t}°" if t else "" for t in range(-120, 121, 60)], fontsize=6.5, color=MUTED)
    ax.set_yticks(np.radians(np.arange(-60, 61, 30)))
    ax.set_yticklabels([f"{t}°" if t else "" for t in range(-60, 61, 30)], fontsize=6.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.4, zorder=0.8)

handles = [Patch(facecolor=to_rgba(MUTED, 0.2), edgecolor=to_rgba(MUTED, 0.8), ls="--", label="accessible sky of the program's telescopes"),
           Patch(facecolor=to_rgba(MUTED, 0.45), edgecolor=to_rgba(MUTED, 0.8), ls="--", label="also within reach of Arecibo, the most sensitive dish (NANOGrav only)"),
           Line2D([], [], marker="o", ls="", color=MUTED, markeredgecolor=SURF, markersize=7, label="pulsars whose acceleration this program measured"),
           Line2D([], [], marker="o", ls="", color=CTX, markersize=4, label="rest of the 52-pulsar sample")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=2.5)
fig.text(0.5, 0.098, r"Galactic longitude $\ell$ increases to the left; Galactic centre at the middle of each panel. "
         r"Q1: $\ell\in(0°,90°)$, Q4: $\ell\in(270°,360°)$.", ha="center", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.14, hspace=0.42, wspace=0.08)
fig.savefig(HERE / "accessible_sky.png", dpi=200)
fig.savefig(PAPER / "figures" / "sky_accessible_by_program.pdf")
print("saved", HERE / "accessible_sky.png", "and", PAPER / "figures/sky_accessible_by_program.pdf")
