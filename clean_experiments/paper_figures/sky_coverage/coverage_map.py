"""Sky coverage of the Donlon+2025 52-pulsar acceleration catalog in Galactic (l, b).

Points: the 52 pulsars, colored by the timing program behind the acceleration channel used
(see programs.py / atnf_refs.py), marker by channel.  Background: declination bands showing
which hemisphere's telescopes can reach each part of the sky; hatched = Arecibo strip.
Writes coverage_map.png here and figures/sky_coverage_lb.pdf in the paper repo.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy.coordinates import SkyCoord
import astropy.units as u
from programs import program, PROGRAM_ORDER

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[3] / "accelerations-paper"

df = pd.read_csv(HERE / "donlon52_refs.csv")
df["program"] = df.tag.map(program)
df["l"] = (df.GL + 180) % 360 - 180
df["dec"] = SkyCoord(l=df.GL.to_numpy() * u.deg, b=df.GB.to_numpy() * u.deg, frame="galactic").icrs.dec.deg
l = df.l.to_numpy()
df["quadrant"] = np.select([(l >= 0) & (l < 90), (l >= -90) & (l < 0), l >= 90, l < -90],
                           ["Q1 (0-90)", "Q4 (270-360)", "Q2 (90-180)", "Q3 (180-270)"], default="")

pd.set_option("display.width", 200)
print("program x quadrant:\n", pd.crosstab(df.program, df.quadrant, margins=True).to_string(), "\n")
bands = pd.cut(df.dec, [-90, -46, -1, 38, 44, 90],
               labels=["<-46 (south only)", "-46..-1", "-1..38 (Arecibo)", "38..44", ">44 (north only)"])
print("program x declination band:\n", pd.crosstab(df.program, bands, margins=True).to_string(), "\n")

PAL = {"NANOGrav": "#0072B2", "PPTA": "#D55E00", "EPTA": "#009E73", "Other": "#E69F00"}   # Okabe-Ito, CVD-validated
MARK = {"PB": ("o", r"binary channel ($\dot P_b$)"), "PS": ("^", r"spin channel ($\dot P$)")}
INK, MUTED, GRID, SURF = "#3b3934", "#6b6860", "#d8d5ce", "#fcfcfb"

fig = plt.figure(figsize=(9.5, 5.6))
ax = fig.add_subplot(111, projection="aitoff")
ax.set_facecolor(SURF)

# declination-band background (x = -l so that l increases to the left, astronomical convention)
xg, yg = np.linspace(-np.pi, np.pi, 721), np.linspace(-np.pi / 2, np.pi / 2, 361)
X, Y = np.meshgrid(xg, yg)
DEC = SkyCoord(l=-np.degrees(X) * u.deg, b=np.degrees(Y) * u.deg, frame="galactic").icrs.dec.deg
ax.contourf(X, Y, ((DEC < -46) | (DEC > 44)).astype(int), levels=[-0.5, 0.5, 1.5], colors=[SURF, "#e6e4df"], zorder=0)
plt.rcParams["hatch.color"], plt.rcParams["hatch.linewidth"] = "#b9b6ae", 0.6
ax.contourf(X, Y, ((DEC > -1) & (DEC < 38)).astype(int), levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=0.5)

def dec_curve(dec0, **kw):
    g = SkyCoord(ra=np.linspace(0, 360, 1441) * u.deg, dec=dec0 * u.deg, frame="icrs").galactic
    x, y = -np.radians((g.l.deg + 180) % 360 - 180), np.radians(g.b.deg)
    jump = np.where(np.abs(np.diff(x)) > np.pi / 2)[0] + 1            # break the line at the l = +-180 wrap
    ax.plot(np.insert(x, jump, np.nan), np.insert(y, jump, np.nan), **kw)

dec_curve(0, color="#8a877f", lw=1.0, ls=":", zorder=1)
for d in (-46, 44): dec_curve(d, color="#8a877f", lw=1.0, ls="--", zorder=1)

for prog in PROGRAM_ORDER:
    for ch, (mk, _) in MARK.items():
        m = (df.program == prog) & (df.channel == ch)
        if m.any():
            ax.scatter(-np.radians(df.l[m]), np.radians(df.GB[m]), s=48, marker=mk, color=PAL[prog],
                       edgecolor=SURF, linewidth=1.2, zorder=3)

corners = {"Q2 (90-180)": (0.01, 0.97, "left", "top"), "Q3 (180-270)": (0.99, 0.97, "right", "top"),
           "Q1 (0-90)": (0.01, 0.03, "left", "bottom"), "Q4 (270-360)": (0.99, 0.03, "right", "bottom")}
for q, (xq, yq, ha, va) in corners.items():
    n = int((df.quadrant == q).sum()); nn = int(((df.quadrant == q) & df.program.isin(["NANOGrav", "EPTA"])).sum())
    lo, hi = q.split("(")[1].rstrip(")").split("-")
    ax.text(xq, yq, f"$\\ell$ = {lo}°–{hi}°\n{n} pulsars, {nn} NANOGrav+EPTA", transform=ax.transAxes, ha=ha, va=va,
            fontsize=8.5, color=INK, zorder=4, bbox=dict(boxstyle="round,pad=0.3", fc=SURF, ec=GRID, lw=0.6))

ax.text(0, np.radians(-6), "GC", ha="center", va="top", fontsize=8, color=INK, zorder=4)
ax.set_xticks(np.radians(np.arange(-150, 151, 30)))
ax.set_xticklabels([f"{-t}°" if t else "" for t in range(-150, 151, 30)], fontsize=7.5, color=MUTED)
ax.set_yticks(np.radians(np.arange(-60, 61, 30)))
ax.set_yticklabels([f"{t}°" for t in range(-60, 61, 30)], fontsize=7.5, color=MUTED)
ax.grid(True, color=GRID, lw=0.5, zorder=0.8)
ax.set_xlabel(r"Galactic longitude $\ell$ (0 = Galactic centre; increases to the left)", fontsize=9, color=INK)
ax.set_ylabel(r"Galactic latitude $b$", fontsize=9, color=INK)

h1 = [Line2D([], [], marker="o", ls="", color=PAL[p], markeredgecolor=SURF, markersize=7, label=p) for p in PROGRAM_ORDER]
h2 = [Line2D([], [], marker=mk, ls="", color=MUTED, markersize=7, label=lab) for mk, lab in MARK.values()]
h3 = [Patch(facecolor="#e6e4df", edgecolor="none", label=r"one hemisphere only ($\delta<-46°$ or $\delta>+44°$)"),
      Patch(facecolor="none", edgecolor="#b9b6ae", hatch="////", label=r"Arecibo strip ($-1°<\delta<+38°$)"),
      Line2D([], [], color="#8a877f", ls=":", label="celestial equator")]
leg1 = ax.legend(handles=h1, loc="lower left", bbox_to_anchor=(-0.02, -0.30), fontsize=8, frameon=False, ncol=2, title="Timing program", title_fontsize=8)
ax.add_artist(leg1)
leg2 = ax.legend(handles=h2, loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=8, frameon=False, title="Channel", title_fontsize=8)
ax.add_artist(leg2)
ax.legend(handles=h3, loc="lower right", bbox_to_anchor=(1.02, -0.32), fontsize=8, frameon=False, title="Sky access", title_fontsize=8)
ax.set_title("Donlon+2025 acceleration catalog: sky coverage in Galactic coordinates", fontsize=10, color=INK, pad=14)
fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.26)
fig.savefig(HERE / "coverage_map.png", dpi=200)
fig.savefig(PAPER / "figures" / "sky_coverage_lb.pdf")
print("saved", HERE / "coverage_map.png", "and", PAPER / "figures/sky_coverage_lb.pdf")
