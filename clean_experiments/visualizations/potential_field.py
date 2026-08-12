"""Real vs recovered potential for ONE noisy_data model.

Setting = hetW (catalog sigmas, importance weight w ~ 1/sigma, L1 loss),
mode = LOS-only, n = 50 sightlines, trial 0. The script REPLAYS the driver's
exact seeding recipe (pool subsample + per-family noise realization), so the
trained model reproduces that specific results.csv row
(rel_err sun4 ~= 0.3036) rather than a lookalike.

It then makes three figures, all Sun-centered (origin at the Sun, r=4 kpc =
the training-ball edge), in a [-6, 6] kpc window:

  fig1_potential_slices.png  3 planes x [true, pred, residual] potential maps
  fig2_radial_profiles.png   Phi vs distance-from-Sun along 4 rays + residual
  fig3_accel_error.png       gauge-free |a| relative-error slices

Why a gauge fix is mandatory: training only ever sees a = -grad(Phi) (and only
its LOS projection), so the ADDITIVE CONSTANT in Phi is unconstrained by the
data. Every comparison below removes it -- maps/profiles are referenced to the
Sun (Phi - Phi_sun); the residual map removes the best-fit in-ball mean.

Run: uv run python clean_experiments/visualizations/potential_field.py

To visualize a different model, edit the N / TRIAL / FAMILY / ARM / LOS_LOSS
constants below (see clean_experiments/noisy_data/run_experiment.py SETTINGS).
"""
from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax import nnx
import galax.potential as gp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "noisy_data"))  # noisy_datasets...
# ...which itself puts ../sun_los_recovery on the path, so `datasets` imports:

from galactoPINNs.models.static_model import StaticModel
from galactoPINNs.train import train_model_static
from noisy_datasets import (
    NOISE_FAMILY_SEED, OBSERVER, apply_los_noise, arm_weights, build_data,
    draw_sigmas_mmsyr, sigma_scale_factor,
)
from datasets import eval_targets

# --- which model: n=50, trial=0, hetW (catalog sigmas, w~1/sigma, L1), LOS ---
N, TRIAL, R_TRAIN, EPOCHS, POOL = 50, 0, 4.0, 1500, 30000
FAMILY, ARM, LOS_LOSS = "het", "W", "l1"
CSV_REF = 0.3036                     # results.csv rel_err(sun4) for this row
KMS_PER_KPCMYR = 977.79222           # 1 kpc/Myr in km/s
U2KMS2 = KMS_PER_KPCMYR ** 2         # potential kpc^2/Myr^2 -> (km/s)^2

LIM = 6.0                            # plot half-window [kpc], Sun-centered
NG = 140                            # grid resolution per plane axis


# ---------------------------------------------------------------------------
# 1. Reproduce the exact trained model
# ---------------------------------------------------------------------------
def train_reference_model():
    tp = gp.MilkyWayPotential()
    regions = {"sun4": (OBSERVER, R_TRAIN)}          # only needed for the API
    (x_pool, a_pool, x_pool_phys, cfg,
     val_scaled, dist_sun, x_obs, a_obs) = build_data(
        tp, POOL, 512, R_TRAIN, regions, include_analytic=False)

    fac = sigma_scale_factor(cfg)                    # mm/s/yr -> scaled units

    # replay the driver's exact subsample + noise realization (run_experiment.py)
    rng = np.random.default_rng(TRIAL)
    idx = rng.choice(x_pool.shape[0], size=N, replace=False)
    x_n, a_n = x_pool[idx], a_pool[idx]
    dx = x_pool_phys[idx] - OBSERVER
    n_vecs = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))

    sig_i = draw_sigmas_mmsyr(
        np.random.default_rng(9000 + 137 * TRIAL + N), N, FAMILY)
    eps_rng = np.random.default_rng(
        7000 + 1009 * TRIAL + 31 * N + NOISE_FAMILY_SEED[FAMILY])
    a_noisy = apply_los_noise(eps_rng, a_n, n_vecs, sig_i * fac)
    weights = arm_weights(sig_i, ARM)

    model = StaticModel(cfg, rngs=nnx.Rngs(TRIAL))
    train_model_static(
        model, optax.adam(1e-3), x_n, a_noisy, EPOCHS, log_every=0,
        importance_weight=weights, los_loss=LOS_LOSS,
        n_vecs=n_vecs, line_of_sight=True)

    # sanity: this must match results.csv
    x_val, a_val = val_scaled["sun4"]
    a_pred = model(x_val, mode="acceleration")["acceleration"]
    rel = float(jnp.mean(jnp.linalg.norm(a_pred - a_val, axis=1)
                         / jnp.linalg.norm(a_val, axis=1)))
    print(f"[check] rel_err(sun4) = {rel:.4f}   (results.csv: {CSV_REF})")
    return model, cfg, tp, np.asarray(x_pool_phys[idx])


# ---------------------------------------------------------------------------
# 2. Field evaluators (all in PHYSICAL units)
# ---------------------------------------------------------------------------
def phi_pred_phys(model, cfg, xyz):
    """Predicted potential [kpc^2/Myr^2] at physical positions (N, 3) kpc."""
    xs = cfg["x_transformer"].transform(jnp.asarray(xyz))
    ps = model(xs, mode="potential")["potential"].reshape(-1)
    return np.atleast_1d(np.asarray(cfg["u_transformer"].inverse_transform(ps)))


def phi_true_phys(tp, xyz):
    """True potential [kpc^2/Myr^2] from the galax MilkyWayPotential."""
    _, pot = eval_targets(tp, np.asarray(xyz))
    return np.atleast_1d(np.asarray(pot))


def accel_rel_error(model, cfg, tp, xyz):
    """|a_pred - a_true| / |a_true| (dimensionless) -- gauge-free."""
    xs = cfg["x_transformer"].transform(jnp.asarray(xyz))
    a_pred_s = model(xs, mode="acceleration")["acceleration"]
    a_pred = np.asarray(cfg["a_transformer"].inverse_transform(a_pred_s))
    a_true, _ = eval_targets(tp, np.asarray(xyz))
    a_true = np.asarray(a_true)
    return (np.linalg.norm(a_pred - a_true, axis=1)
            / np.linalg.norm(a_true, axis=1))


# ---------------------------------------------------------------------------
# 3. Sun-centered plane geometry + overlays
# ---------------------------------------------------------------------------
# name: (axis-0 label, axis-1 label, builder(a, b) -> physical xyz, GC note)
PLANES = {
    "xy": ("x_Sun [kpc]", "y_Sun [kpc]",
           lambda a, b: np.stack([a + OBSERVER[0], b, np.zeros_like(a)], -1),
           "arrow"),        # z=0 disk plane (contains Sun & GC)
    "xz": ("x_Sun [kpc]", "z_Sun [kpc]",
           lambda a, b: np.stack([a + OBSERVER[0], np.zeros_like(a), b], -1),
           "arrow"),        # y=0 meridional plane (contains Sun & GC)
    "yz": ("y_Sun [kpc]", "z_Sun [kpc]",
           lambda a, b: np.stack([np.full_like(a, OBSERVER[0]), a, b], -1),
           "outofplane"),   # x=x_Sun transverse plane (GC out of page)
}


def plane_grid(name):
    g = np.linspace(-LIM, LIM, NG)
    A, B = np.meshgrid(g, g, indexing="xy")
    xyz = PLANES[name][2](A.ravel(), B.ravel())
    return A, B, xyz


def plane_points(name, xyz_phys, slab=0.5):
    """In-plane (a, b) Sun-centered coords of training pts within `slab` kpc."""
    p = np.asarray(xyz_phys) - OBSERVER  # Sun-centered
    if name == "xy":      # z=0 plane
        a, b, perp = p[:, 0], p[:, 1], p[:, 2]
    elif name == "xz":    # y=0 plane
        a, b, perp = p[:, 0], p[:, 2], p[:, 1]
    else:                 # yz plane at x=x_Sun
        a, b, perp = p[:, 1], p[:, 2], p[:, 0]
    m = np.abs(perp) < slab
    return a[m], b[m]


def draw_overlays(ax, gc_note, pts=None):
    if pts is not None:
        ax.scatter(pts[0], pts[1], s=14, c="w", edgecolor="k", lw=0.5,
                   alpha=0.9, zorder=7)
    ax.add_patch(plt.Circle((0, 0), R_TRAIN, fill=False, ls="--",
                            ec="k", lw=1.3, alpha=0.8))
    ax.plot(0, 0, "*", color="gold", mec="k", ms=13, zorder=8)  # Sun at origin
    if gc_note == "arrow":
        ax.annotate("", xy=(LIM * 0.92, 0), xytext=(LIM * 0.55, 0),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
        ax.text(LIM * 0.72, LIM * 0.12, "GC", fontsize=8, ha="center")
    else:
        ax.plot(0, 0, "o", mfc="none", mec="k", ms=18, zorder=6)
        ax.text(0, -LIM * 0.9, r"$\to$GC out of plane", fontsize=7, ha="center")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# 4. Figure 1 -- potential slices (true / pred / residual)
# ---------------------------------------------------------------------------
def figure_potential(model, cfg, tp, train_xyz):
    # gauge: anchor both fields at the Sun for the side-by-side maps
    phi_true_sun = float(phi_true_phys(tp, OBSERVER[None, :])[0]) * U2KMS2
    phi_pred_sun = float(phi_pred_phys(model, cfg, OBSERVER[None, :])[0]) * U2KMS2

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.8))
    for i, name in enumerate(("xy", "xz", "yz")):
        A, B, xyz = plane_grid(name)
        pt = phi_true_phys(tp, xyz).reshape(A.shape) * U2KMS2 - phi_true_sun
        pp = phi_pred_phys(model, cfg, xyz).reshape(A.shape) * U2KMS2 - phi_pred_sun
        # residual: raw (pred - true), best-fit constant removed over r<4 disk
        rr = (pp + phi_pred_sun) - (pt + phi_true_sun)
        rsun = np.sqrt(A ** 2 + B ** 2) <= R_TRAIN
        rr = rr - rr[rsun].mean()
        rms = float(np.sqrt((rr[rsun] ** 2).mean()))

        vmin = min(pt.min(), pp.min())
        vmax = max(pt.max(), pp.max())
        lab0, lab1, _, gc = PLANES[name]
        pts = plane_points(name, train_xyz)

        m0 = axes[i, 0].pcolormesh(A, B, pt, cmap="viridis",
                                   vmin=vmin, vmax=vmax, shading="auto")
        m1 = axes[i, 1].pcolormesh(A, B, pp, cmap="viridis",
                                   vmin=vmin, vmax=vmax, shading="auto")
        rlim = np.abs(rr[rsun]).max() if rsun.any() else np.abs(rr).max()
        m2 = axes[i, 2].pcolormesh(A, B, rr, cmap="RdBu_r",
                                   vmin=-rlim, vmax=rlim, shading="auto")
        for j, m in ((0, m0), (1, m1), (2, m2)):
            draw_overlays(axes[i, j], gc, pts=pts)
            axes[i, j].set_xlabel(lab0)
            axes[i, j].set_ylabel(lab1)
            fig.colorbar(m, ax=axes[i, j], fraction=0.046, pad=0.04)
        axes[i, 0].set_title(f"{name}: TRUE  $\\Phi-\\Phi_\\odot$ [(km/s)$^2$]",
                             fontsize=9)
        axes[i, 1].set_title(f"{name}: PRED  $\\Phi-\\Phi_\\odot$", fontsize=9)
        axes[i, 2].set_title(f"{name}: residual (r<4 RMS={rms:.0f})",
                             fontsize=9)
    fig.suptitle("Recovered vs true potential  |  LOS-only, catalog noise, "
                 "w~1/sigma  |  n=50, trial 0  (Sun-centered)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "fig1_potential_slices.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 5. Figure 2 -- radial profiles from the Sun
# ---------------------------------------------------------------------------
def figure_profiles(model, cfg, tp):
    rays = {
        "Sun -> GC (+x)": np.array([1.0, 0, 0]),
        "Sun -> anti-GC (-x)": np.array([-1.0, 0, 0]),
        "transverse (+y)": np.array([0, 1.0, 0]),
        "vertical (+z)": np.array([0, 0, 1.0]),
    }
    r = np.linspace(0, LIM, 200)
    phi_true_sun = float(phi_true_phys(tp, OBSERVER[None, :])[0]) * U2KMS2
    phi_pred_sun = float(phi_pred_phys(model, cfg, OBSERVER[None, :])[0]) * U2KMS2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(rays)))
    for (name, u), c in zip(rays.items(), colors):
        xyz = OBSERVER[None, :] + r[:, None] * u[None, :]
        pt = phi_true_phys(tp, xyz) * U2KMS2 - phi_true_sun
        pp = phi_pred_phys(model, cfg, xyz) * U2KMS2 - phi_pred_sun
        axes[0].plot(r, pt, "-", color=c, lw=2, label=name)
        axes[0].plot(r, pp, "--", color=c, lw=2)
        axes[1].plot(r, pp - pt, "-", color=c, lw=2, label=name)
    for ax in axes:
        ax.axvline(R_TRAIN, color="k", ls=":", lw=1)
        ax.set_xlabel("distance from Sun [kpc]")
    axes[0].text(R_TRAIN, axes[0].get_ylim()[1], " r=4 (train edge)",
                 fontsize=8, va="top")
    axes[0].set_ylabel(r"$\Phi-\Phi_\odot$ [(km/s)$^2$]")
    axes[0].set_title("solid = true, dashed = pred")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel(r"$\Phi_{pred}-\Phi_{true}$ [(km/s)$^2$]")
    axes[1].set_title("residual along each ray (gauge: 0 at Sun)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Potential vs distance from Sun along rays", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "fig2_radial_profiles.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 6. Figure 3 -- gauge-free acceleration error slices
# ---------------------------------------------------------------------------
def figure_accel(model, cfg, tp, train_xyz):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, name in zip(axes, ("xy", "xz", "yz")):
        A, B, xyz = plane_grid(name)
        e = accel_rel_error(model, cfg, tp, xyz).reshape(A.shape) * 100.0
        lab0, lab1, _, gc = PLANES[name]
        m = ax.pcolormesh(A, B, e, cmap="magma", vmin=0,
                          vmax=np.percentile(e, 98), shading="auto")
        draw_overlays(ax, gc, pts=plane_points(name, train_xyz))
        ax.set_xlabel(lab0)
        ax.set_ylabel(lab1)
        ax.set_title(f"{name}: |a| rel. error [%]", fontsize=9)
        fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Acceleration recovery error (gauge-free; the trained "
                 "quantity)  |  n=50, trial 0", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "fig3_accel_error.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def main():
    model, cfg, tp, train_xyz = train_reference_model()
    figure_potential(model, cfg, tp, train_xyz)
    figure_profiles(model, cfg, tp)
    figure_accel(model, cfg, tp, train_xyz)
    print("done")


if __name__ == "__main__":
    main()
