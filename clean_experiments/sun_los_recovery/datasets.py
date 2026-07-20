"""
Dataset construction for the sun_los_recovery experiment

Training data are density-weighted samples in a small ball around the Sun of ~kpc (pulsar like)
Truth is galax's MilkyWayPotential
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import galax.potential as gp
from galax.potential import density
import coordinax as cx
import unxt as u

from galactoPINNs.data import sample_angles, scale_data

OBSERVER = np.array([-8.1, 0.0, 0.0]) # Sun, galactocentric kpc
GC = np.zeros(3)
HALO_RS = 15.62 # NFW scale radius (kpc)



# Now lets adapt galactoPINNs.data.rejection_sample_sphere (GC-centered) to an arbitrary position

def density_fn_factory(potential):
    """JIT-compiled rho(xyz) for raw (N, 3) positions in kpc."""
    t_q = u.Quantity(0.0, "Myr")

    @jax.jit
    def _rho(xyz):
        pos = cx.CartesianPos3D(
            x=u.Quantity(xyz[:, 0], "kpc"),
            y=u.Quantity(xyz[:, 1], "kpc"),
            z=u.Quantity(xyz[:, 2], "kpc"),
        )
        return density(potential, pos, t=t_q).value

    return _rho


def rejection_sample_ball(density_fn, n, center, radius, seed,
                        batch=65536, grid_n=20, safety=1.3):
    """Density-weighted rejection sampling in a ball around `center` (kpc).

    Generalizes galactoPINNs.data.rejection_sample_sphere (GC-centered) to an
    arbitrary center. Envelope = coarse grid max over the ball, inflated by
    `safety`; an even-count linspace never lands exactly on the GC, where the
    Hernquist bulge/nucleus density of MilkyWayPotential diverges. Proposals
    denser than the envelope are accepted with probability 1, which mildly
    undersamples the central cusp.
    """
    center = np.asarray(center, dtype=np.float64)
    lin = np.linspace(-radius, radius, grid_n)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    mask = X**2 + Y**2 + Z**2 <= radius**2
    grid = np.stack([X[mask], Y[mask], Z[mask]], axis=1) + center
    rho_grid = np.asarray(density_fn(jnp.asarray(grid)))
    rho_grid = rho_grid[np.isfinite(rho_grid)]
    rho_max = safety * float(rho_grid.max())
    if not np.isfinite(rho_max) or rho_max <= 0:
        raise RuntimeError(f"bad rejection envelope: rho_max={rho_max}")

    rng = np.random.default_rng(seed)
    chunks, got = [], 0
    for _ in range(20000):
        if got >= n:
            break
        prop_c = rng.uniform(-radius, radius, size=(batch, 3))
        xyz = prop_c + center
        rho = np.asarray(density_fn(jnp.asarray(xyz)))
        p = np.clip(np.nan_to_num(rho / rho_max, nan=0.0, posinf=1.0), 0.0, 1.0)
        inside = np.sum(prop_c**2, axis=1) <= radius**2
        kept = xyz[inside & (rng.uniform(size=batch) < p)]
        chunks.append(kept)
        got += kept.shape[0]
    else:
        raise RuntimeError(f"rejection sampling stalled: {got}/{n} accepted")
    return np.concatenate(chunks, axis=0)[:n]

#Truth targets, scaling, collocation, anchor

def eval_targets(potential, xyz):
    """True acceleration (N, 3) and potential (N,) at physical positions (kpc)."""
    t_q = u.Quantity(0.0, "Myr")
    pos = cx.CartesianPos3D(
        x=u.Quantity(xyz[:, 0], "kpc"),
        y=u.Quantity(xyz[:, 1], "kpc"),
        z=u.Quantity(xyz[:, 2], "kpc"),
    )
    acc = potential.acceleration(pos, t=t_q)
    pot = np.asarray(potential.potential(pos, t=t_q).value)
    a = np.stack(
        [np.asarray(acc.x.value), np.asarray(acc.y.value), np.asarray(acc.z.value)],
        axis=1,
    )
    return a, pot


def observer_acceleration_phys(true_potential):
    """Synthetic true a at the observer — same potential as the pool, not a catalog."""
    a, _ = eval_targets(true_potential, OBSERVER[None, :])
    return a[0].astype(np.float64)

#Collocation smaple for the rho>=0 prior

def sample_colloc_scaled(key, n_colloc, r_min, r_max, x_transformer):
    """Uniform-in-volume points in the GC-centered shell r in (r_min, r_max) kpc.

    NOT density-weighted: positivity violations live in the low-density
    outskirts, exactly the volume the Sun-ball training data never sees.
    """
    key_r, key_ang = jr.split(key)
    u01 = jr.uniform(key_r, (n_colloc,))
    r = (u01 * (r_max**3 - r_min**3) + r_min**3) ** (1.0 / 3.0)
    theta, phi = sample_angles(key_ang, n_colloc)
    xyz_phys = jnp.stack(
        [r * jnp.sin(phi) * jnp.cos(theta),
         r * jnp.sin(phi) * jnp.sin(theta),
         r * jnp.cos(phi)],
        axis=1,
    )  # (N_c, 3) kpc
    return x_transformer.transform(xyz_phys)  # (N_c, 3) scaled

# Assemble everything
def build_data(true_potential, pool, n_val, r_train, regions, include_analytic):
    """Sun-ball train pool + val regions; scaling fit ONCE on the pool.

    eval_targets ("truth labels at these positions") is called once per point
    set: the train pool, each val region, and (inside
    observer_acceleration_phys) the Sun anchor — different positions each
    time, nothing recomputed.
    """
    density_fn = density_fn_factory(true_potential)

    # --- TRAINING pool: `pool` density-weighted points in the r_train ball
    # around the Sun, plus their truth labels. The driver subsamples
    # n <= pool points from this per run; build_data never picks n.
    x_pool_phys = rejection_sample_ball(density_fn, pool, OBSERVER, r_train, seed=0)
    a_pool_phys, u_pool_phys = eval_targets(true_potential, x_pool_phys)

    # --- VALIDATION sets: one independent ball per region (own center,
    # radius, and seed 100+i, so they never coincide with the train pool).
    # Evaluated only, never trained on. Each needs its own eval_targets
    # call — different positions, different labels.
    val_phys = {}
    for i, (name, (cen, rad)) in enumerate(regions.items()):
        xv = rejection_sample_ball(density_fn, n_val, cen, rad, seed=100 + i)
        av, uv = eval_targets(true_potential, xv)
        val_phys[name] = (xv, av, uv)

    # --- Fit the non-dimensionalization ONCE, on the TRAIN pool only:
    # u* = max|u_train| sets the units; the transformers are constant
    # factors, so val points anywhere (even outside the ball) reuse them.
    # scale_data's API requires x/a/u_val keys, so the first region is
    # passed to satisfy the signature — its values do NOT influence the
    # fit, and every region is (re-)scaled uniformly below.
    first = next(iter(val_phys))
    raw = {
        "x_train": jnp.asarray(x_pool_phys),
        "a_train": jnp.asarray(a_pool_phys),
        "u_train": jnp.asarray(u_pool_phys),
        "x_val": jnp.asarray(val_phys[first][0]),
        "a_val": jnp.asarray(val_phys[first][1]),
        "u_val": jnp.asarray(val_phys[first][2]),
    }
    ab_pot = gp.NFWPotential(m=5.4e11, r_s=HALO_RS, units="galactic")
    scale_config = {"r_s": HALO_RS, "include_analytic": include_analytic,
                    "ab_potential": ab_pot}
    scaled, tf = scale_data(raw, scale_config)
    cfg = {
        "x_transformer": tf["x"], "a_transformer": tf["a"], "u_transformer": tf["u"],
        "r_s": HALO_RS, "include_analytic": include_analytic,
        "scale": "nfw", "depth": 6,
    }

    # Scale ALL val regions with the pool-fitted constant factors (the first
    # region again, uniformly). u_val is dropped: evaluation is on a only.
    val_scaled = {
        name: (tf["x"].transform(jnp.asarray(xv)), tf["a"].transform(jnp.asarray(av)))
        for name, (xv, av, _) in val_phys.items()
    }
    # Physical distance of each val point from the Sun, for error-vs-distance
    # plots downstream.
    dist_sun = {name: np.linalg.norm(xv - OBSERVER, axis=1)
                for name, (xv, _, _) in val_phys.items()}

    # Observer anchor: truth at OBSERVER, scaled with the pool transformers.
    a_obs_phys = observer_acceleration_phys(true_potential)
    x_obs = tf["x"].transform(jnp.asarray(OBSERVER[None, :]))  # (1, 3)
    a_obs = tf["a"].transform(jnp.asarray(a_obs_phys[None, :]))  # (1, 3)

    # x_pool_phys rides along un-scaled: the driver builds sightline unit
    # vectors in physical space (unit directions survive isotropic scaling).
    return (scaled["x_train"], scaled["a_train"], x_pool_phys, cfg,
            val_scaled, dist_sun, x_obs, a_obs)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    true_potential = gp.MilkyWayPotential()
    regions = {"sun4": (OBSERVER, 4.0), "sun15": (OBSERVER, 15.0), "gc15": (GC, 15.0)}
    x_pool, a_pool, x_pool_phys, cfg, val_scaled, dist_sun, x_obs, a_obs = build_data(
        true_potential, pool=2000, n_val=256, r_train=4.0,
        regions=regions, include_analytic=False,
    )
    print(f"x_pool {x_pool.shape}, a_pool {a_pool.shape}, finite: "
          f"{bool(jnp.isfinite(x_pool).all() & jnp.isfinite(a_pool).all())}")
    print(f"|x_pool| scaled max = {float(jnp.linalg.norm(x_pool, axis=1).max()):.4f} "
          f"(phys max r_GC = {np.linalg.norm(x_pool_phys, axis=1).max():.2f} kpc)")
    print(f"x_obs (scaled) = {np.asarray(x_obs).round(4)}")
    print(f"|a_obs| (scaled) = {float(jnp.linalg.norm(a_obs)):.4f}")
    for name in regions:
        d = dist_sun[name]
        print(f"val {name:6s}: {val_scaled[name][0].shape[0]} pts, dist-from-Sun "
              f"[{d.min():.2f}, {d.max():.2f}] kpc")
    colloc = sample_colloc_scaled(jr.PRNGKey(1000), 512, 0.5, 25.0,
                                  cfg["x_transformer"])
    r_c = np.linalg.norm(np.asarray(
        cfg["x_transformer"].inverse_transform(colloc)), axis=1)
    print(f"colloc: {colloc.shape}, phys r in [{r_c.min():.2f}, {r_c.max():.2f}] kpc")
    dx = x_pool_phys[:5] - OBSERVER
    n_vecs = dx / np.linalg.norm(dx, axis=1, keepdims=True)
    print(f"n_vecs row0 = {n_vecs[0].round(4)}, norms = "
          f"{np.linalg.norm(n_vecs, axis=1).round(6)}")

    # Physical coords for plotting (val/colloc come back via inverse transform)
    x_tf = cfg["x_transformer"]
    val_phys = {
        name: np.asarray(x_tf.inverse_transform(val_scaled[name][0]))
        for name in regions
    }
    colloc_phys = np.asarray(x_tf.inverse_transform(colloc))

    # One panel per set on a shared galactocentric x–y plane
    # (subsample the dense train pool for plotting only)
    rng_plot = np.random.default_rng(0)
    train_plot = x_pool_phys[rng_plot.choice(len(x_pool_phys), size=256, replace=False)]
    panels = [
        ("sun4 train", train_plot, OBSERVER, 4.0, "#c0392b", "o"),
        ("sun15 val", val_phys["sun15"], OBSERVER, 15.0, "#2980b9", "o"),
        ("gc15 val", val_phys["gc15"], GC, 15.0, "#27ae60", "o"),
        ("colloc (ρ≥0)", colloc_phys, GC, 25.0, "0.25", "x"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), sharex=True, sharey=True)
    sun, gc = np.asarray(OBSERVER), np.asarray(GC)
    for ax, (name, xy, cen, rad, color, marker) in zip(axes, panels):
        ax.scatter(xy[:, 0], xy[:, 1], s=8 if marker == "o" else 12, c=color,
                   alpha=0.45, lw=0 if marker == "o" else 0.6, marker=marker)
        ax.add_patch(plt.Circle(cen[:2], rad, fill=False, ls="--",
                                ec=color, lw=1.4))
        ax.scatter([sun[0]], [sun[1]], s=100, marker="*", c="gold",
                   edgecolor="k", zorder=6, label="Sun")
        ax.scatter([gc[0]], [gc[1]], s=50, marker="+", c="k",
                   zorder=6, label="GC")
        ax.set_aspect("equal")
        ax.set_xlim(-28, 28)
        ax.set_ylim(-28, 28)
        ax.set_title(name)
        ax.set_xlabel("x [kpc]")
    axes[0].set_ylabel("y [kpc]")
    axes[0].legend(loc="upper right", fontsize=7, markerscale=1.4)
    fig.suptitle("Density-weighted balls + uniform collocation (x–y)", y=1.03)
    fig.tight_layout()

    out = Path(__file__).with_name("region_samples_xy.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
