# GUIDE 1 — `datasets.py`

Goal: one module holding everything data-side — the truth potential's density and
acceleration, the Sun-ball training sampler (change #3), the collocation sampler for
the ρ≥0 prior, the fit-once scaling, and the Sun anchor. Type it piece by piece; each
piece has a self-test with exact expected numbers (all code below was run and verified
on 2026-07-17).

---

## Piece A — truth density + the ball rejection sampler

### A.1 Header, imports, constants

```python
"""Dataset construction for the sun_los_recovery experiment.

Training data are density-weighted samples in a small ball around the Sun
(pulsar-like geometry); truth is galax's MilkyWayPotential.
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

OBSERVER = np.array([-8.1, 0.0, 0.0])  # Sun, galactocentric kpc
GC = np.zeros(3)
HALO_RS = 15.62  # NFW scale radius (kpc) used for non-dimensionalization
```

### A.2 The galax→JAX bridge

galax speaks unit-aware objects (`unxt.Quantity`, `coordinax.CartesianPos3D`); our
samplers speak plain `(N, 3)` kpc arrays. This factory builds the shim once and jits
it (`.value` strips units at the end):

```python
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
```

### A.3 The sampler — the heart of change #3

**Why rejection sampling:** we want points distributed like *stars* (pulsars live
where stars live), i.e. drawn from ρ(x) — but the Milky Way density can't be inverted
for direct sampling. Classic workaround: propose uniformly, accept each proposal with
probability ρ(x)/ρ_max. Accepted points are distributed ∝ ρ.

**The r=0 gotcha:** the envelope ρ_max must bound ρ over the region, but
`MilkyWayPotential`'s Hernquist bulge/nucleus density **diverges at the exact
center** — a naive grid max can return `inf` (killing every acceptance probability).
Defenses, each earning its line: the envelope grid is an **even-count linspace
(grid_n=20) that never lands exactly on the center**; non-finite grid densities are
dropped; `safety=1.3` inflates the max; `p` is clipped to 1, which mildly undersamples
the central cusp (documented, acceptable). For the Sun ball the GC isn't inside — but
the `gc15` eval ball is centered *on* it, so this matters.

Two JAX/numerics idioms to notice: proposals are drawn in a **cube** then masked to
the ball (uniform-in-cube is trivial, uniform-in-ball isn't), and the density is
evaluated on the full fixed-size batch *before* masking so the jitted function
compiles exactly once (JAX recompiles on shape changes).

```python
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
```

### A.4 Self-test

Temporary `__main__` block (piece B replaces it with a bigger one):

```python
if __name__ == "__main__":
    true_potential = gp.MilkyWayPotential()
    density_fn = density_fn_factory(true_potential)
    pts = rejection_sample_ball(density_fn, 500, OBSERVER, 4.0, seed=0)
    r_sun = np.linalg.norm(pts - OBSERVER, axis=1)
    print(f"n={pts.shape}, max dist from Sun = {r_sun.max():.3f} kpc (< 4.0?)")
    print(f"mean |z| = {np.abs(pts[:, 2]).mean():.3f} kpc (uniform ball: ~1.500)")
    print(f"mean r_GC = {np.linalg.norm(pts, axis=1).mean():.3f} kpc (< 8.1 = pulled to GC)")
```

Run: `uv run python clean_experiments/sun_los_recovery/datasets.py`

**Expected (exact — seeds are fixed):**

```
n=(500, 3), max dist from Sun = 3.999 kpc (< 4.0?)
mean |z| = 0.747 kpc (uniform ball: ~1.500)
mean r_GC = 7.292 kpc (< 8.1 = pulled to GC)
```

The last two lines are the physics check that density-weighting works: points hug the
disk plane (|z| well below the uniform-ball value) and lean toward the inner galaxy.

---

## Piece B — truth targets, scaling, collocation, anchor

### B.1 True acceleration + potential at arbitrary points

Same galax-bridge pattern as `density_fn_factory`. `potential.acceleration` returns a
unit-aware vector; we unpack components and strip units.

```python
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
```

Physics checkpoint: `observer_acceleration_phys(gp.MilkyWayPotential())` gives
**(0.00692, 0, 0) kpc/Myr² = 2.145e-10 m/s², pointing +x** (toward the GC from
x = -8.1). The real, *measured* value — Sgr A* apparent proper motion (VLBI), Gaia
quasar aberration drift — is ~0.7 cm/s/yr ≈ 2.2e-10 m/s². This is why the anchor is
legitimate future real data and not a synthetic trick.

### B.2 Collocation sampler for the ρ≥0 prior

Uniform **in volume** in a GC-centered shell — *not* density-weighted, deliberately:
positivity violations live in the low-density outskirts, exactly the volume the
Sun-ball training data never sees. Uniform-in-volume means P(r) ∝ r², whose inverse
CDF is the cube-root formula below (`u01` uniform in [0,1] → r such that shell volume
is uniformly filled). The inner cutoff (default 0.5 kpc in the driver) keeps
collocation away from the central cusp, where the Laplacian is huge and numerically
extreme anyway. Points are returned **already scaled** because that's what
`colloc_x` in `train_step_static` expects.

```python
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
```

### B.3 `build_data` — assemble everything, fit the scaling ONCE

The central lesson: **non-dimensionalization is fit on the training pool only.**
`scale_data` derives the scaling from the training arrays; the x/a/u transformers are
constant factors, so validation points *anywhere* (even far outside the training
ball) can be transformed with the same transformers. Fitting per-region or per-n
would silently change the units under each model and make comparisons meaningless.
The same transformers also scale the anchor. The `ab_pot` NFW enters the scaling
config as the analytic reference for non-dimensionalization (`scale: "nfw"` in the
model config); `include_analytic=False` means the network learns the full potential
rather than a residual on top of the analytic term.

```python
def build_data(true_potential, pool, n_val, r_train, regions, include_analytic):
    """Sun-ball train pool + val regions; scaling fit ONCE on the pool."""
    density_fn = density_fn_factory(true_potential)

    x_pool_phys = rejection_sample_ball(density_fn, pool, OBSERVER, r_train, seed=0)
    a_pool_phys, u_pool_phys = eval_targets(true_potential, x_pool_phys)

    val_phys = {}
    for i, (name, (cen, rad)) in enumerate(regions.items()):
        xv = rejection_sample_ball(density_fn, n_val, cen, rad, seed=100 + i)
        av, uv = eval_targets(true_potential, xv)
        val_phys[name] = (xv, av, uv)

    # Non-dimensionalization is fit on the TRAIN pool only (the transformers
    # are constant factors, so val points anywhere can reuse them).
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

    val_scaled = {
        name: (tf["x"].transform(jnp.asarray(xv)), tf["a"].transform(jnp.asarray(av)))
        for name, (xv, av, _) in val_phys.items()
    }
    dist_sun = {name: np.linalg.norm(xv - OBSERVER, axis=1)
                for name, (xv, _, _) in val_phys.items()}

    # Observer anchor: truth at OBSERVER, scaled with the pool transformers.
    a_obs_phys = observer_acceleration_phys(true_potential)
    x_obs = tf["x"].transform(jnp.asarray(OBSERVER[None, :]))  # (1, 3)
    a_obs = tf["a"].transform(jnp.asarray(a_obs_phys[None, :]))  # (1, 3)

    return (scaled["x_train"], scaled["a_train"], x_pool_phys, cfg,
            val_scaled, dist_sun, x_obs, a_obs)
```

Why return `x_pool_phys` too: the driver builds **sightlines in physical space**
(`dx = x_phys - OBSERVER`, normalized). Unit direction vectors are unchanged by the
isotropic scaling, so they can be used directly against scaled accelerations.

### B.4 Self-test (replaces the piece-A `__main__`)

```python
if __name__ == "__main__":
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
```

**Expected (exact):**

```
x_pool (2000, 3), a_pool (2000, 3), finite: True
|x_pool| scaled max = 0.7701 (phys max r_GC = 12.03 kpc)
x_obs (scaled) = [[-0.5186  0.      0.    ]]
|a_obs| (scaled) = 0.5358
val sun4  : 256 pts, dist-from-Sun [0.16, 4.00] kpc
val sun15 : 256 pts, dist-from-Sun [0.98, 14.97] kpc
val gc15  : 256 pts, dist-from-Sun [1.67, 21.93] kpc
colloc: (512, 3), phys r in [1.52, 24.98] kpc
n_vecs row0 = [ 0.8465 -0.2986 -0.4408], norms = [1. 1. 1. 1. 1.]
```

Worth noticing: `|a_obs| (scaled) = 0.54` — an ordinary O(1) number in scaled units,
which is exactly what the shared-mean anchor design assumes (its 3 components sit in
the same mean as the LOS residuals, so they must live on the same scale). The
first run takes a while (galax import + jit); that's normal.

Done? → `GUIDE_2_driver.md`.
