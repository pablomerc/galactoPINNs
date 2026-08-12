#!/usr/bin/env python
"""FIRE m12i truth-table builder for the fire_los_recovery experiment.

Two stages, both cached under ./cache/ (gitignored):

  prepare : read snapshot 600 (z=0) of m12i_res7100 with gizmo_analysis,
            move every particle to the host-centered disk principal-axis
            frame (physical kpc), attach per-type softenings, and write a
            compact particle table (cache/particles.npz, ~2.5 GB).
            One-time cost: reads the full 7.2 GB snapshot.

  build   : select old-star tracers, draw the training pool / validation
            regions / Sun anchor (seeded, mirroring sun_los_recovery),
            compute "true" accelerations+potentials by direct softened
            summation over ALL ~147M particles (pytreegrav octree, with a
            brute-force JAX cross-validation on a subset), plus a SMOOTHED
            reference field (0.5 kpc Plummer-equivalent softening) from the
            same sweep, run physics cross-checks, and write
            cache/truth_<tag>.npz consumed by datasets.py.

Design decisions (2026-07-29, agreed with Pablo):
  * truth = direct summation (honest, includes clumpy small-scale power),
    smoothed reference = same sum with ~0.5 kpc softening, to separate
    "PINN error" from "unresolvable small-scale structure".
  * tracers = star particles with age > 1 Gyr (MSP-like); --age-min 0
    gives the all-stars ablation.
  * geometry mirrors clean_experiments/sun_los_recovery: Sun at
    (-8.1, 0, 0) kpc in the principal frame, 4 kpc training ball,
    sun4/sun15/gc15 eval regions.

Units: kpc, Myr, Msun throughout (galax "galactic" convention), so
accelerations are kpc/Myr^2 and potentials kpc^2/Myr^2 — exactly what
galactoPINNs.data.scale_data expects. 1 kpc/Myr^2 = 977.79 mm/s/yr.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

FIRE_DIR = Path("/Users/pablom.perez/Desktop/Linas-group/data/fire")
CACHE_DIR = Path(__file__).resolve().parent / "cache"
SNAPSHOT_INDEX = 600  # z = 0

# Geometry — mirrors clean_experiments/sun_los_recovery/datasets.py
OBSERVER = np.array([-8.1, 0.0, 0.0])  # Sun, principal-axis frame, kpc
GC = np.zeros(3)

# GIZMO/FIRE-2 m12i_res7100 force softenings, *Plummer-equivalent* kpc
# (Hopkins+2018 Table 3: star 4 pc, DM 40 pc, gas adaptive with ~1 pc floor;
# we use a fixed 4 pc for gas — only matters within ~10 pc of dense cells).
EPS_PLUMMER = {"gas": 0.004, "star": 0.004, "dark": 0.040, "dark2": 0.040}
# pytreegrav's `softening` is the M4 cubic-spline support radius h;
# GIZMO quotes Plummer-equivalent eps, with h ~= 2.8 eps.
SPLINE_PER_PLUMMER = 2.8

# unit constants (from astropy, evaluated once at import)
import astropy.units as u  # noqa: E402
from astropy.constants import G as _G_SI  # noqa: E402

G = _G_SI.to(u.kpc**3 / (u.Msun * u.Myr**2)).value  # ~4.4985e-12
KMS2_TO_KPC2MYR2 = ((u.km / u.s) ** 2).to(u.kpc**2 / u.Myr**2)  # ~1.0459e-6
KPCMYR2_TO_MMSYR = (u.kpc / u.Myr**2).to(u.mm / u.s / u.yr)  # ~977.79

SPECIES = ["gas", "dark", "dark2", "star"]  # star LAST -> easy index block


# ----------------------------------------------------------------------------
# stage 1: prepare — snapshot -> compact particle table
# ----------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    import gizmo_analysis as gizmo

    t0 = time.time()
    part = gizmo.io.Read.read_snapshots(
        species=SPECIES,
        snapshot_value_kind="index",
        snapshot_values=SNAPSHOT_INDEX,
        simulation_directory=str(args.fire_dir),
        properties=["position", "velocity", "mass", "potential",
                    "form.scalefactor"],
        assign_hosts=True,           # uses track/host_coordinates.hdf5
        assign_hosts_rotation=True,  # needed for .principal props
    )
    print(f"[prepare] snapshot read in {time.time() - t0:.0f} s", flush=True)

    counts = {s: part[s]["mass"].size for s in SPECIES}
    n_tot = sum(counts.values())
    pos = np.empty((n_tot, 3), dtype=np.float32)
    mass = np.empty(n_tot, dtype=np.float32)
    eps = np.empty(n_tot, dtype=np.float32)

    # star-block extras (stars are the LAST block in the concatenation)
    star_age = part["star"].prop("age").astype(np.float32)          # Gyr
    star_pot_stored = part["star"]["potential"].astype(np.float32)  # km^2/s^2

    # stellar v_phi(R) profile for the circular-velocity cross-check,
    # computed from Cartesian principal-frame vectors (no reliance on
    # utilities' cylindrical component ordering)
    xs = part["star"].prop("host.distance.principal")   # (N,3) kpc physical
    vs = part["star"].prop("host.velocity.principal")   # (N,3) km/s
    R = np.hypot(xs[:, 0], xs[:, 1])
    vphi = (xs[:, 0] * vs[:, 1] - xs[:, 1] * vs[:, 0]) / np.maximum(R, 1e-12)
    thin = np.abs(xs[:, 2]) < 1.0
    r_edges = np.arange(1.0, 20.5, 0.5)
    vphi_med, vphi_cnt = [], []
    for lo, hi in zip(r_edges[:-1], r_edges[1:]):
        m = thin & (R >= lo) & (R < hi)
        vphi_med.append(float(np.median(vphi[m])) if m.any() else np.nan)
        vphi_cnt.append(int(m.sum()))
    del vs, R, vphi, thin

    # fill the table species by species, freeing as we go
    offsets, i0 = {}, 0
    for s in SPECIES:
        n = counts[s]
        xp = xs if s == "star" else part[s].prop("host.distance.principal")
        pos[i0:i0 + n] = xp.astype(np.float32)   # centered first -> f32 safe
        mass[i0:i0 + n] = part[s]["mass"]
        eps[i0:i0 + n] = EPS_PLUMMER[s]
        offsets[s] = (i0, i0 + n)
        i0 += n
        del xp
        for k in list(part[s].keys()):
            del part[s][k]
        gc.collect()
        print(f"[prepare] packed {s}: {n:,}", flush=True)
    del xs

    # diagnostics: low-res DM contamination near the host
    d2_lo, d2_hi = offsets["dark2"]
    r_dark2 = np.linalg.norm(pos[d2_lo:d2_hi], axis=1)
    dark2_min_r = float(r_dark2.min())
    dark2_mass_300 = float(mass[d2_lo:d2_hi][r_dark2 < 300.0].sum())

    host = part.host
    meta = {
        "source": "FIRE-2 public release m12i_res7100, snapshot 600 (z=0)",
        "fire_dir": str(args.fire_dir),
        "frame": "host-centered disk principal axes, physical kpc "
                 "(gizmo_analysis 'host.distance.principal')",
        "host_position_comoving_kpc": np.asarray(host["position"]).tolist(),
        "host_velocity_kms": np.asarray(host["velocity"]).tolist(),
        "host_rotation": np.asarray(host["rotation"]).tolist(),
        "host_axis_ratios": np.asarray(host["axis.ratios"]).tolist(),
        "counts": counts,
        "eps_plummer_kpc": EPS_PLUMMER,
        "vphi_profile": {"r_edges_kpc": r_edges.tolist(),
                         "median_vphi_kms": vphi_med, "count": vphi_cnt,
                         "cut": "|Z|<1 kpc, all stars"},
        "dark2_min_r_kpc": dark2_min_r,
        "dark2_mass_within_300kpc_msun": dark2_mass_300,
        "mass_total_msun": float(mass.astype(np.float64).sum()),
    }

    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / "particles.npz"
    np.savez(out, pos=pos, mass=mass, eps=eps,
             star_age=star_age, star_pot_stored=star_pot_stored,
             star_offset=np.array(offsets["star"]),
             meta=np.array(json.dumps(meta)))
    print(f"[prepare] wrote {out} ({out.stat().st_size/1e9:.2f} GB) "
          f"in {time.time() - t0:.0f} s total", flush=True)
    print(f"[prepare] dark2 min r = {dark2_min_r:.1f} kpc, "
          f"dark2 mass(<300 kpc) = {dark2_mass_300:.3e} Msun", flush=True)
    iv = ~np.any(np.isnan(pos), axis=1)
    if not iv.all():
        print(f"[prepare] WARNING: {np.sum(~iv):,} NaN positions!", flush=True)


# ----------------------------------------------------------------------------
# stage 2 helpers: the two summation engines
# ----------------------------------------------------------------------------

def tree_fields(x_q: np.ndarray, pos: np.ndarray, mass: np.ndarray,
                soft_spline: np.ndarray, theta: float,
                quadrupole: bool) -> tuple[np.ndarray, np.ndarray]:
    """Softened summation via pytreegrav octree. Returns (a, phi) in
    kpc/Myr^2 and kpc^2/Myr^2. `soft_spline` is the M4 spline support radius
    per SOURCE particle (kpc)."""
    from pytreegrav import AccelTarget, PotentialTarget

    a, tree = AccelTarget(x_q, pos, mass, softening_source=soft_spline,
                          G=G, theta=theta, parallel=True,
                          quadrupole=quadrupole, return_tree=True)
    phi = PotentialTarget(x_q, pos, mass, softening_source=soft_spline,
                          G=G, theta=theta, parallel=True,
                          quadrupole=quadrupole, tree=tree)
    del tree
    gc.collect()
    return np.asarray(a, dtype=np.float64), np.asarray(phi, dtype=np.float64)


def brute_fields(x_q: np.ndarray, pos: np.ndarray, mass: np.ndarray,
                 eps_plummer: np.ndarray,
                 q_chunk: int = 1024, s_chunk: int = 131072,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Exact O(N*M) Plummer-softened summation, used to cross-validate the
    tree on a query subset (a few hundred points), so speed is secondary:
    everything runs in float64 — the r^2-by-expansion trick would lose
    ~half its digits in f32 for near pairs in wide (15 kpc) query chunks.
    Returns (a, phi) in kpc/Myr^2, kpc^2/Myr^2."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def chunk(qc, sc, m, e2):
        r2 = ((qc * qc).sum(1)[:, None] + (sc * sc).sum(1)[None, :]
              - 2.0 * (qc @ sc.T))
        r2 = jnp.maximum(r2, 0.0)
        inv = jax.lax.rsqrt(r2 + e2[None, :])
        phi = -(inv @ m)
        w = (inv * inv * inv) * m[None, :]
        a = w @ sc - qc * w.sum(1)[:, None]
        return a, phi

    nq, ns = x_q.shape[0], pos.shape[0]
    a_out = np.zeros((nq, 3)); phi_out = np.zeros(nq)
    e2_all = (eps_plummer.astype(np.float64) ** 2)
    for qi in range(0, nq, q_chunk):
        q64 = x_q[qi:qi + q_chunk].astype(np.float64)
        c = q64.mean(0)   # recentering: keeps qq/ss/cross terms O(region^2)
        qc = q64 - c
        a_acc = np.zeros((q64.shape[0], 3)); p_acc = np.zeros(q64.shape[0])
        for si in range(0, ns, s_chunk):
            sl = slice(si, si + s_chunk)
            sc = pos[sl].astype(np.float64) - c
            da, dp = chunk(jnp.asarray(qc), jnp.asarray(sc),
                           jnp.asarray(mass[sl], dtype=jnp.float64),
                           jnp.asarray(e2_all[sl]))
            a_acc += np.asarray(da); p_acc += np.asarray(dp)
        # recentring shifts positions, not differences: a is translation-
        # invariant because w @ sc - qc * sum(w) == sum(w * (sc - qc)).
        a_out[qi:qi + q_chunk] = a_acc; phi_out[qi:qi + q_chunk] = p_acc
    return G * a_out, G * phi_out


# ----------------------------------------------------------------------------
# stage 2: build — query sets, truth fields, cross-checks, cache
# ----------------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    t0 = time.time()
    if args.method == "tree":
        try:
            import pytreegrav  # noqa: F401  (fail fast, before the 3 GB load)
        except ImportError:
            raise SystemExit(
                "pytreegrav (and numba) required for --method tree; run with\n"
                "  uv run --with pytreegrav --with numba python "
                "fire_truth.py build ...\n"
                "or `uv add pytreegrav numba`, or use --method brute")
    tab = np.load(CACHE_DIR / "particles.npz")
    pos, mass, eps = tab["pos"], tab["mass"], tab["eps"]
    star_lo, star_hi = tab["star_offset"]
    star_age, star_pot_stored = tab["star_age"], tab["star_pot_stored"]
    meta_particles = json.loads(str(tab["meta"]))
    print(f"[build] particle table: {pos.shape[0]:,} particles", flush=True)

    # ---- tracers: old stars (MSP-like) --------------------------------------
    star_pos = pos[star_lo:star_hi]
    is_tracer = star_age > args.age_min
    print(f"[build] tracers: {is_tracer.sum():,} stars with "
          f"age > {args.age_min} Gyr", flush=True)

    regions = {"sun4": (OBSERVER, args.r_train),
               "sun15": (OBSERVER, args.r_eval),
               "gc15": (GC, args.r_eval)}

    def in_ball(x, center, radius):
        return np.linalg.norm(x - center[None, :], axis=1) < radius

    # training pool: seeded like sun_los_recovery (pool seed 0)
    cand = np.where(is_tracer & in_ball(star_pos, OBSERVER, args.r_train))[0]
    if cand.size < args.pool:
        raise SystemExit(f"only {cand.size} tracers in the training ball, "
                         f"need {args.pool}")
    pool_idx = np.random.default_rng(args.seed_pool).choice(
        cand, size=args.pool, replace=False)
    print(f"[build] pool: {args.pool:,} of {cand.size:,} in-ball tracers",
          flush=True)

    # validation regions: seeds 100+i, EXCLUDING pool particles (finite
    # particle set -> without exclusion sun4/sun15 val would overlap train)
    pool_mask = np.zeros(star_pos.shape[0], dtype=bool)
    pool_mask[pool_idx] = True
    val_idx = {}
    for i, (name, (center, radius)) in enumerate(regions.items()):
        c = np.where(is_tracer & ~pool_mask & in_ball(star_pos, center,
                                                      radius))[0]
        val_idx[name] = np.random.default_rng(100 + i).choice(
            c, size=min(args.n_val, c.size), replace=False)
        print(f"[build] val {name}: {val_idx[name].size:,} of {c.size:,}",
              flush=True)

    # ---- assemble the query stack ------------------------------------------
    blocks = {"pool": star_pos[pool_idx].astype(np.float64)}
    for name in regions:
        blocks[name] = star_pos[val_idx[name]].astype(np.float64)
    blocks["obs"] = OBSERVER[None, :].astype(np.float64)
    # frame points: "gc" gives the host center's own acceleration (subtracted
    # from every a-label -> host-frame tidal field; kills the net force from
    # the ~2.3e16 Msun of distant low-res box mass, which the real simulation
    # largely cancels via periodic gravity); "ref" re-zeros the potential
    # (an isolated sum has a huge -G M_box / d constant that would corrupt
    # the u* = max|u_train| scaling). Both corrections are identically zero
    # for the analytic MilkyWayPotential of sun_los_recovery.
    blocks["gc"] = GC[None, :].astype(np.float64)
    blocks["ref"] = np.array([[0.0, 0.0, 300.0]])
    x_q = np.vstack(list(blocks.values()))
    slices, j0 = {}, 0
    for k, v in blocks.items():
        slices[k] = slice(j0, j0 + v.shape[0]); j0 += v.shape[0]
    print(f"[build] total query points: {x_q.shape[0]:,}", flush=True)

    # ---- true field (per-type GIZMO softenings) -----------------------------
    soft_true = (eps * SPLINE_PER_PLUMMER).astype(np.float64)
    pos64, m64 = pos.astype(np.float64), mass.astype(np.float64)
    if args.method == "tree":
        t1 = time.time()
        a_true, phi_true = tree_fields(x_q, pos64, m64, soft_true,
                                       args.theta, quadrupole=True)
        print(f"[build] tree TRUE field in {time.time()-t1:.0f} s",
              flush=True)
        # cross-validate against exact brute force on a stratified subset
        nv = args.validate_n
        vsel = np.concatenate([np.arange(slices[k].start,
                                         min(slices[k].start
                                             + max(1, nv // 4),
                                             slices[k].stop))
                               for k in ["pool", *regions]])
        t1 = time.time()
        a_b, phi_b = brute_fields(x_q[vsel], pos, mass, eps)
        rel = (np.linalg.norm(a_true[vsel] - a_b, axis=1)
               / np.linalg.norm(a_b, axis=1))
        validation = {"n": int(vsel.size),
                      "rel_diff_median": float(np.median(rel)),
                      "rel_diff_p95": float(np.percentile(rel, 95)),
                      "rel_diff_max": float(rel.max()),
                      "phi_rel_diff_median": float(np.median(
                          np.abs((phi_true[vsel] - phi_b) / phi_b))),
                      "note": "tree(spline,h=2.8eps) vs brute(Plummer,eps); "
                              "includes kernel-convention difference",
                      "seconds_brute": time.time() - t1}
        print(f"[build] tree-vs-brute on {vsel.size} pts: "
              f"median {validation['rel_diff_median']:.2e}, "
              f"p95 {validation['rel_diff_p95']:.2e}, "
              f"max {validation['rel_diff_max']:.2e}", flush=True)
        # HARD gate: a sign/unit/kernel bug shows up here as O(1) mismatch;
        # refuse to write a corrupt truth cache (override with --force).
        if (validation["rel_diff_median"] > 5e-3
                or validation["rel_diff_max"] > 0.20) and not args.force:
            raise SystemExit(
                f"[build] tree-vs-brute mismatch (median "
                f"{validation['rel_diff_median']:.2e}, max "
                f"{validation['rel_diff_max']:.2e}) — refusing to write the "
                f"truth cache; rerun with --force to override")
    else:
        t1 = time.time()
        a_true, phi_true = brute_fields(x_q, pos, mass, eps)
        validation = {"note": "brute is the primary method; no cross-val"}
        print(f"[build] brute TRUE field in {time.time()-t1:.0f} s",
              flush=True)

    # ---- smoothed reference field -------------------------------------------
    # one uniform softening; Plummer-equivalent --smooth-eps (default 0.5 kpc)
    t1 = time.time()
    soft_smooth = np.full(pos.shape[0],
                          args.smooth_eps * SPLINE_PER_PLUMMER)
    if args.method == "tree":
        a_sm, phi_sm = tree_fields(x_q, pos64, m64, soft_smooth,
                                   args.theta, quadrupole=True)
    else:
        a_sm, phi_sm = brute_fields(
            x_q, pos, mass, np.full(pos.shape[0], args.smooth_eps,
                                    dtype=np.float32))
    print(f"[build] SMOOTH field ({args.smooth_eps} kpc) in "
          f"{time.time()-t1:.0f} s", flush=True)
    del pos64, m64
    gc.collect()

    # ---- frame corrections ---------------------------------------------------
    # a_frame from the SMOOTHED field at the GC (stable against central
    # graininess: the 0.5 kpc kernel averages ~1e5 particles). The potential
    # gets the matching linear tilt +a_frame.x so that -grad(u) equals the
    # frame-corrected a everywhere, then is zeroed at the 300 kpc reference.
    a_frame = a_sm[slices["gc"]][0].copy()
    a_true -= a_frame[None, :]
    a_sm -= a_frame[None, :]
    tilt = x_q @ a_frame
    phi_true += tilt
    phi_sm += tilt
    u0 = float(phi_true[slices["ref"]][0])
    phi_true -= u0
    phi_sm -= u0

    # ---- cross-checks --------------------------------------------------------
    checks = {}
    checks["frame"] = {
        "a_frame_kpcmyr2": a_frame.tolist(),
        "a_frame_mag_mmsyr": float(np.linalg.norm(a_frame) * KPCMYR2_TO_MMSYR),
        "a_frame_over_a_obs": float(
            np.linalg.norm(a_frame)
            / np.linalg.norm(a_true[slices["obs"]])),
        "u0_kpc2myr2": u0,
        "u_pool_minmax": [float(phi_true[slices['pool']].min()),
                          float(phi_true[slices['pool']].max())],
        # post-correction a_true at the GC IS the true-minus-smooth residual
        # there (the meaningful "graininess at the center" number)
        "gc_graininess_mag_mmsyr": float(
            np.linalg.norm(a_true[slices["gc"]]) * KPCMYR2_TO_MMSYR)}
    # (1) summed potential vs the potential GIZMO stored per star particle,
    # on the pool block (pool points ARE star particles). Compare after
    # converting stored km^2/s^2 -> kpc^2/Myr^2; fit slope+offset because the
    # stored value has an arbitrary zero point (and box-scale contributions).
    p_stored = star_pot_stored[pool_idx].astype(np.float64) * KMS2_TO_KPC2MYR2
    p_sum = phi_true[slices["pool"]]
    A = np.vstack([p_stored, np.ones_like(p_stored)]).T
    coef, *_ = np.linalg.lstsq(A, p_sum, rcond=None)
    resid = p_sum - A @ coef
    checks["potential_regression"] = {
        "slope": float(coef[0]), "offset_kpc2myr2": float(coef[1]),
        "scatter_over_std": float(resid.std() / p_sum.std()),
        "corr": float(np.corrcoef(p_stored, p_sum)[0, 1])}
    # (2) circular-velocity profile from the true field at gc15 val points
    xg = x_q[slices["gc15"]]; ag = a_true[slices["gc15"]]
    Rg = np.hypot(xg[:, 0], xg[:, 1])
    aR = -(ag[:, 0] * xg[:, 0] + ag[:, 1] * xg[:, 1]) / np.maximum(Rg, 1e-12)
    vc = np.sqrt(np.maximum(Rg * aR, 0.0)) / np.sqrt(KMS2_TO_KPC2MYR2)  # km/s
    thin = np.abs(xg[:, 2]) < 1.0
    prof = []
    for lo in (2, 4, 6, 8, 10, 12):
        m = thin & (Rg >= lo) & (Rg < lo + 2)
        prof.append([lo + 1.0, float(np.median(vc[m])) if m.any() else np.nan,
                     int(m.sum())])
    checks["vcirc_from_field_kms"] = prof
    # absolute sanity gates that catch sign/G/unit errors even without the
    # brute cross-check (e.g. --method brute runs, or --force)
    vc_solar = [v for R, v, cnt in prof if 6 <= R <= 10 and cnt > 10]
    slope = checks["potential_regression"]["slope"]
    if not args.force:
        if not 0.8 < slope < 1.2:
            raise SystemExit(f"[build] sanity gate: potential regression "
                             f"slope {slope:.3f} outside (0.8, 1.2)")
        if vc_solar and not all(150.0 < v < 350.0 for v in vc_solar):
            raise SystemExit(f"[build] sanity gate: v_circ near the solar "
                             f"radius {vc_solar} outside (150, 350) km/s")
    checks["vphi_profile_from_stars"] = meta_particles["vphi_profile"]
    # (3) the Sun anchor, in intuition units
    a_obs = a_true[slices["obs"]]
    checks["anchor"] = {
        "a_obs_kpcmyr2": a_obs[0].tolist(),
        "a_obs_mag_mmsyr": float(np.linalg.norm(a_obs) * KPCMYR2_TO_MMSYR),
        "a_obs_smooth_mag_mmsyr": float(
            np.linalg.norm(a_sm[slices["obs"]]) * KPCMYR2_TO_MMSYR)}
    # (4) how much small-scale power the smoothing removes, per block
    sm_stats = {}
    for k, s in slices.items():
        if k in ("obs", "gc", "ref"):  # single frame points: ratio degenerate
            continue
        d = (np.linalg.norm(a_true[s] - a_sm[s], axis=1)
             / np.linalg.norm(a_true[s], axis=1))
        sm_stats[k] = {"median": float(np.median(d)),
                       "p90": float(np.percentile(d, 90))}
    checks["smooth_vs_true_rel"] = sm_stats

    # ---- write the truth cache ----------------------------------------------
    meta = {
        "created": args.tag, "snapshot": SNAPSHOT_INDEX,
        "observer_kpc": OBSERVER.tolist(), "gc_kpc": GC.tolist(),
        "regions": {k: [v[0].tolist(), v[1]] for k, v in regions.items()},
        "age_min_gyr": args.age_min, "pool": args.pool, "n_val": args.n_val,
        "seed_pool": args.seed_pool, "seeds_val": "100+i (region order)",
        "method": args.method, "theta": args.theta,
        "smooth_eps_plummer_kpc": args.smooth_eps,
        "eps_plummer_kpc": EPS_PLUMMER,
        "spline_per_plummer": SPLINE_PER_PLUMMER,
        "G_kpc3_msun_myr2": G,
        "units": "x kpc (principal frame), a kpc/Myr^2, phi kpc^2/Myr^2",
        "validation": validation, "checks": checks,
        "particles_meta": meta_particles,
    }
    out = CACHE_DIR / f"truth_{args.tag}.npz"
    payload = {
        "x_pool": x_q[slices["pool"]],
        "a_pool": a_true[slices["pool"]],
        "u_pool": phi_true[slices["pool"]],
        "a_pool_smooth": a_sm[slices["pool"]],
        "x_obs": x_q[slices["obs"]],
        "a_obs": a_true[slices["obs"]],
        "a_obs_smooth": a_sm[slices["obs"]],
        "meta": np.array(json.dumps(meta)),
    }
    for name in regions:
        payload[f"x_val|{name}"] = x_q[slices[name]]
        payload[f"a_val|{name}"] = a_true[slices[name]]
        payload[f"a_val_smooth|{name}"] = a_sm[slices[name]]
        payload[f"u_val|{name}"] = phi_true[slices[name]]
    np.savez(out, **payload)
    print(f"[build] wrote {out}", flush=True)
    print(json.dumps(checks, indent=2), flush=True)
    print(f"[build] done in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="snapshot -> compact particle table")
    p.add_argument("--fire-dir", type=Path, default=FIRE_DIR)

    b = sub.add_parser("build", help="particle table -> truth cache")
    b.add_argument("--pool", type=int, default=30000)
    b.add_argument("--n-val", type=int, default=4096)
    b.add_argument("--r-train", type=float, default=4.0)
    b.add_argument("--r-eval", type=float, default=15.0)
    b.add_argument("--age-min", type=float, default=1.0,
                   help="tracer age cut in Gyr; 0 = all stars ablation")
    b.add_argument("--method", choices=["tree", "brute"], default="tree")
    b.add_argument("--theta", type=float, default=0.5)
    b.add_argument("--smooth-eps", type=float, default=0.5,
                   help="Plummer-equivalent softening of the smoothed "
                        "reference field, kpc")
    b.add_argument("--seed-pool", type=int, default=0)
    b.add_argument("--validate-n", type=int, default=256)
    b.add_argument("--tag", default="old1gyr")
    b.add_argument("--force", action="store_true",
                   help="write the truth cache even if a validation or "
                        "sanity gate fails")

    args = ap.parse_args()
    if args.cmd == "build" and args.validate_n < 4:
        ap.error("--validate-n must be >= 4 (split across 4 query blocks)")
    if args.cmd == "prepare":
        prepare(args)
    else:
        build(args)


if __name__ == "__main__":
    sys.exit(main())
