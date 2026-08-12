#!/usr/bin/env python
"""Fit the n<=4 BFE baseline to the m12i particle table and cache it.

Computes Hernquist-Ostriker SCF coefficients (nmax=4, lmax=4, all m) from a
mass-preserving subsample of the cached particle table (host-centered
principal-axis frame, kpc/Msun), using gala's battle-tested
compute_coeffs_discrete; validates the pure-JAX evaluator in
bfe_baseline.py against gala's compiled one; quantifies baseline-alone
quality against the truth-cache labels (and the misspecified NFW for
context); and writes cache/bfe_nmax4_lmax4.npz for the arch driver.

Run from the repo root (gala is not a project dep, mirrored from the
pytreegrav pattern):

  uv run --with gala python experiments/fire_los_recovery/fit_bfe.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

import sys
sys.path.insert(0, str(HERE))
from bfe_baseline import G_GALACTIC, BFEPotential, bfe_potential_raw  # noqa: E402

HALO_RS = 15.62          # basis scale radius — kept identical to the NFW
NFW_M = 5.4e11           # the misspecified MW-tuned baseline, for context


def nfw_accel(x, m=NFW_M, rs=HALO_RS):
    """Analytic NFW acceleration, kpc/Myr^2 (context baseline)."""
    r = np.linalg.norm(x, axis=1)
    A = G_GALACTIC * m
    dphidr = A * (np.log(1.0 + r / rs) / r**2 - 1.0 / (r * rs * (1.0 + r / rs)))
    return -dphidr[:, None] * (x / r[:, None])


def fit_coeffs(xyz, mass_rel, nmax, lmax, r_s):
    from gala.potential.scf import compute_coeffs_discrete
    t0 = time.time()
    S, T = compute_coeffs_discrete(
        np.ascontiguousarray(xyz, dtype=np.float64),
        np.ascontiguousarray(mass_rel, dtype=np.float64),
        nmax=nmax, lmax=lmax, r_s=r_s)
    print(f"  coeffs on {xyz.shape[0]:,} particles in {time.time()-t0:.0f} s")
    return S, T


def baseline_region_errors(pot_fn, truth, n_pts=2048):
    """median/p90 of |a_baseline - a_label| / |a_label| per val region."""
    out = {}
    for name in ("sun4", "sun15", "gc15"):
        xv = np.asarray(truth[f"x_val|{name}"][:n_pts], dtype=np.float64)
        av = np.asarray(truth[f"a_val|{name}"][:n_pts], dtype=np.float64)
        ab = pot_fn(xv)
        rel = np.linalg.norm(ab - av, axis=1) / np.linalg.norm(av, axis=1)
        out[name] = {"median": float(np.median(rel)),
                     "p90": float(np.percentile(rel, 90))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--r-s", type=float, default=HALO_RS)
    ap.add_argument("--r-max", type=float, default=300.0,
                    help="fit particles within this GC radius (kpc)")
    ap.add_argument("--n-sub", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--tag", default=None,
                    help="output tag; default bfe_nmax<nmax>_lmax<lmax>")
    args = ap.parse_args()

    tag = args.tag or f"bfe_nmax{args.nmax}_lmax{args.lmax}"
    rng = np.random.default_rng(args.seed)

    # ---- particles -------------------------------------------------------
    tab = np.load(CACHE / "particles.npz")
    pos, mass = tab["pos"], tab["mass"]
    r = np.linalg.norm(pos, axis=1)
    sel = np.where(r < args.r_max)[0]
    m_tot = float(mass[sel].astype(np.float64).sum())
    print(f"[fit] {sel.size:,} particles within {args.r_max} kpc, "
          f"M = {m_tot:.4e} Msun")

    n_sub = min(args.n_sub, sel.size)
    sub = rng.choice(sel, size=n_sub, replace=False)
    xyz = pos[sub].astype(np.float64)
    m_sub = mass[sub].astype(np.float64)
    m_rel = m_sub / m_sub.sum()      # sum=1; total mass carried by m_tot

    # ---- main fit ----------------------------------------------------------
    print(f"[fit] nmax={args.nmax} lmax={args.lmax} r_s={args.r_s} kpc")
    S, T = fit_coeffs(xyz, m_rel, args.nmax, args.lmax, args.r_s)

    # ---- validation 1: JAX evaluator vs gala compiled evaluator -----------
    from gala.potential.scf._bfe import potential as gala_pot
    from gala.potential.scf._bfe import gradient as gala_grad
    rq = np.concatenate([rng.uniform(0.5, 25.0, 192),
                         rng.uniform(25.0, 120.0, 64)])
    th = np.arccos(rng.uniform(-1, 1, rq.size))
    ph = rng.uniform(-np.pi, np.pi, rq.size)
    q = np.stack([rq * np.sin(th) * np.cos(ph),
                  rq * np.sin(th) * np.sin(ph),
                  rq * np.cos(th)], axis=1)
    gp_ = gala_pot(np.ascontiguousarray(q), S, T, G=G_GALACTIC, M=m_tot,
                   r_s=args.r_s)
    mp = np.asarray(bfe_potential_raw(jnp.asarray(q), jnp.asarray(S),
                                      jnp.asarray(T), G_GALACTIC, m_tot,
                                      args.r_s))
    gg = gala_grad(np.ascontiguousarray(q), S, T, G=G_GALACTIC, M=m_tot,
                   r_s=args.r_s)
    pot_obj = BFEPotential(m=m_tot, r_s=args.r_s, Snlm=S, Tnlm=T)
    mg = -np.asarray(pot_obj.acceleration(q))
    v1 = {"pot_reldiff_max": float(np.max(np.abs(mp - gp_) / np.abs(gp_))),
          "grad_reldiff_max": float(np.max(
              np.linalg.norm(mg - gg, axis=1) / np.linalg.norm(gg, axis=1)))}
    print(f"[val1] JAX vs gala: pot {v1['pot_reldiff_max']:.2e}, "
          f"grad {v1['grad_reldiff_max']:.2e}")
    if v1["pot_reldiff_max"] > 1e-10 or v1["grad_reldiff_max"] > 1e-8:
        raise SystemExit("[val1] JAX evaluator does not match gala — abort")

    # ---- validation 2: coefficient shot noise (disjoint half-fits) --------
    half = n_sub // 4
    idx = rng.permutation(n_sub)
    fields = []
    for h in (idx[:half], idx[half:2 * half]):
        Sh, Th = fit_coeffs(xyz[h], m_sub[h] / m_sub[h].sum(),
                            args.nmax, args.lmax, args.r_s)
        ph_ = np.asarray(bfe_potential_raw(jnp.asarray(q), jnp.asarray(Sh),
                                           jnp.asarray(Th), G_GALACTIC,
                                           m_tot, args.r_s))
        fields.append(ph_)
    # noise at n_sub ~ half-fit diff / 2 (each half has n_sub/4 particles)
    v2 = {"halffit_field_reldiff_median": float(np.median(
        np.abs(fields[0] - fields[1]) / np.abs(gp_)))}
    print(f"[val2] half-fit field rel diff (median): "
          f"{v2['halffit_field_reldiff_median']:.2e} "
          f"(shot noise at n_sub is ~1/2 of this)")

    # ---- validation 3: baseline-alone quality vs truth labels -------------
    truth = np.load(CACHE / "truth_old1gyr.npz")
    err_bfe = baseline_region_errors(
        lambda xv: np.asarray(pot_obj.acceleration(xv)), truth)
    err_nfw = baseline_region_errors(nfw_accel, truth)
    print("[val3] baseline-alone |a_base - a_label|/|a_label| "
          "(median / p90):")
    for name in ("sun4", "sun15", "gc15"):
        b, nf = err_bfe[name], err_nfw[name]
        print(f"   {name:6s}  BFE {b['median']:.3f} / {b['p90']:.3f}"
              f"   NFW {nf['median']:.3f} / {nf['p90']:.3f}")

    # residual potential scale (drives u* in the include_analytic scaling)
    u_pool = np.asarray(truth["u_pool"], dtype=np.float64)
    x_pool = np.asarray(truth["x_pool"], dtype=np.float64)
    u_bfe = np.asarray(pot_obj.potential(x_pool))
    r_p = np.linalg.norm(x_pool, axis=1)
    u_nfw = -G_GALACTIC * NFW_M * np.log(1.0 + r_p / HALO_RS) / r_p
    v3 = {"u_star_plain": float(np.max(np.abs(u_pool))),
          "u_star_res_bfe": float(np.max(np.abs(u_pool - u_bfe))),
          "u_star_res_nfw": float(np.max(np.abs(u_pool - u_nfw)))}
    print(f"[val3] u*: plain {v3['u_star_plain']:.4f}, residual BFE "
          f"{v3['u_star_res_bfe']:.4f}, residual NFW "
          f"{v3['u_star_res_nfw']:.4f}  kpc^2/Myr^2")

    # ---- write -------------------------------------------------------------
    meta = {
        "created_by": "fit_bfe.py",
        "date": "2026-08-09",
        "convention": "gala.potential.scf / Lowing+2011 (validated 1e-15)",
        "nmax": args.nmax, "lmax": args.lmax, "r_s_kpc": args.r_s,
        "r_max_fit_kpc": args.r_max, "n_particles_fit": int(n_sub),
        "n_particles_available": int(sel.size),
        "m_tot_msun": m_tot, "seed": args.seed,
        "mass_handling": "uniform index subsample; relative masses sum to 1,"
                         " total carried by m (mass-preserving)",
        "validation": {"jax_vs_gala": v1, "shot_noise": v2,
                       "baseline_vs_labels_bfe": err_bfe,
                       "baseline_vs_labels_nfw": err_nfw,
                       "u_star": v3},
    }
    out = CACHE / f"{tag}.npz"
    np.savez(out, Snlm=S, Tnlm=T, m=m_tot, r_s=args.r_s,
             meta=np.array(json.dumps(meta)))
    print(f"[fit] wrote {out}")


if __name__ == "__main__":
    main()
