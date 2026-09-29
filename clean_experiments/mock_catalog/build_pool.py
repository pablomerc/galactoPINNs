#!/usr/bin/env python
"""Build a catalog-like mock pulsar pool from FIRE m12i, with truth labels.

Target distribution of mock pulsar positions:

    p(x)  ∝  n_*(x) * S_r(r_sun) * S_delta(delta)

n_*: old star particles (age > --age-min Gyr) within --r-cand kpc of the observer,
minus the frozen validation particles (sun4 / sun15 / gc15 of the truth cache), so
training and evaluation never share a particle. S_r, S_delta: selection.py, with
r_s and the program counts from catalog.py.

Rejection sampling, one pool per trial t: candidate i is kept iff u_i < p_i with
u = default_rng(t).uniform(size=N_candidates) and p_i = S_r(r_i) S_delta(delta_i).
p <= 1 (both factors peak at 1), so the kept set is an exact unweighted sample of
the target. With --sky none this is exactly the S(r)-only pool of
experiments/repeating_with_rejectancesampling (same candidates, seeds, draws).

Truth labels: pytreegrav summation over all ~147M particles with the frozen
softenings and theta, then the frozen frame/gauge corrections of the truth cache
(a -> a - a_frame, phi -> phi + a_frame.x - u0), for the union of the trial pools
and the observer. Gate: re-summing the first 256 frozen pool particles and the
observer must reproduce the frozen labels to 1e-6, or nothing is written.

Output (--out, default cache/pool_<sky>[_flip-<f>].npz):
    x_union, a_union, u_union   positions [kpc], accelerations [kpc/Myr^2] and
                                potentials [kpc^2/Myr^2] of every particle in any pool
    r_union, l_union, b_union, dec_union, p_union
                                heliocentric distance, sky position, acceptance prob.
    cand_index_union            index into the star block of particles.npz
    idx|<t>                     trial t's pool, as indices into the union arrays
    x_obs, a_obs, u_obs         the observer (Sun) and its true field
    sigma_pool_mmsyr            catalog sigma(a_LOS) values the mock noise resamples
    meta                        JSON: settings, catalog sample, per-trial statistics

Examples (from the galactoPINNs repo root):
    uv run python clean_experiments/mock_catalog/build_pool.py --dry-run   # selection only
    uv run python clean_experiments/mock_catalog/build_pool.py             # S_r S_delta, count
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))                        # mock_catalog package
sys.path.insert(0, str(REPO / "clean_experiments" / "fire_sims"))
import fire_truth as ft                                     # noqa: E402
from mock_catalog.candidates import OBSERVER, PARTICLES, TRUTH, load_candidates  # noqa: E402
from mock_catalog.catalog import LEGACY_SAMPLE, load_catalog, radial_scale, sigma_pool_mmsyr  # noqa: E402
from mock_catalog.selection import BAND_EDGES, SkySelection, acceptance, sky_stats  # noqa: E402

assert np.array_equal(OBSERVER, ft.OBSERVER)
N_GATE = 256


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sky", default="count", choices=["count", "rate", "none"],
                    help="on-sky weighting of S_delta; 'none' = S_r only (default: count)")
    ap.add_argument("--flip", default="none", choices=["none", "l", "b", "lb"],
                    help="mirror the mock sky before computing delta (default: none)")
    ap.add_argument("--n-trials", type=int, default=6)
    ap.add_argument("--r-cand", type=float, default=5.0, help="candidate radius around the observer [kpc]")
    ap.add_argument("--age-min", type=float, default=1.0, help="tracer age cut [Gyr]")
    ap.add_argument("--particles", type=Path, default=PARTICLES)
    ap.add_argument("--truth", type=Path, default=TRUTH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="selection and statistics only: no truth summation, nothing written")
    ap.add_argument("--check-against", type=Path, default=None,
                    help="an earlier pool .npz whose idx|t and labels must be reproduced exactly")
    ap.add_argument("--legacy-sample", action="store_true",
                    help="use the 51-pulsar sample of the 2026-09 pipelines (catalog.LEGACY_SAMPLE) "
                         "instead of the catalog.py defaults; for regression checks")
    return ap.parse_args()


def fmt_stats(s: dict) -> str:
    bands = " ".join(f"{v:.2f}" for v in s["band_share"])
    return (f"Q1 {s['frac_Q1']:.2f}  Q4 {s['frac_Q4']:.2f}  |l|<90 {s['frac_abs_l_lt_90']:.2f}  "
            f"med|b| {s['median_abs_b']:5.1f}  r50 {s['r_p50']:.2f}  r75 {s['r_p75']:.2f}  "
            f"dec bands [{bands}]")


def main() -> None:
    args = parse_args()
    t_start = time.time()
    tag = (f"pool_{args.sky}" + ("" if args.flip == "none" else f"_flip-{args.flip}")
           + ("_legacy51" if args.legacy_sample else ""))
    out = args.out or HERE / "cache" / f"{tag}.npz"

    # ---- catalog sample -> r_s, S_delta, sigma pool --------------------------------
    cat = load_catalog(**LEGACY_SAMPLE) if args.legacy_sample else load_catalog()
    r_s = radial_scale(cat)
    sky = None if args.sky == "none" else SkySelection.from_catalog(cat, args.sky)
    print(f"[catalog] {len(cat['name'])} pulsars (excluded: {', '.join(cat['excluded'])}); "
          f"r_s = {r_s:.4f} kpc", flush=True)
    if sky is not None:
        print(f"[catalog] S_delta ({args.sky} weights) per band "
              f"{np.array2string(BAND_EDGES, precision=0)}: "
              f"{np.array2string(sky.band_S(), precision=2)}", flush=True)

    # ---- frozen truth cache: frame corrections and gate labels -----------------------
    z = np.load(args.truth)
    meta_truth = json.loads(str(z["meta"]))
    fr = meta_truth["checks"]["frame"]
    a_frame = np.asarray(fr["a_frame_kpcmyr2"], dtype=np.float64)
    u0, theta = float(fr["u0_kpc2myr2"]), float(meta_truth["theta"])

    # ---- candidates: old stars within r_cand of the observer, minus validation ---------
    c = load_candidates(args.particles, args.truth, args.r_cand, args.age_min)
    x_cand, cand, tab = c["x"], c["index"], c["tab"]
    acc = acceptance(x_cand, OBSERVER, r_s, sky, args.flip)
    p = acc["p"]
    print(f"[pool] {cand.size:,} candidates (old stars within {args.r_cand} kpc, "
          f"{c['n_val_excluded']:,} validation particles removed); expected pool size "
          f"{p.sum():.0f}", flush=True)

    # ---- per-trial rejection -----------------------------------------------------------
    cat_stats = sky_stats(cat["l_deg"], cat["b_deg"])
    cat_stats.update(r_p50=float(np.median(cat["d_kpc"])), r_p75=float(np.percentile(cat["d_kpc"], 75)))
    pools, trial_stats = {}, {}
    for t in range(args.n_trials):
        u = np.random.default_rng(t).uniform(size=cand.size)
        sel = np.flatnonzero(u < p)
        pools[t] = sel
        s = sky_stats(acc["l"][sel], acc["b"][sel])
        r = acc["r"][sel]
        s.update(r_p50=float(np.median(r)), r_p75=float(np.percentile(r, 75)),
                 r_p90=float(np.percentile(r, 90)), frac_gt4=float(np.mean(r > 4.0)))
        trial_stats[t] = s
        print(f"[pool] trial {t}: {sel.size:5,} kept | {fmt_stats(s)}", flush=True)
    print(f"[pool] catalog     {len(cat['name']):5,}      | {fmt_stats(cat_stats)}", flush=True)

    union = np.unique(np.concatenate([pools[t] for t in range(args.n_trials)]))
    idx_in_union = {t: np.searchsorted(union, pools[t]) for t in range(args.n_trials)}
    print(f"[pool] union of the {args.n_trials} pools: {union.size:,} particles", flush=True)
    if args.dry_run:
        print(f"[dry-run] nothing written ({time.time() - t_start:.0f} s)")
        return

    # ---- truth labels --------------------------------------------------------------------
    x_union = x_cand[union]
    x_gate = np.asarray(z["x_pool"], dtype=np.float64)[:N_GATE]
    x_q = np.vstack([x_union, x_gate, OBSERVER[None, :]])
    sl_union = slice(0, union.size)
    sl_gate = slice(union.size, union.size + N_GATE)
    i_obs = union.size + N_GATE
    pos = tab["pos"]
    print(f"[truth] {x_q.shape[0]:,} query points; tree summation over "
          f"{pos.shape[0]:,} particles (theta={theta})...", flush=True)
    t1 = time.time()
    soft = (tab["eps"] * ft.SPLINE_PER_PLUMMER).astype(np.float64)
    a_q, phi_q = ft.tree_fields(x_q, pos.astype(np.float64), tab["mass"].astype(np.float64),
                                soft, theta, quadrupole=True)
    print(f"[truth] done in {time.time() - t1:.0f} s", flush=True)
    a_q = a_q - a_frame[None, :]
    phi_q = phi_q + x_q @ a_frame - u0

    # ---- gate: the frozen labels must come back ---------------------------------------
    a_old, u_old = np.asarray(z["a_pool"])[:N_GATE], np.asarray(z["u_pool"])[:N_GATE]
    a_obs_old = np.asarray(z["a_obs"], dtype=np.float64)[0]
    rel_a = np.max(np.linalg.norm(a_q[sl_gate] - a_old, axis=1) / np.linalg.norm(a_old, axis=1))
    rel_u = np.max(np.abs(phi_q[sl_gate] - u_old) / np.abs(u_old))
    rel_obs = np.linalg.norm(a_q[i_obs] - a_obs_old) / np.linalg.norm(a_obs_old)
    print(f"[gate] frozen pool[:{N_GATE}] max rel diff: a {rel_a:.1e}, u {rel_u:.1e}; "
          f"observer a {rel_obs:.1e}", flush=True)
    if max(rel_a, rel_u, rel_obs) > 1e-6:
        raise SystemExit("[gate] FAILED: the summation does not reproduce the frozen truth "
                         "labels (pytreegrav must be ==1.1.4); nothing written")

    # ---- optional: reproduce an earlier pool exactly ------------------------------------
    if args.check_against is not None:
        old = np.load(args.check_against)
        same_idx = all(np.array_equal(old[f"idx|{t}"], idx_in_union[t]) for t in range(args.n_trials))
        same_x = np.array_equal(old["x_union"], x_union)
        d_a = float(np.max(np.abs(old["a_union"] - a_q[sl_union]))) if same_x else float("nan")
        print(f"[check] vs {args.check_against.name}: idx identical {same_idx}, "
              f"x_union identical {same_x}, max |a diff| {d_a:.1e}", flush=True)

    # ---- write -------------------------------------------------------------------------
    meta = dict(
        created=time.strftime("%Y-%m-%d %H:%M"),
        sky=args.sky, flip=args.flip, legacy_sample=args.legacy_sample, r_s_kpc=r_s,
        catalog=dict(n=len(cat["name"]), excluded=cat["excluded"],
                     N_k=None if sky is None else sky.N_k,
                     band_edges_deg=BAND_EDGES.tolist(),
                     band_S=None if sky is None else sky.band_S().tolist(),
                     stats=cat_stats),
        r_cand_kpc=args.r_cand, age_min_gyr=args.age_min,
        n_candidates=int(cand.size), n_val_excluded=c["n_val_excluded"],
        n_trials=args.n_trials,
        seeds="pool t: u = default_rng(t).uniform(size=n_candidates), keep u < p",
        theta=theta, frame=dict(a_frame_kpcmyr2=a_frame.tolist(), u0_kpc2myr2=u0),
        gate=dict(n=N_GATE, max_rel_a=float(rel_a), max_rel_u=float(rel_u), rel_obs=float(rel_obs)),
        trial_stats={str(t): s for t, s in trial_stats.items()},
        particles=str(args.particles), truth=str(args.truth),
        pytreegrav=importlib.metadata.version("pytreegrav"),
    )
    payload = dict(
        x_union=x_union, a_union=a_q[sl_union], u_union=phi_q[sl_union],
        r_union=acc["r"][union], l_union=acc["l"][union], b_union=acc["b"][union],
        dec_union=acc["dec"][union], p_union=p[union], cand_index_union=cand[union],
        x_obs=OBSERVER, a_obs=a_q[i_obs], u_obs=np.asarray(phi_q[i_obs]),
        sigma_pool_mmsyr=sigma_pool_mmsyr(cat),
        catalog_names=np.asarray(cat["name"]),
        meta=np.asarray(json.dumps(meta)),
    )
    for t in range(args.n_trials):
        payload[f"idx|{t}"] = idx_in_union[t]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"[pool] wrote {out} ({time.time() - t_start:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
