"""Draw one trial's mock pulsar catalog from a pool built by build_pool.py.

A draw is n positions from trial t's pool, their Sun-relative line-of-sight
accelerations, and catalog-like noise:

    n_hat_i  = (x_i - x_obs) / |x_i - x_obs|
    y_true_i = (a_i - a_obs) . n_hat_i                   what a pulsar measures
    y_i      = y_true_i + sigma_i * eps_i,   eps_i ~ N(0, 1)

with sigma_i resampled (with replacement) from the catalog's sigma(a_LOS) stored
in the pool ('het'), or a constant level ('s1.7' = 1.7 mm/s/yr, 's0' = noiseless).

Seeds follow the earlier FIRE harnesses (fire_sims/noise.py, the rejection
rerun's common.draw_trial), so one (trial, n, family) always gives the same draw:

    positions   default_rng(trial).choice(pool size, n, replace=False)
    sigmas      default_rng(9000 + 137 trial + n)
    eps         default_rng(7000 + 1009 trial + 31 n + NOISE_FAMILY_SEED[family])

Everything is in physical units: kpc, kpc/Myr^2. To train with the relative loss
(galactoPINNs.train, x_obs=...), scale with the model's transformers and pass
the scalar y through the target slot as y * n_hat (see README.md).

    uv run python clean_experiments/mock_catalog/draw.py cache/pool_count.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 1 mm/s/yr = 1 km/s/Myr = 1/977.79222 kpc/Myr^2 (same constant as fire_sims/noise.py)
MMSYR_TO_KPCMYR2 = 1.0 / 977.79222
# sigma families share their eps seed with fire_sims/noise.py
NOISE_FAMILY_SEED = {"s0": 0, "s0.5": 1, "s1.7": 2, "s4.3": 3, "het": 4}


def load_pool(path: str | Path) -> dict:
    """Pool arrays plus the parsed meta, as a plain dict."""
    z = np.load(path)
    pool = {k: z[k] for k in z.files if k != "meta"}
    pool["meta"] = json.loads(str(z["meta"]))
    return pool


def trial_pool(pool: dict, trial: int) -> dict:
    """The candidates trial `trial` draws from (union rows of pool idx|trial)."""
    idx = pool[f"idx|{trial}"]
    return {k: pool[f"{k}_union"][idx] for k in ("x", "a", "u", "r", "l", "b", "dec")}


def draw_sigmas_mmsyr(rng: np.random.Generator, n: int, family: str,
                      sigma_pool_mmsyr: np.ndarray) -> np.ndarray:
    """'het' -> resampled from the catalog sigmas; 'sX' -> constant X mm/s/yr."""
    if family == "het":
        return rng.choice(np.asarray(sigma_pool_mmsyr), size=n, replace=True)
    if not family.startswith("s"):
        raise ValueError(f"family must be 'het' or 's<level>', got {family!r}")
    return np.full(n, float(family[1:]))


def draw_trial(pool: dict, trial: int, n: int, family: str = "het") -> dict:
    """n mock pulsars of trial `trial` with Sun-relative LOS accelerations and noise.

    Returns (physical units): x, a, n_hat, r (heliocentric distance), l, b, dec,
    y_los (noisy, Sun-relative), y_los_true, sigma, sigma_mmsyr, eps_std,
    x_obs, a_obs, idx (rows of the trial pool).
    """
    if family not in NOISE_FAMILY_SEED:
        raise ValueError(f"unknown noise family {family!r}; known: {list(NOISE_FAMILY_SEED)}")
    tp = trial_pool(pool, trial)
    idx = np.random.default_rng(trial).choice(tp["x"].shape[0], size=n, replace=False)
    x, a = tp["x"][idx], tp["a"][idx]
    x_obs, a_obs = pool["x_obs"], pool["a_obs"]
    dx = x - x_obs
    n_hat = dx / np.linalg.norm(dx, axis=1, keepdims=True)
    y_true = np.sum((a - a_obs) * n_hat, axis=1)

    sig_mmsyr = draw_sigmas_mmsyr(np.random.default_rng(9000 + 137 * trial + n), n,
                                  family, pool["sigma_pool_mmsyr"])
    eps_std = np.random.default_rng(7000 + 1009 * trial + 31 * n
                                    + NOISE_FAMILY_SEED[family]).normal(0.0, 1.0, size=n)
    sigma = sig_mmsyr * MMSYR_TO_KPCMYR2
    return dict(x=x, a=a, n_hat=n_hat, r=tp["r"][idx], l=tp["l"][idx], b=tp["b"][idx],
                dec=tp["dec"][idx], y_los=y_true + eps_std * sigma, y_los_true=y_true,
                sigma=sigma, sigma_mmsyr=sig_mmsyr, eps_std=eps_std,
                x_obs=x_obs, a_obs=a_obs, idx=idx)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Summarise one draw from a pool.")
    ap.add_argument("pool", type=Path)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--family", default="het")
    args = ap.parse_args()
    pool = load_pool(args.pool)
    m = pool["meta"]
    print(f"pool: sky={m['sky']} flip={m['flip']} r_s={m['r_s_kpc']:.4f} kpc, "
          f"{m['catalog']['n']} catalog pulsars, {m['n_trials']} trials, "
          f"trial {args.trial} has {pool[f'idx|{args.trial}'].size:,} candidates")
    d = draw_trial(pool, args.trial, args.n, args.family)
    k = 1.0 / MMSYR_TO_KPCMYR2
    print(f"draw trial={args.trial} n={args.n} family={args.family}:")
    print(f"  r_sun [kpc]            median {np.median(d['r']):.2f}, p75 {np.percentile(d['r'], 75):.2f}, "
          f"max {d['r'].max():.2f}")
    print(f"  |y_los_true| [mm/s/yr] median {np.median(np.abs(d['y_los_true'])) * k:.2f}")
    print(f"  sigma [mm/s/yr]        median {np.median(d['sigma_mmsyr']):.2f}")
    print(f"  |a_obs| [mm/s/yr]      {np.linalg.norm(d['a_obs']) * k:.2f}")
