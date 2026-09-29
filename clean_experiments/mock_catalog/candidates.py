"""Candidate mock pulsars: old FIRE m12i star particles near the observer.

Candidates are star particles older than `age_min` Gyr within `r_cand` kpc of the
observer, minus the frozen validation particles of the truth cache (sun4 / sun15 /
gc15), so a mock training pulsar is never also an evaluation point. The order is
that of the star block of particles.npz, which the per-trial rejection draws rely on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIRE_CACHE = REPO / "experiments" / "fire_los_recovery" / "cache"
PARTICLES = FIRE_CACHE / "particles.npz"          # fire_truth.py prepare (~3 GB)
TRUTH = FIRE_CACHE / "truth_old1gyr.npz"          # fire_truth.py build
OBSERVER = np.array([-8.1, 0.0, 0.0])             # Sun, disk principal-axis frame [kpc]
VAL_REGIONS = ("sun4", "sun15", "gc15")


def rows_view(a: np.ndarray) -> np.ndarray:
    """(N, 3) float64 -> (N,) structured view, so np.isin matches whole rows."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a.view([("x", "f8"), ("y", "f8"), ("z", "f8")]).ravel()


def load_candidates(particles: Path = PARTICLES, truth: Path = TRUTH,
                    r_cand: float = 5.0, age_min: float = 1.0) -> dict:
    """Candidate positions and bookkeeping.

    Returns x (N, 3) float64 [kpc], index (N,) into the star block of particles.npz,
    n_val_excluded, and the open particle table `tab` (for the truth summation).
    """
    z = np.load(truth)
    meta = json.loads(str(z["meta"]))
    assert np.allclose(meta["observer_kpc"], OBSERVER), meta["observer_kpc"]
    x_val = np.vstack([z[f"x_val|{k}"] for k in VAL_REGIONS])

    tab = np.load(particles)
    s0, s1 = tab["star_offset"]
    x_star = tab["pos"][s0:s1].astype(np.float64)
    d_sun = np.linalg.norm(x_star - OBSERVER, axis=1)
    cand = np.flatnonzero((tab["star_age"] > age_min) & (d_sun < r_cand))
    in_val = np.isin(rows_view(x_star[cand]), rows_view(x_val))
    cand = cand[~in_val]
    return dict(x=x_star[cand], index=cand, n_val_excluded=int(in_val.sum()), tab=tab)
