"""Selection function of the mock catalog: S(x) = S_r(r_sun) * S_delta(delta).

Radial part. S_r(r) = exp(-r / r_s) with r_s from catalog.radial_scale: how the
chance of a pulsar being timed and having a measured acceleration falls with its
heliocentric distance.

On-sky part. The catalog is a union of timing programs, and each can only point at
the declination band its telescopes reach. The chance that a pulsar at declination
delta enters the catalog is a mixture of the programs' footprints,

    S_delta(delta)  ~  sum_k  w_k  1_k(delta),        normalised to max S_delta = 1,

with 1_k the indicator of contributor k's band and one of two weightings:

    count : w_k = N_k / N_tot     each contributor's share of the catalog (default)
    rate  : w_k = N_k / Omega_k   pulsars per unit of reachable sky

Contributors (NANOGrav is split by whether Arecibo could see the pulsar):

    Arecibo (NANOGrav)      -1 < delta < +38
    Green Bank (NANOGrav)   delta > -46, outside the Arecibo strip
    EPTA                    delta > -39   (Nancay sets the southern limit)
    PPTA                    delta < +27   (Parkes)
    Other                   all sky       (dedicated timing studies)

Ported from ../paper_figures/sky_selection/selection.py; with the same 52 pulsars
it gives the same S_delta to machine precision (checked when run as a script).

Mock geometry. A simulated star at x is "observed" from OBSERVER: its (l, b) are
the direction x - OBSERVER in the simulation frame (x toward the galaxy centre,
z along the disk axis), and delta follows from (l, b) by the J2000 rotation, as if
the simulated galaxy were the Milky Way. The frame fixes the disk plane but not
the sense of rotation or which side is north, so `flip` can mirror l and/or b.

Run as a script for the S_delta table of the current catalog sample:
    uv run python clean_experiments/mock_catalog/selection.py
"""
from __future__ import annotations

import numpy as np

# telescope declination limits [deg]; sources in ../paper_figures/sky_coverage/accessible_sky.py
DEC_GBT, DEC_NANCAY, DEC_ARECIBO_LO, DEC_PARKES, DEC_ARECIBO_HI = -46.0, -39.0, -1.0, 27.0, 38.0
BAND_EDGES = np.array([-90.0, DEC_GBT, DEC_NANCAY, DEC_ARECIBO_LO, DEC_PARKES, DEC_ARECIBO_HI, 90.0])

# J2000 equatorial <- Galactic rotation constants
_DEC_NGP, _L_NCP = np.radians(27.12825), np.radians(122.93192)


# ----------------------------------------------------------------------------- geometry
def heliocentric_lb(x: np.ndarray, observer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian positions (N, 3) -> Galactic (l, b) [deg] seen from `observer`; l in (-180, 180]."""
    d = np.asarray(x, dtype=np.float64) - observer
    l = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    b = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
    return l, b


def flip_lb(l: np.ndarray, b: np.ndarray, flip: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """Mirror the mock sky in l and/or b ('none', 'l', 'b', 'lb')."""
    if flip not in ("none", "l", "b", "lb"):
        raise ValueError(f"flip must be none, l, b or lb, got {flip!r}")
    if "l" in flip:
        l = -l
    if "b" in flip:
        b = -b
    return l, b


def declination(l_deg, b_deg):
    """J2000 declination [deg] of Galactic (l, b) [deg]; arrays of any shape."""
    l, b = np.radians(np.asarray(l_deg, float)), np.radians(np.asarray(b_deg, float))
    return np.degrees(np.arcsin(np.sin(_DEC_NGP) * np.sin(b)
                                + np.cos(_DEC_NGP) * np.cos(b) * np.cos(_L_NCP - l)))


_RA_NGP = np.radians(192.85948)


def equatorial(l_deg, b_deg):
    """J2000 (RA, Dec) [deg] of Galactic (l, b) [deg]; RA in [0, 360)."""
    l, b = np.radians(np.asarray(l_deg, float)), np.radians(np.asarray(b_deg, float))
    ra = _RA_NGP + np.arctan2(np.cos(b) * np.sin(_L_NCP - l),
                              np.cos(_DEC_NGP) * np.sin(b) - np.sin(_DEC_NGP) * np.cos(b) * np.cos(_L_NCP - l))
    return np.degrees(ra) % 360.0, declination(l_deg, b_deg)


# ----------------------------------------------------------------------------- radial part
def radial_selection(r: np.ndarray, r_s: float) -> np.ndarray:
    """S_r(r) = exp(-r / r_s); S_r(0) = 1."""
    return np.exp(-np.asarray(r) / r_s)


# ----------------------------------------------------------------------------- on-sky part
def arecibo_strip(dec):
    return (dec > DEC_ARECIBO_LO) & (dec < DEC_ARECIBO_HI)


# contributor -> indicator function of declination [deg]
REGIONS = {
    "Arecibo (NANOGrav)":    arecibo_strip,
    "Green Bank (NANOGrav)": lambda d: (d > DEC_GBT) & ~arecibo_strip(d),
    "EPTA":                  lambda d: d > DEC_NANCAY,
    "PPTA":                  lambda d: d < DEC_PARKES,
    "Other":                 lambda d: np.ones(np.shape(d), dtype=bool),
}


def cap_fraction(dec_lo, dec_hi):
    """Fraction of the sphere between two declinations: (sin hi - sin lo) / 2."""
    return 0.5 * (np.sin(np.radians(dec_hi)) - np.sin(np.radians(dec_lo)))


def sky_fraction(region: str) -> float:
    """Omega_k: fraction of the sphere that contributor `region` can reach."""
    if region == "Arecibo (NANOGrav)":
        return cap_fraction(DEC_ARECIBO_LO, DEC_ARECIBO_HI)
    if region == "Green Bank (NANOGrav)":
        return cap_fraction(DEC_GBT, 90) - cap_fraction(DEC_ARECIBO_LO, DEC_ARECIBO_HI)
    if region == "EPTA":
        return cap_fraction(DEC_NANCAY, 90)
    if region == "PPTA":
        return cap_fraction(-90, DEC_PARKES)
    return 1.0


def contributor(program: str, dec: float) -> str:
    """Timing program + declination -> contributor (splits NANOGrav at the Arecibo strip)."""
    if program != "NANOGrav":
        return program
    return "Arecibo (NANOGrav)" if arecibo_strip(dec) else "Green Bank (NANOGrav)"


class SkySelection:
    """S_delta built from per-contributor catalog counts N_k.

    >>> sel = SkySelection.from_catalog(cat)
    >>> sel.S(dec)                      # acceptance probability in [0, 1], count weights
    """

    KINDS = ("count", "rate")

    def __init__(self, N_k: dict[str, int], kind: str = "count"):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind = kind
        self.N_k = {k: int(N_k.get(k, 0)) for k in REGIONS}
        self.N_tot = sum(self.N_k.values())
        self.Omega_k = {k: float(sky_fraction(k)) for k in REGIONS}

    @classmethod
    def from_catalog(cls, cat: dict, kind: str = "count") -> "SkySelection":
        dec = declination(cat["l_deg"], cat["b_deg"])
        regions = [contributor(p, d) for p, d in zip(cat["program"], dec)]
        return cls({k: regions.count(k) for k in REGIONS}, kind)

    def weights(self) -> dict[str, float]:
        if self.kind == "rate":
            return {k: self.N_k[k] / self.Omega_k[k] for k in REGIONS}
        return {k: self.N_k[k] / self.N_tot for k in REGIONS}

    def _raw(self, dec, w):
        return sum(w[k] * REGIONS[k](dec) for k in REGIONS)

    def S(self, dec) -> np.ndarray:
        """Selection probability at declination(s) `dec` [deg], normalised to max 1."""
        w = self.weights()
        band_mid = 0.5 * (BAND_EDGES[1:] + BAND_EDGES[:-1])        # S is constant per band
        return self._raw(np.asarray(dec, float), w) / np.max(self._raw(band_mid, w))

    def band_S(self) -> np.ndarray:
        """S_delta in each declination band (BAND_EDGES)."""
        return self.S(0.5 * (BAND_EDGES[1:] + BAND_EDGES[:-1]))


# ----------------------------------------------------------------------------- joint
def acceptance(x: np.ndarray, observer: np.ndarray, r_s: float,
               sky: SkySelection | None, flip: str = "none") -> dict[str, np.ndarray]:
    """Joint acceptance probability p = S_r(r) * S_delta(delta) of positions x (N, 3).

    sky=None keeps the radial part only (S_delta = 1). Returns p with the r, l, b,
    delta it was computed from (l, b after `flip`).
    """
    r = np.linalg.norm(np.asarray(x, dtype=np.float64) - observer, axis=1)
    l, b = flip_lb(*heliocentric_lb(x, observer), flip)
    dec = declination(l, b)
    p = radial_selection(r, r_s)
    if sky is not None:
        p = p * sky.S(dec)
    return dict(p=p, r=r, l=l, b=b, dec=dec)


def sky_stats(l, b) -> dict:
    """Summary statistics used to compare a mock sample with the catalog."""
    l, b = np.asarray(l, float), np.asarray(b, float)
    l = (l + 180.0) % 360.0 - 180.0                 # catalog l in [0, 360) -> (-180, 180]
    dec = declination(l, b)
    return dict(
        n=int(l.size),
        frac_Q1=float(np.mean((l > 0) & (l < 90))),
        frac_Q4=float(np.mean((l > -90) & (l < 0))),
        frac_abs_l_lt_90=float(np.mean(np.abs(l) < 90)),
        median_abs_b=float(np.median(np.abs(b))),
        band_share=[float(np.mean((dec >= lo) & (dec < hi)))
                    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])],
    )


def _check_against_paper_figures(cat_full_52: dict) -> None:
    """Same 52 pulsars -> the same S_delta as ../paper_figures/sky_selection/selection.py."""
    import importlib.util
    from pathlib import Path
    old = Path(__file__).resolve().parents[1] / "paper_figures" / "sky_selection" / "selection.py"
    if not old.exists():
        return
    spec = importlib.util.spec_from_file_location("paper_sky_selection", old)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dec = np.linspace(-89.9, 89.9, 3601)
    for kind in ("count", "rate"):
        new = SkySelection.from_catalog(cat_full_52, kind)
        old_sel = mod.SkySelection(new.N_k)
        assert np.allclose(new.S(dec), old_sel.S(dec, kind), rtol=0, atol=1e-15), kind
    # and the N_k agree with the paper table built from donlon52_refs.csv
    df_regions = mod.load_pulsars()["region"].value_counts().to_dict()
    assert {k: v for k, v in new.N_k.items() if v} == df_regions, (new.N_k, df_regions)
    print("[check] 52-pulsar S_delta (count and rate) == paper_figures/sky_selection/selection.py")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mock_catalog.catalog import load_catalog, radial_scale

    cat = load_catalog()
    r_s = radial_scale(cat)
    print(f"{len(cat['name'])} pulsars, r_s = {r_s:.4f} kpc; "
          f"S_r(1) = {radial_selection(1.0, r_s):.3f}, S_r(2) = {radial_selection(2.0, r_s):.4f}, "
          f"S_r(4) = {radial_selection(4.0, r_s):.1e}")
    dec_cat = declination(cat["l_deg"], cat["b_deg"])
    n_obs = np.histogram(dec_cat, bins=BAND_EDGES)[0]
    sels = {k: SkySelection.from_catalog(cat, k) for k in SkySelection.KINDS}
    print("\ncontributor counts N_k:", sels["count"].N_k)
    print(f"\n{'dec band':>14s} {'sky frac':>9s} {'S count':>8s} {'S rate':>7s} {'N obs':>6s}")
    om = np.diff(np.sin(np.radians(BAND_EDGES))) / 2
    for j in range(len(BAND_EDGES) - 1):
        print(f"{BAND_EDGES[j]:+6.0f}..{BAND_EDGES[j + 1]:+4.0f}   {om[j]:9.3f} "
              f"{sels['count'].band_S()[j]:8.2f} {sels['rate'].band_S()[j]:7.2f} {n_obs[j]:6d}")
    # the paper-figures table used the 52 pulsars left after the duplicate only
    _check_against_paper_figures(load_catalog(exclude={"J0737-3039B": "duplicate"},
                                              max_frac_dist_err=None))
