#!/usr/bin/env python
"""On-sky selection function S(delta) of the Donlon+2025 pulsar catalog.

The catalog is a union of timing programs, and each program can only point at the
declination band its telescopes reach.  We model the probability that a pulsar at
declination delta enters the catalog as a mixture of the programs' footprints,

    S(delta)  ~  sum_k  w_k  1_k(delta),          normalised to max S = 1,

with 1_k the indicator of contributor k's band and two choices of weight:

    rate  : w_k = N_k / Omega_k   pulsars per unit sky  (the paper's choice; the
                                  Poisson maximum-likelihood estimate if each program
                                  samples its band uniformly -- it returns every N_k
                                  exactly when applied to an isotropic sky)
    count : w_k = N_k / N_tot     each program's share of the catalog (Nathaniel's
                                  version; a flatter S that does *not* reproduce the
                                  N_k it was built from)

Contributors (the NANOGrav pulsars are split by whether Arecibo could see them):
    Arecibo (NANOGrav)      -1 < delta < +38
    Green Bank (NANOGrav)   delta > -46, outside the Arecibo strip
    EPTA                    delta > -39   (Nancay sets the union's southern limit)
    PPTA                    delta < +27   (Parkes)
    Other                   all sky       (dedicated timing studies, various dishes)

The program behind each pulsar is the ATNF reference tag on its acceleration channel
(sky_coverage/donlon52_refs.csv, built by sky_coverage/atnf_refs.py) mapped through
sky_coverage/programs.py.  Declinations are J2000, from Galactic (l, b) by the standard
rotation (agrees with astropy to 0.02 arcsec; no astropy dependency).

Run as a script to print the contributor table and S(delta) per declination band:

    python sky_selection.py                      # uses sky_coverage/donlon52_refs.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFS_CSV = HERE / "sky_coverage" / "donlon52_refs.csv"

# telescope limits [deg]; see sky_coverage/accessible_sky.py for the sources
DEC_GBT, DEC_NANCAY, DEC_ARECIBO_LO, DEC_PARKES, DEC_ARECIBO_HI = -46.0, -39.0, -1.0, 27.0, 38.0
BAND_EDGES = np.array([-90.0, DEC_GBT, DEC_NANCAY, DEC_ARECIBO_LO, DEC_PARKES, DEC_ARECIBO_HI, 90.0])

# J2000 equatorial <- Galactic rotation constants
_DEC_NGP, _L_NCP = np.radians(27.12825), np.radians(122.93192)


def declination(l_deg, b_deg):
    """J2000 declination [deg] of Galactic (l, b) [deg]; arrays of any shape."""
    l, b = np.radians(np.asarray(l_deg, float)), np.radians(np.asarray(b_deg, float))
    return np.degrees(np.arcsin(np.sin(_DEC_NGP) * np.sin(b)
                                + np.cos(_DEC_NGP) * np.cos(b) * np.cos(_L_NCP - l)))


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
REGION_LABEL = {  # for tables / captions
    "Arecibo (NANOGrav)": "-1 < dec < +38", "Green Bank (NANOGrav)": "dec > -46, outside Arecibo",
    "EPTA": "dec > -39", "PPTA": "dec < +27", "Other": "all sky",
}


def cap_fraction(dec_lo, dec_hi):
    """Fraction of the sphere between two declinations: (sin hi - sin lo) / 2."""
    return 0.5 * (np.sin(np.radians(dec_hi)) - np.sin(np.radians(dec_lo)))


def sky_fraction(region):
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


def load_pulsars(refs_csv=REFS_CSV):
    """The 52 catalog pulsars with (l, b, dec, program, region) from the provenance table."""
    import pandas as pd
    sys.path.insert(0, str(refs_csv.parent))
    from programs import program as program_of_tag  # sky_coverage/programs.py
    df = pd.read_csv(refs_csv)
    df["dec"] = declination(df["GL"].to_numpy(float), df["GB"].to_numpy(float))
    df["program"] = df["tag"].map(program_of_tag)
    df["region"] = np.where(df["program"] == "NANOGrav",
                            np.where(arecibo_strip(df["dec"]), "Arecibo (NANOGrav)", "Green Bank (NANOGrav)"),
                            df["program"])
    return df


class SkySelection:
    """S(delta) built from the catalog's program attribution.

    >>> sel = SkySelection.from_refs()
    >>> sel.S(dec, "rate")            # acceptance probability in [0, 1] at declination dec
    """
    WEIGHTS = ("rate", "count")

    def __init__(self, N_k: dict[str, int]):
        self.N_k = {k: int(N_k.get(k, 0)) for k in REGIONS}
        self.N_tot = sum(self.N_k.values())
        self.Omega_k = {k: float(sky_fraction(k)) for k in REGIONS}

    @classmethod
    def from_refs(cls, refs_csv=REFS_CSV):
        df = load_pulsars(refs_csv)
        return cls(df["region"].value_counts().to_dict())

    def weights(self, kind="rate", normalised=False):
        if kind == "rate":
            w = {k: self.N_k[k] / self.Omega_k[k] for k in REGIONS}
        elif kind == "count":
            w = {k: self.N_k[k] / self.N_tot for k in REGIONS}
        else:
            raise ValueError(f"kind must be one of {self.WEIGHTS}, got {kind!r}")
        if normalised:
            m = max(w.values())
            w = {k: v / m for k, v in w.items()}
        return w

    def _raw(self, dec, w):
        return sum(w[k] * REGIONS[k](dec) for k in REGIONS)

    def S(self, dec, kind="rate"):
        """Selection probability at declination(s) `dec` [deg], normalised to max 1."""
        w = self.weights(kind)
        band_mid = 0.5 * (BAND_EDGES[1:] + BAND_EDGES[:-1])         # S is constant per band
        return self._raw(np.asarray(dec, float), w) / np.max(self._raw(band_mid, w))

    def band_table(self, kind="rate"):
        """Per declination band: sky fraction, S, and the catalog counts S predicts on an isotropic sky."""
        mid = 0.5 * (BAND_EDGES[1:] + BAND_EDGES[:-1])
        om = np.diff(np.sin(np.radians(BAND_EDGES))) / 2
        S = self.S(mid, kind)
        return dict(edges=BAND_EDGES, sky_fraction=om, S=S, N_pred=self.N_tot * S * om / np.sum(S * om))

    def predicted_counts(self, kind="rate"):
        """Counts each contributor would supply on an isotropic sky (scaled to N_tot): the
        self-consistency check.  'rate' returns N_k exactly; 'count' does not."""
        w = self.weights(kind)
        raw = {k: w[k] * self.Omega_k[k] for k in REGIONS}
        tot = sum(raw.values())
        return {k: self.N_tot * v / tot for k, v in raw.items()}

    def summary(self, dec_cat=None) -> str:
        lines = [f"{'contributor':24s}{'band':>28s}{'N_k':>5s}{'Omega_k':>9s}{'rate':>12s}{'count':>12s}{'N_pred rate':>12s}{'N_pred cnt':>11s}"]
        wr, wc = self.weights("rate", True), self.weights("count", True)
        rr, rc = self.weights("rate"), self.weights("count")
        pr, pc = self.predicted_counts("rate"), self.predicted_counts("count")
        for k in REGIONS:
            lines.append(f"{k:24s}{REGION_LABEL[k]:>28s}{self.N_k[k]:5d}{self.Omega_k[k]:9.3f}"
                         f"{rr[k]:6.1f} ({wr[k]:.2f}){rc[k]:6.2f} ({wc[k]:.2f}){pr[k]:12.1f}{pc[k]:11.1f}")
        lines.append(f"\n{'dec band':14s}{'sky frac':>9s}{'S rate':>8s}{'S count':>9s}{'N_pred rate':>12s}{'N_pred cnt':>11s}"
                     + ("{:>7s}".format("N obs") if dec_cat is not None else ""))
        br, bc = self.band_table("rate"), self.band_table("count")
        n_obs = np.histogram(dec_cat, bins=BAND_EDGES)[0] if dec_cat is not None else None
        for j in range(len(BAND_EDGES) - 1):
            lines.append(f"{BAND_EDGES[j]:+4.0f}..{BAND_EDGES[j+1]:+3.0f}   {br['sky_fraction'][j]:9.3f}{br['S'][j]:8.2f}"
                         f"{bc['S'][j]:9.2f}{br['N_pred'][j]:12.1f}{bc['N_pred'][j]:11.1f}"
                         + (f"{n_obs[j]:7d}" if n_obs is not None else ""))
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", type=Path, default=REFS_CSV, help="sky_coverage/donlon52_refs.csv")
    a = ap.parse_args()
    df = load_pulsars(a.refs)
    sel = SkySelection(df["region"].value_counts().to_dict())
    print(f"{len(df)} pulsars; programs: {df['program'].value_counts().to_dict()}\n")
    print(sel.summary(dec_cat=df["dec"].to_numpy()))
