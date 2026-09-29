"""The Donlon+2025 pulsar acceleration catalog, reduced to the sample the mocks copy.

One catalog definition feeds every mock ingredient, so they cannot drift apart:

    r_s            radial selection scale, mean heliocentric distance / 3
    sigma pool     the per-pulsar sigma(a_LOS) the mock noise is resampled from
    program counts the N_k behind the on-sky selection S(delta)

Rows. The CSV has one row per catalog entry (53), a units row under the header.
Two knobs define the sample; edit them (and nothing else) to change it:

    EXCLUDE            named entries that are not independent measurements
    MAX_FRAC_DIST_ERR  drop pulsars with DIST_ERR / DIST above this, i.e. whose
                       distance is not measured (DIST = 1/parallax, so the ratio is
                       the inverse parallax significance); None disables the cut

Channel. Binary orbital decay (ALOS_PB) where present, else spin-down (ALOS_PS),
the rule that reproduces Donlon+2025's selection.

Program. The timing program behind each pulsar comes from the ATNF reference tag
of its acceleration channel (../paper_figures/sky_coverage/donlon52_refs.csv,
mapped by ../paper_figures/sky_coverage/programs.py). NANOGrav pulsars are split
into the Arecibo strip and Green Bank by declination, as in selection.py.

Run as a script for the sample table and self-checks:
    uv run python clean_experiments/mock_catalog/catalog.py
"""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CATALOG_CSV = HERE / "donlon2025_catalog.csv"      # copy of Linas-group/data/pulsars/data.csv
SKY_COVERAGE = HERE.parent / "paper_figures" / "sky_coverage"
REFS_CSV = SKY_COVERAGE / "donlon52_refs.csv"
PROGRAMS_PY = SKY_COVERAGE / "programs.py"

EXCLUDE = {
    "J0737-3039B": "second pulsar of the double pulsar: same binary, same row as "
                   "J0737-3039A, so one measurement, not two",
}
# parallax below 1 sigma -> distance not measured. With 1.0 this drops J1721-2457,
# J1946+3417, J2229+2643, J2302+4442 and J2322-2650 (J0931-1902 sits at exactly 1.00
# and is kept), leaving 47 pulsars.
MAX_FRAC_DIST_ERR: float | None = 1.0

# the 51-pulsar sample of the 2026-09 pipelines (J1721-2457 dropped by name, no ratio
# cut), kept for regression checks: load_catalog(**LEGACY_SAMPLE)
LEGACY_SAMPLE = dict(
    exclude={**EXCLUDE, "J1721-2457": "distance 10 +- 20 kpc (legacy exclusion by name)"},
    max_frac_dist_err=None,
)

# 1 mm/s/yr = 1 km/s/Myr = 1/977.79222 kpc/Myr^2 (same constant as fire_sims/noise.py)
MMSYR_TO_KPCMYR2 = 1.0 / 977.79222


def _float(raw: str) -> float:
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        return math.nan


def _programs_by_name(refs_csv: Path = REFS_CSV, programs_py: Path = PROGRAMS_PY) -> dict[str, str]:
    """Pulsar name -> timing program (NANOGrav / EPTA / PPTA / Other)."""
    spec = importlib.util.spec_from_file_location("sky_coverage_programs", programs_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(refs_csv, newline="") as fh:
        return {row["name"]: mod.program(row["tag"]) for row in csv.DictReader(fh)}


def load_catalog(exclude: dict[str, str] | None = None,
                 max_frac_dist_err: float | None | str = "default",
                 csv_path: Path = CATALOG_CSV) -> dict:
    """The catalog sample as aligned arrays.

    Keys: name (list), l_deg, b_deg, d_kpc, sigma_d_kpc, a_los_mmsyr, sigma_mmsyr,
    channel ("PB"/"PS"), program, excluded (dict name -> reason).
    Defaults: the module's EXCLUDE and MAX_FRAC_DIST_ERR.
    """
    exclude = dict(EXCLUDE if exclude is None else exclude)
    frac_cut = MAX_FRAC_DIST_ERR if max_frac_dist_err == "default" else max_frac_dist_err
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, units, data = rows[0], rows[1], rows[2:]
    ix = {k: i for i, k in enumerate(header)}
    assert units[ix["DIST"]].strip("()") == "kpc", units[ix["DIST"]]
    assert units[ix["ALOS_PB"]].strip("()") == "mm/s/yr", units[ix["ALOS_PB"]]
    assert units[ix["ALOS_PS"]].strip("()") == "mm/s/yr", units[ix["ALOS_PS"]]
    programs = _programs_by_name()

    out = {k: [] for k in ("name", "l_deg", "b_deg", "d_kpc", "sigma_d_kpc",
                           "a_los_mmsyr", "sigma_mmsyr", "channel", "program")}
    seen = set()
    for row in data:
        name = row[ix["NAME"]].strip() if row else ""
        if not name:
            continue
        seen.add(name)
        if name in exclude:
            continue
        d, sd = _float(row[ix["DIST"]]), _float(row[ix["DIST_ERR"]])
        if frac_cut is not None and sd / d > frac_cut:
            exclude[name] = (f"DIST_ERR / DIST = {sd:.2f} / {d:.2f} = {sd / d:.2f} > "
                             f"{frac_cut:g}: distance not measured")
            continue
        a_pb, s_pb = _float(row[ix["ALOS_PB"]]), _float(row[ix["ALOS_PB_ERR"]])
        a_ps, s_ps = _float(row[ix["ALOS_PS"]]), _float(row[ix["ALOS_PS_ERR"]])
        if math.isfinite(a_pb) and math.isfinite(s_pb):
            channel, a, s = "PB", a_pb, s_pb
        elif math.isfinite(a_ps) and math.isfinite(s_ps):
            channel, a, s = "PS", a_ps, s_ps
        else:
            raise ValueError(f"{name}: no usable acceleration channel")
        out["name"].append(name)
        out["l_deg"].append(_float(row[ix["GL"]]))
        out["b_deg"].append(_float(row[ix["GB"]]))
        out["d_kpc"].append(d)
        out["sigma_d_kpc"].append(sd)
        out["a_los_mmsyr"].append(a)
        out["sigma_mmsyr"].append(s)
        out["channel"].append(channel)
        # the refs table lists the double pulsar once, as J0737-3039A
        out["program"].append(programs.get(name) or programs[name[:-1] + "A"])

    unknown = set(exclude) - seen
    if unknown:
        raise ValueError(f"EXCLUDE names not in the catalog: {sorted(unknown)}")
    cat = {k: (v if k in ("name", "channel", "program") else np.asarray(v, dtype=np.float64))
           for k, v in out.items()}
    cat["excluded"] = dict(exclude)
    return cat


def radial_scale(cat: dict) -> float:
    """r_s of S(r) = exp(-r / r_s): for a locally uniform tracer density the
    distances follow r^2 exp(-r / r_s), a Gamma(3, r_s) law whose mean is 3 r_s."""
    return float(cat["d_kpc"].mean() / 3.0)


def sigma_pool_mmsyr(cat: dict) -> np.ndarray:
    """Sorted per-pulsar sigma(a_LOS) [mm/s/yr] that the mock noise resamples."""
    return np.sort(cat["sigma_mmsyr"])


def _self_check(cat: dict) -> None:
    """Check the loader against the numbers of the earlier pipelines."""
    # (1) the legacy 51-pulsar sample gave this r_s in experiments/repeating_with_rejectancesampling
    r_legacy = radial_scale(load_catalog(**LEGACY_SAMPLE))
    assert r_legacy == 0.5048954248366012, r_legacy
    print(f"[check] legacy 51-pulsar sample: r_s = {r_legacy!r} kpc (matches the S(r) pool build)")
    # (2) the old hard-coded sigma list (frozen copy in experiments/fire_los_recovery/noise.py)
    # is the 53 rows minus the two 3-sigma outliers J0125-2327, J1400-1431; so ours must equal
    # that list plus those two, minus every pulsar this sample excludes
    old_path = HERE.parents[1] / "experiments" / "fire_los_recovery" / "noise.py"
    if old_path.exists():
        spec = importlib.util.spec_from_file_location("fire_sims_noise", old_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        full = load_catalog(exclude={}, max_frac_dist_err=None)
        s_of = dict(zip(full["name"], full["sigma_mmsyr"]))
        expect = sorted(list(mod.CATALOG_SIGMAS_MMSYR)
                        + [s_of["J0125-2327"], s_of["J1400-1431"]])
        for gone in cat["excluded"]:
            expect.remove(min(expect, key=lambda v: abs(v - s_of[gone])))
        assert np.allclose(sigma_pool_mmsyr(cat), np.sort(expect), rtol=0, atol=1e-4)
        print(f"[check] sigma pool == frozen noise.py list + {{J0125-2327, J1400-1431}} "
              f"- the {len(cat['excluded'])} excluded pulsars")


if __name__ == "__main__":
    cat = load_catalog()
    n = len(cat["name"])
    print(f"{n} pulsars kept; excluded:")
    for k, v in cat["excluded"].items():
        print(f"  {k}: {v}")
    s = sigma_pool_mmsyr(cat)
    print(f"r_s = mean distance / 3 = {radial_scale(cat):.4f} kpc")
    print(f"channels: PB {cat['channel'].count('PB')}, PS {cat['channel'].count('PS')}")
    print("programs:", {p: cat["program"].count(p) for p in ("NANOGrav", "EPTA", "PPTA", "Other")})
    print(f"distance [kpc]: median {np.median(cat['d_kpc']):.2f}, max {cat['d_kpc'].max():.2f}")
    print(f"sigma(a_LOS) [mm/s/yr]: min {s.min():.3f}, median {np.median(s):.3f}, "
          f"mean {s.mean():.3f}, max {s.max():.3f}, 1/<1/sigma> {1 / np.mean(1 / s):.3f}; "
          f"p10/p50/p75 {np.percentile(s, 10):.2f} / {np.percentile(s, 50):.2f} / {np.percentile(s, 75):.2f}")
    _self_check(cat)
