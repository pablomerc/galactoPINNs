"""Attribute each pulsar's acceleration in the Donlon+2025 catalog to a timing publication.

Downloads the ATNF Pulsar Catalogue package (cached in atnf_cache/, gitignored), reads the
reference tag on PBDOT (binary channel) or F1 (spin channel) for each of the 52 pulsars used in
the paper, and writes donlon52_refs.csv + atnf_version.txt.
"""
import re, tarfile, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                                   # .../Linas-group
CACHE = HERE / "atnf_cache"; CACHE.mkdir(exist_ok=True)
URL = "https://www.atnf.csiro.au/research/pulsar/psrcat/downloads/psrcat_pkg.tar.gz"
tgz = CACHE / "psrcat_pkg.tar.gz"
if not tgz.exists():
    print("downloading", URL); urllib.request.urlretrieve(URL, tgz)
if not (CACHE / "psrcat_tar" / "psrcat.db").exists():
    tarfile.open(tgz).extractall(CACHE)
DB = CACHE / "psrcat_tar" / "psrcat.db"

text = open(DB, encoding="latin-1").read()
version = re.search(r"#CATALOGUE\s+(\S+)", text).group(1)
cat = {}
for rec in text.split("@-----------------------------------------------------------------"):
    d = {}
    for line in rec.splitlines():
        if not line or line.startswith("#"): continue
        parts = line.split(); key = parts[0]
        if key in ("PSRJ", "PSRB"): d[key] = parts[1]
        elif key in ("PBDOT", "F1", "P1"):
            d[key] = parts[1]; d[key + "_ref"] = parts[-1] if len(parts) > 2 else ""
        elif key == "SURVEY": d["SURVEY"] = parts[1]
    if "PSRJ" in d:
        cat[d["PSRJ"]] = d
        if "PSRB" in d: cat[d["PSRB"]] = d               # e.g. B1913+16 -> J1915+1606

df = pd.read_csv(ROOT / "data/pulsars/data.csv", skiprows=[1])
df = df[df["NAME"] != "J0737-3039B"].reset_index(drop=True)   # companion of J0737-3039A, not used
rows = []
for _, p in df.iterrows():
    d = cat.get(p.NAME, {})
    binary = not np.isnan(p.ALOS_PB)                     # paper uses the P_b-dot channel where available
    pb_ref, f1_ref = d.get("PBDOT_ref", ""), d.get("F1_ref", d.get("P1_ref", ""))
    rows.append(dict(name=p.NAME, psrj=d.get("PSRJ", ""), GL=p.GL, GB=p.GB,
                     channel="PB" if binary else "PS", pbdot_ref=pb_ref, f1_ref=f1_ref,
                     tag=pb_ref if (binary and pb_ref) else f1_ref, survey=d.get("SURVEY", "")))
out = pd.DataFrame(rows)
out.to_csv(HERE / "donlon52_refs.csv", index=False)
(HERE / "atnf_version.txt").write_text(version + "\n")
missing = out[out.tag == ""]
print(f"ATNF psrcat {version}: {len(out)} pulsars; tags:\n{out.tag.value_counts().to_string()}")
if len(missing): print("NO TAG FOUND for:", list(missing.name))
