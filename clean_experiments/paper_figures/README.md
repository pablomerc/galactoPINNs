# paper_figures

Figures for the accelerations paper. `sampling.ipynb` is the m12i notebook behind the
sampling-comparison figures; `sampling_suite.py` is its script version for a suite of FIRE-2
galaxies (m12i, m12b, m12c, m12f, m12m, m12r, m12w, ...).

## Sampling suite on the cluster

Requirements: numpy, scipy, pandas, matplotlib (scienceplots optional, for the paper style);
`gizmo_analysis` only for `--fire-dir`. Each simulation directory needs the public-release
`track/host_coordinates.hdf5` (used for the principal-axis frame, as in `fire_sims/fire_truth.py`).

```bash
cd clean_experiments/paper_figures
CAT=/path/to/donlon2025/data.csv          # v3 catalog (53 rows + units row)
for sim in m12i m12b m12c m12f m12m m12r m12w; do
  python sampling_suite.py --sim $sim --fire-dir /path/to/fire/${sim}_res7100 --catalog $CAT
done
```

Per simulation this writes
- `cache/<sim>_oldstars_r5_age1.npz` — old (> 1 Gyr) stars within 5 kpc of the observer (gitignored);
- `figs/<sim>_sampling6_rs-fixed.{pdf,png}` and `figs/<sim>_sampling6_rs-mle.{pdf,png}` — the
  6-panel figure (A: uniform pool vs catalog, B: rejection-sampled vs catalog) for the two
  acceptance scales;
- `results/<sim>_sampling_stats.json` — r_s values, acceptance counts, and summary statistics
  (median r, fraction |l| < 90, median |b|, median |z|) of catalog / pool / both rejection samples.

Acceptance scales of S(r) = exp(-r / r_s):
- `fixed`: r_s = mean(r_cat[r < 10 kpc]) / 3, the closed form for stars uniform in volume
  (Gamma(3) law), 0.50 kpc for the Donlon+2025 catalog, identical for all simulations;
- `mle`: maximises the catalog likelihood under n_sim(r) S(r) with n_sim the simulation's own
  candidate density within `--r-cand` (default 5 kpc), so it differs per simulation.
Both use the same uniform draws (seed 0), so the two rejection samples are nested.

A `fire_truth.py prepare` particle table can be used instead of a snapshot: `--particles cache/particles.npz`.
