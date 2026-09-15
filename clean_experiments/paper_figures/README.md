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

## Sky selection suite on the cluster

The declination selection S(δ) of the paper's appendix (`sky_selection.py`) applied on top of
S(r) for every simulation, with both weightings (`rate`: w_k = N_k/Ω_k, the paper's choice;
`count`: w_k = N_k/N_tot). Requirements as above plus `pandas`; no astropy needed.

```bash
cd clean_experiments/paper_figures
python sky_selection.py                                  # prints the contributor table and S(δ) per band
for sim in m12i m12b m12c m12f m12m m12r m12w; do        # or: sbatch sky_selection_suite.slurm
  python sampling_suite.py      --sim $sim --fire-dir /path/to/fire/${sim}_res7100 --catalog $CAT
  python sky_selection_suite.py --sim $sim --catalog $CAT          # reuses cache/<sim>_oldstars_r5_age1.npz
done
python sky_selection_summary.py                          # grid: rows = simulations, columns = samples
```

Per simulation this writes
- `figs/<sim>_sky2d_rate.{pdf,png}`, `figs/<sim>_sky2d_count.{pdf,png}` — the sky-density figure
  (catalog, uniform pool, S(r) sample, joint S(r) S(δ) sample; 50% / 90% contours);
- `results/<sim>_sky_selection_stats.json` — acceptance, quadrant fractions, |l| < 90, median |b|,
  Arecibo-strip share and the share per declination band, for every sample;
- `results/<sim>_sky_samples.npz` — the (l, b) samples, read by `sky_selection_summary.py`,
  which writes `figs/suite_sky2d.{pdf,png}` and prints the statistics table.

Options: `--rs-mle` uses the per-simulation MLE of r_s instead of the closed form; `--flip {l,b,lb}`
mirrors the mock before computing δ, to test the sign conventions of the simulation frame
(S(δ) is symmetric in neither l nor b); `--weights rate` runs one weighting only. The S(r) and
S(δ) rejections use seeds `--seed` and `--seed + 1`; both weightings share the S(δ) draw, so the
samples are nested. The per-pulsar program attribution comes from
`sky_coverage/donlon52_refs.csv` + `sky_coverage/programs.py` (Appendix table of the paper).
