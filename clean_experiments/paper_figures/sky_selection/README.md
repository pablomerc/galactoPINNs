# sky_selection

The declination selection S(δ) of the paper's appendix (`selection.py`) applied on top of
S(r) for every FIRE-2 m12 galaxy, with both weightings (`rate`: w_k = N_k/Ω_k, the paper's
choice; `count`: w_k = N_k/N_tot). Requirements as in `../README.md` plus `pandas`; no astropy.

## On the cluster

```bash
cd clean_experiments/paper_figures/sky_selection
python selection.py                                      # prints the contributor table and S(δ) per band
for sim in m12i m12b m12c m12f m12m m12r m12w; do        # or: sbatch suite.slurm
  python ../sampling_suite.py --sim $sim --fire-dir /path/to/fire/${sim}_res7100 --catalog $CAT
  python suite.py             --sim $sim --catalog $CAT   # reuses ../cache/<sim>_oldstars_r5_age1.npz
done
python summary.py                                        # grid: rows = simulations, columns = samples
```

Per simulation this writes
- `figs/<sim>_sky2d_rate.{pdf,png}`, `figs/<sim>_sky2d_count.{pdf,png}` — the sky-density figure
  (catalog, uniform pool, S(r) sample, joint S(r) S(δ) sample; 50% / 90% contours);
- `results/<sim>_sky_selection_stats.json` — acceptance, quadrant fractions, |l| < 90, median |b|,
  Arecibo-strip share and the share per declination band, for every sample;
- `results/<sim>_sky_samples.npz` — the (l, b) samples, read by `summary.py`, which writes
  `figs/suite_sky2d.{pdf,png}` and prints the statistics table.

Options: `--rs-mle` uses the per-simulation MLE of r_s instead of the closed form; `--flip {l,b,lb}`
mirrors the mock before computing δ, to test the sign conventions of the simulation frame
(S(δ) is symmetric in neither l nor b); `--weights rate` runs one weighting only. The S(r) and
S(δ) rejections use seeds `--seed` and `--seed + 1`; both weightings share the S(δ) draw, so the
samples are nested. The per-pulsar program attribution comes from
`../sky_coverage/donlon52_refs.csv` + `../sky_coverage/programs.py` (Appendix table of the paper).
