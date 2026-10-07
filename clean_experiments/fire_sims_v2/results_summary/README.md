# fire_sims_v2 results summary (2026-09-29)

All runs: PINN IV (BFE n,l <= 4 baseline + MLP residual), heteroscedastic catalog noise with
1/sigma weights, chi-stop (Morozov), 6 trials, `val_split_check.py`.

| label | pool | LOS loss | mode | n | where |
|---|---|---|---|---|---|
| a_uniform_abs | uniform old stars in 4 kpc, 51 sigma (legacy) | absolute | LOS+Sun+rho | 50-500 | `experiments/repeating_with_rejectancesampling` (= frozen champion) |
| b_Sr51_abs | S(r), r_s = 0.505, 51 sigma | absolute | LOS+Sun+rho | 50-500 | same |
| c_SrSd47_abs | S(r)S(delta) count, 47 pulsars, r_s = 0.474 | absolute | LOS+Sun+rho | 50-500 | `results_ablation_abs_LOSSunrho` |
| d_SrSd47_rel | same | **relative** | LOS+Sun+rho | 50-2000 | `results_LOSSunrho{,_large}` |
| rel_LOSrho | same | relative | LOS+rho | 50-500 | `results_LOSrho` |
| rel_LOS / abs_LOS | same | relative / absolute | LOS | 50-500 | `results_LOS`, `results_ablation_abs_LOS` |

n = 1000 and 2000 used 2500 epochs (so the chi-stop can cross); the column `rec4_e1500` in
those summary.csv files is the last epoch, 2500.

Files: `table.md` (median [IQR] per setup and n), `paired.md` (per-trial ratios on identical
draws), `summary.png/pdf`, `all_rows.csv` (every metric per model).

Metrics (post hoc from the chi-stop checkpoints):
- `rec4`, `rec15`: the trainer's metric, mean relative error of the absolute acceleration on 512
  star particles within 4 / 15 kpc of the Sun.
- `g...`: the same for the observer-referenced field a(x) - a(x_sun) + a_sun(true), invariant
  under a -> a + c, i.e. what the Sun-relative data constrain (equivalently: the field after
  an exact post hoc Sun anchor).
- `sun_off`: |a_model(x_sun) - a_sun| in mm/s/yr, the constant the absolute metrics carry.
- bubbles: uniform-in-volume balls of radius R around the Sun; densities by Gauss's law vs
  particle truth, `dex`: median |log10| error on midplane pixels.
- BFE baseline alone: the same model with the MLP output set to zero.

Produced by the (uncommitted) evaluator in `.claude-tmp/fire_sims_v2_eval/`
(`eval_ckpts.py`, `aggregate.py`), which reuses `metrics_block` of the rejection rerun; gated:
it reproduces that rerun's metrics exactly and every run's own rec4 at the chi-stop (< 2e-6).
The absolute-loss ablation script is `.claude-tmp/fire_sims_v2_ablation/val_split_check_abs.py`
(this folder's `val_split_check.py` with the old absolute residual).
