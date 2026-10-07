# NFW vs BFE analytic baseline (2026-09-29)

Same runs as ../results_summary but with `--baseline nfw`: the MW-tuned NFW halo
(m = 5.4e11 Msun, r_s = 15.62 kpc), not fitted to m12i, in place of the n,l <= 4 BFE fitted to
m12i. Everything else identical (PINN IV, hetW noise, chi-stop, same draws and inits per trial).

| label | loss | mode | n | run folder |
|---|---|---|---|---|
| nfw_d_rel | relative | LOS+Sun+rho | 50-2000 | `results_nfw_LOSSunrho{,_large}` |
| nfw_rel_LOS | relative | LOS | 50-500 | `results_nfw_LOS` |
| nfw_c_abs | absolute (ablation) | LOS+Sun+rho | 50-500 | `results_nfw_ablation_abs_LOSSunrho` |
| nfw_a_uniform_abs | absolute, old uniform pool | LOS+Sun+rho | 50-500 | `experiments/fire_los_recovery/results_val_split_check_rho_nfw` (frozen) |

`table.md` (medians [IQR] incl. the BFE twins and both baselines alone), `paired.md` (NFW/BFE
per-trial ratios on identical draws), `summary.png/pdf`. Evaluator: `.claude-tmp/fire_sims_v2_eval/`
(`eval_ckpts.py` NFWField, gated vs the frozen NFW runs' rec4; `aggregate_nfw.py`).
