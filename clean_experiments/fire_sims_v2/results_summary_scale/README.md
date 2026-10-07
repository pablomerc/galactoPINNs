# Scale-function ablation (2026-09-30)

**Question.** The IV model is `Φ = Φ_baseline + s(r) · u_nn(x)` with
`s(r) = ln(1 + r/r_s) / (r/r_s)`, r from the Galactic centre, r_s = 15.62 kpc
(`cfg["scale"] = "nfw"`, [layers.py](../../../src/galactoPINNs/layers.py)). s is the NFW
*potential* shape normalized to 1 at the centre (0.98 at 0.5 kpc, 0.81 at the Sun, 0.60 at
25 kpc). Does removing it (`scale = "one"`, `Φ = Φ_baseline + u_nn`) change the acceleration
or density errors?

**Setup.** Everything else is the v2 champion: relative LOS loss, LOS+Sun+ρ, 47-pulsar
S(r)S(δ) pool, hetW noise, χ-stop, 6 trials, n = 50–500 (1500 epochs) and 1000–2000 (2500
epochs), for the BFE and the NFW baseline. The scale layer has no parameters, so both variants
start from identical weights on identical draws: every comparison is paired per trial.
Script: `.claude-tmp/fire_sims_v2_ablation/val_split_check_scale.py` (the v2 script plus a
`--scale` flag; `--scale nfw` reproduces the v2 run exactly). Runs: `results_scale1_*`,
`results_nfw_scale1_*`; default-scale runs from `results_LOSSunrho*`, `results_nfw_LOSSunrho*`.
Evaluation: `.claude-tmp/fire_sims_v2_eval/{eval_ckpts.py, aggregate_scale.py}` (model rebuild
gated at 0.0 difference; every run's own rec4 reproduced to < 2e-6).

**Files.** `table.md` medians [IQR], `paired.md` per-trial ratios (no scaling / default, and the
number of trials better without scaling), `summary.png` (rows: BFE, NFW baseline).

## Result: the scale function barely matters

- **Acceleration where the data are (R = 0.5, 2 kpc): no effect**, both baselines, all n
  (paired ratios 0.95–1.08, mostly 0.98–1.03).
- **Edge of the data and beyond: the scaling helps a little.** BFE: R = 4 kpc 7–10% worse
  without it at n ≥ 200 (1/6 trials better); 15 kpc 12–15% worse at n = 500 only. NFW: 15 kpc
  6–22% worse at n ≥ 500 (0–2/6 better), where the far field is 75–100% wrong either way.
- **Density: noisy, no consistent direction.** Median |ρ_tot| error ratios at R = 2 kpc
  swing 0.59–1.62 between n with wide IQRs; midplane dex unchanged (0.97–1.05); no negative-
  density pixels in either variant except a few at n ≥ 1000. Only systematic shift: BFE,
  R = 4 kpc, ρ_tot +3.4 points higher without scaling (23/30 trials).
- **Sun offset** (soft-anchor residual) 1.3–1.9× larger without scaling for the BFE at
  n ≥ 200; for the NFW it goes either way.

**Why so little.** Over the evaluated volume s varies only between 1 and 0.6, so it is nearly
a constant rescaling of the network output, which Adam absorbs. Its one structural effect is
that a constant network output c becomes physical: −c∇s is a radial pull and c∇²s ∝
−c/[r̃(1+r̃)²] an NFW-shaped density. That one-parameter "halo mass" mode helps only away from
the data, where nothing else constrains the correction; near the data the pulsars pin the field
either way.

**χ-stop caveat.** With the NFW baseline at n ≥ 1000 both variants mostly never reach the χ
target in 2500 epochs (default 2/6 and 1/6 crossed, no scaling 2/6 and 0/6; final χ
0.81–0.82 vs 0.798) and fall back to the last epoch, so those rows compare equal budgets
rather than equal fits.
