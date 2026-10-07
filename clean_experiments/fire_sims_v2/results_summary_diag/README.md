# Diagnostic runs (2026-09-29): chi-stop by region, noisy Sun anchor

Scratch variant `.claude-tmp/fire_sims_v2_diag/val_split_check_diag.py` of ../val_split_check.py
(BFE, relative loss, LOS+Sun+rho, no val split): records per-epoch errors in uniform bubbles
R = 0.5 / 2 / 4 kpc (absolute and observer-referenced) and the Sun offset, and optionally makes
the Sun anchor a noisy measurement (`--sun-noise-frac 0.1`: rms vector error 10% of |a_sun|,
one draw per trial, seed 31337 + trial; the pulsar data stay relative to the TRUE a_sun).
Gate: with no Sun noise it reproduces the v2 runs' chi-stop epoch and rec4 for 30/30 models.

Runs: `results_diag_sun_exact{,_large}`, `results_diag_sun_noise10{,_large}`.
`report.md` = tables; `trajectories.png` = error vs epoch per region (each normalized to its own
best, median of 6 trials, 15-epoch running mean for display), chi-stop marked.
Analysis: `.claude-tmp/fire_sims_v2_diag/analyze_diag.py`.
