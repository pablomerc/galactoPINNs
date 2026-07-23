# noisy_data on an AMD-GPU cluster — hand-off for the cluster session

**Read this first if you're a Claude Code session (or Pablo) picking this up on the AMD cluster.**
It's the port of the `noisy_data` experiment from a sequential CPU driver to a GPU-parallel one.
Everything below the "Feasibility / install / speedup" line was researched and adversarially
fact-checked against AMD's own docs on 2026-07-22; corrections from that pass are folded in.

---

## What was built (and already verified on CPU)

| File | Role |
|------|------|
| `run_experiment.py` | Original **sequential** driver — one model at a time, 228 fits × 1500 epochs. Ground truth. |
| `run_experiment_vmap.py` | **NEW GPU-parallel driver.** Trains every `(trial × setting × mode)` fit at once as **two batched ensembles** via `jax.vmap` over stacked params + `jax.lax.scan` over epochs. Same CLI, same `results.csv`/`per_point_errors.npz` format. |
| `run_amd.slurm` | SLURM template for one AMD GPU. |

**Why the vmap driver is trustworthy:** its `--self-test` trains the same models both ways
(this driver vs `galactoPINNs.train.train_model_static`) and asserts per-model parameter
agreement. On CPU it passes at **~1e-8 reldiff across every arm** — plain L1, whitened-W,
χ²/sq-C, anchor+ρ, and the 3D reference. It reproduces the full pipeline; end-to-end `results.csv`
values match the sequential driver to float32 trajectory noise (2e-7 at 30 epochs, growing with
epochs as adam amplifies op-reorder differences — **this is expected, not a bug**; the sibling
`sun_los_recovery/run_experiment_vmap.py` documents the same).

**The design, in one paragraph:** all 9 settings of a given `(n, trial)` share the *same*
geometry (`x_n`, sightlines) — they differ only in the noise realization of the target, the
per-point importance weights, and whether the loss is `l1` or `sq`. So they collapse into one
computation graph with per-model switches `(w3d, wlos, anchor_on, λ_anchor, λ_rho, sq_on)` plus a
per-model importance-weight vector `w_ip (N,)`. Models are split into a **no-ρ group** (LOS + the
3D ref) and a **ρ group** (LOS+Sun+ρ, which pays the Laplacian/Hessian cost) so the no-ρ models
never compute a Hessian.

---

## Do this first, in order

1. **Check the cluster's ROCm minor version — this decides everything.**
   ```bash
   cat /opt/rocm/.info/version 2>/dev/null || rocminfo | grep -i version || module avail rocm
   ```
   - **ROCm 7.2.x → good**, jax 0.8.2 is supported.
   - **ROCm 6.x → hard blocker.** ROCm 6.4 tops out at jax 0.4.35; this repo needs `jax>=0.8.2`.
     You must get a ROCm-7.2 module/container, or use the container route (below), which carries its own.

2. **Get JAX-on-ROCm.** Preferred = AMD's prebuilt container (exact tested combo, avoids host matching):
   ```bash
   apptainer pull docker://rocm/jax:rocm7.2.4-jax0.8.2-py3.12    # or -py3.11
   ```
   Pip route (only if the system ROCm minor matches the plugin's target):
   ```bash
   pip install --upgrade "jax[rocm7-local]"      # -local = ROCm already installed on the host
   ```
   Then reinstall the repo's other deps (`uv sync` / `pip install -e .`) **without** letting them pull a
   CPU/CUDA jaxlib over the ROCm one. Confirm the backend:
   ```bash
   python -c "import jax; print(jax.default_backend(), jax.devices())"   # want: 'gpu' / [Rocm...]
   ```

3. **Verify correctness on this GPU build before trusting any science:**
   ```bash
   cd clean_experiments/noisy_data
   python run_experiment_vmap.py --self-test
   ```
   Expect PASS. **Caveat:** GPUs are nondeterministic run-to-run (atomics in GEMM/scatter, on both
   CUDA and ROCm), so validate by **tolerance, not bit-equality**. The check's threshold is 1e-4; a
   GPU reldiff of ~1e-5..1e-4 is fine. If it "fails" at ~1e-3 that's still just op-ordering drift,
   not a wrong answer — sanity is that the number is *tiny* (≤~1e-3), not O(1).

4. **Run it** (from inside this dir, so `import noisy_datasets` resolves — same as the sequential driver):
   ```bash
   python run_experiment_vmap.py --matmul-precision highest         # writes ./results_vmap/
   ```
   Compare `results_vmap/results.csv` to the sequential `results.csv` **statistically** (medians per
   `n×setting×mode`), never bit-exact. Useful knobs: `--n-grid`, `--settings`, `--trials`, `--epochs`,
   `--outdir`, `--jax-cache DIR`.

---

## What speedup to expect (honest)

There is **no public benchmark for this exact hot path** (trace-of-Hessian inside a gradient, under
vmap+scan, on ROCm). **Micro-benchmark it** — the driver prints per-group wall time; compare one MI
GCD against CPU-XLA. Structural model:

- **The win is NOT raw TFLOPS.** A single width-128 MLP on 50–1000 points uses only ~2–3% of a GPU.
  On **CPU, batching measured ~1× (no gain)** on this workload — so the GPU is the whole point, but
  *because it fills otherwise-idle capacity*, not because of peak flops.
- **`vmap` batching is the dominant lever:** packing ~18 models (each group) into one kernel trains
  them in ≈ the wall-clock of *one* model while below saturation → **throughput ≈ number of models,
  up to ~18× and nearly free.** 18 small MLPs won't even saturate an MI250X/MI300X, so batching all
  trials together (the driver already does) is essentially free headroom.
- **`lax.scan` is what makes tiny-per-step work GPU-viable** — it fuses the 1500-epoch loop into one
  program, killing Python/host dispatch overhead. (Note: XLA lowers `scan` to a device-side
  while-loop, so per-iteration *kernel launches* remain; if per-step work is too tiny, `unroll=` helps.)
- **Apply a ~10–30% ROCm-vs-CUDA software-maturity discount** as a loose prior (from LLM benchmarks,
  not this pattern).

**Anchor number:** the full sequential run is **~41 min on a 14-core M-series laptop CPU**
(dominated by the 108 `LOS+Sun+ρ` fits at ~19 s each — Hessian over 512 collocation points). The
GPU batched runner should beat that meaningfully on one MI GCD, but **measure before quoting a factor.**
More trials for tighter error bars are close to free on GPU (raise `--trials`).

---

## Correctness caveats (ROCm-specific)

- **fp32 matmuls stay true fp32 on ROCm** — it does *not* silently downcast to a TF32-equivalent the
  way A100/H100 do under `Precision.DEFAULT`. Good for a curvature/Hessian penalty. The driver already
  sets `--matmul-precision highest` by default; keep it.
- **fp64 is available** if the Laplacian ever needs it: `jax.config.update("jax_enable_x64", True)`
  (fp64 is an Instinct strength). Not currently needed.
- **Determinism:** run-to-run nondeterminism is expected (see step 3). It's a reproducibility property,
  not a bug, and is harmless for the `relu(-laplacian)` hinge. If you ever need bit-reproducibility,
  the flags exist (`--xla_gpu_exclude_nondeterministic_ops`, older `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
  + `TF_DETERMINISTIC_OPS=1`) at a throughput cost — **verify the exact flag name against the jaxlib
  0.8.2 build**, I couldn't pin a version-specific source.

---

## SLURM essentials (see `run_amd.slurm`)

- **1 "GPU" = 1 GCD = 64 GB** on MI250X (a board is 2 GCDs / 128 GB, each seen as a separate device).
  Don't assume 128 GB / 2 devices. 18 small MLPs fit trivially in 64 GB.
- Use portable `--gres=gpu:1`. For a lone single-task job, Slurm's cgroup already isolates the GCD —
  no `ROCR_VISIBLE_DEVICES` wrapper needed (that's only for packing multiple tasks/node). For device
  selection prefer `ROCR_VISIBLE_DEVICES` over `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`.
- **Point `JAX_COMPILATION_CACHE_DIR` at a shared FS path** — the big fused program is slow to compile;
  cache is keyed by HLO + backend + jaxlib version + XLA flags (so array tasks with identical shapes
  hit it). Keep it in your own space (cache is trusted = code exec).
- **n-grid sweep → job array** (`#SBATCH --array=0-N%K`), one `--gres=gpu:1` per task, index by
  `SLURM_ARRAY_TASK_ID`; `results.csv` files concatenate cleanly. Share the compile cache across tasks.

---

## Uncertainty ledger — verify, don't assume

1. **Cluster ROCm minor** (6.x = blocker). *Check first.*
2. **Exact determinism flag for jaxlib 0.8.2** — unverified at version granularity.
3. **This hessian-in-grad hot path's performance** — no public benchmark; micro-benchmark yourself.
4. **`--xla_gpu_enable_command_buffer=` workaround** — version-specific; test if still needed.
5. All TFLOPS→wall-clock and ROCm-vs-CUDA numbers are general/transformer priors, not measured for a
   small-MLP higher-order-AD PINN.

---

*Provenance: `run_experiment_vmap.py` generalizes `../sun_los_recovery/run_experiment_vmap.py` (same
vmap/scan/group-by-ρ mechanics) with two additions `noisy_data` needs — a per-model importance-weight
vector and a per-model `l1`/`sq` residual switch. CPU self-test + end-to-end diff verified 2026-07-22.*
