# GUIDE 0 — src changes: `importance_weight` pass-through + `los_loss`

Goal: two small, **default-off** edits to `src/galactoPINNs/train.py` that the
noisy experiment's weighted arms need. At the defaults nothing changes —
verified 2026-07-20 against the pre-edit src: training LOS+Sun+rho for 120
epochs gave `max|Δparam| = 0.0` (bit-exact) for both edits at their defaults.

Prereq: none — this comes before everything else in `noisy_data`.

## Why these two changes

**1. Per-point weights exist in src but are unreachable.** `train_step_static`
already applies `importance_weight` to the per-point residual
(`train.py:275-276`), but `train_model_static` — the epoch driver every
experiment calls — neither accepts nor forwards it. It silently vanishes. One
pass-through kwarg fixes that.

**2. The LOS data term is L1; there is no χ² option.** `mean|Δa_los|` is
median regression: unbiased under symmetric noise and robust to the catalog's
heavy σ tail (0.12 → 18.2 mm/s/yr) — a *feature* worth keeping as the default.
But the statistically efficient estimator under Gaussian noise is squared
residuals weighted by 1/σ². A `los_loss: "l1" | "sq"` switch gives us the
proper heteroscedastic-likelihood arm to compare against.

How the pieces compose into the three training arms of the experiment:

| arm | `los_loss` | `importance_weight` | statistical meaning |
|---|---|---|---|
| plain | `"l1"` (default) | `None` (default) | median regression, σ ignored |
| W | `"l1"` | `w ∝ 1/σ`, mean-norm. | whitened L1 (`E|r|/σ` constant per point) |
| C | `"sq"` | `w ∝ 1/σ²`, mean-norm. | whitened χ² (Gaussian likelihood) |
| LLH | `"l1"` | `w = 1/σ` (scaled), unnorm. + `lambda_anchor = 1/σ_sun` | Laplace NLL, physics-set, zero tuned λs |

Two design rules that matter:

- **Whitening matches the residual power**: for `|r|` the noise-equalizing
  weight is `1/σ`; for `r²` it is `1/σ²`. Passing `1/σ²` to the L1 loss
  over-punishes precise points.
- **Weights are mean-normalized to 1** (done driver-side, GUIDE 2). Then
  `weights=ones ≡ unweighted` exactly, and the anchor-vs-data balance inside
  the shared LOS+anchor mean is untouched. This is why the pass-through can
  default to `None` with zero behavior change.

## Edit 1 — `train_step_static`

**(a) `los_loss` must be a static jit argument** (it selects a Python branch at
trace time). `train.py:71`:

```python
# before
@nnx.jit(static_argnames=("target", "ramp_kind", "balance_grads", "line_of_sight"))
# after
@nnx.jit(static_argnames=("target", "ramp_kind", "balance_grads", "line_of_sight", "los_loss"))
```

**(b) signature** — insert one parameter right after `line_of_sight`
(`train.py:79-80`):

```python
    line_of_sight: bool = False,
    los_loss: Literal["l1", "sq"] = "l1",
    lambda_rel: float = 1.0,
```

**(c) validation** — right after the existing `line_of_sight`/`n_vecs` check
(`train.py:214-215`), add:

```python
    if los_loss not in ("l1", "sq"):
        raise ValueError(f"unknown los_loss '{los_loss}' (use 'l1' or 'sq').")
    if los_loss != "l1" and not line_of_sight:
        raise ValueError("los_loss='sq' modifies the line-of-sight loss; it requires line_of_sight=True.")
```

Raise, don't ignore: the T1/T2 harness bug (a flag that silently added zero)
is exactly the failure mode this prevents. A `los_loss="sq"` that silently
does nothing on a 3D run would poison every downstream comparison.

**(d) the LOS residual** (`train.py:274`):

```python
# before
        per_point = jnp.abs(a_los_pred - a_los_true)   # (N,)
# after
        resid = a_los_pred - a_los_true                # (N,)
        per_point = jnp.abs(resid) if los_loss == "l1" else resid**2
```

**(e) the anchor residual** (`train.py:282`):

```python
# before
            anchor_abs = jnp.abs(a_anchor_pred - anchor_a).reshape(-1)  # (3M,)
# after
            anchor_res = (a_anchor_pred - anchor_a).reshape(-1)         # (3M,)
            anchor_abs = jnp.abs(anchor_res) if los_loss == "l1" else anchor_res**2
```

The anchor **must** switch with the data term: both live inside one shared
mean (`(Σ per_point + λ Σ anchor) / (N + 3M)`), so their units have to match —
mixing `|r|` data terms with `r²` anchor terms would make `lambda_anchor`
meaningless.

Note what is *not* touched: `_acc_loss` (the 3D branch) — the χ² arm is an
LOS-experiment question, and the (c) guard makes misuse loud.

## Edit 2 — `train_model_static`

**(a) signature** — after `line_of_sight` (`train.py:530-531`):

```python
    line_of_sight: bool = False,
    los_loss: Literal["l1", "sq"] = "l1",
    importance_weight: Array | None = None,
    anchor_x: Array | None = None,
```

**(b) forward both in the epoch loop** — in the `train_step_static` call
(`train.py:660-662`):

```python
                n_vecs=n_vecs,
                line_of_sight=line_of_sight,
                los_loss=los_loss,
                importance_weight=importance_weight,
                anchor_x=anchor_x,
```

**(c) docstrings** — add the two parameter entries to both functions'
docstrings (copy the style of the existing `importance_weight` entry in
`train_step_static`, `train.py:154-157`; for `los_loss`: *"Functional form of
the per-point LOS data term: `"l1"` (default, mean absolute error — median
regression) or `"sq"` (squared residuals; combine with
`importance_weight ∝ 1/σ²` for a whitened χ² data term). Requires
`line_of_sight=True`. The anchor residuals switch with it so both terms share
units inside the common mean."*)

## Verify (freeze-baselines pattern)

Save as `.claude-tmp/verify_guide0.py` and run with
`uv run python .claude-tmp/verify_guide0.py`. All numbers below are exact on
this machine (CPU, seeds fixed) — they were captured against the verified
patch and must reproduce identically with your edit.

```python
"""Verify the GUIDE_0 edits to galactoPINNs.train (defaults must be no-ops)."""
import os, sys
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "clean_experiments", "sun_los_recovery"))
import galax.potential as gp
from datasets import OBSERVER, GC, build_data, sample_colloc_scaled
from galactoPINNs.models.static_model import StaticModel
from galactoPINNs.train import create_optimizer, train_model_static, train_step_static

EPOCHS, N = 120, 50
params_of = lambda m: nnx.to_pure_dict(nnx.split(m)[1])
max_diff = lambda p, q: max(float(jnp.abs(a - b).max()) for a, b in
                            zip(jax.tree.leaves(p), jax.tree.leaves(q)))

true_potential = gp.MilkyWayPotential()
regions = {"sun4": (OBSERVER, 4.0), "sun15": (OBSERVER, 15.0), "gc15": (GC, 15.0)}
(x_pool, a_pool, x_pool_phys, cfg, val_scaled, dist_sun, x_obs, a_obs) = build_data(
    true_potential, pool=2000, n_val=256, r_train=4.0, regions=regions,
    include_analytic=False)
idx = np.random.default_rng(0).choice(x_pool.shape[0], size=N, replace=False)
x_n, a_n = x_pool[idx], a_pool[idx]
dx = x_pool_phys[idx] - OBSERVER
n_vecs = jnp.asarray(dx / np.linalg.norm(dx, axis=1, keepdims=True))
x_colloc = sample_colloc_scaled(jr.PRNGKey(1000), 512, 0.5, 25.0, cfg["x_transformer"])
kw = dict(n_vecs=n_vecs, line_of_sight=True, anchor_x=x_obs, anchor_a=a_obs,
          lambda_anchor=1.0, colloc_x=x_colloc, lambda_rho=1.0)

# 1) weights=ones == weights=None, bit-exact (mean-normalization no-op)
mA = StaticModel(cfg, rngs=nnx.Rngs(0))
train_model_static(mA, optax.adam(1e-3), x_n, a_n, EPOCHS, log_every=0, **kw)
mC = StaticModel(cfg, rngs=nnx.Rngs(0))
train_model_static(mC, optax.adam(1e-3), x_n, a_n, EPOCHS, log_every=0,
                   importance_weight=jnp.ones(N), **kw)
d = max_diff(params_of(mA), params_of(mC))
print(f"1) ones == None:      max|dparam| = {d:.3e}");  assert d == 0.0

# 2) sq loss == hand-computed chi2-style value at step 0
sig = jnp.asarray(np.random.default_rng(1).uniform(0.5, 3.0, size=N))
w = 1.0 / sig**2; w = w / w.mean()
mD = StaticModel(cfg, rngs=nnx.Rngs(0)); mRef = StaticModel(cfg, rngs=nnx.Rngs(0))
loss = train_step_static(mD, create_optimizer(mD, optax.adam(1e-3)), x_n, a_n,
                         los_loss="sq", importance_weight=w, step=0,
                         total_steps=1, **kw)
resid = jnp.sum(mRef(x_n)["acceleration"] * n_vecs, 1) - jnp.sum(a_n * n_vecs, 1)
anch = (mRef(x_obs)["acceleration"] - a_obs).reshape(-1)
lap = mRef.compute_laplacian(x_colloc)
exp = ((jnp.sum(w * resid**2) + jnp.sum(anch**2)) / (resid.size + anch.size)
       + jnp.mean(jax.nn.relu(-lap)))
print(f"2) sq == hand-coded:  loss = {float(loss):.6f}, |d| = {abs(float(loss-exp)):.3e}")
assert abs(float(loss - exp)) < 1e-7

# 3) misuse raises
for bad in (dict(los_loss="sq"), dict(los_loss="huber", n_vecs=n_vecs, line_of_sight=True)):
    m = StaticModel(cfg, rngs=nnx.Rngs(0))
    try:
        train_step_static(m, create_optimizer(m, optax.adam(1e-3)), x_n, a_n, **bad)
    except ValueError as e:
        print(f"3) raises: {e}")
    else:
        raise AssertionError(f"did not raise for {bad}")

# 4) sq training converges on noisy targets
eps = jnp.asarray(np.random.default_rng(2).normal(0.0, 1.0, size=N)) * 0.05
a_noisy = a_n + eps[:, None] * n_vecs
mF = StaticModel(cfg, rngs=nnx.Rngs(0)); optF = create_optimizer(mF, optax.adam(1e-3))
l0 = float(train_step_static(mF, optF, x_n, a_noisy, los_loss="sq",
                             importance_weight=w, step=0, total_steps=EPOCHS, **kw))
for s in range(1, EPOCHS):
    lF = train_step_static(mF, optF, x_n, a_noisy, los_loss="sq",
                           importance_weight=w, step=s, total_steps=EPOCHS, **kw)
print(f"4) sq training: loss {l0:.4f} -> {float(lF):.4f}")
assert float(lF) < l0
print("ALL GUIDE_0 CHECKS PASSED")
```

Expected output (exact):

```
1) ones == None:      max|dparam| = 0.000e+00
2) sq == hand-coded:  loss = 0.145195, |d| = 0.000e+00
3) raises: los_loss='sq' modifies the line-of-sight loss; it requires line_of_sight=True.
3) raises: unknown los_loss 'huber' (use 'l1' or 'sq').
4) sq training: loss 0.1463 -> 0.0015
ALL GUIDE_0 CHECKS PASSED
```

Check 1 proves the pass-through defaults are inert; check 2 pins the sq-loss
math to the formula; the completing check that *defaults reproduce pre-edit
behavior* is GUIDE 2's freeze baseline — the σ=0 rows of the new driver must
match the parent's `results.csv` bit-exactly (the private quick-scan already
reproduced all 81 overlapping rows exactly).

## What this unlocks

- `train_model_static(..., importance_weight=w)` — heteroscedastic weighting
  from any experiment, no custom loops.
- `train_model_static(..., los_loss="sq", importance_weight=w2)` — the χ² arm.
- The Bayesian path (`bayesian_tools.py`) already accepts `sigma_a` +
  `importance_weight`; nothing to do there for phase 2.
