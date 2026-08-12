# visualizations — recovered vs. true potential

`potential_field.py` takes **one** trained model from the
[`noisy_data`](../noisy_data) experiment and shows what its recovered
gravitational potential Φ looks like next to the truth
(`galax.MilkyWayPotential`).

Default model: **LOS-only, catalog noise, `w ∝ 1/σ`** (`setting = hetW`),
`n = 50` sightlines, `trial = 0` — the honest ~51-pulsar regime.

```bash
uv run python clean_experiments/visualizations/potential_field.py
```

Runtime ≈ 15 s (rebuilds the pool, trains 1500 epochs, renders 3 figures).
Outputs land next to the script:

| file | what it shows |
|---|---|
| `fig1_potential_slices.png` | 3 planes × [true Φ, pred Φ, residual] heatmaps |
| `fig2_radial_profiles.png`  | Φ vs distance-from-Sun along 4 rays + residual |
| `fig3_accel_error.png`      | gauge-free \|a\| relative-error slices |

---

## The one thing you must understand: gauge freedom

Training only ever sees acceleration **a = −∇Φ** — and in LOS mode only its
projection onto the sightline. Adding any **constant** to Φ leaves the loss
unchanged, so **the absolute zero-point of Φ_pred is not determined by the
data.** A raw `Φ_pred − Φ_true` map would be dominated by that meaningless
offset.

Every comparison in the script removes it, two ways:

- **Maps & profiles** are referenced to the Sun: we plot `Φ − Φ(Sun)` for both
  fields, so they share a zero at the observer.
- **The residual map** removes the best-fit constant over the training disk
  (subtract the mean of `Φ_pred − Φ_true` for `r < 4 kpc`), then reports the
  in-ball RMS. This is the fairest "how wrong is the shape" measure.

`fig3` (acceleration) needs none of this — it's the trained, gauge-free
quantity, and it's exactly what `results.csv` reports as `rel_err`.

---

## How to read the figures

**Coordinate frame:** Sun-centered. The origin is the Sun (gold ★); the dashed
circle at `r = 4 kpc` is the training-ball edge; the window is `[−6, 6] kpc`, so
the outer ring is a 2-kpc extrapolation margin. The GC is 8.1 kpc away in the
`+x` direction (arrow, or "out of plane" for the transverse `yz` panel). White
dots are the training sightline points whose out-of-plane offset is `< 0.5 kpc`
— they show *where the data actually is*.

**The three planes** all pass through the Sun:
- `xy` (z=0) — the disk plane (Sun & GC lie in it); most training points project
  here.
- `xz` (y=0) — the meridional plane (Sun & GC in it).
- `yz` (x=x_Sun) — the transverse plane; the Sun→GC line is the plane normal.

**fig1.** True and pred columns look near-identical — the deep smooth well
toward the GC dominates. The **residual column is the payload**: small inside
`r < 4`, growing outside, and *anisotropic* (it has lobes, not rings) because
LOS data constrains Sun-radial gradients far better than transverse ones.

**fig2.** Solid = true, dashed = pred. They track tightly to `r = 4`, then peel
apart — the picture of "constrained inside the ball, extrapolating beyond." The
residual panel shows the Sun→GC and anti-GC radial rays extrapolate worst, while
the vertical (+z) ray stays near zero.

**fig3.** Low error (dark) hugs the training disk; a bright error tongue points
toward the GC (`+x`, ~5 kpc out) where Φ steepens and extrapolation fails
hardest.

---

## How the "exact model" is reproduced

The driver never saves models — only `results.csv` and per-point errors. So the
script **retrains** one model, replaying the driver's seeding recipe verbatim
(see [`run_experiment.py`](../noisy_data/run_experiment.py)):

- `POOL = 30000` and `np.random.default_rng(TRIAL)` for the subsample — the pool
  **size must match**, or `rng.choice` picks different points.
- per-family noise seeds (`9000 + 137·trial + n`, `7000 + …`) so the noise
  realization is identical.

A built-in check prints `rel_err(sun4)` and the `results.csv` reference; they
agree to ~0.4 % (platform float nondeterminism), confirming it's *that* model.

Units: potential is shown in **(km/s)²** — galax returns kpc²/Myr²; we multiply
by `977.79²`.

---

## Visualizing a different model

Edit the constants near the top of `potential_field.py`:

```python
N, TRIAL = 50, 0                      # sightline count / trial
FAMILY, ARM, LOS_LOSS = "het", "W", "l1"   # -> setting hetW
```

`(FAMILY, ARM, LOS_LOSS)` map to the driver's `SETTINGS` labels, e.g.
`("het", "plain", "l1") = het`, `("het", "C", "sq") = hetC`,
`("s1.7", "W", "l1") = s1.7W`, `("s0", "plain", "l1") = s0` (noiseless parent
baseline). To see the **LOS+Sun+ρ** mode instead of LOS-only, the training call
would need the anchor/collocation kwargs added (see the driver's `lsr_kw`).
