# How the GP in `06_gp.ipynb` works — a pen-and-paper walkthrough

This document is meant to be worked through **on paper**. It derives every formula
that `clean_experiments/fire_sims/06_gp.ipynb` uses, in the order the notebook uses
them, and maps each result to the exact function in the code. Boxes marked
**📄 Exercise** are the parts to actually do by hand; compact worked answers are
collected at the end so you can check yourself.

Companion: `gp_tutorial.ipynb` in this folder builds the same machinery in code,
with fill-in-the-blank cells. Do the two in either order — they reference each other.

---

## 0. The problem and the objects

**Data.** For one trial at $n=50$: pulsar positions $p_j \in \mathbb{R}^3$ (kpc),
unit sightlines from the Sun $\hat u_j = (p_j - x_\odot)/\lVert p_j - x_\odot\rVert$,
and noisy line-of-sight accelerations

$$y_j = \hat u_j \cdot a(p_j) + \varepsilon_j, \qquad
\varepsilon_j \sim N(0, \sigma_j^2), \qquad a = -\nabla\Phi .$$

The catalog $\sigma_j$ are in mm/s/yr and converted to kpc/Myr² (`MMSYR_TO_KPCMYR2`).
Three extra rows are the **Sun anchor**: the same functional with
$\hat u = e_x, e_y, e_z$ at $x_\odot$, with an assumed
$\sigma_\odot = 0.1$ mm/s/yr. So the data vector has $n+3 = 53$ entries, and —
this is the key structural fact — **every entry is a linear functional of $\Phi$.**

**Residualization.** The GP does not model $\Phi$ itself but the residual against
the frozen analytic baseline $\Phi_{AB}$ (in `06_gp.ipynb`, the sim-fitted
$n,l\le4$ BFE):

$$\delta\Phi = \Phi - \Phi_{AB}, \qquad
r_j = y_j - \hat u_j \cdot a_{AB}(p_j).$$

Why: the GP prior below is zero-mean, so whatever mean field you believe in must be
subtracted first; and the baseline is what the posterior reverts to far from data,
so it carries the extrapolation (the same "baseline is the extrapolation lever"
lesson as the NN ladder). Every prediction at the end is
$\text{field} = \text{AB baseline} + \delta(\text{GP posterior})$. In the code:
`build_rows` (cell 3) does the subtraction; `replay_trial` (cell 1) reconstructs
bit-exactly the data the NN trained on, so GP and NN are compared on identical draws.

**Prior.** $\delta\Phi \sim \mathcal{GP}(0,\; k)$ with the squared-exponential
(SE / RBF) kernel, amplitude included:

$$k(x, x') = A^2 \exp\!\left(-\frac{\lVert x-x'\rVert^2}{2\ell^2}\right).$$

Two hyperparameters only: $\ell$ (kpc) — the correlation length of the residual
field — and $A$ (units of $\delta\Phi$, kpc²/Myr²) — its typical size. Both are fit
by evidence (§6). One bookkeeping note for reading the code: `kfun` in cell 3 is
this kernel at *unit* amplitude, multiplied by `A2` at assembly time — so the
evidence grid can reuse one expensive `K_unit` matrix across all amplitudes. Same
mathematics, factored differently; on paper the $A^2$ stays inside $k$.

---

## 1. Warm-up: Gaussian conditioning, the one formula everything rests on

If a jointly Gaussian pair $(t, d)$ has

$$\begin{pmatrix} t \\ d \end{pmatrix} \sim
N\!\left( 0,\;
\begin{pmatrix} C_{tt} & C_{td} \\ C_{dt} & C_{dd} \end{pmatrix} \right),$$

then conditioning on $d = y$ gives

$$\boxed{\;\mathbb{E}[t \mid y] = C_{td}\, C_{dd}^{-1}\, y, \qquad
\operatorname{Var}[t \mid y] = C_{tt} - C_{td}\, C_{dd}^{-1}\, C_{dt}.\;}$$

In our case $d$ = the 53 rows (so $C_{dd} = K + N$ with $N = \operatorname{diag}(\sigma_j^2)$,
the noise being independent of the field) and $t$ = whatever you want to predict.

> **📄 Exercise 1.** Derive the boxed formula. Write the joint density, treat $d=y$
> as fixed, and complete the square in $t$. Equivalently: verify that
> $t - C_{td}C_{dd}^{-1}d$ is uncorrelated with $d$ (hence independent — Gaussian!),
> and read the mean and variance off that decomposition. The second route is two
> lines and is the one worth remembering. Every step of both routes written out
> in full (the two lemmas, the Schur block inverse, the completed square, the
> marginal integral): `ex1_gaussian_conditioning.md` in this folder.

Note what the variance formula does **not** contain: $y$. The error bars of a
linear-Gaussian experiment depend only on its *design* (geometry, prior, noise
levels). That single fact powers the whole of §9.

---

## 2. From vectors to fields: linear functionals of a GP

A GP is the function-space version of the above: any finite collection of *linear
functionals* of $f \sim \mathcal{GP}(0, k)$ is jointly Gaussian, with

$$\operatorname{Cov}\big(L_1[f],\, L_2[f]\big) = L_1^{x}\, L_2^{x'}\, k(x, x'),$$

i.e. apply the first functional to $k$ in its **first** argument and the second in
its **second** argument. Evaluation $f(x)$, directional derivatives
$\hat u\cdot\nabla f(p)$, and the Laplacian $\nabla^2 f(x)$ are all linear
functionals, so **derivatives of a GP are again a GP** (for kernels smooth enough —
the SE kernel is $C^\infty$, which we need since the density variance takes four
derivatives of $k$).

Intuition for why the rule holds: a derivative is a limit of finite differences,
finite differences are linear combinations of evaluations, and covariance is
bilinear — so the derivative operators pass through onto each argument of $k$.

**This is "the derivative trick":** we never differentiate data or fit gradients
numerically. We differentiate the *kernel*, analytically (in the code:
by autodiff), and then everything is ordinary $53\times 53$ linear algebra.

---

## 3. Every row and every target, with the signs done carefully

The observable and targets, as functionals of $\delta\Phi$ (this is cell 2's table,
now with the minus-sign bookkeeping written out):

| object | functional of $\delta\Phi$ | covariance expression | order |
|---|---|---|---|
| data row $j$ | $-\hat u_j\cdot\nabla\,\delta\Phi(p_j)$ | rows $i,j$: $(\hat u_i\cdot\nabla_x)(\hat u_j\cdot\nabla_{x'})\,k$ | 2nd |
| potential target | $\delta\Phi(x)$ | vs row $j$: $-\hat u_j\cdot\nabla_{x'} k(x, p_j)$ | 1st |
| acceleration target | $-\partial_k\,\delta\Phi(x)$ | vs row $j$: $\partial_{x_k}\big(\hat u_j\cdot\nabla_{x'}k\big)$ | 2nd |
| density target | $\nabla^2\delta\Phi(x)/4\pi G$ | vs row $j$: $-\nabla_x^2\big(\hat u_j\cdot\nabla_{x'}k\big)$ | 3rd |
| density prior var | — | $\nabla_x^2\nabla_{x'}^2\, k\big|_{x=x'}$ | 4th |

Sign rules to internalize:

- A data row carries one minus (from $a = -\nabla\Phi$). **Row–row covariances have
  two rows ⇒ the minuses cancel** — that's the comment in `cov_rr`.
- Potential-vs-row and density-vs-row have **one** row each ⇒ one surviving minus.
- Acceleration-vs-row has a minus from the target *and* one from the row ⇒ they
  cancel again (`cov_avec_row` has no minus).

The 53×53 data covariance is then $C = K + N$ where
$K_{ij} = (\hat u_i\cdot\nabla_x)(\hat u_j\cdot\nabla_{x'})k(p_i, p_j)$ — the Sun
rows are nothing special, just $\hat u = e_k$ at $x_\odot$ with their own $\sigma$.

---

## 4. The SE kernel derivatives, by hand — the core paper section

Everything in the table reduces to derivatives of the SE kernel. The code gets them
by autodiff; **your job is to get them by hand**, because the closed forms are (a)
short, (b) the unit test for the code, and (c) where all the physical intuition
lives. Work with the separation vector

$$r \equiv x - x', \qquad k = A^2 e^{-\lVert r\rVert^2 / 2\ell^2},$$

writing every result as $(\text{prefactor})\cdot k$ — then the $A^2$ rides along
inside $k$ automatically.

A workhorse identity you will use twice — for a radial function $g(s)$ with
$s = \lVert r\rVert^2$:

$$\nabla g = 2 g'(s)\, r, \qquad \nabla^2 g = 2d\, g'(s) + 4s\, g''(s)
\quad (d = \text{dimension} = 3).$$

### 4.1 First derivatives

> **📄 Exercise 2.** Chain rule. Show
> $$\nabla_x k = -\frac{r}{\ell^2}\,k, \qquad \nabla_{x'} k = +\frac{r}{\ell^2}\,k.$$
> (Opposite signs because $k$ depends only on $x - x'$.) Hence the potential-vs-row
> covariance (`cov_val_row`), with $r = x - p_j$:
> $$\operatorname{Cov}\big(\delta\Phi(x), \text{row}_j\big)
>   = -\hat u_j\cdot\nabla_{x'}k = -\frac{(\hat u_j\cdot r)}{\ell^2}\,k .$$

Read it: a datum saying "the field slopes this way along $\hat u_j$" pulls the
potential *up* on one side of $p_j$ and *down* on the other — the covariance is
odd in $\hat u_j\cdot r$ and dies off with the kernel.

### 4.2 Second derivatives — the row–row matrix

> **📄 Exercise 3.** Differentiate Exercise 2's result once more to get the full
> Hessian in mixed arguments:
> $$\frac{\partial^2 k}{\partial x\,\partial x'} =
>   \left(\frac{I}{\ell^2} - \frac{r\,r^\top}{\ell^4}\right) k,$$
> and therefore the row–row covariance (`cov_rr`):
> $$\operatorname{Cov}(\text{row}_i, \text{row}_j) =
>   \left[\frac{\hat u_i\cdot\hat u_j}{\ell^2}
>   - \frac{(\hat u_i\cdot r)(\hat u_j\cdot r)}{\ell^4}\right] k,
>   \qquad r = p_i - p_j.$$

Sanity checks to do in your head:

- At $r = 0$: $\operatorname{Cov} = (\hat u_i\cdot\hat u_j)\,A^2/\ell^2$ (using
  $k(0) = A^2$). So the **prior
  variance of one acceleration component is $A^2/\ell^2$** — go find the literal
  `A2 / ell ** 2` in `a_post` (cell 3). Same measurement direction ⇒ full variance;
  orthogonal sightlines through the same point ⇒ *zero* prior covariance. That is
  the transverse-information problem of LOS data expressed in one formula.
- Two parallel sightlines separated *along* their common direction
  ($\hat u_i = \hat u_j \parallel r$): covariance $\propto (1 - \lVert r\rVert^2/\ell^2)k$ —
  goes *negative* beyond $\lVert r\rVert = \ell$ (a slope up here means a slope
  down past the correlation length, since the field turns over). Separated
  *transverse* to it: $\propto k > 0$ always.

### 4.3 Third derivatives — density vs a data row

> **📄 Exercise 4.** Show that the density-target covariance (`cov_lap_row`), with
> $r = x - p_j$, is
> $$\operatorname{Cov}\big(\nabla^2\delta\Phi(x), \text{row}_j\big)
>  = -\nabla_x^2\!\left[\frac{(\hat u_j\cdot r)}{\ell^2} k\right]\cdot(-1)
>  = \frac{(\hat u_j\cdot r)}{\ell^4}\left(d + 2 - \frac{\lVert r\rVert^2}{\ell^2}\right) k
>  \;\overset{d=3}{=}\;
>  \frac{(\hat u_j\cdot r)}{\ell^4}\left(5 - \frac{\lVert r\rVert^2}{\ell^2}\right) k .$$
> Cleanest route: apply $\nabla^2(\varphi\psi) = \psi\nabla^2\varphi
> + 2\nabla\varphi\cdot\nabla\psi + \varphi\nabla^2\psi$ with
> $\varphi = \hat u_j\cdot r$ (so $\nabla\varphi = \hat u_j$, $\nabla^2\varphi = 0$)
> and $\psi = k/\ell^2$, using the workhorse identity for $\nabla^2 k$.

Sanity checks:

- **Zero at $r = 0$.** An acceleration measurement at $x$ says *nothing* about the
  density at $x$ itself — curvature is only constrained by the *pattern* of
  neighboring slopes. This is the honest form of "ρ is two derivatives from the
  data".
- Zero crossing at $\lVert r\rVert = \sqrt{5}\,\ell$, and maximal sensitivity at
  order-$\ell$ separations along the sightline: the data that constrain
  $\rho_\odot$ are the *nearby* ones (foreshadowing fig 9's "nearby, vertical,
  precise").

### 4.4 Fourth derivatives — the density prior variance

> **📄 Exercise 5.** First show (workhorse identity again)
> $$\nabla^2_{x'} k = \frac{1}{\ell^2}\left(\frac{\lVert r\rVert^2}{\ell^2} - d\right)k,$$
> then apply $\nabla_x^2$ to that and evaluate at $r = 0$ to get
> $$\nabla_x^2 \nabla_{x'}^2\, k\,\big|_{x=x'} = \frac{d(d+2)}{\ell^4}\,A^2
>   \;\overset{d=3}{=}\; \frac{15\,A^2}{\ell^4}.$$
> This is $A^2\times$ `lap_lap0` — the code computes the unit-amplitude number
> with two `jax.hessian` traces and multiplies by `A2` where it is used.

### 4.5 The scaling ladder

Collect the prior standard deviations you have now derived:

$$\sigma(\delta\Phi) = A, \qquad
\sigma(\delta a_k) = \frac{A}{\ell}, \qquad
\sigma(\nabla^2\delta\Phi) = \frac{\sqrt{15}\,A}{\ell^2}
\;\;\Rightarrow\;\;
\sigma(\delta\rho) = \frac{\sqrt{15}}{4\pi G}\frac{A}{\ell^2}.$$

Each derivative order costs a factor $\ell$. This *is* the statement "density is
the hardest target": with $\ell$ of a few kpc fit by evidence, the prior allows —
and the data barely constrain — short-scale curvature, and a 0.3 kpc thin disk
lives entirely below the resolvable scale (fig 5's diagnosis).

---

## 5. What the code actually does: autodiff instead of your formulas

Cell 3 of `06_gp.ipynb` never writes §4's closed forms. It writes the *functionals*
and lets JAX differentiate:

```python
def kfun(x1, x2, ell):                       # unit-amplitude SE kernel
    return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / ell ** 2)

def cov_rr(p1, u1, p2, u2, ell):
    """row-row: (u1 . grad_x)(u2 . grad_x') k — the minus signs cancel."""
    g = lambda a: jnp.dot(jax.grad(kfun, argnums=1)(a, p2, ell), u2)
    return jnp.dot(jax.grad(g)(p1), u1)
```

Read `cov_rr` against Exercise 3: `jax.grad(kfun, argnums=1)` is $\nabla_{x'}k$;
dotting with `u2` applies row $j$'s functional; wrapping that in a function of the
*first* argument and taking `jax.grad` again applies row $i$'s. The other
covariances are the same pattern one order up or down: `cov_val_row` is Exercise 2
(note its explicit minus — the single surviving sign of §3), `cov_avec_row` is the
Hessian contracted on one side only, `cov_lap_row` takes
`jnp.trace(jax.hessian(...))` of `cov_val_row` (Exercise 4), and `lap_lap0` nests
two Hessian traces (Exercise 5). `K_unit` is a doubly-`vmap`ped `cov_rr` — the full
53×53 matrix in one line.

Why autodiff if the formulas are four lines each? Because the *code* stays
sign-mistake-proof and kernel-agnostic (swap `kfun`, everything downstream is still
correct), while **your §4 results are its unit test** — the tutorial notebook makes
you check them against finite differences.

`build_rows` then stacks positions `P` (50 pulsars + Sun×3), directions `U`
(sightlines + $e_x, e_y, e_z$), residual labels `r`, and noise variances — the
data-side half of §1's formula.

---

## 6. Fitting $\ell$ and $A$: the evidence

With $C(\ell, A) = K + N$ (when you need $A$ explicit, write $K = A^2 K_u$ with
$K_u$ the unit-amplitude kernel matrix — the code's factorization), the marginal
likelihood of the residual labels
is a plain multivariate Gaussian, so

$$-\log p(r \mid \ell, A) =
\underbrace{\tfrac12\, r^\top C^{-1} r}_{\text{data fit}}
+ \underbrace{\tfrac12 \log\det C}_{\text{complexity}}
+ \tfrac{53}{2}\log 2\pi .$$

The two active terms fight: a huge-$A$, tiny-$\ell$ prior can fit anything (first
term ↓) but pays in volume (second term ↑) — Occam's razor is built in, no
validation set needed. `neg_log_evidence` computes it via Cholesky
($C = LL^\top$): solve triangular systems for $r^\top C^{-1}r$ and read
$\log\det C = 2\sum_i \log L_{ii}$ off the diagonal. `fit_hypers` is then just a
13×17 grid over $\ell \in [0.5, 12]$ kpc (geometric) × $\log_{10}A \in [-5, -1]$,
plus a 1-D bounded refinement of $\log_{10}A$ at the winning $\ell$. One Cholesky
per grid point at 53×53: the whole fit is well under a second — which is why the
notebook fits all six trials while the NN could afford one.

> **📄 Exercise 6 (optional).** Differentiate the evidence with respect to $A^2$
> (use $\partial \log\det C = \operatorname{tr}(C^{-1}\partial C)$ and
> $\partial C^{-1} = -C^{-1}\partial C\, C^{-1}$) and set it to zero. Interpret the
> resulting condition $r^\top C^{-1} K_u C^{-1} r = \operatorname{tr}(C^{-1}K_u)$ as
> "the data's whitened power in the prior's directions matches its expectation".

---

## 7. The posterior evaluators (`make_gp`)

Everything expensive is done once: Cholesky of $C = K + N$, and
$\alpha = C^{-1} r$. Then §1's formula, specialized per target, gives each
evaluator in cell 3:

| evaluator | cross-cov vector $K_x$ | mean | variance |
|---|---|---|---|
| `phi_post` | `cov_val_row`$(x, p_j, \hat u_j)$ | $K_x^\top\alpha$ | $A^2 - K_x^\top C^{-1} K_x$ |
| `a_post` (per comp.) | `cov_avec_row` | $K_x^\top\alpha$ | $A^2/\ell^2 - \ldots$ |
| `lap_post` | `cov_lap_row` | $K_x^\top\alpha$ | $15A^2/\ell^4 - \ldots$ |

Note the prior-variance column is exactly §4.5's ladder. Costs: $O(53^3)$ once,
then per target point $O(53)$ for the mean and $O(53^2)$ for the variance (one
`cho_solve`). The final physical fields are the AB baseline + these residual
posteriors, with unit conversions from §10.

---

## 8. The gauge: what gradient data cannot know

Every row is a derivative, so no datum constrains the additive constant of
$\delta\Phi$: the posterior variance of $\delta\Phi(x)$ at any single point stays
at order $A^2$ no matter how much data you have. Quoting that as "the" error bar
of a $\Phi$ map would be dishonest in the *other* direction — the map's zero point
was never claimed. The measured object is the **difference functional**

$$t = \delta\Phi(x) - \delta\Phi(x_\odot),$$

which is itself linear, so §1 applies directly:

> **📄 Exercise 7.** For two jointly Gaussian posterior quantities show
> $$\operatorname{Var}[t] = \operatorname{Var}[\delta\Phi(x)]
> + \operatorname{Var}[\delta\Phi(x_\odot)]
> - 2\operatorname{Cov}[\delta\Phi(x), \delta\Phi(x_\odot)],$$
> with the posterior cross-covariance
> $\operatorname{Cov} = k(x, x_\odot) - K_x^\top C^{-1} K_\odot$.
> Then read `phi_gauged` in cell 3 line by line — the `cov_xs` line is precisely
> this cross term. The unconstrained constant cancels in $t$, and its variance
> honestly shrinks with data.

This is the band drawn in fig 2, and the same gauge freedom you'll rediscover in
Part 5 of the tutorial notebook with 1-D derivative data.

---

## 9. Fig 9: scoring the next measurement before it is measured

Because $\operatorname{Var}[t\mid y]$ never contains $y$ (§1), the value of a
*hypothetical* 54th row — a new pulsar at position $p$, sightline $\hat u$,
precision $\sigma_{\rm new}$ — is computable in closed form before anyone measures
it. Let the target be $t = \nabla^2\delta\Phi(x_\odot)$ (i.e. $\rho_\odot$, up to
$4\pi G$ and the deterministic $\Phi_{AB}$ part), and define

- $c_t$: the target-vs-rows covariance vector, entries from Ex. 4 evaluated at
  $(x_\odot, p_j, \hat u_j)$ (in the code, `A2 * cov_lap_row(...)`), with prior
  variance $v_t = 15A^2/\ell^4$,
- $k_n$: the candidate's covariances with the existing 53 rows (Ex. 3),
- $c_{tn}$: candidate vs target (Ex. 4 again), and $v_n = A^2/\ell^2$ the
  candidate's own prior variance.

> **📄 Exercise 8.** Append the candidate to the data covariance,
> $$C_+ = \begin{pmatrix} C & k_n \\ k_n^\top & v_n + \sigma_{\rm new}^2 \end{pmatrix},$$
> and use the block (Schur-complement) inverse to show
> $$\operatorname{Var}_{54} = \operatorname{Var}_{53} -
> \frac{\big(c_{tn} - c_t^\top C^{-1} k_n\big)^2}
>      {v_n + \sigma_{\rm new}^2 - k_n^\top C^{-1} k_n}.$$
> Hint: you only need the last row/column of $C_+^{-1}$, and
> $\operatorname{Var}_{54} = v_t - (c_t^\top, c_{tn})\, C_+^{-1} (c_t^\top, c_{tn})^\top$.

Read the two pieces:

- **Numerator** — the candidate's covariance with the target *after projecting out
  what the 53 rows already know* ($C^{-1}k_n$ is the representation of the
  candidate in the existing data; $c_t^\top C^{-1}k_n$ is the part of its
  target-relevance already covered). Redundancy is penalized automatically —
  duplicating your best existing pulsar scores far below a genuinely new
  direction, and the notebook prints that check.
- **Denominator** — the candidate's *own* fresh variance plus its noise: a noisy
  or predictable row is worth little.

The score reported is $\Delta\text{bits} = \tfrac12\log_2
(\operatorname{Var}_{53}/\operatorname{Var}_{54})$, and cell 29 evaluates it
exhaustively on an $(\ell, b)$ sky grid in ~0.1 s — no gradient ascent. Two
framing points worth holding on to: this is **c-optimality** (variance of a *named*
science target — a whole-sky "most uncertain point" map is degenerate for a
stationary kernel and answers nothing), and the tempting alternative
$\partial L/\partial x_i$ is residual chasing with expectation zero under a
correct model (`experiments/gp_audit/REPORT.md` shows it selecting the
$|\chi|\approx 10$ systematics outliers on the real catalog). The physics the map
returns — vertical, nearby, and above all *precise* — is Exercise 4's covariance
shape plus the $N_{\rm eff}\approx 3.5$ lesson.

---

## 10. Units ledger

| quantity | internal unit | display unit | conversion in code |
|---|---|---|---|
| positions, $\ell$ | kpc | kpc | — |
| acceleration, rows, $\sigma_j$ | kpc/Myr² | mm/s/yr | `MMSYR_TO_KPCMYR2`, `KPCMYR2_TO_MMSYR` |
| $\delta\Phi$, $A$ | kpc²/Myr² | (km/s)² | `U2KMS2` |
| $\nabla^2\delta\Phi$ | 1/Myr² | — | — |
| density | $M_\odot$/kpc³ via $\rho = \nabla^2\Phi/4\pi G$ | $M_\odot$/pc³ | `FOUR_PI_G`, then `TO_MSUN_PC3` $=10^{-9}$ |

$G$ is taken in kpc³/($M_\odot$·Myr²) from astropy. The GP works natively in
physical units — no transformers anywhere; the only assembly is baseline + residual.

---

## 11. What the Gaussian cannot say (why the PINN exists)

Worth stating after all this closed-form elegance: the GP posterior is the *exact*
Bayesian answer only within its assumptions — Gaussian prior, linear observables,
Gaussian noise. Three things live outside them: the inequality $\rho \ge 0$
(non-Gaussian information — fig 4's masked negative-density pixels are the
Gaussian being honestly Gaussian); non-Gaussian FIRE texture; and any structure
below $\ell$ (the thin disk — fig 5 shows the truth *outside* the ±2σ band at the
midplane, a prior-*class* failure, not a noise-level one). That triple is the
entire justification budget for the neural network at $n=50$, since on field
recovery alone the GP matches or beats it. That is the "floor" argument of
notebook 06 and `docs/gp-wiener-audit.md`.

---

## Worked answers

**Ex. 1.** Let $z = t - C_{td}C_{dd}^{-1}d$. Then
$\operatorname{Cov}(z, d) = C_{td} - C_{td}C_{dd}^{-1}C_{dd} = 0$, so $z \perp d$
(jointly Gaussian ⇒ independent). Hence
$t = z + C_{td}C_{dd}^{-1}d$ with $z$ unaffected by conditioning:
$\mathbb{E}[t|y] = C_{td}C_{dd}^{-1}y$ and
$\operatorname{Var}[t|y] = \operatorname{Var}(z) = C_{tt} - C_{td}C_{dd}^{-1}C_{dt}$. ∎

**Ex. 2.** With $s = \lVert r\rVert^2$, $k = e^{-s/2\ell^2}$,
$\nabla_x s = 2r$, $\nabla_{x'} s = -2r$:
$\nabla_x k = -\tfrac{r}{\ell^2}k$, $\nabla_{x'} k = +\tfrac{r}{\ell^2}k$.
Then $-\hat u_j\cdot\nabla_{x'}k = -\tfrac{(\hat u_j\cdot r)}{\ell^2}k$. ∎

**Ex. 3.** $\partial_{x'}\!\left(-\tfrac{r}{\ell^2}k\right)$: the $r$ factor gives
$+I/\ell^2\,k$ (since $\partial r/\partial x' = -I$), the $k$ factor gives
$-\tfrac{r}{\ell^2}\otimes \tfrac{r}{\ell^2}k$. Total
$\big(\tfrac{I}{\ell^2} - \tfrac{rr^\top}{\ell^4}\big)k$; contract with
$\hat u_i, \hat u_j$. ∎

**Ex. 4.** $\nabla^2\big[(\hat u\cdot r)\,\tfrac{k}{\ell^2}\big]
= 2\hat u\cdot\nabla\tfrac{k}{\ell^2} + (\hat u\cdot r)\nabla^2\tfrac{k}{\ell^2}
= -\tfrac{2(\hat u\cdot r)}{\ell^4}k
+ \tfrac{(\hat u\cdot r)}{\ell^4}\big(\tfrac{s}{\ell^2} - d\big)k
= \tfrac{(\hat u\cdot r)}{\ell^4}\big(\tfrac{s}{\ell^2} - d - 2\big)k$.
The target covariance is minus this (one surviving row sign):
$\tfrac{(\hat u\cdot r)}{\ell^4}\big(d + 2 - \tfrac{s}{\ell^2}\big)k$. ∎

**Ex. 5.** Workhorse identity with $g = e^{-s/2\ell^2}$:
$g' = -\tfrac{1}{2\ell^2}g$, so
$\nabla^2_{x'}k = 2d\,g' + 4s\,g'' = \tfrac{1}{\ell^2}\big(\tfrac{s}{\ell^2}-d\big)k$.
Now let $h(s) = \tfrac{1}{\ell^2}\big(\tfrac{s}{\ell^2}-d\big)e^{-s/2\ell^2}$ and
apply the identity again in $x$ at $s=0$: $\nabla_x^2 h\big|_0 = 2d\,h'(0)$ with
$h'(0) = \tfrac{1}{\ell^4}\big(1 + \tfrac d2\big)$, giving $d(d+2)/\ell^4$. ∎

**Ex. 6.** With $C = A^2K_u + N$:
$\tfrac{\partial}{\partial A^2}\big[\tfrac12 r^\top C^{-1}r
+ \tfrac12\log\det C\big]
= -\tfrac12 r^\top C^{-1}K_uC^{-1}r + \tfrac12\operatorname{tr}(C^{-1}K_u) = 0$.
With whitened residuals $w = L^{-1}r$: the power of $w$ projected onto the prior's
eigendirections equals its expected value — the amplitude is set where surprise
vanishes. ∎

**Ex. 7.** Bilinearity of covariance on $t = t_1 - t_2$:
$\operatorname{Var}[t] = \operatorname{Var}[t_1] + \operatorname{Var}[t_2]
- 2\operatorname{Cov}[t_1, t_2]$, each term the *posterior* one; the posterior
cross-covariance is §1's formula with mixed target vectors:
$k(x, x_\odot) - K_x^\top C^{-1}K_\odot$. ∎

**Ex. 8.** Write $\operatorname{Var}_{54} = v_t - c_+^\top C_+^{-1} c_+$ with
$c_+ = (c_t^\top, c_{tn})^\top$. Block inversion with Schur complement
$s = v_n + \sigma^2_{\rm new} - k_n^\top C^{-1}k_n$:
$$C_+^{-1} = \begin{pmatrix}
C^{-1} + \tfrac1s C^{-1}k_nk_n^\top C^{-1} & -\tfrac1s C^{-1}k_n \\
-\tfrac1s k_n^\top C^{-1} & \tfrac1s \end{pmatrix}.$$
Expanding $c_+^\top C_+^{-1}c_+$ gives
$c_t^\top C^{-1}c_t + \tfrac1s\big(c_t^\top C^{-1}k_n - c_{tn}\big)^2$, i.e.
$\operatorname{Var}_{54} = \operatorname{Var}_{53} - \tfrac1s\big(c_{tn}
- c_t^\top C^{-1}k_n\big)^2$. Non-negative gain, zero iff the candidate is
target-redundant. ∎
