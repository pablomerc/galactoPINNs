# Exercise 1, with no magic: Gaussian conditioning from the density up

Companion to `GP_WALKTHROUGH.md` §1. The walkthrough's Exercise 1 asks you to
derive

$$\mathbb{E}[t \mid d=y] = C_{td}\, C_{dd}^{-1}\, y, \qquad
\operatorname{Var}[t \mid d=y] = C_{tt} - C_{td}\, C_{dd}^{-1}\, C_{dt}$$

for a zero-mean jointly Gaussian pair $(t, d)$ with covariance blocks
$C_{tt}, C_{td}, C_{dd}$. There are two honest routes. Route A is two lines and
rests on two lemmas; Route B writes out the density, the block inverse, the
completed square, and the marginalization integral — everything, once. Do B on
paper a single time; carry A with you forever.

Notation note: your $(y_a, y_b)$ is our $(t, d)$, and $\Sigma_{AB}\Sigma_{BB}^{-1}$
is $C_{td}C_{dd}^{-1}$.

---

## 1. The two lemmas (the actual primitives)

Everything Gaussian reduces to these. Both are proved with the characteristic
function of a Gaussian vector $v \sim N(\mu, \Sigma)$ in $\mathbb{R}^n$:

$$\varphi_v(u) \equiv \mathbb{E}\,e^{iu^\top v} = e^{iu^\top\mu - \tfrac12 u^\top \Sigma u},$$

together with the fact that the characteristic function determines the
distribution.

**Lemma 1 (linear images are Gaussian).** For any $m\times n$ matrix $A$:
$Av \sim N(A\mu,\, A\Sigma A^\top)$.

*Proof.* $\varphi_{Av}(w) = \mathbb{E}\,e^{iw^\top Av} = \varphi_v(A^\top w)
= e^{iw^\top A\mu - \tfrac12 w^\top A\Sigma A^\top w}$, which is the
characteristic function of $N(A\mu, A\Sigma A^\top)$. ∎

**Lemma 2 (uncorrelated + jointly Gaussian ⇒ independent).** If $(z, d)$ is
jointly Gaussian and $\operatorname{Cov}(z, d) = 0$, then $z$ and $d$ are
independent.

*Proof.* The joint covariance is block diagonal, so the quadratic form in the
joint characteristic function splits:
$\varphi_{(z,d)}(u_1, u_2) = \varphi_z(u_1)\,\varphi_d(u_2)$, and a factorizing
characteristic function is equivalent to independence. ∎

*(Caution: Lemma 2 needs **joint** Gaussianity. The classic counterexample —
$z = s\,x$ with $x \sim N(0,1)$ and $s = \pm1$ a fair coin — has $z$ Gaussian,
$x$ Gaussian, $\operatorname{Cov}(z,x)=0$, yet $z, x$ wildly dependent; the
pair is not jointly Gaussian.)*

## 2. Route A: decorrelate and read off

Define the residual of the best linear guess,

$$z \;=\; t - C_{td}C_{dd}^{-1} d.$$

By Lemma 1 (applied to the linear map $(t,d) \mapsto (z,d)$), the pair $(z,d)$
is jointly Gaussian. Its cross-covariance:

$$\operatorname{Cov}(z, d) = C_{td} - C_{td}C_{dd}^{-1}C_{dd} = 0,$$

so by Lemma 2, $z \perp d$. Now write $t = z + C_{td}C_{dd}^{-1}d$ and condition
on $d = y$: the second term becomes the constant $C_{td}C_{dd}^{-1}y$, and $z$
is *unaffected by the conditioning* (independence), keeping its unconditional
law $N(0, \operatorname{Var}(z))$ with

$$\operatorname{Var}(z) = C_{tt} - 2\,C_{td}C_{dd}^{-1}C_{dt}
 + C_{td}C_{dd}^{-1}C_{dd}C_{dd}^{-1}C_{dt}
 = C_{tt} - C_{td}C_{dd}^{-1}C_{dt}.$$

Hence $t \mid y \sim N\!\big(C_{td}C_{dd}^{-1}y,\;
C_{tt} - C_{td}C_{dd}^{-1}C_{dt}\big)$. ∎

Note what was *used*: covariance algebra only — no density anywhere. Hold that
thought for §5.

## 3. Route B: the density, the block inverse, the completed square

Let $C = \begin{pmatrix} C_{tt} & C_{td} \\ C_{dt} & C_{dd} \end{pmatrix}$ and
write the joint density

$$p(t, d) = \frac{1}{(2\pi)^{(m+n)/2}\,(\det C)^{1/2}}
\exp\!\left(-\tfrac12 \begin{pmatrix} t \\ d\end{pmatrix}^{\!\top} C^{-1}
\begin{pmatrix} t \\ d\end{pmatrix}\right).$$

Everything now hinges on inverting $C$ in blocks.

### 3.1 Block LDU factorization (the Schur complement)

Define the shorthand $M = C_{td}C_{dd}^{-1}$ and the **Schur complement**

$$S \;=\; C_{tt} - C_{td}C_{dd}^{-1}C_{dt}.$$

> **📄 Verify by block multiplication** (this is Gaussian elimination done in
> blocks — subtract $M\times$(second block row) from the first):
> $$C = \begin{pmatrix} I & M \\ 0 & I \end{pmatrix}
> \begin{pmatrix} S & 0 \\ 0 & C_{dd} \end{pmatrix}
> \begin{pmatrix} I & 0 \\ C_{dd}^{-1}C_{dt} & I \end{pmatrix}.$$

Two immediate payoffs. Determinant (each triangular factor has determinant 1):

$$\det C = \det S \cdot \det C_{dd}.$$

And the inverse, by inverting the three factors in reverse order (a unit
triangular block matrix inverts by flipping the sign of its off-diagonal
block):

$$C^{-1} = \begin{pmatrix} I & 0 \\ -C_{dd}^{-1}C_{dt} & I \end{pmatrix}
\begin{pmatrix} S^{-1} & 0 \\ 0 & C_{dd}^{-1} \end{pmatrix}
\begin{pmatrix} I & -M \\ 0 & I \end{pmatrix}.$$

### 3.2 The quadratic form splits — completing the square, organized

Sandwich the vector $(t, d)$ around that factored inverse. The right factor
maps $(t, d) \mapsto (t - Md,\; d)$, and the left factor is its transpose, so

> **📄 Verify by expanding:**
> $$\begin{pmatrix} t \\ d\end{pmatrix}^{\!\top} C^{-1}
> \begin{pmatrix} t \\ d\end{pmatrix}
> \;=\; (t - Md)^\top S^{-1} (t - Md) \;+\; d^\top C_{dd}^{-1} d.$$

This *is* "completing the square in $t$" — the Schur factorization is the
bookkeeping that does it for you. With the determinant identity, the
normalization splits the same way, and the joint density factors exactly:

$$\boxed{\;p(t, d) \;=\;
\underbrace{N\!\big(t;\; Md,\; S\big)}_{p(t\mid d)}\;\cdot\;
\underbrace{N\!\big(d;\; 0,\; C_{dd}\big)}_{p(d)}.\;}$$

### 3.3 Conditioning and "the integral", both for free

- **Conditioning** is division: $p(t\mid y) = p(t, y)/p(y)$. Fix $d = y$ in
  the box; the second factor cancels, leaving
  $p(t \mid y) = N\!\big(t;\; C_{td}C_{dd}^{-1}y,\; S\big)$ — the walkthrough's
  formula, mean and covariance both. ∎
- **Marginalization** is the integral you asked about — and after the square
  is completed it is trivial: $\int p(t, d)\, dt$ integrates the *first*
  factor, a normalized Gaussian in $t$, to exactly 1, leaving
  $p(d) = N(0, C_{dd})$. (Consistency check: the marginal of a block of a
  Gaussian is the Gaussian with that block's covariance — no new computation
  was ever needed.)

## 4. Sanity check in one dimension

Scalars with correlation $\rho$: $C_{tt} = \sigma_t^2$,
$C_{dd} = \sigma_d^2$, $C_{td} = \rho\,\sigma_t\sigma_d$. The formulas give

$$\mathbb{E}[t\mid y] = \rho\,\frac{\sigma_t}{\sigma_d}\,y, \qquad
\operatorname{Var}[t\mid y] = \sigma_t^2\,(1 - \rho^2)$$

— regression to the mean, and a variance that shrinks with $\rho^2$
*independently of $y$*. The whole "posterior variance is data-value-free"
theme of the walkthrough (§1, §9) is already visible in this one line.

## 5. Which route "counts" — and why the GP case forces the answer

For finite-dimensional vectors the two routes are equivalent and both fully
rigorous. For the GP, only Route A survives: an infinite-dimensional function
space carries **no Lebesgue density**, so "write the joint density of
$\delta\Phi$ and complete the square" cannot even be *stated*. What one
actually does — and what the walkthrough's §2 does — is apply the
finite-dimensional result to the finitely many linear functionals ever probed
(53 data rows + your targets), which is legitimate because a GP is *defined*
by its finite-dimensional marginals. Route A is the derivation whose every
step is covariance algebra, so it passes to that setting verbatim.

One more reason Route B was still worth your ink: the block-inverse identity
of §3.1, applied with the blocks (existing 53 rows | one candidate row), is
*exactly* the computation of walkthrough **Exercise 8** — the fig-9 rank-1
update that scores the next measurement. You have now already done the hard
part of it.
