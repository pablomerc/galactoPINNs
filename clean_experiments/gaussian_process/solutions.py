"""Solutions to the fill-ins in gp_tutorial.ipynb.

Rule: a CHECK cell must beat you twice before you open this file.
Each block is the fill-in cell exactly as it should read when complete.
"""


# ==========================================================================
# =========================== FILL-IN 1 ===========================
def se_kernel(x1, x2, A, ell):
    """Squared-exponential kernel  k(x1, x2) = A^2 exp(-(x1-x2)^2 / (2 ell^2)).

    Write it as a single jnp expression (no ifs/loops): then it works on
    scalars AND broadcasts over arrays, and jax can differentiate it.
    """
    return A**2 * jnp.exp(-0.5 * (x1 - x2)**2 / ell**2)


def kernel_matrix(xs1, xs2, A, ell):
    """(n,) and (m,) point arrays -> (n, m) matrix  K[i, j] = k(xs1[i], xs2[j]).

    Hint: broadcasting — feed se_kernel  xs1[:, None]  and  xs2[None, :].
    """
    return se_kernel(xs1[:, None], xs2[None, :], A, ell)


# ==========================================================================
# =========================== FILL-IN 2 ===========================
def gp_posterior(X, y, Xstar, A, ell, sigma):
    """GP regression posterior:  X (n,), y (n,), Xstar (m,) -> mean (m,), cov (m, m).

    Use np.linalg.solve(C, ...) — never an explicit inverse.
    """
    Kxx = np.asarray(kernel_matrix(X, X, A, ell))
    Ksx = np.asarray(kernel_matrix(Xstar, X, A, ell))
    Kss = np.asarray(kernel_matrix(Xstar, Xstar, A, ell))
    C = Kxx + sigma**2 * np.eye(len(X))
    mean = Ksx @ np.linalg.solve(C, y)
    cov = Kss - Ksx @ np.linalg.solve(C, Ksx.T)
    return mean, cov


# ==========================================================================
# =========================== FILL-IN 3 ===========================
import jax.scipy.linalg as jsl

def neg_log_evidence(y, K, nvar):
    """-log p(y)  for  y ~ N(0, C),  C = K + diag(nvar).   (all jnp arrays)

    Steps:
        C     = K + diag(nvar)
        L     = cholesky of C                    (jnp.linalg.cholesky)
        alpha = C^{-1} y                         (jsl.cho_solve((L, True), y))
        return  0.5 * y.alpha  +  sum(log diag L)  +  0.5 * n * log(2 pi)

    Question to answer on paper: why is  sum(log diag L) = 0.5 log det C ?
    """
    C = K + jnp.diag(nvar)
    L = jnp.linalg.cholesky(C)
    alpha = jsl.cho_solve((L, True), y)
    return (0.5 * jnp.dot(y, alpha) + jnp.sum(jnp.log(jnp.diag(L)))
            + 0.5 * y.shape[0] * jnp.log(2 * jnp.pi))


# ==========================================================================
# =========================== FILL-IN 4 ===========================
def fit_hypers(X, y, nvar, ells, As):
    """Exhaustive grid search over (ell, A): return (best_ell, best_A, best_nle).

    For each pair: build the kernel matrix, score it with neg_log_evidence,
    keep the argmin. Brute force is fine here (06_gp.ipynb factors out
    A^2 * K_unit to reuse the expensive part — same idea, optimized).
    """
    best = None
    for ell in ells:
        for A in As:
            K = jnp.asarray(kernel_matrix(X, X, A, ell))
            nle = float(neg_log_evidence(jnp.asarray(y), K, jnp.asarray(nvar)))
            if best is None or nle < best[2]:
                best = (float(ell), float(A), nle)
    return best


# ==========================================================================
# =========================== FILL-IN 5 ===========================
def k_dd(t1, t2, A, ell):
    """Cov(f'(t1), f'(t2)) = d^2 k / dt1 dt2 — one jax.grad per argument.

    No hand-derived formulas allowed: build it from se_kernel with
    jax.grad (argnums=...). The check cell holds you to finite differences.
    """
    return jax.grad(jax.grad(se_kernel, argnums=0), argnums=1)(t1, t2, A, ell)


def k_fd(t, t2, A, ell):
    """Cov(f(t), f'(t2)) = d k / dt2   (derivative on the SECOND argument)."""
    return jax.grad(se_kernel, argnums=1)(t, t2, A, ell)


# ==========================================================================
# =========================== FILL-IN 6 ===========================
def se2(x1, x2, A, ell):        # provided: the same kernel, x now in R^2
    return A**2 * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / ell**2)

def cov_rr(p1, u1, p2, u2, A, ell):
    """Row-row covariance:  (u1 . grad_x)(u2 . grad_x') k(x, x').

    Same pattern as 06_gp.ipynb's cov_rr: an inner lambda that dots
    jax.grad(se2, argnums=1) with u2, then jax.grad of THAT wrt its (first)
    argument, dotted with u1. The two row minus signs cancel.
    """
    g = lambda a: jnp.dot(jax.grad(se2, argnums=1)(a, p2, A, ell), u2)
    return jnp.dot(jax.grad(g)(p1), u1)

def cov_val_row(x, pj, uj, A, ell):
    """Cov(Phi(x), row_j) = -uj . grad_x' k(x, pj) — ONE surviving minus."""
    return -jnp.dot(jax.grad(se2, argnums=1)(x, pj, A, ell), uj)
