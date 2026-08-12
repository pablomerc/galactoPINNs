"""Pure-JAX Hernquist-Ostriker basis-function-expansion (BFE) baseline.

A drop-in replacement for the analytic NFW `ab_potential` used by the
architecture ladder (PINN IV/V): a low-order SCF expansion (n <= 4, l <= 4)
fit to the m12i particle table itself, i.e. a *well-specified* smooth
baseline, in contrast to the deliberately misspecified MW-tuned NFW.

Conventions follow gala.potential.scf / Lowing et al. (2011):

    Phi(r,th,ph) = G M / r_s * sum_nlm  Phi_nl(s) * Ntilde_lm P_lm(cos th)
                     * [S_nlm cos(m ph) + T_nlm sin(m ph)]
    Phi_nl(s)    = - s^l (1+s)^-(2l+1) C_n^(2l+3/2)((s-1)/(s+1))
    Ntilde_lm    = sqrt((2l+1) (l-m)!/(l+m)!)        (sqrt(4pi) folded in)

with P_lm in the scipy convention (Condon-Shortley phase). Verified against
gala's compiled evaluator to ~1e-15 relative in both potential and gradient,
per-(l,m) and full-field (.claude-tmp/bfe_build/calibrate_convention.py,
re-run as part of fit_bfe.py --validate).

Interface contract (mirrors how galax potentials are consumed here):
  * `.potential(x, t)` with a raw (N, 3) kpc array returns a raw (N,) array
    in kpc^2/Myr^2  — the StaticModel / TrainableGalaxPotential path;
  * `.potential(pos, t)` with a coordinax CartesianPos3D returns an unxt
    Quantity — the galactoPINNs.data.scale_data path (u* residual fit);
  * constructible as `PotClass(m=..., r_s=..., units="galactic")` so that
    TrainableGalaxPotential can train (log10 m, r_s) for PINN V — m rescales
    the whole expansion, r_s dilates its radial scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import astropy.units as _au
from astropy.constants import G as _G_SI

G_GALACTIC = float(_G_SI.to(_au.kpc**3 / (_au.Msun * _au.Myr**2)).value)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_COEFFS = CACHE_DIR / "bfe_nmax4_lmax4.npz"


# ---------------------------------------------------------------------------
# basis evaluation (validated against gala to machine precision)
# ---------------------------------------------------------------------------

def _legendre_plm_table(x, sin_theta, lmax):
    """Associated Legendre P_lm(x), 0<=m<=l<=lmax, scipy/CS convention."""
    P = {(0, 0): jnp.ones_like(x)}
    for m in range(1, lmax + 1):
        dfact = 1.0
        for k in range(1, 2 * m, 2):
            dfact *= k
        P[(m, m)] = ((-1.0) ** m) * dfact * sin_theta ** m
    for m in range(0, lmax):
        P[(m + 1, m)] = x * (2 * m + 1) * P[(m, m)]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            P[(l, m)] = ((2 * l - 1) * x * P[(l - 1, m)]
                         - (l + m - 1) * P[(l - 2, m)]) / (l - m)
    return P


def _gegenbauer_table(xi, alpha, nmax):
    """C_n^(alpha)(xi), n=0..nmax, standard three-term recurrence."""
    C = [jnp.ones_like(xi)]
    if nmax >= 1:
        C.append(2.0 * alpha * xi)
    for n in range(2, nmax + 1):
        C.append((2.0 * xi * (n + alpha - 1.0) * C[n - 1]
                  - (n + 2.0 * alpha - 2.0) * C[n - 2]) / n)
    return C


def bfe_potential_raw(xyz, Snlm, Tnlm, G, M, r_s):
    """HO-expansion potential at xyz (..., 3). Differentiable; kpc^2/Myr^2
    when (G, M, r_s) are in (kpc^3/Msun/Myr^2, Msun, kpc)."""
    nmax = Snlm.shape[0] - 1
    lmax = Snlm.shape[1] - 1
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = jnp.sqrt(x * x + y * y + z * z)
    r_safe = jnp.maximum(r, 1e-12)
    s = r_safe / r_s
    xi = (s - 1.0) / (s + 1.0)
    cth = z / r_safe
    sth = jnp.sqrt(jnp.maximum(1.0 - cth * cth, 1e-30))
    phi_az = jnp.arctan2(y, x)

    P = _legendre_plm_table(cth, sth, lmax)

    total = jnp.zeros_like(r)
    for l in range(lmax + 1):
        Cn = _gegenbauer_table(xi, 2.0 * l + 1.5, nmax)
        rad_l = -(s ** l) * (1.0 + s) ** (-(2 * l + 1))
        for m in range(l + 1):
            fac = 1.0
            for k in range(l - m + 1, l + m + 1):
                fac *= k
            norm = np.sqrt((2 * l + 1) / fac)
            radsum = jnp.zeros_like(r)
            for n in range(nmax + 1):
                radsum = radsum + (Snlm[n, l, m] * jnp.cos(m * phi_az)
                                   + Tnlm[n, l, m] * jnp.sin(m * phi_az)) * Cn[n]
            total = total + rad_l * norm * P[(l, m)] * radsum
    return G * M / r_s * total


# ---------------------------------------------------------------------------
# the potential object
# ---------------------------------------------------------------------------

def _coordinax_to_kpc_array(pos):
    """CartesianPos3D -> raw (N, 3) jnp array in kpc."""
    import unxt as u
    from unxt.quantity import AllowValue
    return jnp.stack([u.ustrip(AllowValue, "kpc", pos.x),
                      u.ustrip(AllowValue, "kpc", pos.y),
                      u.ustrip(AllowValue, "kpc", pos.z)], axis=-1)


class BFEPotential(eqx.Module):
    """Fixed-coefficient SCF potential, duck-typed for this experiment."""

    m: jax.Array       # total-mass scale, Msun
    r_s: jax.Array     # basis scale radius, kpc
    Snlm: jax.Array    # (nmax+1, lmax+1, lmax+1)
    Tnlm: jax.Array

    def __init__(self, m, r_s, Snlm, Tnlm, units="galactic"):
        if units not in ("galactic", None):
            raise ValueError("BFEPotential only supports galactic units")
        self.m = jnp.asarray(m)
        self.r_s = jnp.asarray(r_s)
        self.Snlm = jnp.asarray(Snlm)
        self.Tnlm = jnp.asarray(Tnlm)

    def _phi(self, xyz):
        return bfe_potential_raw(xyz, self.Snlm, self.Tnlm,
                                 G_GALACTIC, self.m, self.r_s)

    def potential(self, positions, t=0):
        """kpc^2/Myr^2. Raw array in -> raw array out; coordinax in ->
        unxt Quantity out (the scale_data path calls .ustrip on it)."""
        if hasattr(positions, "x"):          # coordinax CartesianPos3D
            import unxt as u
            xyz = _coordinax_to_kpc_array(positions)
            return u.Quantity(self._phi(xyz), "kpc2/Myr2")
        return self._phi(jnp.asarray(positions))

    def acceleration(self, positions, t=0):
        """-grad(potential), kpc/Myr^2, raw (N, 3) arrays only."""
        xyz = jnp.asarray(positions)
        g = jax.vmap(jax.grad(lambda p: self._phi(p[None, :])[0]))(
            jnp.atleast_2d(xyz))
        return -(g if xyz.ndim > 1 else g[0])


def make_bfe_class(Snlm, Tnlm):
    """Class with (m, r_s) constructor closing over fixed coefficients —
    the PotClass handed to TrainableGalaxPotential for PINN V."""
    S = jnp.asarray(Snlm)
    T = jnp.asarray(Tnlm)

    class _BFEFixedCoeffs(BFEPotential):
        def __init__(self, m, r_s, units="galactic"):
            super().__init__(m=m, r_s=r_s, Snlm=S, Tnlm=T, units=units)

    return _BFEFixedCoeffs


def load_bfe(path=DEFAULT_COEFFS):
    """-> (BFEPotential instance, PotClass for PINN V, meta dict)."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    pot = BFEPotential(m=float(z["m"]), r_s=float(z["r_s"]),
                       Snlm=z["Snlm"], Tnlm=z["Tnlm"])
    return pot, make_bfe_class(z["Snlm"], z["Tnlm"]), meta
