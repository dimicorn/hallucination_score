"""Pixel- and feature-level prior/data attribution at a reconstruction x_hat.

Pixel level (Sampson & Melchior 2023, eq. 5): Delta F = diag(F_P) - diag(F_D), positive
where the prior's curvature dominates.

Feature level, local directional curvature (plan section C1): for feature phi_k with
gradient g_k at x_hat and unit direction v_k = g_k / ||g_k||,

    I_D = v^T F_D v >= 0,   I_P = v^T F_P v (any sign),
    R   = I_P / (I_P + I_D),   D = (I_P - I_D) / (I_P + I_D).

F_P already includes the prior weight lambda (prior.py), so R is the lambda-weighted
ratio. These are *directional* quantities: curvature along one image direction with all
other directions held fixed. R is the local prior-curvature share along v, not the
fraction of the feature supplied by the prior, and I_D is not the data's marginal
information about phi_k once nuisance directions may vary.

R is only reported when I_P >= 0 and the total curvature is not ~0. On the feature's own
scale, sigma_loc = ||g|| / sqrt(I_P + I_D) and sigma_data = ||g|| / sqrt(I_D) are the
widths of phi_k along the line x_hat + a v. By Cauchy-Schwarz, ||g|| / sqrt(v^T M v) <=
sqrt(g^T M^-1 g) for positive definite M, so each is a lower bound on the corresponding
nuisance-adjusted width when computed from the same matrix (sigma_loc uses the raw F_P
while the Laplace sigma_post of laplace.py uses its positive part, so sigma_loc <=
sigma_post fails in a few rows). sigma_data stays finite even when g has components the
data cannot see at all (null space of A); laplace.py reports that null-space fraction
and evaluate.py flags features dominated by it. A feature is flagged weakly constrained
when sigma_data exceeds its spread over the reference population; the flag ignores the
prior, whose curvature alone keeps sigma_loc small even with no data at all.

Because the prior curvature scales with the diffusion smoothing (see prior.py), R's level
depends on t. We therefore also report kappa = I_P / I_cap and kappa_u (the share of the
u-space Tweedie ceiling, prior.kappa_u), and the absolute data curvature I_D.
``diagonal_attribution`` is the same ratio with the off-diagonal curvature discarded.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import curvature, features


def pixel_scores(obs, prior, x, n_probes=1024, key=None, exact=False, dense=None):
    """Return dict with diag_D, diag_P and delta = diag_P - diag_D.

    ``dense`` = (F_D, F_P) as already-built dense matrices (implies exact)."""
    if dense is not None:
        diag_d = np.diag(dense[0]).reshape(x.shape)
    elif obs.blur_cutoff is None:
        diag_d = np.asarray(obs.curvature_diag())
    else:
        diag_d = np.diag(curvature.exact_matrix(obs.curvature_matvec, x.shape)).reshape(x.shape)
    mv_p = lambda v: prior.curvature_matvec(x, v)
    if dense is not None:
        diag_p = np.diag(dense[1]).reshape(x.shape)
    elif exact:
        diag_p = np.diag(curvature.exact_matrix(mv_p, x.shape)).reshape(x.shape)
    else:
        key = jax.random.PRNGKey(0) if key is None else key
        diag_p, _ = curvature.hutchinson_diag(mv_p, x.shape, n_probes, key)
    diag_cap = np.asarray(prior.curvature_cap_matvec(x, jnp.ones_like(x)))
    return {"diag_D": diag_d, "diag_P": diag_p, "delta": diag_p - diag_d, "diag_cap": diag_cap, "kappa": diag_p / diag_cap}


INVALIDATING_FLAGS = {"negative_prior_curvature", "near_zero_total"}


@dataclass
class FeatureAttribution:
    name: str
    value: float
    grad_norm: float
    I_D: float
    I_P: float
    I_cap: float  # smoothing-scale ceiling on I_P (prior.curvature_cap_matvec)
    flags: list = field(default_factory=list)
    kappa_u: float = np.nan  # share of the u-space Tweedie ceiling (prior.kappa_u)

    @property
    def kappa(self):
        """Fraction of the ceiling used by the prior: ~1 prior pins the direction, ~0 flat."""
        return self.I_P / self.I_cap if self.I_cap > 0 else np.nan

    @property
    def total(self):
        return self.I_D + self.I_P

    @property
    def sigma_local(self):
        return self.grad_norm / np.sqrt(self.total) if self.total > 0 else np.inf

    @property
    def sigma_data(self):
        return self.grad_norm / np.sqrt(self.I_D) if self.I_D > 0 else np.inf

    @property
    def valid(self):
        """R and D are defined (weakly constrained features keep R, but are flagged)."""
        return not set(self.flags) & INVALIDATING_FLAGS

    @property
    def R(self):
        return self.I_P / self.total if self.valid else np.nan

    @property
    def D(self):
        return (self.I_P - self.I_D) / self.total if self.valid else np.nan

    def as_dict(self):
        return {
            "feature": self.name,
            "value": self.value,
            "grad_norm": self.grad_norm,
            "I_D": self.I_D,
            "I_P": self.I_P,
            "I_cap": self.I_cap,
            "kappa": self.kappa,
            "kappa_u": self.kappa_u,
            "total": self.total,
            "sigma_local": self.sigma_local,
            "sigma_data": self.sigma_data,
            "R": self.R,
            "D": self.D,
            "flags": ",".join(self.flags),
        }


def feature_attribution(obs, prior, x, population_std=None, rel_tol=1e-6):
    """C1 diagnostic for every feature in ``features.FEATURES``."""
    out = []
    for name, f in features.FEATURES.items():
        g = features.GRADIENTS[name](x)
        gn = float(jnp.linalg.norm(g))
        if not gn > 0:  # zero or non-finite gradient: all curvatures 0, flagged near_zero_total
            g, gn = jnp.zeros_like(g), 0.0
        v = g / gn if gn > 0 else g
        i_d = curvature.quadratic_form(obs.curvature_matvec, v)
        i_p = curvature.quadratic_form(lambda u: prior.curvature_matvec(x, u), v)
        i_cap = curvature.quadratic_form(lambda u: prior.curvature_cap_matvec(x, u), v)
        fa = FeatureAttribution(name, float(f(x)), gn, i_d, i_p, i_cap, kappa_u=prior.kappa_u(x, v) if gn > 0 else np.nan)
        scale = abs(i_d) + abs(i_p)
        if i_p < 0:
            fa.flags.append("negative_prior_curvature")
        if fa.total <= rel_tol * scale or fa.total <= 0:
            fa.flags.append("near_zero_total")
        elif population_std is not None and fa.sigma_data > population_std[name]:
            fa.flags.append("weakly_constrained")
        out.append(fa)
    return out


def diagonal_attribution(diag_d, diag_p, v):
    """The directional ratio with off-diagonal curvature discarded (review ablation):
    R_diag = v^T diag(F_P) v / (v^T diag(F_D) v + v^T diag(F_P) v), and the v^2-weighted
    mean of the per-pixel ratio diag(F_P) / (diag(F_P) + diag(F_D))."""
    w = np.asarray(v, np.float64).ravel() ** 2
    d, p = np.asarray(diag_d, np.float64).ravel(), np.asarray(diag_p, np.float64).ravel()
    i_d, i_p = float(w @ d), float(w @ p)
    total = d + p
    pixel = np.divide(p, total, out=np.full_like(total, np.nan), where=total > 0)
    ok = np.isfinite(pixel)
    return {
        "R_diag": i_p / (i_p + i_d) if i_p + i_d > 0 else np.nan,
        "pixel_ratio_vw": float(w[ok] @ pixel[ok] / w[ok].sum()) if w[ok].sum() > 0 else np.nan,
    }
