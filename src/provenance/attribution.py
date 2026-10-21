"""Pixel- and feature-level prior/data attribution at a reconstruction x_hat.

Pixel level (Sampson & Melchior 2023, eq. 5): Delta F = diag(F_P) - diag(F_D), positive
where the prior's curvature dominates.

Feature level, local directional curvature (plan section C1): for feature phi_k with
gradient g_k at x_hat and unit direction v_k = g_k / ||g_k||,

    I_D = v^T F_D v >= 0,   I_P = v^T F_P v (any sign),
    R   = I_P / (I_P + I_D),   D = (I_P - I_D) / (I_P + I_D).

R is only reported when I_P >= 0 and the total curvature is not ~0. To put curvature
on a feature's own scale we convert it into a local feature std along the line
x_hat + a v: sigma_loc = ||g|| / sqrt(I_P + I_D) for the total curvature and
sigma_data = ||g|| / sqrt(I_D) for the data alone. A feature is flagged weakly
constrained when sigma_data exceeds its spread over the reference population, i.e. when
the measurements alone do not pin it down better than knowing it is a galaxy. The flag
uses the data curvature only: the learned prior's curvature keeps sigma_loc small even
with no data at all, where the prior-dominated reconstruction can still be far off.
This is a local sensitivity diagnostic, not a decomposition of the marginal information
in phi_k (sigma_loc is a lower bound on the Laplace marginal std sqrt(g^T H^-1 g)).

Because I_P is capped by the diffusion smoothing scale (see prior.py), R depends on
the prior's evaluation time t. We therefore also report kappa = I_P / I_cap, the
t-normalised strength of the prior along v, and the absolute data support I_D.
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

    @property
    def kappa(self):
        """Fraction of the ceiling used by the prior: ~1 prior pins the direction, ~0 flat."""
        return self.I_P / self.I_cap

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
            "I_D": self.I_D,
            "I_P": self.I_P,
            "I_cap": self.I_cap,
            "kappa": self.kappa,
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
        v = g / gn if gn > 0 else g  # zero gradient (e.g. asymmetry of a symmetric image): flagged near_zero_total
        i_d = curvature.quadratic_form(obs.curvature_matvec, v)
        i_p = curvature.quadratic_form(lambda u: prior.curvature_matvec(x, u), v)
        i_cap = curvature.quadratic_form(lambda u: prior.curvature_cap_matvec(x, u), v)
        fa = FeatureAttribution(name, float(f(x)), gn, i_d, i_p, i_cap)
        scale = abs(i_d) + abs(i_p)
        if i_p < 0:
            fa.flags.append("negative_prior_curvature")
        if fa.total <= rel_tol * scale or fa.total <= 0:
            fa.flags.append("near_zero_total")
        elif population_std is not None and fa.sigma_data > population_std[name]:
            fa.flags.append("weakly_constrained")
        out.append(fa)
    return out
