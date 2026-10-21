"""Learned galaxy prior: the pretrained HSC score network from galaxygrad 0.0.19.

The network was trained on peak-normalised SCARLET models mapped to
u = log(1 + x) / 0.1 (this is the transform scarlet2's NNPrior applies; samples from
the network's probability-flow ODE, mapped back through it, have background ~0 and
peak ~0.9, matching peak-normalised galaxies). We change variables back to x:

    log p_x(x) = log p_u(T(x)) + sum_i log T'(x_i),   T'(x) = 1 / (0.1 (1 + x))
    s_x(x)     = T'(x) * s_u(T(x)) - 1 / (1 + x)

The second term (the log-Jacobian) is omitted by scarlet2's NNPrior; it is kept here so
that s_x is the score of a density on x. F_P = -ds_x/dx is a learned Jacobian and is
neither guaranteed symmetric nor positive semidefinite.

Diffusion time ``t``: the paper evaluates the score at t=0, but the training loss is
weighted by 1 - exp(-t), which vanishes at t=0, so the network is effectively
untrained there (pure noise gets as large a t=0 score as real galaxies). We default to
a small t > 0 (set in pipeline.Config) and keep t=0 available for comparison.

Curvature cap. The network is a VP-SDE score with int_beta(t) = t, so u_t = a u_0 + s e
with a = exp(-t/2), s^2 = 1 - exp(-t). Tweedie's second-order identity gives

    -Hess_u log p_t(u) = (I - a dD/du) / s^2,     D(u) = E[u_0 | u_t = u],

and dD/du = a Cov[u_0 | u_t] / s^2 is PSD, so for the exact smoothed prior every
directional curvature in u is at most 1/s^2. This is an upper bound only: between modes
the curvature can be negative. In x the exact identity is

    -Hess_x log p_x = lambda [J F_u J + diag(s_x,i / (1 + x_i))],   J = diag(T'(x)),

with F_u = -Hess_u log p_t (unweighted) and s_x the unweighted x-space score.
``curvature_cap_matvec`` is lambda J (I / s^2) J, i.e. it drops the second, score-
dependent diagonal term (about 1 % of the cap at t = 0.5, mostly negative), so
kappa = I_P / I_cap is a heuristic normalisation by the smoothing scale, not a bounded
fraction. ``curvature_u_matvec`` gives F_u itself: the learned network exceeds the u-space
ceiling by a few per cent along a handful of directions (largest eigenvalue of s^2 F_u
about 1.03-1.06 on HSC reconstructions at t = 0.5), so kappa_u > 1 flags score error.

The prior weight lambda multiplies the score, so F_P, I_P and the cap all include it and
kappa does not depend on it.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from galaxygrad import HSC_ScoreNet32

SIGMA_U = 0.1


@dataclass(frozen=True)
class ScorePrior:
    t: float = 0.01
    weight: float = 1.0  # multiplies log p, i.e. prior strength lambda
    jacobian_term: bool = True

    def score(self, x, t=None):
        u = jnp.log1p(x) / SIGMA_U
        s_u = HSC_ScoreNet32(u[None].astype(jnp.float32), t=self.t if t is None else t)[0]
        s_x = s_u / (SIGMA_U * (1 + x))
        if self.jacobian_term:
            s_x = s_x - 1 / (1 + x)
        return self.weight * s_x

    @property
    def alpha(self):
        return jnp.exp(-self.t / 2)

    @property
    def sigma2(self):
        return -jnp.expm1(-self.t)

    def curvature_cap_matvec(self, x, v):
        """(weight / s^2) (dT/dx)^2 v: the smoothing-scale ceiling on F_P in x-space."""
        return self.weight * v / (self.sigma2 * (SIGMA_U * (1 + x)) ** 2)

    def curvature_matvec(self, x, v):
        """F_P v = -J_s(x) v, one forward-mode pass through the network."""
        return -jax.jvp(self.score, (x,), (v,))[1]

    def network_score(self, u, t=None):
        """Unweighted score of p_t in the network's coordinates u = log(1 + x) / 0.1."""
        return HSC_ScoreNet32(u[None].astype(jnp.float32), t=self.t if t is None else t)[0]

    def curvature_u_matvec(self, x, w):
        """F_u w = -d s_u / du w at u = T(x): the unweighted curvature that Tweedie bounds
        by 1 / s^2."""
        u = jnp.log1p(x) / SIGMA_U
        return -jax.jvp(self.network_score, (u,), (w.astype(jnp.float32),))[1]

    def kappa_u(self, x, v):
        """s^2 (Jv)^T F_u (Jv) / ||Jv||^2: the share of the u-space Tweedie ceiling used
        along the image direction v (<= 1 for an exact score)."""
        jv = v / (SIGMA_U * (1 + x))
        return float(self.sigma2 * jnp.vdot(jv, self.curvature_u_matvec(x, jv)) / jnp.vdot(jv, jv))

    def with_weight(self, weight):
        return ScorePrior(self.t, weight, self.jacobian_term)
