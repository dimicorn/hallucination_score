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

and dD/du = a Cov[u_0 | u_t] / s^2 is PSD, so for the true smoothed prior every
directional curvature lies in [~0, 1/s^2]: the prior can never look more constraining
than the smoothing scale allows. ``curvature_cap_matvec`` is that 1/s^2 ceiling mapped
to x-space (dominant term of the change of variables); I_P / I_cap is the fraction of
the ceiling that the learned prior actually uses along a direction.
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

    def with_weight(self, weight):
        return ScorePrior(self.t, weight, self.jacobian_term)
