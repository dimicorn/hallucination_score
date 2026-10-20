"""MAP reconstruction: maximise log p(y|x) + lambda log p(x) subject to x >= 0.

Only the score of the prior is available, so we run projected Adam on the gradient
directly rather than on a loss value.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def map_estimate(obs, prior, y, x0=None, steps=1500, lr=1e-2, anneal_from=None):
    """Returns (x_map, norm of the projected gradient at the last step).

    With ``anneal_from`` the prior's diffusion time is decreased geometrically from
    that value to ``prior.t`` over the first 2/3 of the steps (coarse-to-fine), then
    held at ``prior.t``, so the returned point is a stationary point of the final
    objective.
    """
    x = jnp.clip(y, 0, None) if x0 is None else jnp.asarray(x0)
    n_anneal = 2 * steps // 3 if anneal_from else 0
    times = jnp.full(steps, prior.t, jnp.float32)
    if n_anneal:
        # geometric schedule cannot reach t=0; stop at 1e-3 and hold prior.t afterwards
        times = times.at[:n_anneal].set(jnp.geomspace(anneal_from, max(prior.t, 1e-3), n_anneal))
    x, gnorm = _run(obs, prior, y, x, times, lr)
    return x, float(gnorm)


@eqx.filter_jit
def _run(obs, prior, y, x, times, lr):
    opt = optax.adam(optax.cosine_decay_schedule(lr, times.shape[0], alpha=0.01))

    def step(carry, t):
        x, state = carry
        g = -(obs.score(x, y) + prior.score(x, t))
        # pixels pinned at the x >= 0 bound by a positive gradient are stationary
        projected = jnp.where((x > 0) | (g < 0), g, 0.0)
        updates, state = opt.update(g, state)
        x = jnp.clip(optax.apply_updates(x, updates), 0, None)
        return (x, state), jnp.linalg.norm(projected)

    (x, _), gnorms = jax.lax.scan(step, (x, opt.init(x)), times)
    return x, gnorms[-1]
