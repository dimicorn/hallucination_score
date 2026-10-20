"""Curvature estimators built only from matrix-vector products.

``hutchinson_diag`` reproduces Algorithm 1 of Sampson & Melchior (2023):
diag(F) ~ E_z[z * (F z)] for Rademacher z. ``exact_matrix`` materialises F from n
matvecs and is affordable for 32x32 images (n = 1024), which gives an exact reference.
"""

import jax
import jax.numpy as jnp
import numpy as np


def hutchinson_diag(matvec, shape, n_probes, key, batch=128, checkpoints=()):
    """Diagonal estimate from ``n_probes`` Rademacher probes.

    Returns (estimate, {n: estimate_after_n_probes for n in checkpoints}).
    """
    mv = jax.jit(jax.vmap(matvec))
    total = jnp.zeros(shape)
    seen = 0
    snapshots = {}
    checkpoints = sorted(c for c in checkpoints if c <= n_probes)
    while seen < n_probes:
        b = min(batch, n_probes - seen)
        if checkpoints and seen < checkpoints[0] < seen + b:
            b = checkpoints[0] - seen
        key, sub = jax.random.split(key)
        z = jax.random.rademacher(sub, (b, *shape), dtype=jnp.float32)
        total = total + jnp.sum(z * mv(z), axis=0)
        seen += b
        while checkpoints and checkpoints[0] == seen:
            snapshots[seen] = np.asarray(total / seen)
            checkpoints.pop(0)
    return np.asarray(total / seen), snapshots


def exact_matrix(matvec, shape, batch=256):
    """Dense F (n x n) with F[:, i] = F e_i."""
    n = int(np.prod(shape))
    mv = jax.jit(jax.vmap(matvec))
    cols = []
    for start in range(0, n, batch):
        idx = jnp.arange(start, min(start + batch, n))
        basis = jax.nn.one_hot(idx, n, dtype=jnp.float32).reshape(-1, *shape)
        cols.append(np.asarray(mv(basis)).reshape(len(idx), n))
    return np.concatenate(cols).T


def quadratic_form(matvec, v):
    return float(jnp.vdot(v, matvec(v)))
