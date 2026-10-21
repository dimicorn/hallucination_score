"""Linear Gaussian observation model y = M (k * x) + eps, eps ~ N(0, sigma^2 / M).

Because the model is linear, the data curvature F_D = A^T Sigma^{-1} A is known in
closed form. It serves as ground truth for the stochastic curvature estimators and
for identifiability checks (||Sigma^{-1/2} A v||^2 = v^T F_D v).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import convolve2d


class Observation(eqx.Module):
    psf: jnp.ndarray  # odd-sized kernel
    weights: jnp.ndarray  # inverse variance per pixel, 0 where unobserved
    blur_cutoff: float | None = eqx.static_field(default=None)  # Fourier low-pass (cycles/px)

    def apply(self, x):
        """Noiseless observed image A x (unobserved pixels are still rendered)."""
        y = convolve2d(x, self.psf, mode="same")
        if self.blur_cutoff is not None:
            y = _lowpass(y, self.blur_cutoff)
        return y

    def adjoint(self, r):
        if self.blur_cutoff is not None:
            r = _lowpass(r, self.blur_cutoff)
        return convolve2d(r, self.psf[::-1, ::-1], mode="same")

    def simulate(self, x, key):
        noise = jax.random.normal(key, x.shape) / jnp.sqrt(jnp.where(self.weights > 0, self.weights, 1.0))
        return jnp.where(self.weights > 0, self.apply(x) + noise, 0.0)

    def log_likelihood(self, x, y):
        return -0.5 * jnp.sum(self.weights * (self.apply(x) - y) ** 2)

    def score(self, x, y):
        return self.adjoint(self.weights * (y - self.apply(x)))

    def curvature_matvec(self, v):
        """F_D v = A^T W A v."""
        return self.adjoint(self.weights * self.apply(v))

    def curvature_diag(self):
        """Exact diag(F_D) for the pure-PSF model: (W correlated with k^2)."""
        if self.blur_cutoff is not None:
            raise NotImplementedError("closed form only without the Fourier low-pass")
        return convolve2d(self.weights, (self.psf**2)[::-1, ::-1], mode="same")


def _lowpass(img, cutoff):
    k = jnp.fft.fftfreq(img.shape[0])
    kk = jnp.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(img) * (kk <= cutoff)))


def make_observation(source, mask=None, noise_scale=1.0, blur_cutoff=None):
    """Observation of ``source`` with its real PSF and (scaled) real noise level."""
    shape = source.truth.shape
    mask = np.ones(shape) if mask is None else np.asarray(mask, dtype=np.float64)
    sigma = source.sigma * noise_scale
    return Observation(
        psf=jnp.asarray(source.psf, jnp.float32),
        weights=jnp.asarray(mask / sigma**2, jnp.float32),
        blur_cutoff=blur_cutoff,
    )


# ----------------------------- masks ----------------------------- #


def half_plane_mask(shape, fraction, axis=0):
    """Remove the first ``fraction`` of rows (axis=0) or columns (axis=1)."""
    m = np.ones(shape)
    n = int(round(fraction * shape[axis]))
    if axis == 0:
        m[:n, :] = 0
    else:
        m[:, :n] = 0
    return m


def annulus_mask(shape, r_in, r_out=np.inf, center=None):
    """Remove pixels with r_in <= r < r_out (r_in=0 masks the core)."""
    c = (np.array(shape) - 1) / 2 if center is None else np.asarray(center)
    yy, xx = np.indices(shape)
    r = np.hypot(yy - c[0], xx - c[1])
    return np.where((r >= r_in) & (r < r_out), 0.0, 1.0)


def area_mask(shape, fraction, kind, center=None):
    """Masks parameterised by the removed area fraction.

    kind = "half-plane": contiguous strip from one edge.
    kind = "outer": everything outside a centred disc (removes the outskirts).
    kind = "core": a centred disc (removes the core).
    """
    if kind == "half-plane":
        return half_plane_mask(shape, fraction)
    if fraction == 0:
        return np.ones(shape)
    c = (np.array(shape) - 1) / 2 if center is None else np.asarray(center)
    yy, xx = np.indices(shape)
    r = np.hypot(yy - c[0], xx - c[1]).ravel()
    n_mask = int(round(fraction * r.size))
    order = np.argsort(-r if kind == "outer" else r, kind="stable")
    m = np.ones(r.size)
    m[order[:n_mask]] = 0
    return m.reshape(shape)


def random_mask(shape, fraction, kind, rng):
    """Masks that do not look at the source: a ``fraction`` of the pixels chosen at
    random (kind="pixels"), or a disc of that area at a random position (kind="blob").
    The control for the reference-based ring masks of ``flux_mask``."""
    if fraction == 0:
        return np.ones(shape)
    if kind == "blob":
        return area_mask(shape, fraction, "core", center=rng.uniform(0, np.array(shape) - 1))
    m = np.ones(int(np.prod(shape)))
    m[rng.choice(m.size, int(round(fraction * m.size)), replace=False)] = 0
    return m.reshape(shape)


def flux_mask(truth, fraction, kind):
    """Remove ``fraction`` of the true source flux, starting from the core
    (kind="core") or from the outskirts (kind="outer"), in rings about the centroid.

    Both the centroid and the flux fractions come from the reference (truth) image, so
    these are deliberately controlled, oracle masks rather than realistic missing data,
    and they are rings, not the brightest or faintest pixels as such."""
    truth = np.asarray(truth)
    if fraction == 0:
        return np.ones(truth.shape)
    yy, xx = np.indices(truth.shape)
    f = truth.sum()
    c = np.array([(truth * yy).sum() / f, (truth * xx).sum() / f])
    r = np.hypot(yy - c[0], xx - c[1]).ravel()
    order = np.argsort(r if kind == "core" else -r, kind="stable")
    removed = np.cumsum(truth.ravel()[order]) / f
    n_mask = int(np.searchsorted(removed, fraction - 1e-12) + 1)
    m = np.ones(r.size)
    m[order[:n_mask]] = 0
    return m.reshape(truth.shape)
