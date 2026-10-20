"""Smooth, differentiable galaxy morphology features of a 32x32 image.

All features are defined on the image itself (Route 1 in the plan). Positions are in
pixels, measured from the flux-weighted centroid. Moments use a fixed circular Gaussian
window W (sigma 6 px, centred on the stamp) because unweighted moments are dominated
by faint, extended reconstructed halos; like any fixed-window moment this biases
shapes towards round for large sources. Definitions:

flux           sum_i x_i (unwindowed)
size           windowed second-moment radius sqrt(Qxx + Qyy) [px]
e1, e2         (Qxx - Qyy) / (Qxx + Qyy),  2 Qxy / (Qxx + Qyy)
concentration  F(r < 0.5 R) / F(r < 2 R), soft (sigmoid) apertures, R = size
asymmetry      sum (x - R180 x)^2 / (2 sum x^2), rotation about the centroid with
               bilinear interpolation
hf_power       fraction of Fourier power at |k| > 0.25 cycles/px (half Nyquist)

Ellipticity is reported as (e1, e2) because the position angle 0.5 atan2(e2, e1) has
an undefined gradient for round sources; ``derived`` adds |e| and the PA for display.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

NAMES = ("flux", "size", "e1", "e2", "concentration", "asymmetry", "hf_power")
APERTURE_SOFTNESS = 0.5  # px
WINDOW_SIGMA = 6.0  # px
HF_CUTOFF = 0.25  # cycles / px


def _grid(shape):
    yy, xx = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing="ij")
    return yy.astype(jnp.float32), xx.astype(jnp.float32)


def _window(shape):
    yy, xx = _grid(shape)
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    return jnp.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * WINDOW_SIGMA**2))


def _moments(x):
    yy, xx = _grid(x.shape)
    x = x * _window(x.shape)
    f = jnp.sum(x)
    cy, cx = jnp.sum(x * yy) / f, jnp.sum(x * xx) / f
    dy, dx = yy - cy, xx - cx
    qxx = jnp.sum(x * dx**2) / f
    qyy = jnp.sum(x * dy**2) / f
    qxy = jnp.sum(x * dx * dy) / f
    return f, cy, cx, dy, dx, qxx, qyy, qxy


def flux(x):
    return jnp.sum(x)


def size(x):
    _, _, _, _, _, qxx, qyy, _ = _moments(x)
    return jnp.sqrt(qxx + qyy)


def e1(x):
    _, _, _, _, _, qxx, qyy, _ = _moments(x)
    return (qxx - qyy) / (qxx + qyy)


def e2(x):
    _, _, _, _, _, qxx, qyy, qxy = _moments(x)
    return 2 * qxy / (qxx + qyy)


def concentration(x):
    _, _, _, dy, dx, qxx, qyy, _ = _moments(x)
    r = jnp.sqrt(dy**2 + dx**2)
    radius = jnp.sqrt(qxx + qyy)
    inner = jnp.sum(x * jax.nn.sigmoid((0.5 * radius - r) / APERTURE_SOFTNESS))
    outer = jnp.sum(x * jax.nn.sigmoid((2.0 * radius - r) / APERTURE_SOFTNESS))
    return inner / outer


def asymmetry(x):
    # rotate about the unwindowed centroid: the windowed one is pulled to the stamp centre
    yy, xx = _grid(x.shape)
    cy, cx = jnp.sum(x * yy) / jnp.sum(x), jnp.sum(x * xx) / jnp.sum(x)
    rotated = map_coordinates(x, [2 * cy - yy, 2 * cx - xx], order=1, mode="constant")
    return jnp.sum((x - rotated) ** 2) / (2 * jnp.sum(x**2))


def hf_power(x):
    k = jnp.fft.fftfreq(x.shape[0])
    kk = jnp.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    p = jnp.abs(jnp.fft.fft2(x)) ** 2
    return jnp.sum(jnp.where(kk > HF_CUTOFF, p, 0.0)) / jnp.sum(p)


FEATURES = dict(zip(NAMES, (flux, size, e1, e2, concentration, asymmetry, hf_power)))


def measure(x):
    return {name: float(f(x)) for name, f in FEATURES.items()}


def derived(values):
    import numpy as np

    return {
        "ellipticity": float(np.hypot(values["e1"], values["e2"])),
        "position_angle_deg": float(np.degrees(0.5 * np.arctan2(values["e2"], values["e1"]))),
    }


def gradients(x):
    """d phi_k / dx for every feature, evaluated at x."""
    return {name: jax.grad(f)(x) for name, f in FEATURES.items()}
