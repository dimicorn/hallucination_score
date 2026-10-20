"""Smooth, differentiable galaxy morphology features of a 32x32 image.

All features are defined on the image itself (Route 1 in the plan). Moments use an
adaptive elliptical Gaussian window (Bernstein & Jarvis 2002, as in HSM): starting from
a round window at the stamp centre, iterate

    centre <- centroid of x W,    M <- 2 * (second moments of x W) + MIN_VAR I

where W is the Gaussian with covariance M about the centre. For a Gaussian source the
fixed point is the source's own covariance, so size and ellipticity are unbiased, and
the window (hence every gradient) follows the source's extent instead of a fixed stamp
scale. A fixed wide window put almost all of each feature's gradient on background
pixels ~10 px out, so all moment-based features measured the same annulus. The
iteration is unrolled with a fixed count, so gradients include the window's dependence
on x. Definitions, with M and the centre at the fixed point:

flux           2 sum_i x_i W_i (adaptive flux; exact for a Gaussian source)
size           sqrt(Mxx + Myy) [px] (second-moment radius for a Gaussian source)
e1, e2         (Mxx - Myy) / (Mxx + Myy),  2 Mxy / (Mxx + Myy)
concentration  F(r < 0.5 R) / F(r < 2 R), soft (sigmoid) apertures about the adaptive
               centre, R = size
asymmetry      sum W' (x - R180 x)^2 / (2 sum W' x^2), rotation about the adaptive centre
               with bilinear interpolation; W' is the adaptive window widened by
               ASYM_WINDOW_SCALE so that lopsided outskirts count
hf_power       fraction of Fourier power at |k| > 0.25 cycles/px (half Nyquist)

Ellipticity is reported as (e1, e2) because the position angle 0.5 atan2(e2, e1) has
an undefined gradient for round sources; ``derived`` adds |e| and the PA for display.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

NAMES = ("flux", "size", "e1", "e2", "concentration", "asymmetry", "hf_power")
APERTURE_SOFTNESS = 0.5  # px
N_ITER = 30  # adaptive-window iterations; the error contracts ~2x per step for Gaussians
INIT_SIGMA = 3.0  # px, starting window
MIN_VAR = 0.01  # px^2, keeps the window positive definite
ASYM_WINDOW_SCALE = 2.0
HF_CUTOFF = 0.25  # cycles / px


def _grid(shape):
    yy, xx = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing="ij")
    return yy.astype(jnp.float32), xx.astype(jnp.float32)


def _window(yy, xx, cy, cx, myy, mxx, mxy):
    dy, dx = yy - cy, xx - cx
    det = myy * mxx - mxy**2
    return jnp.exp(-0.5 * (mxx * dy**2 + myy * dx**2 - 2 * mxy * dx * dy) / det)


def _weighted_moments(x, w, yy, xx):
    """(sum x w, centroid, second moments) of the weighted image x w."""
    xw = x * w
    f = jnp.sum(xw)
    cy, cx = jnp.sum(xw * yy) / f, jnp.sum(xw * xx) / f
    dy, dx = yy - cy, xx - cx
    return f, cy, cx, jnp.sum(xw * dy**2) / f, jnp.sum(xw * dx**2) / f, jnp.sum(xw * dx * dy) / f


def adaptive_moments(x):
    """Returns (sum x W, cy, cx, Myy, Mxx, Mxy) at the fixed point of the adaptive window."""
    yy, xx = _grid(x.shape)

    def step(_, s):
        w = _window(yy, xx, *s)
        _, cy, cx, qyy, qxx, qxy = _weighted_moments(x, w, yy, xx)
        return cy, cx, 2 * qyy + MIN_VAR, 2 * qxx + MIN_VAR, 2 * qxy

    init = ((x.shape[0] - 1) / 2, (x.shape[1] - 1) / 2, INIT_SIGMA**2, INIT_SIGMA**2, 0.0)
    init = tuple(jnp.asarray(v, jnp.float32) for v in init)
    cy, cx, myy, mxx, mxy = jax.lax.fori_loop(0, N_ITER, step, init)
    f = jnp.sum(x * _window(yy, xx, cy, cx, myy, mxx, mxy))
    return f, cy, cx, myy, mxx, mxy


def flux(x):
    f, *_ = adaptive_moments(x)
    return 2 * f


def size(x):
    _, _, _, myy, mxx, _ = adaptive_moments(x)
    return jnp.sqrt(mxx + myy)


def e1(x):
    _, _, _, myy, mxx, _ = adaptive_moments(x)
    return (mxx - myy) / (mxx + myy)


def e2(x):
    _, _, _, myy, mxx, mxy = adaptive_moments(x)
    return 2 * mxy / (mxx + myy)


def concentration(x):
    _, cy, cx, myy, mxx, _ = adaptive_moments(x)
    yy, xx = _grid(x.shape)
    r = jnp.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = jnp.sqrt(mxx + myy)
    inner = jnp.sum(x * jax.nn.sigmoid((0.5 * radius - r) / APERTURE_SOFTNESS))
    outer = jnp.sum(x * jax.nn.sigmoid((2.0 * radius - r) / APERTURE_SOFTNESS))
    return inner / outer


def asymmetry(x):
    _, cy, cx, myy, mxx, mxy = adaptive_moments(x)
    yy, xx = _grid(x.shape)
    s2 = ASYM_WINDOW_SCALE**2
    w = _window(yy, xx, cy, cx, s2 * myy, s2 * mxx, s2 * mxy)
    rotated = map_coordinates(x, [2 * cy - yy, 2 * cx - xx], order=1, mode="constant")
    return jnp.sum(w * (x - rotated) ** 2) / (2 * jnp.sum(w * x**2))


def hf_power(x):
    k = jnp.fft.fftfreq(x.shape[0])
    kk = jnp.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    p = jnp.abs(jnp.fft.fft2(x)) ** 2
    return jnp.sum(jnp.where(kk > HF_CUTOFF, p, 0.0)) / jnp.sum(p)


# jitted once: un-jitted calls would retrace the adaptive-window loop every time
FEATURES = {name: jax.jit(f) for name, f in zip(NAMES, (flux, size, e1, e2, concentration, asymmetry, hf_power))}
GRADIENTS = {name: jax.jit(jax.grad(f)) for name, f in FEATURES.items()}


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
    return {name: g(x) for name, g in GRADIENTS.items()}
