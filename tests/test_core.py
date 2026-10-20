import jax
import jax.numpy as jnp
import numpy as np
import pytest

from provenance import curvature, features
from provenance.forward import Observation, area_mask, half_plane_mask
from provenance.prior import ScorePrior

N = 32


def gaussian_psf(size=9, sigma=1.5):
    r = np.arange(size) - size // 2
    k = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma**2))
    return jnp.asarray(k / k.sum(), jnp.float32)


def gaussian_galaxy(sx=3.0, sy=3.0, theta=0.0, center=(15.5, 15.5)):
    yy, xx = np.indices((N, N), dtype=np.float64)
    dy, dx = yy - center[0], xx - center[1]
    c, s = np.cos(theta), np.sin(theta)
    u, w = c * dx + s * dy, -s * dx + c * dy
    return jnp.asarray(np.exp(-0.5 * ((u / sx) ** 2 + (w / sy) ** 2)), jnp.float32)


@pytest.fixture(params=[None, 0.2])
def obs(request):
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, (N, N)) * half_plane_mask((N, N), 0.5)
    return Observation(gaussian_psf(), jnp.asarray(w, jnp.float32), blur_cutoff=request.param)


def test_adjoint(obs):
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    x, r = jax.random.normal(k1, (N, N)), jax.random.normal(k2, (N, N))
    np.testing.assert_allclose(jnp.vdot(obs.apply(x), r), jnp.vdot(x, obs.adjoint(r)), rtol=1e-4)


def test_closed_form_diag_matches_exact():
    w = jnp.asarray(half_plane_mask((N, N), 0.5) * 3.0, jnp.float32)
    obs = Observation(gaussian_psf(), w)
    exact = np.diag(curvature.exact_matrix(obs.curvature_matvec, (N, N))).reshape(N, N)
    np.testing.assert_allclose(obs.curvature_diag(), exact, rtol=1e-4, atol=1e-6)


def test_masked_region_is_null_space():
    obs = Observation(gaussian_psf(size=9), jnp.asarray(half_plane_mask((N, N), 0.5), jnp.float32))
    v = jnp.zeros((N, N)).at[:10, :].set(1.0)  # >= psf radius away from observed rows
    assert curvature.quadratic_form(obs.curvature_matvec, v) == pytest.approx(0.0, abs=1e-8)
    assert jnp.sum(obs.weights * obs.apply(v) ** 2) == pytest.approx(0.0, abs=1e-8)


def test_hutchinson_converges_to_diag():
    rng = np.random.default_rng(2)
    b = rng.normal(size=(64, 64))
    m = jnp.asarray(b @ b.T / 64 + np.diag(rng.uniform(1, 3, 64)), jnp.float32)
    est, snaps = curvature.hutchinson_diag(lambda z: m @ z, (64,), 4096, jax.random.PRNGKey(0), checkpoints=(64, 4096))
    err = lambda d: np.linalg.norm(d - np.diag(m)) / np.linalg.norm(np.diag(m))
    assert err(est) < 0.05
    assert err(snaps[4096]) < err(snaps[64])


def test_area_mask_fractions():
    for kind in ("half-plane", "outer", "core"):
        for f in (0.0, 0.25, 0.5, 0.75):
            assert 1 - area_mask((N, N), f, kind).mean() == pytest.approx(f, abs=0.02)


def test_features_on_gaussians():
    w2 = features.WINDOW_SIGMA**2
    windowed = lambda s: s**2 * w2 / (s**2 + w2)  # Gaussian x Gaussian window variance
    round_ = gaussian_galaxy(3.0, 3.0)
    vals = features.measure(round_)
    assert vals["size"] == pytest.approx(np.sqrt(2 * windowed(3.0)), rel=0.02)
    assert abs(vals["e1"]) < 1e-3 and abs(vals["e2"]) < 1e-3
    assert vals["asymmetry"] < 1e-3

    # elongated along x: e1 = (sx^2 - sy^2) / (sx^2 + sy^2) of the windowed variances
    e_expected = (windowed(4.0) - windowed(2.0)) / (windowed(4.0) + windowed(2.0))
    assert features.e1(gaussian_galaxy(4.0, 2.0)) == pytest.approx(e_expected, abs=0.01)
    assert features.e2(gaussian_galaxy(4.0, 2.0, theta=np.pi / 4)) == pytest.approx(e_expected, abs=0.01)

    # shape features are flux-invariant; flux scales linearly
    scaled = features.measure(2.5 * round_)
    assert scaled["flux"] == pytest.approx(2.5 * vals["flux"], rel=1e-5)
    for name in ("size", "e1", "e2", "concentration", "asymmetry", "hf_power"):
        assert scaled[name] == pytest.approx(vals[name], rel=1e-4, abs=1e-6)

    # off-centre symmetric source is still symmetric (rotation is about the centroid)
    assert features.asymmetry(gaussian_galaxy(3.0, 2.0, center=(13.3, 17.6))) < 5e-3
    # a lopsided source is not
    lopsided = round_ + 0.5 * gaussian_galaxy(1.5, 1.5, center=(15.5, 21.0))
    assert features.asymmetry(lopsided) > 0.02
    # sharper sources have more high-frequency power and higher concentration
    sharp = gaussian_galaxy(1.0, 1.0)
    assert features.hf_power(sharp) > vals["hf_power"]
    assert features.concentration(sharp + 0.05 * gaussian_galaxy(8.0, 8.0)) > vals["concentration"]


def test_feature_gradients_finite():
    x = gaussian_galaxy(3.0, 2.0, theta=0.3)
    for name, g in features.gradients(x).items():
        assert np.all(np.isfinite(g)), name


def test_prior_curvature_is_score_jacobian():
    prior = ScorePrior()
    x = 0.8 * gaussian_galaxy(2.5, 2.0)
    v = jax.random.normal(jax.random.PRNGKey(3), (N, N)) * 0.01
    s = prior.score(x)
    assert s.shape == (N, N) and np.all(np.isfinite(s))
    eps = 1e-2
    fd = -(prior.score(x + eps * v) - prior.score(x - eps * v)) / (2 * eps)
    np.testing.assert_allclose(prior.curvature_matvec(x, v), fd, rtol=0.05, atol=0.05 * float(jnp.abs(fd).max()))


def test_prior_curvature_below_smoothing_cap():
    """Tweedie: for the true smoothed prior, v^T F_P v <= v^T F_cap v. Check the
    learned prior respects this on a galaxy (up to small transform terms)."""
    prior = ScorePrior(t=0.05)
    x = 0.8 * gaussian_galaxy(2.5, 2.0)
    for k in jax.random.split(jax.random.PRNGKey(4), 4):
        v = jax.random.normal(k, (N, N))
        v = v / jnp.linalg.norm(v)
        i_p = curvature.quadratic_form(lambda u: prior.curvature_matvec(x, u), v)
        i_cap = curvature.quadratic_form(lambda u: prior.curvature_cap_matvec(x, u), v)
        assert i_p <= 1.1 * i_cap
