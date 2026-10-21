import jax.numpy as jnp
import numpy as np
import pytest

from provenance import attribution, laplace
from provenance.forward import Observation
from provenance.prior import ScorePrior
from test_core import POP_STD, N, gaussian_galaxy, gaussian_psf


def _spd(rng, n, scale=1.0):
    a = rng.normal(size=(n, n))
    return scale * (a @ a.T + n * np.eye(n))


def test_posterior_covariance_matches_direct_inverse():
    rng = np.random.default_rng(0)
    f_d, f_p = _spd(rng, 6), _spd(rng, 6, 0.3)
    g = rng.normal(size=(3, 6))
    rows, _, corr = laplace.feature_laplace(f_d, f_p, g, np.zeros(3), np.zeros(3), np.eye(3))
    post = g @ np.linalg.inv(f_d + f_p) @ g.T
    np.testing.assert_allclose([r["sigma_post"] for r in rows], np.sqrt(np.diag(post)), rtol=1e-4)  # up to the eps ridge
    np.testing.assert_allclose(np.diag(corr), 1.0)
    s = np.sqrt(np.diag(post))
    np.testing.assert_allclose(corr, post / np.outer(s, s), rtol=1e-4)
    assert all(r["neg_mode_share"] == 0 for r in rows)


def test_directional_sigma_is_a_lower_bound_on_the_marginal():
    """sigma_loc = ||g|| / sqrt(v^T H v) <= sqrt(g^T H^-1 g) for positive definite H."""
    rng = np.random.default_rng(1)
    h = _spd(rng, 8)
    g = rng.normal(size=(4, 8))
    rows, _, _ = laplace.feature_laplace(h, np.zeros((8, 8)) + 1e-9 * np.eye(8), g, np.zeros(4), np.zeros(4), np.eye(4))
    for gk, r in zip(g, rows):
        sigma_loc = np.linalg.norm(gk) / np.sqrt(gk @ h @ gk / (gk @ gk))
        assert sigma_loc <= r["sigma_post"] * (1 + 1e-5)


def test_posterior_std_does_not_depend_on_the_ridge(monkeypatch):
    """Where data or prior constrain every direction, eps only regularises."""
    rng = np.random.default_rng(2)
    f_d, f_p = _spd(rng, 6), _spd(rng, 6, 0.3)
    g = rng.normal(size=(3, 6))
    stds = []
    for floor in (1e-8, 1e-6, 1e-4):
        monkeypatch.setattr(laplace, "EIG_FLOOR", floor)
        rows, _, _ = laplace.feature_laplace(f_d, f_p, g, np.zeros(3), np.zeros(3), np.eye(3))
        stds.append([r["sigma_post"] for r in rows])
    np.testing.assert_allclose(stds[0], stds[-1], rtol=1e-3)


def test_population_terms_and_joint_quantities():
    # independent features: posterior variances 0.25 and 1.0, population variance 1
    f_d, f_p = np.diag([3.0, 0.5]), np.diag([1.0, 0.5])
    values, mean = np.array([0.5, 0.0]), np.zeros(2)
    rows, joint, _ = laplace.feature_laplace(f_d, f_p, np.eye(2), values, mean, np.eye(2))
    assert rows[0]["info_gain_pop"] == pytest.approx(0.5 * np.log(4.0), rel=1e-4)
    assert rows[1]["info_gain_pop"] == pytest.approx(0.0, abs=1e-5)
    # KL[N(0.5, 0.25) || N(0, 1)] = 1/2 (log 4 + 0.25 + 0.25 - 1)
    assert rows[0]["kl_post_pop"] == pytest.approx(0.5 * (np.log(4.0) + 0.5 - 1), rel=1e-4)
    assert joint["info_gain_pop_joint"] == pytest.approx(rows[0]["info_gain_pop"] + rows[1]["info_gain_pop"])
    assert joint["kl_post_pop_joint"] == pytest.approx(rows[0]["kl_post_pop"] + rows[1]["kl_post_pop"])


def test_negative_prior_curvature_is_dropped_and_reported():
    """The prior's negative mode is replaced by 0: along it only the data constrain."""
    f_p = np.diag([2.0, -0.5, 1.0])
    f_d = np.diag([0.0, 4.0, 0.0])
    g = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rows, _, _ = laplace.feature_laplace(f_d, f_p, g, np.zeros(2), np.zeros(2), np.eye(2))
    assert rows[0]["sigma_post"] == pytest.approx(0.5, rel=1e-4)
    assert rows[0]["neg_mode_share"] == pytest.approx(1.0)
    assert rows[1]["neg_mode_share"] == pytest.approx(0.0)
    assert rows[1]["sigma_post"] == pytest.approx(np.sqrt(0.5), rel=1e-4)


def test_data_never_increase_posterior_std_with_indefinite_prior_curvature():
    """Regression: inverting H and F_P through |eigenvalues| separately let adding data
    increase a feature's variance for ~40 % of features when F_P was indefinite."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = rng.normal(size=(12, 12))
        f_p = a + a.T  # indefinite
        b = rng.normal(size=(12, 4))
        f_d = b @ b.T  # rank-deficient, like masked data
        g = rng.normal(size=(5, 12))
        with_data, _, _ = laplace.feature_laplace(f_d, f_p, g, np.zeros(5), np.zeros(5), np.eye(5))
        without, _, _ = laplace.feature_laplace(np.zeros((12, 12)), f_p, g, np.zeros(5), np.zeros(5), np.eye(5))
        assert all(w["sigma_post"] <= wo["sigma_post"] * (1 + 1e-9) for w, wo in zip(with_data, without))


def test_laplace_with_the_learned_prior():
    """On a galaxy with native-like data: finite posterior stds, at least the
    directional sigma_loc wherever negative modes do not contribute, and finite
    information gains."""
    prior = ScorePrior(t=0.2, weight=0.5)
    x = 0.8 * gaussian_galaxy(2.5, 2.0, theta=0.4)
    obs = Observation(gaussian_psf(), jnp.full((N, N), 850.0, jnp.float32))
    f_d, f_p = laplace.dense_curvatures(obs, prior, x)
    np.testing.assert_allclose(f_p, f_p.T)
    attrs = attribution.feature_attribution(obs, prior, x, population_std=POP_STD)
    from provenance import features

    grads = features.gradients(x)
    g = np.stack([np.asarray(grads[a.name]).ravel() for a in attrs])
    rows, joint, corr = laplace.feature_laplace(f_d, f_p, g, [a.value for a in attrs], np.zeros(len(attrs)), np.diag([POP_STD[a.name] ** 2 for a in attrs]))
    assert np.all(np.isfinite(corr)) and np.isfinite(joint["info_gain_pop_joint"])
    for a, r in zip(attrs, rows):
        assert np.isfinite(r["sigma_post"]) and np.isfinite(r["info_gain_pop"]), a.name
        if r["neg_mode_share"] < 1e-3:
            assert a.sigma_local <= r["sigma_post"] * 1.01, a.name
