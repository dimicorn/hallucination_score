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


def test_degenerate_feature_gets_nan_rows_without_touching_the_others():
    rng = np.random.default_rng(4)
    f_d, f_p = _spd(rng, 6), _spd(rng, 6, 0.3)
    g = rng.normal(size=(3, 6))
    g_bad = g.copy()
    g_bad[1] = 0.0
    g_bad[2, 0] = np.nan
    rows, joint, corr = laplace.feature_laplace(f_d, f_p, g_bad, np.zeros(3), np.zeros(3), np.eye(3))
    ref, _, _ = laplace.feature_laplace(f_d, f_p, g[:1], np.zeros(1), np.zeros(1), np.eye(1))
    assert rows[0]["sigma_post"] == pytest.approx(ref[0]["sigma_post"])
    for r in rows[1:]:
        assert all(np.isnan(v) for v in r.values())
    assert np.isnan(joint["info_gain_pop_joint"]) and np.isnan(joint["kl_post_pop_joint"])
    assert np.isnan(corr[1, 1]) and np.isfinite(corr[0, 0])


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


def test_review_counterexample_null_space_and_effective_share():
    """The review's 2x2 case: the data see x1 only, phi = x1 + x2. The directional
    sigma_data = ||g|| / sqrt(v^T F_D v) = 2 is finite, yet half of ||g||^2 lies in the
    null space of F_D; along the least-favourable direction H^-1 g the prior supplies
    almost all of the feature's precision, while the directional R says the opposite."""
    f_d, f_p = np.diag([1.0, 0.0]), 0.01 * np.eye(2)
    g = np.array([[1.0, 1.0]])
    rows, _, _ = laplace.feature_laplace(f_d, f_p, g, np.zeros(1), np.zeros(1), np.eye(1))
    v = g[0] / np.linalg.norm(g[0])
    r_directional = (v @ f_p @ v) / (v @ (f_d + f_p) @ v)
    assert np.linalg.norm(g[0]) / np.sqrt(v @ f_d @ v) == pytest.approx(2.0)
    assert rows[0]["eta_null"] == pytest.approx(0.5)
    assert r_directional == pytest.approx(0.02, abs=1e-3)
    assert rows[0]["R_eff"] == pytest.approx(0.99, abs=0.01)


def test_effective_split_sums_to_the_posterior_precision():
    rng = np.random.default_rng(5)
    f_d, f_p = _spd(rng, 7), _spd(rng, 7, 0.3)
    g = rng.normal(size=(3, 7))
    rows, _, _ = laplace.feature_laplace(f_d, f_p, g, np.zeros(3), np.zeros(3), np.eye(3))
    h_inv = np.linalg.inv(f_d + f_p)
    for gk, r in zip(g, rows):
        w = h_inv @ gk
        i_d, i_p = w @ f_d @ w / (gk @ w) ** 2, w @ f_p @ w / (gk @ w) ** 2
        assert i_d + i_p == pytest.approx(1 / r["sigma_post"] ** 2, rel=1e-4)
        assert r["R_eff"] == pytest.approx(i_p / (i_p + i_d), rel=1e-4)
        assert r["eta_null"] == pytest.approx(0.0, abs=1e-9)  # F_D positive definite


def test_free_subspace_and_unclipped_widths():
    rng = np.random.default_rng(6)
    f_d, f_p = _spd(rng, 6), _spd(rng, 6, 0.3)
    g = rng.normal(size=(2, 6))
    free = np.array([True, True, False, True, False, True])
    rows, joint, _ = laplace.feature_laplace(f_d, f_p, g, np.zeros(2), np.zeros(2), np.eye(2), free=free)
    h = f_d + f_p
    for gk, r in zip(g, rows):
        gf = gk[free]
        assert r["sigma_post_free"] == pytest.approx(np.sqrt(gf @ np.linalg.solve(h[np.ix_(free, free)], gf)), rel=1e-4)
        assert r["sigma_post_free"] <= r["sigma_post"] * (1 + 1e-9)  # conditioning narrows
        assert r["grad_share_bound"] == pytest.approx((gk[~free] ** 2).sum() / (gk**2).sum())
        assert r["sigma_post_raw"] == pytest.approx(r["sigma_post"], rel=1e-6)  # nothing clipped here
    assert joint["h_raw_min_eig"] > 0

    # an indefinite prior that makes F_D + F_P indefinite: no unclipped Laplace width
    rows, joint, _ = laplace.feature_laplace(np.diag([1.0, 0.0]), np.diag([1.0, -0.5]), np.array([[1.0, 0.0]]), np.zeros(1), np.zeros(1), np.eye(1))
    assert joint["h_raw_min_eig"] < 0 and np.isnan(rows[0]["sigma_post_raw"])


def test_assess_rows_have_the_review_diagnostics(monkeypatch):
    """End to end on a synthetic galaxy observed through a strong Fourier low-pass:
    HF power's gradient is almost entirely in the null space of F_D (its numerator lives
    above 0.25 cycles/px), so it is flagged even if its directional sigma_data is small."""
    from types import SimpleNamespace

    from provenance import evaluate, features
    from provenance.pipeline import Case

    truth = np.asarray(0.8 * gaussian_galaxy(2.5, 2.0, theta=0.4))
    x_hat = jnp.asarray(np.where(truth > 1e-3, truth, 0.0), jnp.float32)  # background at the bound
    obs = Observation(gaussian_psf(), jnp.full((N, N), 850.0, jnp.float32), blur_cutoff=0.07)
    case = Case(obs, jnp.zeros((N, N)), x_hat, truth, 1e-5)
    src = SimpleNamespace(truth=truth, source_id="synthetic", snr=50.0)
    monkeypatch.setattr(evaluate, "population_moments", lambda: (np.zeros(len(features.NAMES)), np.diag([POP_STD[n] ** 2 for n in features.NAMES])))
    rows = evaluate.assess(case, src, ScorePrior(t=0.5, weight=0.5), POP_STD, {"condition": "lowpass"}, ref_grads=features.gradients(x_hat))
    by = {r["feature"]: r for r in rows}

    assert by["hf_power"]["eta_null"] > 0.9 and "null_space_dominated" in by["hf_power"]["flags"]
    for r in rows:
        assert r["frac_at_bound"] > 0.5 and r["opt_grad_norm"] == 1e-5
        if r["grad_norm"] == 0:  # asymmetry of this exactly symmetric galaxy: NaN rows by design
            continue
        assert 0 <= r["grad_share_bound"] <= 1 and np.isfinite(r["kappa_u"])
        assert r["I_D_fixed"] == pytest.approx(r["I_D"], rel=1e-3) and r["cos_ref"] == pytest.approx(1.0, abs=1e-5)
        assert 0 <= r["R_diag"] <= 1 and 0 <= r["R_eff"] <= 1
