"""Feature-space Laplace diagnostics from the dense image-space curvatures.

Plan C1 (full feature-space curvature) and C3 (posterior information gain). Around the
reconstruction x_hat, take the Laplace approximation p(x|y) ~ N(x_hat, H^-1) with
H = F_D + F_P, and linearise the features, phi(x) ~ phi(x_hat) + G (x - x_hat) with G the
(k x n) matrix of feature gradients. The feature posterior is then

    Sigma_post = G H^-1 G^T                      (k x k, cross-feature terms included)

Per feature this gives the Laplace marginal std sigma_post (the directional sigma_loc of
attribution.py is a lower bound on it), the information gain in nats

    info_gain_pop = 1/2 log(sigma_pop^2 / Sigma_post)

and the KL divergence KL[N(phi(x_hat), Sigma_post) || N(mu_pop, Sigma_pop)], with the
reference population standing in for the prior's distribution of features p(phi).
Joint (all-feature) versions use log-determinants. The gain is measured against the
population, not against the local curvature of the prior alone: that curvature has
(near-)null directions, so the prior-only Laplace Gaussian is improper and a gain
against it is set by the regularising eps below, not by the data.

F_P is a learned Jacobian. It is symmetrised and replaced by its positive part F_P+
(negative eigenvalues set to 0), and H = F_D + F_P+ gets a small ridge eps I before
inversion (eps = EIG_FLOOR times its largest eigenvalue). sigma_post does not depend
on eps wherever the data or the prior constrain the feature. ``neg_mode_share``
reports how much of the prior's curvature along each feature gradient was dropped
(share of g^T |F_P| g from negative modes), so cases where the Laplace picture is
unreliable can be excluded.
"""

import numpy as np

from . import curvature

EIG_FLOOR = 1e-6


def dense_curvatures(obs, prior, x):
    """(F_D, F_P) as dense, symmetrised n x n arrays at x."""
    f_d = curvature.exact_matrix(obs.curvature_matvec, x.shape)
    f_p = curvature.exact_matrix(lambda v: prior.curvature_matvec(x, v), x.shape)
    return 0.5 * (f_d + f_d.T), 0.5 * (f_p + f_p.T)


def _inverse_form(matrix, eps, g):
    """G (M + eps I)^-1 G^T for symmetric positive semidefinite M."""
    w, u = np.linalg.eigh(matrix)
    z = u.T @ g.T  # (n, k)
    return z.T @ (z / np.maximum(w + eps, eps)[:, None])


def feature_laplace(f_d, f_p, grads, values, pop_mean, pop_cov):
    """Per-feature rows and joint quantities for one case.

    grads: (k, n) feature gradients; values: (k,) phi(x_hat); pop_mean (k,), pop_cov (k, k).
    Returns (list of per-feature dicts, dict of joint quantities, posterior correlation).
    """
    g = np.asarray(grads, np.float64).reshape(len(values), -1)
    values = np.asarray(values, np.float64)
    pop_mean, pop_cov = np.asarray(pop_mean, np.float64), np.asarray(pop_cov, np.float64)

    w_p, u_p = np.linalg.eigh(np.asarray(f_p, np.float64))
    z_p = (u_p.T @ g.T) ** 2  # (n, k)
    neg_share = (np.abs(w_p[w_p < 0])[:, None] * z_p[w_p < 0]).sum(axis=0) / np.maximum((np.abs(w_p)[:, None] * z_p).sum(axis=0), 1e-300)
    f_p_pos = (u_p * np.maximum(w_p, 0)) @ u_p.T
    h = np.asarray(f_d, np.float64) + f_p_pos
    eps = EIG_FLOOR * max(np.linalg.eigvalsh(h).max(), 1e-300)
    post = _inverse_form(h, eps, g)

    var_post, var_pop = np.diag(post), np.diag(pop_cov)
    kl = 0.5 * (np.log(var_pop / var_post) + (var_post + (values - pop_mean) ** 2) / var_pop - 1)
    rows = [
        {
            "sigma_post": float(np.sqrt(var_post[i])),
            "info_gain_pop": float(0.5 * np.log(var_pop[i] / var_post[i])),
            "kl_post_pop": float(kl[i]),
            "neg_mode_share": float(neg_share[i]),
        }
        for i in range(len(values))
    ]

    _, logdet_post = np.linalg.slogdet(post)
    _, logdet_pop = np.linalg.slogdet(pop_cov)
    pop_inv = np.linalg.inv(pop_cov)
    d = values - pop_mean
    joint = {
        "info_gain_pop_joint": float(0.5 * (logdet_pop - logdet_post)),
        "kl_post_pop_joint": float(0.5 * (np.trace(pop_inv @ post) + d @ pop_inv @ d - len(values) + logdet_pop - logdet_post)),
    }
    s = np.sqrt(var_post)
    return rows, joint, post / np.outer(s, s)
