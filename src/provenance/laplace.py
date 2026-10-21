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
NULL_TOL = 1e-6  # eigenvalues of F_D below NULL_TOL * max count as its (numerical) null space
NULL_DOMINATED = 0.5  # null-space share of ||g||^2 above which a feature is flagged


def dense_curvatures(obs, prior, x):
    """(F_D, F_P) as dense, symmetrised n x n arrays at x."""
    f_d = curvature.exact_matrix(obs.curvature_matvec, x.shape)
    f_p = curvature.exact_matrix(lambda v: prior.curvature_matvec(x, v), x.shape)
    return 0.5 * (f_d + f_d.T), 0.5 * (f_p + f_p.T)


def feature_laplace(f_d, f_p, grads, values, pop_mean, pop_cov, free=None):
    """Per-feature rows and joint quantities for one case.

    grads: (k, n) feature gradients; values: (k,) phi(x_hat); pop_mean (k,), pop_cov (k, k);
    free: optional boolean (n,) mask of pixels not at the x >= 0 bound.
    Returns (list of per-feature dicts, dict of joint quantities, posterior correlation).

    Per feature, besides sigma_post, info_gain_pop (the population-relative log-width
    reduction 1/2 log(sigma_pop^2 / sigma_post^2), not a data-only information gain), the
    KL and neg_mode_share:
      eta_null         share of ||g||^2 in the numerical null space of F_D (what the data
                       cannot see); the feature is not identifiable from the data alone
                       when it is > 0, whatever sigma_data says
      R_eff            prior share of the Laplace feature precision 1/sigma_post^2, i.e.
                       the directional split taken along w = H^-1 g instead of g
      sigma_post_raw   sigma_post with F_P unclipped (NaN if F_D + sym F_P is not positive
                       definite); the clipping can only narrow sigma_post
      sigma_post_free  the Gaussian restricted to the free pixels (conditioning on the
                       pixels at the positivity bound); the truncated answer lies between
      grad_share_bound share of ||g||^2 on pixels at the bound
    """
    g = np.asarray(grads, np.float64).reshape(len(values), -1)
    values = np.asarray(values, np.float64)
    # features with a zero or non-finite gradient or value have no Laplace posterior: NaN rows,
    # and the joint quantities over all features are NaN too
    ok = np.all(np.isfinite(g), axis=1) & (np.abs(g).sum(axis=1) > 0) & np.isfinite(values)
    g = np.where(ok[:, None], np.nan_to_num(g), 0.0)
    g_norm2 = np.maximum((g**2).sum(axis=1), 1e-300)
    pop_mean, pop_cov = np.asarray(pop_mean, np.float64), np.asarray(pop_cov, np.float64)
    f_d, f_p = np.asarray(f_d, np.float64), np.asarray(f_p, np.float64)
    nan = lambda a: np.where(ok, a, np.nan)

    w_p, u_p = np.linalg.eigh(f_p)
    z_p = (u_p.T @ g.T) ** 2  # (n, k)
    neg_share = (np.abs(w_p[w_p < 0])[:, None] * z_p[w_p < 0]).sum(axis=0) / np.maximum((np.abs(w_p)[:, None] * z_p).sum(axis=0), 1e-300)
    f_p_pos = (u_p * np.maximum(w_p, 0)) @ u_p.T
    h = f_d + f_p_pos
    w_h, u_h = np.linalg.eigh(h)
    eps = EIG_FLOOR * max(w_h.max(), 1e-300)
    z = u_h.T @ g.T  # (n, k)
    hinv_g = u_h @ (z / np.maximum(w_h + eps, eps)[:, None])  # (n, k) = H^-1 g^T
    post = g @ hinv_g
    var_post, var_pop = nan(np.diag(post)), np.diag(pop_cov)

    # prior share of the feature precision, along the least-favourable direction H^-1 g
    norm = np.maximum(np.diag(post) ** 2, 1e-300)
    i_d_eff = np.einsum("nk,nm,mk->k", hinv_g, f_d, hinv_g) / norm
    i_p_eff = np.einsum("nk,nm,mk->k", hinv_g, f_p_pos, hinv_g) / norm
    r_eff = nan(i_p_eff / np.maximum(i_p_eff + i_d_eff, 1e-300))

    w_d, u_d = np.linalg.eigh(f_d)
    null = w_d <= NULL_TOL * max(w_d.max(), 1e-300)
    eta_null = nan(((u_d[:, null].T @ g.T) ** 2).sum(axis=0) / g_norm2)

    w_r, u_r = np.linalg.eigh(f_d + f_p)
    if w_r.min() + eps > 0:
        z_r = u_r.T @ g.T
        var_raw = nan((z_r**2 / (w_r + eps)[:, None]).sum(axis=0))
    else:
        var_raw = np.full(len(values), np.nan)

    var_free = share_bound = np.full(len(values), np.nan)
    if free is not None:
        free = np.asarray(free, bool).ravel()
        share_bound = nan((g[:, ~free] ** 2).sum(axis=1) / g_norm2)
        if free.any():
            gf = g[:, free]
            h_ff = h[np.ix_(free, free)] + eps * np.eye(int(free.sum()))
            var_free = nan(np.einsum("kn,nk->k", gf, np.linalg.solve(h_ff, gf.T)))

    kl = 0.5 * (np.log(var_pop / var_post) + (var_post + (values - pop_mean) ** 2) / var_pop - 1)
    rows = [
        {
            "sigma_post": float(np.sqrt(var_post[i])),
            "info_gain_pop": float(0.5 * np.log(var_pop[i] / var_post[i])),
            "kl_post_pop": float(kl[i]),
            "neg_mode_share": float(nan(neg_share)[i]),
            "eta_null": float(eta_null[i]),
            "R_eff": float(r_eff[i]),
            "sigma_post_raw": float(np.sqrt(var_raw[i])),
            "sigma_post_free": float(np.sqrt(var_free[i])),
            "grad_share_bound": float(share_bound[i]),
        }
        for i in range(len(values))
    ]

    _, logdet_post = np.linalg.slogdet(post) if ok.all() else (0.0, np.nan)
    _, logdet_pop = np.linalg.slogdet(pop_cov)
    pop_inv = np.linalg.inv(pop_cov)
    d = np.where(ok, values, np.nan) - pop_mean
    joint = {
        "info_gain_pop_joint": float(0.5 * (logdet_pop - logdet_post)),
        "kl_post_pop_joint": float(0.5 * (np.trace(pop_inv @ post) + d @ pop_inv @ d - len(values) + logdet_pop - logdet_post)),
        "h_raw_min_eig": float(w_r.min()),
        "h_cond": float((w_h.max() + eps) / (max(w_h.min(), 0) + eps)),
    }
    s = np.sqrt(var_post)
    return rows, joint, post / np.outer(s, s)
