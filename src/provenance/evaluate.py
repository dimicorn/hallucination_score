"""Per-case evaluation shared by the Phase C/D experiments.

For one (source, degradation, noise seed) case: reconstruct, then record for every
feature its reconstructed and true value, the C1 local curvature diagnostic, the
feature-space Laplace diagnostics (posterior std, information gain, cross-feature
posterior correlations) and a pixel-level summary of the same case for comparison with
the baseline map. The dense curvature matrices are built once and shared by the pixel
and Laplace steps.
"""

import json
from functools import lru_cache

import numpy as np

from . import attribution, features, laplace
from .pipeline import results_dir, run_case

FOOTPRINT = 0.02


def _population_file():
    return json.loads((results_dir() / "feature_population.json").read_text())


def population_std():
    return {k: v["std"] for k, v in _population_file()["population"].items()}


@lru_cache(maxsize=None)
def population_moments():
    """(mean, covariance) of the features over the reference population, in NAMES
    order. Files written before the covariance was saved give a diagonal one."""
    stats = _population_file()
    mean = np.array([stats["population"][n]["mean"] for n in features.NAMES])
    if "covariance" in stats:
        cov = np.array([[stats["covariance"][a][b] for b in features.NAMES] for a in features.NAMES])
    else:
        cov = np.diag([stats["population"][n]["std"] ** 2 for n in features.NAMES])
    return mean, cov


def evaluate(source, cfg, pop_std, condition, pixel=True, with_laplace=True, prior=None, ref_grads=None, **case_kwargs):
    """Reconstruct one case and assess it with the same prior."""
    prior = cfg.prior() if prior is None else prior
    case = run_case(source, cfg, prior=prior, **case_kwargs)
    return assess(case, source, prior, pop_std, condition, pixel, with_laplace, ref_grads), case


def assess(case, source, prior, pop_std, condition, pixel=True, with_laplace=True, ref_grads=None):
    """Rows (one per feature) for a reconstructed case, with the attribution computed
    under ``prior`` (which need not be the prior the case was reconstructed with).

    ``ref_grads`` (feature name -> gradient, e.g. from the undegraded reconstruction of the
    same source and noise draw) adds I_D_fixed, the data curvature along that fixed
    direction, and cos_ref, so a change of I_D can be split into the effect of the
    operator and the drift of the feature gradient."""
    truth_vals = features.measure(np.asarray(source.truth, np.float32))
    attrs = attribution.feature_attribution(case.obs, prior, case.x_hat, population_std=pop_std)
    x = np.asarray(case.x_hat)
    common = {"source": source.source_id, "snr": source.snr, **condition, "opt_grad_norm": float(case.grad_norm), "frac_at_bound": float(np.mean(x <= 0))}
    grads = features.gradients(case.x_hat)
    units = {a.name: np.asarray(grads[a.name], np.float64).ravel() / a.grad_norm if a.grad_norm > 0 else np.zeros(x.size) for a in attrs}
    per_feature = {a.name: {} for a in attrs}
    dense = laplace.dense_curvatures(case.obs, prior, case.x_hat) if (pixel or with_laplace) else None
    if pixel:
        pix = attribution.pixel_scores(case.obs, prior, case.x_hat, dense=dense)
        foot = source.truth > FOOTPRINT
        common["pixel_frac_prior_footprint"] = float(np.mean(pix["delta"][foot] > 0))
        common["pixel_median_dF_footprint"] = float(np.median(pix["delta"][foot]))
        ratio = pix["diag_P"] / (pix["diag_P"] + pix["diag_D"])
        common["pixel_ratio_footprint"] = float(np.nanmean(ratio[foot]))
        for a in attrs:
            per_feature[a.name].update(attribution.diagonal_attribution(pix["diag_D"], pix["diag_P"], units[a.name]))
    if ref_grads is not None:
        for a in attrs:
            ref = np.asarray(ref_grads[a.name], np.float64).ravel()
            ref = ref / np.linalg.norm(ref) if np.linalg.norm(ref) > 0 else ref
            mv = np.asarray(case.obs.curvature_matvec(ref.reshape(x.shape).astype(np.float32)), np.float64).ravel()
            per_feature[a.name].update({"I_D_fixed": float(ref @ mv), "cos_ref": float(ref @ units[a.name])})
    if with_laplace:
        mean, cov = population_moments()
        lap_rows, joint, corr = laplace.feature_laplace(
            *dense, np.stack([np.asarray(grads[a.name]).ravel() for a in attrs]), [a.value for a in attrs], mean, cov, free=x > 0
        )
        common.update(joint)
        for i, (a, lap) in enumerate(zip(attrs, lap_rows)):
            per_feature[a.name].update(lap)
            per_feature[a.name].update({f"corr_post_{b.name}": float(corr[i, j]) for j, b in enumerate(attrs)})
    common["relerr_image"] = float(np.linalg.norm(x - source.truth) / np.linalg.norm(source.truth))
    rows = []
    for a in attrs:
        row = dict(common)
        row.update(a.as_dict())
        row.update(per_feature[a.name])
        if row.get("eta_null", 0) > laplace.NULL_DOMINATED:  # mostly invisible to the data, whatever sigma_data says
            row["flags"] = ",".join(filter(None, [row["flags"], "null_space_dominated"]))
        row["truth_value"] = truth_vals[a.name]
        row["error_std_units"] = (a.value - truth_vals[a.name]) / pop_std[a.name]
        rows.append(row)
    return rows
