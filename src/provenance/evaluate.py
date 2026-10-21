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


def evaluate(source, cfg, pop_std, condition, pixel=True, with_laplace=True, prior=None, **case_kwargs):
    """Reconstruct one case and assess it with the same prior."""
    prior = cfg.prior() if prior is None else prior
    case = run_case(source, cfg, prior=prior, **case_kwargs)
    return assess(case, source, prior, pop_std, condition, pixel, with_laplace), case


def assess(case, source, prior, pop_std, condition, pixel=True, with_laplace=True):
    """Rows (one per feature) for a reconstructed case, with the attribution computed
    under ``prior`` (which need not be the prior the case was reconstructed with)."""
    truth_vals = features.measure(np.asarray(source.truth, np.float32))
    attrs = attribution.feature_attribution(case.obs, prior, case.x_hat, population_std=pop_std)
    common = {"source": source.source_id, "snr": source.snr, **condition}
    dense = laplace.dense_curvatures(case.obs, prior, case.x_hat) if (pixel or with_laplace) else None
    if pixel:
        pix = attribution.pixel_scores(case.obs, prior, case.x_hat, dense=dense)
        foot = source.truth > FOOTPRINT
        common["pixel_frac_prior_footprint"] = float(np.mean(pix["delta"][foot] > 0))
        common["pixel_median_dF_footprint"] = float(np.median(pix["delta"][foot]))
    lap_rows = [{}] * len(attrs)
    if with_laplace:
        grads = features.gradients(case.x_hat)
        mean, cov = population_moments()
        lap_rows, joint, corr = laplace.feature_laplace(
            *dense, np.stack([np.asarray(grads[a.name]).ravel() for a in attrs]), [a.value for a in attrs], mean, cov
        )
        common.update(joint)
    common["relerr_image"] = float(np.linalg.norm(case.x_hat - source.truth) / np.linalg.norm(source.truth))
    rows = []
    for i, (a, lap) in enumerate(zip(attrs, lap_rows)):
        row = dict(common)
        row.update(a.as_dict())
        row.update(lap)
        if with_laplace:
            row.update({f"corr_post_{b.name}": float(corr[i, j]) for j, b in enumerate(attrs)})
        row["truth_value"] = truth_vals[a.name]
        row["error_std_units"] = (a.value - truth_vals[a.name]) / pop_std[a.name]
        rows.append(row)
    return rows
