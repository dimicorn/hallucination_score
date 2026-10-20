"""Per-case evaluation shared by the Phase C/D experiments.

For one (source, degradation, noise seed) case: reconstruct, then record for every
feature its reconstructed and true value, the C1 local curvature diagnostic, and a
pixel-level summary of the same case for comparison with the baseline map.
"""

import json

import numpy as np

from . import attribution, features
from .pipeline import RESULTS, run_case

FOOTPRINT = 0.02


def population_std():
    stats = json.loads((RESULTS / "feature_population.json").read_text())["population"]
    return {k: v["std"] for k, v in stats.items()}


def evaluate(source, cfg, pop_std, condition, pixel=True, prior=None, **case_kwargs):
    prior = cfg.prior() if prior is None else prior
    case = run_case(source, cfg, prior=prior, **case_kwargs)
    truth_vals = features.measure(np.asarray(source.truth, np.float32))
    attrs = attribution.feature_attribution(case.obs, prior, case.x_hat, population_std=pop_std)
    common = {"source": source.source_id, "snr": source.snr, **condition}
    if pixel:
        pix = attribution.pixel_scores(case.obs, prior, case.x_hat, exact=True)
        foot = source.truth > FOOTPRINT
        common["pixel_frac_prior_footprint"] = float(np.mean(pix["delta"][foot] > 0))
        common["pixel_median_dF_footprint"] = float(np.median(pix["delta"][foot]))
    common["relerr_image"] = float(np.linalg.norm(case.x_hat - source.truth) / np.linalg.norm(source.truth))
    rows = []
    for a in attrs:
        row = dict(common)
        row.update(a.as_dict())
        row["truth_value"] = truth_vals[a.name]
        row["error_std_units"] = (a.value - truth_vals[a.name]) / pop_std[a.name]
        rows.append(row)
    return rows, case
