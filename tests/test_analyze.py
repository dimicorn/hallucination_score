"""Helpers of experiments/analyze.py on small synthetic result tables."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance.attribution import FeatureAttribution
from provenance.features import NAMES

_spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parents[1] / "experiments" / "analyze.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

POP_STD = {name: 1.0 for name in NAMES}


GRAD_NORM = 2.0


def _rows(**common):
    """One row per feature with the columns phase_d writes (I_D = 100, I_P = 50)."""
    out = []
    for name in NAMES:
        row = FeatureAttribution(name, value=0.0, grad_norm=GRAD_NORM, I_D=100.0, I_P=50.0, I_cap=100.0).as_dict()
        row.update(common, truth_value=0.0, error_std_units=0.1)
        out.append(row)
    return out


def _prepare(df):
    """What _load adds to a table read from disk (rows here may have changed I_D)."""
    df = df.copy()
    df["flags"] = df["flags"].fillna("")
    df["log10_I_D"] = np.log10(df["I_D"].clip(lower=1e-12))
    df["abs_error"] = df["error_std_units"].abs()
    df["sigma_data"] = GRAD_NORM / np.sqrt(df["I_D"])
    df["sigma_data_rel"] = df["sigma_data"] / df["feature"].map(POP_STD)
    df["low_support"] = df["sigma_data_rel"] > 1
    return df


def test_load_derives_sigma_data_for_old_tables(tmp_path, monkeypatch):
    """Tables written before the sigma_data column existed get it from sigma_local and
    total, and low support is recomputed from the data curvature."""
    attrs = [
        FeatureAttribution("flux", 1.0, 2.0, I_D=400.0, I_P=50.0, I_cap=100.0),  # well constrained by data
        FeatureAttribution("size", 1.0, 2.0, I_D=1.0, I_P=500.0, I_cap=600.0),  # held by the prior only
        FeatureAttribution("e1", 1.0, 2.0, I_D=0.0, I_P=500.0, I_cap=600.0),  # no data at all
    ]
    rows = [a.as_dict() | {"error_std_units": 0.0, "flags": "weakly_constrained"} for a in attrs]
    pd.DataFrame(rows).drop(columns="sigma_data").to_csv(tmp_path / "phase_d_masking.csv", index=False)
    monkeypatch.setattr(analyze, "results_dir", lambda: tmp_path)

    df = analyze._load("masking", POP_STD)
    np.testing.assert_allclose(df["sigma_data"], [a.sigma_data for a in attrs])
    # the old flag column is ignored: size is held by the prior (sigma_local small) but
    # the data alone leave it unconstrained
    assert df["low_support"].tolist() == [False, True, True]
    assert analyze._load("snr", POP_STD) is None


def test_response_correlates_within_sources():
    """Two sources on very different I_D scales, each losing support steadily: the
    within-source correlation is -1, which pooling the sources would hide."""
    rows = []
    for source, scale in (("a", 1e4), ("b", 1e-2)):
        for severity in range(6):
            for row in _rows(source=source, seed=0, severity=severity):
                row["I_D"] = scale * 0.5**severity
                rows.append(row)
    df = _prepare(pd.DataFrame(rows))
    out = analyze.response(df, "severity")
    for name in NAMES:
        assert out[name]["rho_severity_log10_I_D"] == pytest.approx(-1.0)


def test_paired_change_against_base_case():
    rows = []
    for kind, factor in (("none", 1.0), ("core", 0.5), ("outer", 0.01)):
        for row in _rows(source="a", seed=0, mask_kind=kind, mask_fraction=0.0 if kind == "none" else 0.5):
            row["I_D"] *= factor
            rows.append(row)
    df = _prepare(pd.DataFrame(rows))
    out = analyze.paired_change(df, df.mask_kind == "none", ["mask_kind", "mask_fraction"])
    assert out["core/0.5"]["flux"] == pytest.approx(np.log10(0.5), abs=1e-3)  # rounded to 3 decimals
    assert out["outer/0.5"]["size"] == pytest.approx(-2.0, abs=1e-3)


def test_mask_selectivity_detects_opposite_feature_orderings():
    """Core masks remove most information from the first feature, outer masks from the
    last: the rank correlation of the two orderings is -1."""
    loss = np.linspace(0.1, 1.0, len(NAMES))
    rows = []
    for kind, d in (("none", np.zeros(len(NAMES))), ("core", -loss[::-1]), ("outer", -loss)):
        for row, dlog in zip(_rows(source="a", seed=0, mask_kind=kind, mask_fraction=0.0 if kind == "none" else 0.5, pixel_frac_prior_footprint=1.0), d):
            row["I_D"] *= 10**dlog
            rows.append(row)
    out = analyze.mask_selectivity(_prepare(pd.DataFrame(rows)))
    assert out["rank_corr_core_vs_outer"]["0.5"] == pytest.approx(-1.0)


def test_low_support_check_splits_errors_by_flag():
    rows = []
    for i, (i_d, err) in enumerate([(400.0, 0.1), (400.0, 0.2), (1e-6, 2.0), (1e-6, 3.0)]):
        for row in _rows(source=str(i), seed=0):
            row.update(I_D=i_d, error_std_units=err)
            rows.append(row)
    out = analyze.low_support_check([_prepare(pd.DataFrame(rows))])
    assert out["flux"]["frac_flagged"] == pytest.approx(0.5)
    assert out["flux"]["median_abs_error_flagged"] == pytest.approx(2.5)
    assert out["flux"]["median_abs_error_unflagged"] == pytest.approx(0.15)


def test_calibration_counts_errors_within_one_sigma():
    """Errors of 0.5, 1.5, 2.5 and 3.5 against sigma_post = 2 (pop. std 1): half of the
    cases fall within 1 sigma. Cases dominated by negative-curvature modes are left out."""
    rows = []
    for i, err in enumerate([0.5, 1.5, 2.5, 3.5, 0.1]):
        for row in _rows(source=str(i), seed=0):
            row.update(error_std_units=err, sigma_post=2.0, neg_mode_share=0.9 if i == 4 else 0.0)
            rows.append(row)
    df = _prepare(pd.DataFrame(rows))
    df["sigma_post_rel"] = df["sigma_post"]
    df["sigma_local_rel"] = df["sigma_local"]
    out = analyze.calibration({"masking": df})
    assert out["masking"]["frac_cases_used"] == pytest.approx(0.8)
    assert out["masking"]["flux"]["sigma_post_coverage_1sigma"] == pytest.approx(0.5)
    assert out["masking"]["flux"]["sigma_post_median_z"] == pytest.approx(1.0)
