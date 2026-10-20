"""Figures and quantitative summaries for the Phase C/D experiments.

Reads results/phase_d_*.csv (whichever exist) and writes results/analysis.json plus
figures. Questions addressed (plan section 7):

Q1  Does data support I_D fall as feature-relevant measurements are removed?
    Spearman rho(severity, I_D) per feature and degradation.
Q2  Prior dominance vs low support: R and kappa reported alongside sigma_loc.
Q3  Stability: spread over noise draws (16-84% bands in figures).
Q4  What the feature level adds: features are ranked differently by core vs outer
    masks that remove the same flux, while the pixel summary is one number.
Q5  Agreement with the intervention diagnostic: rho(R, prior sensitivity S).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from provenance.features import NAMES
from provenance.pipeline import RESULTS, save_json

LOW_SUPPORT_FLAGS = ("weakly_constrained", "near_zero_total")


def _load(name):
    path = RESULTS / f"phase_d_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["flags"] = df["flags"].fillna("")
    df["log10_I_D"] = np.log10(df["I_D"].clip(lower=1e-12))
    df["abs_error"] = df["error_std_units"].abs()
    df["low_support"] = df["flags"].str.contains("|".join(LOW_SUPPORT_FLAGS))
    return df


def _rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def response(df, severity, groups=()):
    """Spearman correlations of each diagnostic with degradation severity, per feature."""
    out = {}
    for keys, sub in df.groupby(list(groups) + ["feature"]) if groups else df.groupby("feature"):
        key = "/".join(map(str, keys if isinstance(keys, tuple) else (keys,)))
        out[key] = {
            "rho_severity_log10_I_D": _rho(sub[severity].values, sub["log10_I_D"].values),
            "rho_severity_R": _rho(sub[severity].values, sub["R"].values),
            "rho_severity_abs_error": _rho(sub[severity].values, sub["abs_error"].values),
            "frac_low_support": float(sub["low_support"].mean()),
        }
    return out


def error_prediction(df):
    """Does the diagnostic flag the features that are reconstructed badly?"""
    out = {}
    for feat, sub in df.groupby("feature"):
        out[feat] = {
            "rho_R_abs_error": _rho(sub["R"].values, sub["abs_error"].values),
            "rho_sigma_local_abs_error": _rho(np.log10(sub["sigma_local"].values), sub["abs_error"].values),
            "rho_pixel_frac_abs_error": _rho(sub["pixel_frac_prior_footprint"].values, sub["abs_error"].values) if "pixel_frac_prior_footprint" in sub else np.nan,
        }
    return out


def curves(df, x, by, path, title, logx=False, diagnostics=("R", "kappa", "log10_I_D", "abs_error")):
    kinds = sorted(df[by].unique()) if by else [None]
    fig, axes = plt.subplots(len(kinds), len(diagnostics), figsize=(4.2 * len(diagnostics), 3.2 * len(kinds)), squeeze=False)
    for i, kind in enumerate(kinds):
        sub = df if kind is None else df[df[by] == kind]
        for j, diag in enumerate(diagnostics):
            ax = axes[i, j]
            for feat in NAMES:
                g = sub[sub.feature == feat].groupby(x)[diag]
                med, lo, hi = g.median(), g.quantile(0.16), g.quantile(0.84)
                ax.plot(med.index, med.values, marker="o", ms=3, label=feat)
                ax.fill_between(med.index, lo.values, hi.values, alpha=0.12)
            if "pixel_frac_prior_footprint" in sub and diag == "R":
                g = sub[sub.feature == NAMES[0]].groupby(x)["pixel_frac_prior_footprint"].median()
                ax.plot(g.index, g.values, "k--", lw=1.5, label="pixel: frac $\\Delta F>0$")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(x)
            ax.set_title(f"{diag}" + (f"  [{kind}]" if kind is not None else ""), fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def feature_level_vs_pixel(df, level=0.5):
    """At equal removed flux, compare feature rankings under core vs outer masks."""
    out = {}
    sel = df[(df.mask_fraction == level) & df.mask_kind.isin(["core", "outer"])]
    med = sel.groupby(["mask_kind", "feature"])[["R", "log10_I_D", "abs_error"]].median()
    pix = sel.groupby("mask_kind")["pixel_frac_prior_footprint"].median()
    for kind in ("core", "outer"):
        out[kind] = {
            "pixel_frac_prior_footprint": float(pix.get(kind, np.nan)),
            "R": med.loc[kind]["R"].round(4).to_dict(),
            "log10_I_D": med.loc[kind]["log10_I_D"].round(3).to_dict(),
            "abs_error": med.loc[kind]["abs_error"].round(3).to_dict(),
        }
    a, b = (np.array([out[k]["log10_I_D"][f] for f in NAMES]) for k in ("core", "outer"))
    out["rank_corr_I_D_core_vs_outer"] = _rho(a, b)
    return out


def intervention(df_prior, pop_std=None):
    """Prior sensitivity S = |phi(lambda=2) - phi(lambda=0.5)| / (2 * pop_std), per
    doubling of lambda, compared with R at lambda=1 for the same observation."""
    from provenance.evaluate import population_std

    pop_std = population_std() if pop_std is None else pop_std
    keys = ["source", "degradation", "level", "feature"]
    piv = df_prior.pivot_table(index=keys, columns="prior_weight", values="value").reset_index()
    base = df_prior[df_prior.prior_weight == 1.0].set_index(keys)
    piv = piv.set_index(keys)
    piv["S"] = (piv[2.0] - piv[0.5]).abs() / (2 * piv.index.get_level_values("feature").map(pop_std).values)
    piv["R"] = base["R"]
    piv["kappa"] = base["kappa"]
    piv["log10_I_D"] = np.log10(base["I_D"].clip(lower=1e-12))
    piv = piv.reset_index()
    out = {"per_feature": {}, "pooled": {}}
    for feat, sub in piv.groupby("feature"):
        out["per_feature"][feat] = {
            "rho_R_S": _rho(sub["R"].values, sub["S"].values),
            "rho_log10_I_D_S": _rho(sub["log10_I_D"].values, sub["S"].values),
            "median_S": float(sub["S"].median()),
        }
    # pooled: within each case, does the feature ranking by R match the ranking by S?
    within = [
        _rho(g["R"].values, g["S"].values) for _, g in piv.groupby(["source", "degradation", "level"]) if g["R"].notna().sum() >= 5
    ]
    out["pooled"]["median_within_case_rho_R_S"] = float(np.nanmedian(within)) if within else np.nan
    out["pooled"]["n_cases"] = len(within)
    return out, piv


def plot_intervention(piv, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for feat in NAMES:
        sub = piv[piv.feature == feat]
        axes[0].scatter(sub["R"], sub["S"], s=8, label=feat)
        axes[1].scatter(sub["log10_I_D"], sub["S"], s=8, label=feat)
    for ax, xl in zip(axes, ("R (curvature, $\\lambda$=1)", "log10 $I_D$")):
        ax.set(xlabel=xl, ylabel="prior sensitivity S [pop. std per doubling of $\\lambda$]", yscale="log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    summary = {}
    masking = _load("masking")
    if masking is not None:
        m = masking.copy()
        # the unmasked case is level 0 of every mask kind
        base = m[m.mask_kind == "none"]
        m = pd.concat([m[m.mask_kind != "none"]] + [base.assign(mask_kind=k) for k in ("half-plane", "core", "outer")])
        summary["masking_response"] = response(m, "flux_removed", groups=("mask_kind",))
        summary["masking_error_prediction"] = error_prediction(m)
        summary["masking_feature_vs_pixel"] = feature_level_vs_pixel(m)
        curves(m, "mask_fraction", "mask_kind", RESULTS / "fig_masking.png", "Experiment 1: progressive masking (median, 16-84% over sources x noise draws)")
    snr = _load("snr")
    if snr is not None:
        summary["snr_response"] = response(snr, "noise_scale")
        summary["snr_error_prediction"] = error_prediction(snr)
        curves(snr, "noise_scale", None, RESULTS / "fig_snr.png", "Experiment 2: noise sweep (noise std x native)", logx=True)
    blur = _load("blur")
    if blur is not None:
        blur["inv_cutoff"] = 1 / blur["blur_cutoff"]
        summary["blur_response"] = response(blur, "inv_cutoff")
        curves(blur, "inv_cutoff", None, RESULTS / "fig_blur.png", "Experiment 3: Fourier low-pass (1 / cutoff; 0 = no low-pass)")
    prior = _load("prior")
    if prior is not None:
        summary["intervention"], piv = intervention(prior)
        plot_intervention(piv, RESULTS / "fig_intervention.png")
    save_json(summary, "analysis.json")
    import json

    print(json.dumps(summary, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
