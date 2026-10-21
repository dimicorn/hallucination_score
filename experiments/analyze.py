"""Figures and quantitative summaries for the Phase C/D experiments.

Reads results/phase_d_*.csv (whichever exist) and writes results/analysis.json plus
figures. Questions addressed (plan section 7):

Q1  Does data support I_D fall as feature-relevant measurements are removed?
    Spearman rho(severity, I_D) within each source, median over sources, per feature
    and degradation; paired change in log10 I_D against the undegraded case.
Q2  Prior dominance vs low support: R and kappa reported alongside the data-only
    support sigma_data; low support = sigma_data above the population spread.
Q3  Stability: spread over noise draws (16-84% bands in figures).
Q4  What the feature level adds: core and outer masks remove information from
    different features (paired I_D loss per feature), while the pixel summary is one
    number; per-feature baseline attribution and bias with full data.
Q5  Agreement with the intervention diagnostic: rho(R, prior sensitivity S).

The low-support flag is recomputed here from I_D and the population spread rather than
read from the CSV, so results written before the flag changed are analysed the same way.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from provenance.evaluate import population_std
from provenance.features import NAMES
from provenance.pipeline import parse_args, results_dir, save_json

EXPERIMENT_COLORS = {"masking": "C0", "snr": "C1", "blur": "C2"}


def _load(name, pop_std):
    path = results_dir() / f"phase_d_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["flags"] = df["flags"].fillna("")
    df["log10_I_D"] = np.log10(df["I_D"].clip(lower=1e-12))
    df["abs_error"] = df["error_std_units"].abs()
    if "sigma_data" not in df:  # written before the column existed: ||g|| = sigma_local sqrt(total)
        grad_norm = df["sigma_local"] * np.sqrt(df["total"].clip(lower=0))
        df["sigma_data"] = np.where(df["I_D"] > 0, grad_norm / np.sqrt(df["I_D"].clip(lower=1e-300)), np.inf)
    df["sigma_data_rel"] = df["sigma_data"] / df["feature"].map(pop_std)
    df["low_support"] = (df["sigma_data_rel"] > 1) | df["flags"].str.contains("near_zero_total")
    return df


def _rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def _median(values):
    """Median of the finite values; NaN (without a warning) if there are none."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _paired(df, is_base):
    """Rows not in the base condition, with d_log10_I_D against the base case of the
    same source, noise draw and feature."""
    keys = ["source", "seed", "feature"]
    base = df[is_base].set_index(keys)["log10_I_D"]
    rest = df[~is_base].copy()
    rest["d_log10_I_D"] = rest["log10_I_D"].values - base.reindex(pd.MultiIndex.from_frame(rest[keys])).values
    return rest


def response(df, severity, groups=()):
    """Within-source Spearman correlations of each diagnostic with degradation severity
    (median over sources), per feature. Pooling sources would mostly measure how
    sources differ, not how one source responds."""
    out = {}
    for keys, sub in df.groupby(list(groups) + ["feature"]):
        key = "/".join(map(str, keys))
        per_source = sub.groupby("source")
        out[key] = {
            f"rho_severity_{d}": _median([_rho(g[severity].values, g[d].values) for _, g in per_source])
            for d in ("log10_I_D", "R", "abs_error")
        }
        out[key]["frac_low_support"] = float(sub["low_support"].mean())
    return out


def paired_change(df, is_base, index):
    """Median paired change in log10 I_D against the undegraded case."""
    tab = _paired(df, is_base).pivot_table(index=index, columns="feature", values="d_log10_I_D", aggfunc="median")
    return {"/".join(map(str, k if isinstance(k, tuple) else (k,))): row.round(3).to_dict() for k, row in tab[list(NAMES)].iterrows()}


def baseline(df):
    """Per-feature attribution and error with full data at native noise."""
    cols = ["log10_I_D", "kappa", "R", "sigma_data_rel", "error_std_units", "abs_error"]
    med = df.groupby("feature")[cols].median().loc[list(NAMES)]
    out = {feat: row.round(4).to_dict() for feat, row in med.iterrows()}
    out["rho_R_vs_abs_bias_across_features"] = _rho(med["R"].values, med["error_std_units"].abs().values)
    return out


def error_prediction(df):
    """Does the diagnostic flag the features that are reconstructed badly?"""
    out = {}
    for feat, sub in df.groupby("feature"):
        out[feat] = {
            "rho_R_abs_error": _rho(sub["R"].values, sub["abs_error"].values),
            "rho_sigma_local_abs_error": _rho(np.log10(sub["sigma_local"].values), sub["abs_error"].values),
            "rho_sigma_data_abs_error": _rho(np.log10(sub["sigma_data"].values), sub["abs_error"].values),
            "rho_pixel_frac_abs_error": _rho(sub["pixel_frac_prior_footprint"].values, sub["abs_error"].values) if "pixel_frac_prior_footprint" in sub else np.nan,
        }
    return out


def low_support_check(dfs):
    """Are rows flagged low-support actually reconstructed worse?"""
    rows = pd.concat([df for df in dfs if df is not None])
    out = {}
    for feat, sub in rows.groupby("feature"):
        flagged, rest = sub[sub["low_support"]], sub[~sub["low_support"]]
        out[feat] = {
            "frac_flagged": float(sub["low_support"].mean()),
            "median_abs_error_flagged": float(flagged["abs_error"].median()) if len(flagged) else np.nan,
            "median_abs_error_unflagged": float(rest["abs_error"].median()) if len(rest) else np.nan,
        }
    return out


def mask_selectivity(m):
    """Q4: do masks of equal flux remove information from different features?
    Per-feature paired I_D loss for each mask, and the rank correlation between the
    feature orderings of core and outer masks at each level (low = selective)."""
    paired = _paired(m, m["mask_kind"] == "none")
    tab = paired.pivot_table(index=["mask_kind", "mask_fraction"], columns="feature", values="d_log10_I_D", aggfunc="median")
    out = {"rank_corr_core_vs_outer": {}, "pixel_frac_prior_footprint": {}}
    for level in sorted(m.loc[m["mask_kind"] == "core", "mask_fraction"].unique()):
        if ("core", level) in tab.index and ("outer", level) in tab.index:
            out["rank_corr_core_vs_outer"][str(level)] = _rho(tab.loc[("core", level)].values, tab.loc[("outer", level)].values)
    pix = m.groupby(["mask_kind", "mask_fraction"])["pixel_frac_prior_footprint"].median()
    out["pixel_frac_prior_footprint"] = {f"{k}/{f}": float(v) for (k, f), v in pix.items()}
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


def plot_error_prediction(dfs, path):
    """Per feature: reconstruction error against R (top) and against the data-only
    support sigma_data / population std (bottom; right of the line = flagged)."""
    fig, axes = plt.subplots(2, len(NAMES), figsize=(3.0 * len(NAMES), 6), sharey="row")
    for j, feat in enumerate(NAMES):
        for name, df in dfs.items():
            sub = df[df.feature == feat]
            axes[0, j].scatter(sub["R"], sub["abs_error"], s=4, alpha=0.4, color=EXPERIMENT_COLORS[name], label=name)
            axes[1, j].scatter(sub["sigma_data_rel"].clip(upper=1e3), sub["abs_error"], s=4, alpha=0.4, color=EXPERIMENT_COLORS[name])
        axes[1, j].axvline(1, color="k", lw=1, ls="--")
        axes[0, j].set(title=feat, xlabel="R", yscale="log")
        axes[1, j].set(xlabel="$\\sigma_{data}$ / pop. std", xscale="log", yscale="log")
    axes[0, 0].set_ylabel("|error| [pop. std]")
    axes[1, 0].set_ylabel("|error| [pop. std]")
    axes[0, 0].legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def intervention(df_prior, pop_std):
    """Prior sensitivity S = |phi(lambda=2) - phi(lambda=0.5)| / (2 * pop_std), per
    doubling of lambda, compared with R at lambda=1 for the same observation."""
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
    out["pooled"]["median_within_case_rho_R_S"] = _median(within)
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
    parse_args()
    pop_std = population_std()
    summary = {}
    masking, snr, blur, prior = (_load(name, pop_std) for name in ("masking", "snr", "blur", "prior"))
    if masking is not None:
        m = masking.copy()
        summary["baseline_full_data"] = baseline(m[m.mask_kind == "none"])
        # the unmasked case is level 0 of every mask kind
        base = m[m.mask_kind == "none"]
        m = pd.concat([m[m.mask_kind != "none"]] + [base.assign(mask_kind=k) for k in ("half-plane", "core", "outer")])
        summary["masking_response"] = response(m, "flux_removed", groups=("mask_kind",))
        summary["masking_paired_d_log10_I_D"] = paired_change(masking, masking.mask_kind == "none", ["mask_kind", "mask_fraction"])
        summary["masking_selectivity"] = mask_selectivity(masking)
        summary["masking_error_prediction"] = error_prediction(m)
        curves(m, "mask_fraction", "mask_kind", results_dir() / "fig_masking.png", "Experiment 1: progressive masking (median, 16-84% over sources x noise draws)")
    if snr is not None:
        summary["snr_response"] = response(snr, "noise_scale")
        summary["snr_error_prediction"] = error_prediction(snr)
        curves(snr, "noise_scale", None, results_dir() / "fig_snr.png", "Experiment 2: noise sweep (noise std x native)", logx=True)
    if blur is not None:
        blur["inv_cutoff"] = 1 / blur["blur_cutoff"]
        summary["blur_response"] = response(blur, "inv_cutoff")
        summary["blur_paired_d_log10_I_D"] = paired_change(blur, np.isinf(blur.blur_cutoff), "blur_cutoff")
        curves(blur, "inv_cutoff", None, results_dir() / "fig_blur.png", "Experiment 3: Fourier low-pass (1 / cutoff; 0 = no low-pass)")
    sweeps = {k: v for k, v in (("masking", masking), ("snr", snr), ("blur", blur)) if v is not None}
    if sweeps:
        summary["low_support_check"] = low_support_check(sweeps.values())
        plot_error_prediction(sweeps, results_dir() / "fig_error_prediction.png")
    if prior is not None:
        summary["intervention"], piv = intervention(prior, pop_std)
        plot_intervention(piv, results_dir() / "fig_intervention.png")
    path = save_json(summary, "analysis.json")
    print(f"wrote {path}; sections: {', '.join(summary)}")


if __name__ == "__main__":
    main()
