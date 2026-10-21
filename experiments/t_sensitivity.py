"""Feature-level sensitivity to the prior's diffusion time t (question 3 of the plan).

The prior curvature is capped by ~1/s_t^2, so R can move with t alone. Two modes, on
the eval split, for a few degradations:

    fixed  reconstruct once with the tuned config, then compute the attribution with the
           prior evaluated at each t
    refit  reconstruct again with the prior at each t (same weight and annealing), and
           compute the attribution at that t

The question is whether the per-feature conclusions (which features are data- or
prior-dominated, how much the data add) survive a change of t. For each t and mode,
the within-case Spearman correlation between the feature ranking at t and at the tuned
t is reported, for R, log sigma_post and info_gain_pop.

    uv run python experiments/t_sensitivity.py [--quick]

Writes results/t_sensitivity.csv (one row per case, mode, t, feature),
results/t_sensitivity.json and results/fig_t_sensitivity.png.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # default threading makes 1024^2 eigh ~100x slower

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from provenance import data, forward
from provenance.evaluate import assess, population_std
from provenance.features import NAMES
from provenance.pipeline import Config, is_quick, parse_args, results_dir, run_case, save_json

N_SOURCES = 12
T_EVAL = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
CONDITIONS = ("none", "half-plane", "core", "outer", "noise")
MASK_FRACTION = 0.5
NOISE_SCALE = 4.0
RANKED = ("R", "log10_sigma_post", "info_gain_pop")


def use_quick_settings():
    global N_SOURCES, T_EVAL, CONDITIONS
    N_SOURCES, T_EVAL, CONDITIONS = 1, (0.05, 0.5), ("none", "half-plane")


def eval_sources():
    _, ev = data.split(data.load_sources())
    return ev[:: max(1, len(ev) // N_SOURCES)][:N_SOURCES]


def case_kwargs(src, condition):
    if condition == "none":
        return {}
    if condition == "half-plane":
        return {"mask": forward.half_plane_mask(src.truth.shape, MASK_FRACTION)}
    if condition in ("core", "outer"):
        return {"mask": forward.flux_mask(src.truth, MASK_FRACTION, condition)}
    return {"noise_scale": NOISE_SCALE}


def run(cfg, pop, t_grid):
    rows = []
    for src in eval_sources():
        for cond in CONDITIONS:
            kw = case_kwargs(src, cond)
            fixed = run_case(src, cfg, **kw)
            for t in t_grid:
                prior = replace(cfg, prior_t=t).prior()
                rows += [r | {"mode": "fixed", "t": t} for r in assess(fixed, src, prior, pop, {"condition": cond})]
                refit = fixed if t == cfg.prior_t else run_case(src, replace(cfg, prior_t=t), prior=prior, **kw)
                rows += [r | {"mode": "refit", "t": t} for r in assess(refit, src, prior, pop, {"condition": cond})]
        print(src.source_id, flush=True)
    return pd.DataFrame(rows)


def summarise(df, t_ref):
    df = df.assign(log10_sigma_post=np.log10(df["sigma_post"]))
    out = {"t_ref": t_ref, "rank_agreement": {}, "median_by_t": {}}
    ref = df[(df["mode"] == "fixed") & (df["t"] == t_ref)].set_index(["source", "condition", "feature"]).sort_index()
    for (mode, t), sub in df.groupby(["mode", "t"]):
        agree = {}
        for col in RANKED:
            rhos = []
            for (src, cond), g in sub.groupby(["source", "condition"]):
                a = g.set_index("feature")[col].reindex(NAMES).values
                b = ref.loc[(src, cond)][col].reindex(NAMES).values
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 5 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0:
                    rhos.append(spearmanr(a[ok], b[ok]).statistic)
            agree[col] = float(np.median(rhos)) if rhos else np.nan
        out["rank_agreement"][f"{mode}/{t}"] = agree
    med = df.groupby(["mode", "condition", "t", "feature"])[["R", "kappa", "log10_sigma_post", "info_gain_pop", "error_std_units"]].median()
    out["median_by_t"] = {"/".join(map(str, k)): v.round(4).to_dict() for k, v in med.iterrows()}
    return out


def plot(df, path):
    df = df.assign(log10_sigma_post=np.log10(df["sigma_post"]))
    cols = ("R", "kappa", "info_gain_pop")
    fig, axes = plt.subplots(2, len(cols), figsize=(4.4 * len(cols), 7), squeeze=False)
    for i, mode in enumerate(("fixed", "refit")):
        sub = df[(df["mode"] == mode) & (df["condition"] == "none")]
        for j, col in enumerate(cols):
            ax = axes[i, j]
            for feat in NAMES:
                g = sub[sub.feature == feat].groupby("t")[col]
                med, lo, hi = g.median(), g.quantile(0.16), g.quantile(0.84)
                ax.plot(med.index, med.values, marker="o", ms=3, label=feat)
                ax.fill_between(med.index, lo.values, hi.values, alpha=0.12)
            ax.set(xscale="log", xlabel="t", title=f"{col}  [{mode}, full data]")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parse_args()
    if is_quick():
        use_quick_settings()
    cfg = Config.load()
    t_grid = sorted(set(T_EVAL) | {cfg.prior_t})
    df = run(cfg, population_std(), t_grid)
    df.to_csv(results_dir() / "t_sensitivity.csv", index=False)
    summary = summarise(df, cfg.prior_t)
    save_json(summary, "t_sensitivity.json")
    plot(df, results_dir() / "fig_t_sensitivity.png")
    print(pd.DataFrame(summary["rank_agreement"]).T.round(2))


if __name__ == "__main__":
    main()
