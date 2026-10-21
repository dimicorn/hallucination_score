"""Publication figures for the write-up, from the Phase D results.

    uv run python experiments/paper_figures.py [--quick]

Writes results/paper/fig_{main,errors,example,tsens}.{pdf,png}:

main     R per feature against degradation severity (half-plane, core, outer and random-
         pixel masks, noise), with the pixel score as a dashed line; lines are medians over
         galaxies, bands 95 % galaxy-bootstrap intervals. Core/outer masks remove rings about
         the reference centroid until a fraction of the reference flux is gone.
errors   R against the diagonal-only R_diag per feature with full data, median errors of
         features flagged weakly constrained or not, and how often the error lies within
         1 sigma_local / sigma_post (95 % galaxy-bootstrap intervals)
tsens    (with results/t_sensitivity.*) R and information gain against the prior's
         diffusion time t, and the agreement of each t's feature ranking with the tuned t
example  one source observed with full data and with its faint outskirts masked: the
         reconstructions look alike and the pixel map is prior-dominated in both, but
         the feature-level data support differs. The pair is chosen from the masking
         results and the two cases are reconstructed again (they are not stored).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # default threading makes 1024^2 eigh ~100x slower

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import _load, _rho
from provenance import attribution, data, forward
from provenance.evaluate import FOOTPRINT, population_std
from provenance.features import NAMES
from provenance.pipeline import Config, is_quick, parse_args, results_dir, run_case

TEXT_WIDTH = 6.8  # inches, A4 with 1.9 cm margins
LABELS = {"flux": "flux", "size": "size", "e1": "$e_1$", "e2": "$e_2$", "concentration": "conc.", "asymmetry": "asym.", "hf_power": "HF"}
COLORS = dict(zip(NAMES, plt.get_cmap("tab10").colors))
EXAMPLE_KIND, EXAMPLE_FRACTION = "outer", 0.5
N_BOOT = 200

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _grid(ax, axis="both"):
    ax.grid(axis=axis, color="0.85", lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    out = results_dir() / "paper"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def _galaxy_median(df, x, y):
    """Median over galaxies of each galaxy's median of y at every x, with a 95 % galaxy-
    bootstrap band (noise draws of one galaxy are not independent)."""
    per = df.groupby([x, "source"])[y].median().unstack()
    rng = np.random.default_rng(0)
    cols = per.columns.to_numpy()
    boots = np.stack([per[rng.choice(cols, len(cols))].median(axis=1).to_numpy() for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return per.index.to_numpy(), per.median(axis=1).to_numpy(), lo, hi


def _feature_stat_ci(df, stat):
    """stat(rows of one feature) per feature, with a 95 % galaxy-bootstrap interval."""
    out = {}
    for feat in NAMES:
        sub = df[df.feature == feat].reset_index(drop=True)
        groups = [g.index.to_numpy() for _, g in sub.groupby("source")]
        rng = np.random.default_rng(0)
        vals = [stat(sub.loc[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])]) for _ in range(N_BOOT)]
        out[feat] = (stat(sub), *np.nanpercentile(vals, [2.5, 97.5]))
    return out


def _median_lines(ax, df, x, y="R"):
    for feat in NAMES:
        xs, med, lo, hi = _galaxy_median(df[df.feature == feat], x, y)
        ax.fill_between(xs, lo, hi, color=COLORS[feat], alpha=0.15, lw=0)
        ax.plot(xs, med, marker="o", ms=2.5, lw=1.2, color=COLORS[feat], label=LABELS[feat])
    xs, med, _, _ = _galaxy_median(df[df.feature == NAMES[0]], x, "pixel_frac_prior_footprint")
    ax.plot(xs, med, "k--", lw=1.2, label="pixel score")


def fig_main(masking, snr):
    base = masking[masking.mask_kind == "none"]
    panels = [("half-plane", "half-plane"), ("core", "core rings"), ("outer", "outer rings"), ("random-pixels", "random pixels")]
    panels = [(k, t) for k, t in panels if (masking.mask_kind == k).any()]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(TEXT_WIDTH, 1.8), sharey=True)
    for ax, (kind, title) in zip(axes, panels):
        _median_lines(ax, pd.concat([base, masking[masking.mask_kind == kind]]), "mask_fraction")
        ax.set(title=title, xlabel="masked fraction")
        ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", ".25", ".5", ".75"])
    if snr is not None:
        _median_lines(axes[-1], snr, "noise_scale")
        axes[-1].set(title="noise", xlabel="noise scale", xscale="log")
    axes[0].set(ylabel="$R$ (prior curv. share)", ylim=(-0.03, 1.05))
    for ax in axes:
        _grid(ax)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, handlelength=1.5, columnspacing=1.0)
    _save(fig, "fig_main")


def _bars_with_ci(ax, x, stats, offset, width, color, label):
    vals = np.array([stats[f][0] for f in NAMES])
    err = np.array([[stats[f][0] - stats[f][1], stats[f][2] - stats[f][0]] for f in NAMES]).T
    ax.bar(x + offset, vals, width, color=color, label=label, yerr=np.clip(err, 0, None), error_kw={"lw": 0.8, "capsize": 1.5})


def fig_errors(sweeps):
    x = np.arange(len(NAMES))
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.0))
    masking = sweeps["masking"]
    full = masking[masking.mask_kind == "none"]
    if "R_diag" in full:
        _bars_with_ci(axes[0], x, _feature_stat_ci(full, lambda d: d["R"].median()), -0.2, 0.4, "C0", "$R$ (full curvature)")
        _bars_with_ci(axes[0], x, _feature_stat_ci(full, lambda d: d["R_diag"].median()), 0.2, 0.4, "0.6", r"$R_{\rm diag}$ (diagonals only)")
    axes[0].set(xticks=x, ylabel="prior share, full data", ylim=(0, 1.05))
    axes[0].legend(frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(0, 1.04), borderaxespad=0)

    rows = pd.concat(list(sweeps.values()))
    _bars_with_ci(axes[1], x, _feature_stat_ci(rows[rows.low_support], lambda d: d["abs_error"].median()), -0.2, 0.4, "C3", "weakly constrained by data")
    _bars_with_ci(axes[1], x, _feature_stat_ci(rows[~rows.low_support], lambda d: d["abs_error"].median()), 0.2, 0.4, "C2", "supported by data")
    axes[1].set(xticks=x, ylabel=r"median |error| [$\sigma_{\rm pop}$]", yscale="log")
    axes[1].legend(frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(0, 1.04), borderaxespad=0)

    ok = rows[rows["neg_mode_share"] <= 0.1]
    for k, (col, color, label) in enumerate((("sigma_local", "0.6", r"$\sigma_{\rm local}$ (directional)"), ("sigma_post", "C0", r"$\sigma_{\rm post}$ (Laplace)"))):
        _bars_with_ci(axes[2], x, _feature_stat_ci(ok, lambda d, c=col: np.mean(d["abs_error"] <= d[f"{c}_rel"])), (k - 0.5) * 0.4, 0.4, color, label)
    axes[2].axhline(0.68, color="k", ls="--", lw=0.8)
    axes[2].set(xticks=x, ylabel=r"fraction within $\pm1\sigma$", ylim=(0, 1))
    axes[2].legend(frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(0, 1.04), borderaxespad=0)
    for ax in axes:
        ax.set_xticklabels([LABELS[f] for f in NAMES], rotation=45, ha="right")
        _grid(ax, "y")
    fig.tight_layout(w_pad=0.8)
    _save(fig, "fig_errors")


def choose_example(masking):
    """Source (noise draw 0) whose full-data and masked reconstructions differ least in
    image error while the shape features lose the most data support."""
    shape = ["size", "e1", "e2", "concentration"]
    m = masking[(masking.seed == 0) & masking.feature.isin(shape)]
    full = m[m.mask_kind == "none"].groupby("source")[["log10_I_D", "relerr_image"]].mean()
    masked = m[(m.mask_kind == EXAMPLE_KIND) & (m.mask_fraction == EXAMPLE_FRACTION)].groupby("source")[["log10_I_D", "relerr_image"]].mean()
    score = (full["log10_I_D"] - masked["log10_I_D"]) - 20 * (full["relerr_image"] - masked["relerr_image"]).abs()
    return score.idxmax()


def fig_example(masking, pop_std):
    source_id = choose_example(masking)
    src = next(s for s in data.load_sources() if s.source_id == source_id)
    cfg = Config.load()
    prior = cfg.prior()
    cases = [("full data", None), ("outskirts masked", forward.flux_mask(src.truth, EXAMPLE_FRACTION, EXAMPLE_KIND))]
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.0))
    images = fig.add_gridspec(2, 3, left=0.03, right=0.52, bottom=0.04, top=0.9, wspace=0.06, hspace=0.1)
    bar_ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.67, right=0.99, bottom=0.2, top=0.9)[0])
    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad("0.85")
    x = np.arange(len(NAMES))
    for i, (label, mask) in enumerate(cases):
        case = run_case(src, cfg, mask=mask, seed=0)
        pix = attribution.pixel_scores(case.obs, prior, case.x_hat, exact=True)
        attrs = attribution.feature_attribution(case.obs, prior, case.x_hat, population_std=pop_std)
        y = np.where(np.asarray(case.obs.weights) > 0, np.asarray(case.y), np.nan)
        delta = pix["delta"]
        lim = np.nanpercentile(np.abs(delta), 99)
        panels = [(y, "observation", {"cmap": viridis}), (np.asarray(case.x_hat), "reconstruction", {"cmap": "viridis"}), (delta, r"pixel $\Delta F$", {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim})]
        for j, (img, title, kw) in enumerate(panels):
            ax = fig.add_subplot(images[i, j])
            im = ax.imshow(img, origin="lower", **kw)
            ax.contour(src.truth > FOOTPRINT, levels=[0.5], colors="w" if j < 2 else "k", linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(label)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        bar_ax.bar(x + (i - 0.5) * 0.4, [a.R for a in attrs], 0.4, color=("C0", "C1")[i], label=label)
        print(f"example {label}: S/N {src.snr:.0f}, R", {a.name: round(a.R, 2) for a in attrs})
    bar_ax.set(xticks=x, ylabel="$R$ (prior share)", ylim=(0, 1.3), yticks=np.linspace(0, 1, 6), title="feature level")
    bar_ax.set_xticklabels([LABELS[f] for f in NAMES])
    bar_ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.2, columnspacing=1.0, borderaxespad=0.3)
    _grid(bar_ax, "y")
    print("example source:", source_id)
    _save(fig, "fig_example")


def fig_tsens():
    csv, js = results_dir() / "t_sensitivity.csv", results_dir() / "t_sensitivity.json"
    if not (csv.exists() and js.exists()):
        print("no t-sensitivity results, skipping fig_tsens")
        return
    df = pd.read_csv(csv)
    full = df[(df["mode"] == "fixed") & (df["condition"] == "none")]
    agree = json.loads(js.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.0))  # talk figure; not in the write-up
    for feat in NAMES:
        sub = full[full.feature == feat].groupby("t")
        axes[0].plot(sub["R"].median(), marker="o", ms=2.5, lw=1.2, color=COLORS[feat], label=LABELS[feat])
        axes[1].plot(sub["info_gain_pop"].median(), marker="o", ms=2.5, lw=1.2, color=COLORS[feat])
    for col, label, color in (("R", "$R$", "C0"), ("info_gain_pop", "information gain", "C3")):
        for mode, ls in (("fixed", "-"), ("refit", "--")):
            pts = sorted((float(k.split("/")[1]), v[col]) for k, v in agree["rank_agreement"].items() if k.startswith(mode + "/"))
            axes[2].plot(*zip(*pts), ls=ls, marker="o", ms=2.5, lw=1.2, color=color, label=f"{label}, {mode}")
    t_ref = agree["t_ref"]
    for ax in axes:
        _grid(ax)
        ax.axvline(t_ref, color="0.7", lw=0.8, zorder=0)
        ax.set(xscale="log", xlabel="diffusion time $t$")
    axes[0].set(ylabel="$R$ (prior share)", ylim=(-0.03, 1.05), title="prior share, full data")
    axes[1].set(ylabel="info. gain [nats]", title="information gain, full data")
    axes[2].set(ylabel="rank agreement", ylim=(0.3, 1.02), title=f"feature ranking vs $t={t_ref:g}$")
    axes[2].legend(frameon=False, loc="lower right", handlelength=1.6, labelspacing=0.15, fontsize=6)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, handlelength=1.5, columnspacing=1.0)
    _save(fig, "fig_tsens")


def main():
    parse_args()
    pop_std = population_std()
    masking, snr, blur = (_load(name, pop_std) for name in ("masking", "snr", "blur"))
    if masking is None:
        raise SystemExit(f"no phase_d_masking.csv in {results_dir()}")
    fig_main(masking, snr)
    fig_errors({k: v for k, v in (("masking", masking), ("snr", snr), ("blur", blur)) if v is not None})
    fig_example(masking, pop_std)
    fig_tsens()


if __name__ == "__main__":
    main()
