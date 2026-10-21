"""Publication figures for the write-up, from the Phase D results.

    uv run python experiments/paper_figures.py [--quick]

Writes results/paper/fig_{main,errors,example}.{pdf,png}:

main     R per feature against degradation severity (half-plane, brightest-flux and
         faintest-flux masks, noise), with the pixel score as a dashed line
errors   how well R and the pixel score predict each feature's error (Spearman, masking
         and noise), and median errors of features flagged weakly constrained or not
example  one source observed with full data and with its faint outskirts masked: the
         reconstructions look alike and the pixel map is prior-dominated in both, but
         the feature-level data support differs. The pair is chosen from the masking
         results and the two cases are reconstructed again (they are not stored).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import _load, _rho
from provenance import attribution, data, forward
from provenance.evaluate import FOOTPRINT, population_std
from provenance.features import NAMES
from provenance.pipeline import Config, is_quick, parse_args, results_dir, run_case

TEXT_WIDTH = 6.6  # inches, A4 with 2.1 cm margins
LABELS = {"flux": "flux", "size": "size", "e1": "$e_1$", "e2": "$e_2$", "concentration": "conc.", "asymmetry": "asym.", "hf_power": "HF"}
COLORS = dict(zip(NAMES, plt.get_cmap("tab10").colors))
EXAMPLE_KIND, EXAMPLE_FRACTION = "outer", 0.5

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _save(fig, name):
    out = results_dir() / "paper"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def _median_lines(ax, df, x, y="R"):
    for feat in NAMES:
        med = df[df.feature == feat].groupby(x)[y].median()
        ax.plot(med.index, med.values, marker="o", ms=2.5, lw=1.2, color=COLORS[feat], label=LABELS[feat])
    pix = df[df.feature == NAMES[0]].groupby(x)["pixel_frac_prior_footprint"].median()
    ax.plot(pix.index, pix.values, "k--", lw=1.2, label="pixel score")


def fig_main(masking, snr):
    base = masking[masking.mask_kind == "none"]
    panels = [("half-plane", "half-plane mask"), ("core", "brightest-flux mask"), ("outer", "faintest-flux mask")]
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 1.75), sharey=True)
    for ax, (kind, title) in zip(axes, panels):
        _median_lines(ax, pd.concat([base, masking[masking.mask_kind == kind]]), "mask_fraction")
        ax.set(title=title, xlabel="masked fraction")
    if snr is not None:
        _median_lines(axes[3], snr, "noise_scale")
        axes[3].set(title="noise", xlabel="noise scale", xscale="log")
    axes[0].set(ylabel="$R$ (prior share)", ylim=(-0.03, 1.05))
    fig.tight_layout(w_pad=0.4, rect=(0, 0.1, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, handlelength=1.5, columnspacing=1.0)
    _save(fig, "fig_main")


def fig_errors(sweeps):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0))
    x = np.arange(len(NAMES))
    bars = []
    for name in ("masking", "snr"):
        df = sweeps.get(name)
        if df is None:
            continue
        rho_r = [_rho(df[df.feature == f]["R"].values, df[df.feature == f]["abs_error"].values) for f in NAMES]
        rho_p = [_rho(df[df.feature == f]["pixel_frac_prior_footprint"].values, df[df.feature == f]["abs_error"].values) for f in NAMES]
        bars.append((name, rho_r, rho_p))
    width = 0.8 / max(1, 2 * len(bars))
    for i, (name, rho_r, rho_p) in enumerate(bars):
        hatch = None if name == "masking" else "///"
        axes[0].bar(x + (2 * i - len(bars) + 0.5) * width, rho_r, width, color="C0", hatch=hatch, edgecolor="white", label=f"$R$, {name}")
        axes[0].bar(x + (2 * i - len(bars) + 1.5) * width, rho_p, width, color="0.6", hatch=hatch, edgecolor="white", label=f"pixel, {name}")
    axes[0].set(xticks=x, ylabel="Spearman with |error|")
    axes[0].set_xticklabels([LABELS[f] for f in NAMES])
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.08), handlelength=1.2, columnspacing=0.6, borderaxespad=0)
    axes[0].set_ylim(top=1.0)

    rows = pd.concat(list(sweeps.values()))
    flagged = rows[rows.low_support].groupby("feature")["abs_error"].median().reindex(NAMES)
    unflagged = rows[~rows.low_support].groupby("feature")["abs_error"].median().reindex(NAMES)
    axes[1].bar(x - 0.2, flagged.values, 0.4, color="C3", label="weakly constrained")
    axes[1].bar(x + 0.2, unflagged.values, 0.4, color="C2", label="supported by data")
    axes[1].set(xticks=x, ylabel=r"median |error| [$\sigma_{\rm pop}$]", yscale="log")
    axes[1].set_xticklabels([LABELS[f] for f in NAMES])
    axes[1].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.08), borderaxespad=0)
    fig.tight_layout(w_pad=1.0)
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
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.2))
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
    bar_ax.set(xticks=x, ylabel="$R$ (prior share)", ylim=(0, 1.05), title="feature level")
    bar_ax.set_xticklabels([LABELS[f] for f in NAMES])
    bar_ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    print("example source:", source_id)
    _save(fig, "fig_example")


def main():
    parse_args()
    pop_std = population_std()
    masking, snr, blur = (_load(name, pop_std) for name in ("masking", "snr", "blur"))
    if masking is None:
        raise SystemExit(f"no phase_d_masking.csv in {results_dir()}")
    fig_main(masking, snr)
    fig_errors({k: v for k, v in (("masking", masking), ("snr", snr), ("blur", blur)) if v is not None})
    fig_example(masking, pop_std)


if __name__ == "__main__":
    main()
