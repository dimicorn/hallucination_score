"""Phase A: reproduce the pixel-level hallucination score and its sanity checks.

1. Half-mask control over several eval sources and noise draws: Delta F should be
   positive (prior-dominated) on the masked half of the source and negative on the
   observed half; with no mask the two halves should look alike.
2. Hutchinson stability: diag(F_P) from k Rademacher probes vs the exact diagonal
   (affordable at 32x32), over several probe seeds.
3. Properties of the learned prior curvature: asymmetry of the score Jacobian and the
   fraction of negative eigenvalues of its symmetric part.
4. Sensitivity to the prior's evaluation time t: with each reconstruction held fixed,
   re-evaluate diag(F_P) at several t. The Tweedie cap 1/s_t^2 implies Delta F's sign
   can be moved by t alone; this quantifies by how much.
"""

import time

import jax
import matplotlib.pyplot as plt
import numpy as np

from provenance import attribution, curvature, data, forward
from provenance.pipeline import RESULTS, Config, run_case, save_json
from provenance.prior import ScorePrior

N_SOURCES = 12
SEEDS = (0, 1, 2)
PROBES = (16, 64, 256, 1024, 4000)
PROBE_SEEDS = range(5)
T_EVAL = (0.01, 0.02, 0.05, 0.1, 0.2)
FOOTPRINT = 0.02  # fraction of peak defining the source footprint
SHAPE = (32, 32)
HALF = forward.half_plane_mask(SHAPE, 0.5)  # rows 0..15 unobserved


def eval_sources():
    _, ev = data.split(data.load_sources())
    return ev[:: max(1, len(ev) // N_SOURCES)][:N_SOURCES]


def region_stats(delta, truth):
    foot = truth > FOOTPRINT
    out = {}
    for name, region in (("masked_half", HALF == 0), ("observed_half", HALF == 1)):
        sel = foot & region
        out[f"median_dF_{name}"] = float(np.median(delta[sel])) if sel.any() else np.nan
        out[f"frac_prior_{name}"] = float(np.mean(delta[sel] > 0)) if sel.any() else np.nan
    return out


def control_experiment(cfg, sources):
    rows, examples = [], {}
    for src in sources:
        for seed in SEEDS:
            for cond, mask in (("full", None), ("half", HALF)):
                case = run_case(src, cfg, mask=mask, seed=seed)
                pix = attribution.pixel_scores(case.obs, cfg.prior(), case.x_hat, exact=True)
                row = {"source": src.source_id, "snr": src.snr, "seed": seed, "condition": cond}
                row["relerr"] = float(np.linalg.norm(case.x_hat - src.truth) / np.linalg.norm(src.truth))
                row.update(region_stats(pix["delta"], src.truth))
                rows.append(row)
                if seed == 0 and src is sources[0]:
                    examples[cond] = (case, pix)
        print(src.source_id, "done", flush=True)
    return rows, examples


def hutchinson_experiment(cfg, sources):
    out = []
    for src in sources[:3]:
        case = run_case(src, cfg, mask=HALF)
        prior = cfg.prior()
        mv = lambda v: prior.curvature_matvec(case.x_hat, v)
        t0 = time.time()
        dense = curvature.exact_matrix(mv, SHAPE)
        t_exact = time.time() - t0
        exact_diag = np.diag(dense).reshape(SHAPE)
        exact_delta = exact_diag - np.asarray(case.obs.curvature_diag())
        sym = 0.5 * (dense + dense.T)
        eig = np.linalg.eigvalsh(sym)
        props = {
            "source": src.source_id,
            "asymmetry_rel_fro": float(np.linalg.norm(dense - dense.T) / np.linalg.norm(dense)),
            "frac_negative_eigs": float(np.mean(eig < 0)),
            "min_eig": float(eig.min()),
            "max_eig": float(eig.max()),
            "frac_negative_diag": float(np.mean(exact_diag < 0)),
            "exact_seconds": t_exact,
            "convergence": [],
        }
        foot = src.truth > FOOTPRINT
        for ps in PROBE_SEEDS:
            t0 = time.time()
            _, snaps = curvature.hutchinson_diag(mv, SHAPE, max(PROBES), jax.random.PRNGKey(100 + ps), checkpoints=PROBES)
            t_h = time.time() - t0
            for k, est in snaps.items():
                d = est - np.asarray(case.obs.curvature_diag())
                props["convergence"].append({
                    "probe_seed": ps,
                    "n_probes": k,
                    "relerr_diag_P": float(np.linalg.norm(est - exact_diag) / np.linalg.norm(exact_diag)),
                    "sign_agreement_dF_footprint": float(np.mean(np.sign(d[foot]) == np.sign(exact_delta[foot]))),
                    "seconds_4000": t_h,
                })
        out.append(props)
        print("hutchinson", src.source_id, "done", flush=True)
    return out


def t_sensitivity(cfg, sources):
    rows = []
    for src in sources:
        case = run_case(src, cfg, mask=HALF)
        for t in T_EVAL:
            prior = ScorePrior(t=t, weight=cfg.prior_weight)
            pix = attribution.pixel_scores(case.obs, prior, case.x_hat, exact=True)
            foot = src.truth > FOOTPRINT
            row = {"source": src.source_id, "t_eval": t}
            row.update(region_stats(pix["delta"], src.truth))
            row["median_kappa_footprint"] = float(np.median(pix["kappa"][foot]))
            row["median_kappa_background"] = float(np.median(pix["kappa"][~foot]))
            rows.append(row)
        print("t-sensitivity", src.source_id, "done", flush=True)
    return rows


def summarise(rows):
    summary = {}
    for cond in ("full", "half"):
        sel = [r for r in rows if r["condition"] == cond]
        s = {}
        for key in ("median_dF_masked_half", "median_dF_observed_half", "frac_prior_masked_half", "frac_prior_observed_half", "relerr"):
            vals = np.array([r[key] for r in sel], dtype=float)
            s[key] = {"median": float(np.nanmedian(vals)), "q16": float(np.nanpercentile(vals, 16)), "q84": float(np.nanpercentile(vals, 84))}
        if cond == "half":
            s["n_cases_masked_prior_dominated"] = int(sum(r["median_dF_masked_half"] > 0 for r in sel))
            s["n_cases_observed_data_dominated"] = int(sum(r["median_dF_observed_half"] < 0 for r in sel))
            s["n_cases"] = len(sel)
        summary[cond] = s
    return summary


def plot_baseline(examples, path):
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for row, cond in enumerate(("full", "half")):
        case, pix = examples[cond]
        panels = [
            (np.asarray(case.y), "observation", "viridis", None),
            (np.asarray(case.x_hat), "reconstruction", "viridis", None),
            (pix["diag_D"], r"diag $F_D$", "magma", None),
            (pix["diag_P"], r"diag $F_P$", "magma", None),
            (pix["delta"], r"$\Delta F$ = diag $F_P$ $-$ diag $F_D$", "RdBu_r", "sym"),
        ]
        for ax, (im, title, cmap, norm) in zip(axes[row], panels):
            kw = {}
            if norm == "sym":
                m = np.abs(im).max()
                kw = {"vmin": -m, "vmax": m}
            h = ax.imshow(im, origin="lower", cmap=cmap, **kw)
            ax.contour(case.truth, levels=[FOOTPRINT], colors="w" if cmap != "RdBu_r" else "k", linewidths=0.6)
            plt.colorbar(h, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
            ax.set_title(f"{title}\n({cond} data)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Pixel-level baseline (contour: true source footprint; half: bottom rows unobserved)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_convergence(hutch, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for props in hutch:
        conv = props["convergence"]
        for key, ax in (("relerr_diag_P", axes[0]), ("sign_agreement_dF_footprint", axes[1])):
            ks = sorted({c["n_probes"] for c in conv})
            vals = np.array([[c[key] for c in conv if c["n_probes"] == k] for k in ks])
            ax.errorbar(ks, vals.mean(1), yerr=vals.std(1), marker="o", capsize=3, label=props["source"])
    ks = np.array(PROBES)
    first = hutch[0]["convergence"]
    e0 = np.mean([c["relerr_diag_P"] for c in first if c["n_probes"] == ks[0]])
    axes[0].plot(ks, e0 * np.sqrt(ks[0] / ks), "k--", lw=0.8, label=r"$\propto k^{-1/2}$")
    axes[0].set(xscale="log", yscale="log", xlabel="probes k", ylabel=r"$\|\hat d - d\| / \|d\|$", title="Hutchinson diag($F_P$) error vs exact")
    axes[1].set(xscale="log", xlabel="probes k", ylabel="fraction", title=r"sign($\Delta F$) agreement with exact (footprint)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    cfg = Config.load()
    sources = eval_sources()
    rows, examples = control_experiment(cfg, sources)
    hutch = hutchinson_experiment(cfg, sources)
    tsens = t_sensitivity(cfg, sources)
    summary = summarise(rows)
    summary["t_sensitivity"] = {
        str(t): {k: float(np.nanmedian([r[k] for r in tsens if r["t_eval"] == t])) for k in tsens[0] if k not in ("source", "t_eval")}
        for t in T_EVAL
    }
    RESULTS.mkdir(exist_ok=True)
    plot_baseline(examples, RESULTS / "fig_phase_a_baseline.png")
    plot_convergence(hutch, RESULTS / "fig_phase_a_hutchinson.png")
    save_json({"config": cfg, "summary": summary, "rows": rows, "hutchinson": hutch, "t_sensitivity": tsens}, "phase_a.json")
    print(summary)
    for h in hutch:
        print({k: v for k, v in h.items() if k != "convergence"})


if __name__ == "__main__":
    main()
