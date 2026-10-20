"""Phase B: feature distributions on the clean reference (truth) images.

Saves the population mean/std of every feature (used to put feature-level diagnostics
on a common scale) and checks each autodiff gradient against central finite
differences on a few real galaxies.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance import data, features
from provenance.pipeline import RESULTS, save_json


def gradient_check(x, f, n_dirs=5, eps=1e-3, seed=0):
    g = jax.grad(f)(x)
    errs = []
    for k in jax.random.split(jax.random.PRNGKey(seed), n_dirs):
        v = jax.random.normal(k, x.shape)
        v = v / jnp.linalg.norm(v)
        fd = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
        errs.append(abs(float(jnp.vdot(g, v)) - float(fd)) / (abs(float(fd)) + 1e-8))
    return float(np.median(errs))


def main():
    sources = data.load_sources()
    table = pd.DataFrame([{"source": s.source_id, "snr": s.snr, **features.measure(jnp.asarray(s.truth, jnp.float32))} for s in sources])
    stats = {n: {"mean": float(table[n].mean()), "std": float(table[n].std()), "q16": float(table[n].quantile(0.16)), "q84": float(table[n].quantile(0.84))} for n in features.NAMES}

    checks = {}
    for name, f in features.FEATURES.items():
        # float64 would be cleaner; float32 FD with eps=1e-3 is accurate to ~1e-2 relative
        checks[name] = float(np.median([gradient_check(jnp.asarray(s.truth, jnp.float32), f, seed=i) for i, s in enumerate(sources[:10])]))

    corr = table[list(features.NAMES)].corr(method="spearman")
    save_json({"n_sources": len(sources), "population": stats, "gradient_check_median_relerr": checks, "spearman_corr": corr.round(3).to_dict()}, "feature_population.json")
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "feature_reference_table.csv", index=False)

    fig, axes = plt.subplots(1, len(features.NAMES) + 1, figsize=(22, 3))
    for ax, n in zip(axes, features.NAMES):
        ax.hist(table[n], bins=30, color="0.4")
        ax.set_title(n)
    im = axes[-1].imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[-1].set_xticks(range(len(features.NAMES)), features.NAMES, rotation=90, fontsize=7)
    axes[-1].set_yticks(range(len(features.NAMES)), features.NAMES, fontsize=7)
    axes[-1].set_title("Spearman")
    plt.colorbar(im, ax=axes[-1])
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_phase_b_features.png", dpi=130)
    print(pd.DataFrame(stats).T.round(4))
    print("gradient checks (median rel err):", {k: round(v, 4) for k, v in checks.items()})
    print(corr.round(2))


if __name__ == "__main__":
    main()
