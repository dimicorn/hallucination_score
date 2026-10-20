"""Phase C/D controlled experiments on the eval split.

    uv run python experiments/phase_d.py masking   # Experiment 1 (priority 1)
    uv run python experiments/phase_d.py snr       # Experiment 2
    uv run python experiments/phase_d.py blur      # Experiment 3 (Fourier low-pass)
    uv run python experiments/phase_d.py prior     # C2: prior-strength intervention

Several names run one after another (e.g. `phase_d.py masking snr`). Add --quick to
any of them for a smoke test (1 source, 1 seed, short fits).

Each writes results/phase_d_<name>.csv with one row per (case, feature).
"""

import argparse

import numpy as np
import pandas as pd

from provenance import data, forward
from provenance.evaluate import evaluate, population_std
from provenance.pipeline import Config, is_quick, parse_args, results_dir

N_SOURCES = 12
SHAPE = (32, 32)
MASK_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
NOISE_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
BLUR_CUTOFFS = (None, 0.2, 0.15, 0.1, 0.07)
SEEDS = (0, 1, 2)
BLUR_SEEDS = (0, 1)
PRIOR_WEIGHTS = (0.5, 1.0, 2.0)  # the intervention analysis needs all three


def use_quick_settings():
    global N_SOURCES, MASK_FRACTIONS, NOISE_SCALES, BLUR_CUTOFFS, SEEDS, BLUR_SEEDS
    N_SOURCES, MASK_FRACTIONS, NOISE_SCALES, BLUR_CUTOFFS = 1, (0.0, 0.5), (1.0, 4.0), (None, 0.1)
    SEEDS = BLUR_SEEDS = (0,)


def eval_sources():
    _, ev = data.split(data.load_sources())
    return ev[:: max(1, len(ev) // N_SOURCES)][:N_SOURCES]


def masks_for(src):
    """(kind, nominal fraction, mask) for every masking condition of one source."""
    yield "none", 0.0, None
    for f in MASK_FRACTIONS[1:]:
        yield "half-plane", f, forward.half_plane_mask(SHAPE, f)
        yield "core", f, forward.flux_mask(src.truth, f, "core")
        yield "outer", f, forward.flux_mask(src.truth, f, "outer")


def run_masking(cfg, pop):
    rows = []
    for src in eval_sources():
        for kind, frac, mask in masks_for(src):
            flux_removed = 0.0 if mask is None else float((src.truth * (1 - mask)).sum() / src.truth.sum())
            for seed in SEEDS:
                cond = {"experiment": "masking", "mask_kind": kind, "mask_fraction": frac, "flux_removed": flux_removed, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, mask=mask, seed=seed)[0]
        print(src.source_id, flush=True)
    return rows


def run_snr(cfg, pop):
    rows = []
    for src in eval_sources():
        for ns in NOISE_SCALES:
            for seed in SEEDS:
                cond = {"experiment": "snr", "noise_scale": ns, "snr_effective": src.snr / ns, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, noise_scale=ns, seed=seed)[0]
        print(src.source_id, flush=True)
    return rows


def run_blur(cfg, pop):
    rows = []
    for src in eval_sources():
        for cut in BLUR_CUTOFFS:
            for seed in BLUR_SEEDS:
                cond = {"experiment": "blur", "blur_cutoff": np.inf if cut is None else cut, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, blur_cutoff=cut, seed=seed)[0]
        print(src.source_id, flush=True)
    return rows


def run_prior(cfg, pop):
    """Same observation, prior weight lambda varied; degradations from Exp. 1 and 2."""
    rows = []
    for src in eval_sources():
        conditions = [("half-plane", f, forward.half_plane_mask(SHAPE, f), 1.0) for f in MASK_FRACTIONS]
        conditions += [("noise", ns, None, ns) for ns in NOISE_SCALES]
        for kind, level, mask, ns in conditions:
            for w in PRIOR_WEIGHTS:
                cond = {"experiment": "prior", "degradation": kind, "level": level, "prior_weight": w, "seed": 0}
                rows += evaluate(src, cfg, pop, cond, pixel=False, prior=cfg.prior(weight=w), mask=mask, noise_scale=ns, seed=0)[0]
        print(src.source_id, flush=True)
    return rows


RUNNERS = {"masking": run_masking, "snr": run_snr, "blur": run_blur, "prior": run_prior}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=RUNNERS, metavar="{" + ",".join(RUNNERS) + "}")
    args = parse_args(ap)
    if is_quick():
        use_quick_settings()
    cfg, pop = Config.load(), population_std()
    for name in args.experiments:
        rows = RUNNERS[name](cfg, pop)
        path = results_dir() / f"phase_d_{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print("wrote", path)


if __name__ == "__main__":
    main()
