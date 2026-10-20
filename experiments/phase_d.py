"""Phase C/D controlled experiments on the eval split.

    uv run python experiments/phase_d.py masking   # Experiment 1 (priority 1)
    uv run python experiments/phase_d.py snr       # Experiment 2
    uv run python experiments/phase_d.py blur      # Experiment 3 (Fourier low-pass)
    uv run python experiments/phase_d.py prior     # C2: prior-strength intervention

Each writes results/phase_d_<name>.csv with one row per (case, feature).
"""

import argparse

import numpy as np
import pandas as pd

from provenance import data, forward
from provenance.evaluate import evaluate, population_std
from provenance.pipeline import RESULTS, Config

N_SOURCES = 12
SHAPE = (32, 32)
MASK_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
NOISE_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
BLUR_CUTOFFS = (None, 0.2, 0.15, 0.1, 0.07)
PRIOR_WEIGHTS = (0.5, 1.0, 2.0)


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


def run_masking(cfg, pop, seeds=(0, 1, 2)):
    rows = []
    for src in eval_sources():
        for kind, frac, mask in masks_for(src):
            flux_removed = 0.0 if mask is None else float((src.truth * (1 - mask)).sum() / src.truth.sum())
            for seed in seeds:
                cond = {"experiment": "masking", "mask_kind": kind, "mask_fraction": frac, "flux_removed": flux_removed, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, mask=mask, seed=seed)[0]
        print(src.source_id, flush=True)
    return rows


def run_snr(cfg, pop, seeds=(0, 1, 2)):
    rows = []
    for src in eval_sources():
        for ns in NOISE_SCALES:
            for seed in seeds:
                cond = {"experiment": "snr", "noise_scale": ns, "snr_effective": src.snr / ns, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, noise_scale=ns, seed=seed)[0]
        print(src.source_id, flush=True)
    return rows


def run_blur(cfg, pop, seeds=(0, 1)):
    rows = []
    for src in eval_sources():
        for cut in BLUR_CUTOFFS:
            for seed in seeds:
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
    ap.add_argument("experiment", choices=RUNNERS)
    args = ap.parse_args()
    rows = RUNNERS[args.experiment](Config(), population_std())
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"phase_d_{args.experiment}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
