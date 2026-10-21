"""Phase C/D controlled experiments on the eval split.

    uv run python experiments/phase_d.py masking   # Experiment 1 (priority 1)
    uv run python experiments/phase_d.py snr       # Experiment 2
    uv run python experiments/phase_d.py blur      # Experiment 3 (Fourier low-pass)
    uv run python experiments/phase_d.py prior     # C2: prior-strength intervention

Several names run one after another (e.g. `phase_d.py masking snr`). Add --quick to
any of them for a smoke test (1 source, 1 seed, short fits).

--n-sources N takes N eval sources spread over the eval split (default 12). --chunk k/K
runs every K-th of them starting at k and writes phase_d_<name>.part<k>of<K>.csv, so
the sources can be shared between processes or GPUs; --merge then combines the K parts
into phase_d_<name>.csv once all of them exist:

    for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k uv run python experiments/phase_d.py masking --n-sources 100 --chunk $k/4 & done; wait
    uv run python experiments/phase_d.py masking --merge

Each writes results/phase_d_<name>.csv with one row per (case, feature). Masking, SNR and
low-pass cases also get I_D_fixed / cos_ref against the feature gradients of the
undegraded reconstruction of the same source and noise draw.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # default threading makes 1024^2 eigh ~100x slower

import argparse
import zlib

import numpy as np
import pandas as pd

from provenance import data, features, forward
from provenance.evaluate import evaluate, population_std
from provenance.pipeline import Config, is_quick, parse_args, results_dir, run_case

N_SOURCES = 12
CHUNK = (0, 1)  # (k, K): this process runs eval sources k, k + K, k + 2K, ...
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
    """N_SOURCES eval sources spread evenly over the eval split, then this process's chunk."""
    _, ev = data.split(data.load_sources())
    chosen = ev[:: max(1, len(ev) // N_SOURCES)][:N_SOURCES]
    k, n_chunks = CHUNK
    return chosen[k::n_chunks]


_REFERENCE = {}


def reference_grads(src, cfg, seed):
    """Feature gradients of the undegraded reconstruction (same source, noise draw)."""
    key = (src.source_id, seed)
    if key not in _REFERENCE:
        _REFERENCE[key] = features.gradients(run_case(src, cfg, seed=seed).x_hat)
    return _REFERENCE[key]


def masks_for(src, seed):
    """(kind, nominal fraction, mask) for every masking condition of one source. The
    random-pixel and random-blob masks ignore the source (control for the reference-based
    core/outer rings) and are drawn per source and noise draw."""
    rng = np.random.default_rng([zlib.crc32(src.source_id.encode()), seed])
    yield "none", 0.0, None
    for f in MASK_FRACTIONS[1:]:
        yield "half-plane", f, forward.half_plane_mask(SHAPE, f)
        yield "core", f, forward.flux_mask(src.truth, f, "core")
        yield "outer", f, forward.flux_mask(src.truth, f, "outer")
        yield "random-pixels", f, forward.random_mask(SHAPE, f, "pixels", rng)
        yield "random-blob", f, forward.random_mask(SHAPE, f, "blob", rng)


def run_masking(cfg, pop):
    rows = []
    for src in eval_sources():
        for seed in SEEDS:
            for kind, frac, mask in masks_for(src, seed):
                flux_removed = 0.0 if mask is None else float((src.truth * (1 - mask)).sum() / src.truth.sum())
                cond = {"experiment": "masking", "mask_kind": kind, "mask_fraction": frac, "flux_removed": flux_removed, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, mask=mask, seed=seed, ref_grads=reference_grads(src, cfg, seed))[0]
        print(src.source_id, flush=True)
    return rows


def run_snr(cfg, pop):
    rows = []
    for src in eval_sources():
        for ns in NOISE_SCALES:
            for seed in SEEDS:
                cond = {"experiment": "snr", "noise_scale": ns, "snr_effective": src.snr / ns, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, noise_scale=ns, seed=seed, ref_grads=reference_grads(src, cfg, seed))[0]
        print(src.source_id, flush=True)
    return rows


def run_blur(cfg, pop):
    rows = []
    for src in eval_sources():
        for cut in BLUR_CUTOFFS:
            for seed in BLUR_SEEDS:
                cond = {"experiment": "blur", "blur_cutoff": np.inf if cut is None else cut, "seed": seed}
                rows += evaluate(src, cfg, pop, cond, blur_cutoff=cut, seed=seed, ref_grads=reference_grads(src, cfg, seed))[0]
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


def merge(name):
    """Combine phase_d_<name>.part<k>of<K>.csv (all K present) into phase_d_<name>.csv."""
    parts = sorted(results_dir().glob(f"phase_d_{name}.part*of*.csv"))
    counts = {int(p.stem.rsplit("of", 1)[1]) for p in parts}
    if len(counts) != 1:
        raise SystemExit(f"{name}: expected the parts of one split, found {[p.name for p in parts]}")
    n_chunks = counts.pop()
    missing = [k for k in range(n_chunks) if not (results_dir() / f"phase_d_{name}.part{k}of{n_chunks}.csv").exists()]
    if missing:
        raise SystemExit(f"{name}: parts {missing} of {n_chunks} are missing")
    path = results_dir() / f"phase_d_{name}.csv"
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(path, index=False)
    print("wrote", path, f"from {n_chunks} parts")


def main():
    global N_SOURCES, CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=RUNNERS, metavar="{" + ",".join(RUNNERS) + "}")
    ap.add_argument("--n-sources", type=int, default=None, help=f"eval sources, spread over the eval split (default {N_SOURCES}; 1 with --quick)")
    ap.add_argument("--chunk", default="0/1", help="k/K: run every K-th source starting at k")
    ap.add_argument("--merge", action="store_true", help="combine the part files of a chunked run")
    args = parse_args(ap)
    if args.merge:
        for name in args.experiments:
            merge(name)
        return
    k, n_chunks = (int(v) for v in args.chunk.split("/"))
    if not 0 <= k < n_chunks:
        raise SystemExit(f"--chunk {args.chunk}: need 0 <= k < K")
    CHUNK = (k, n_chunks)
    if is_quick():
        use_quick_settings()
    if args.n_sources is not None:
        N_SOURCES = args.n_sources
    cfg, pop = Config.load(), population_std()
    for name in args.experiments:
        rows = RUNNERS[name](cfg, pop)
        suffix = "" if n_chunks == 1 else f".part{k}of{n_chunks}"
        path = results_dir() / f"phase_d_{name}{suffix}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print("wrote", path)


if __name__ == "__main__":
    main()
