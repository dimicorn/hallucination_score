"""Choose the prior's diffusion time, weight and annealing on the dev split only.

Selection rule (fixed before looking at results): the setting minimising the mean of
the median relative reconstruction errors for the full and half-masked observations.
"""

import itertools

import numpy as np

from provenance import data, forward
from provenance.pipeline import QUICK_STEPS, Config, is_quick, parse_args, run_case, save_json

# t = 0.2 won the first grid at its edge, hence 0.5 and 1.0
GRID = {"prior_t": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], "prior_weight": [0.0, 0.5, 1.0], "anneal_from": [None, 2.0]}
QUICK_GRID = {"prior_t": [0.0, 0.05], "prior_weight": [0.0, 1.0], "anneal_from": [None, 2.0]}
MASKS = {"full": None, "half": forward.half_plane_mask((32, 32), 0.5)}


def main():
    parse_args()
    dev, _ = data.split(data.load_sources())
    grid, steps = (QUICK_GRID, QUICK_STEPS) if is_quick() else (GRID, Config.steps)
    if is_quick():
        dev = dev[:1]
    rows = []
    for t, lam, ann in itertools.product(*grid.values()):
        if lam == 0.0 and (t, ann) != (0.05, None):
            continue  # likelihood-only baseline, run once
        cfg = Config(prior_t=t, prior_weight=lam, anneal_from=ann, steps=steps)
        errs = {k: [] for k in MASKS}
        for src in dev:
            for name, mask in MASKS.items():
                case = run_case(src, cfg, mask=mask)
                errs[name].append(float(np.linalg.norm(case.x_hat - src.truth) / np.linalg.norm(src.truth)))
        row = {"prior_t": t, "prior_weight": lam, "anneal_from": ann}
        row.update({f"relerr_{k}": float(np.median(v)) for k, v in errs.items()})
        row["score"] = (row["relerr_full"] + row["relerr_half"]) / 2
        rows.append(row)
        print(row, flush=True)
    best = min((r for r in rows if r["prior_weight"] > 0), key=lambda r: r["score"])
    print("best:", best)
    save_json({"rows": rows, "best": best, "n_dev": len(dev)}, "tune_prior.json")


if __name__ == "__main__":
    main()
