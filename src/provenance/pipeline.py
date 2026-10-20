"""Shared configuration and the simulate -> reconstruct step used by all experiments."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import forward, reconstruct
from .prior import ScorePrior

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
QUICK_STEPS = 200
_quick = False


def parse_args(parser=None):
    """Parse the command line shared by all experiment scripts (adds --quick)."""
    global _quick
    parser = argparse.ArgumentParser() if parser is None else parser
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 source, 1 seed, short fits; writes to results/quick/")
    args = parser.parse_args()
    _quick = args.quick
    if _quick:
        print(f"[quick] smoke-test mode, writing to {results_dir()}")
    return args


def is_quick():
    return _quick


def results_dir():
    path = RESULTS_ROOT / "quick" if _quick else RESULTS_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Config:
    # provisional defaults; experiments use Config.load(), which takes the setting
    # chosen on the dev split by experiments/tune_prior.py when that has been run
    prior_t: float = 0.05
    prior_weight: float = 1.0
    anneal_from: float | None = 2.0
    steps: int = 1500
    lr: float = 1e-2

    @classmethod
    def load(cls):
        path = results_dir() / "tune_prior.json"
        if path.exists():
            best = json.loads(path.read_text())["best"]
            cfg = cls(prior_t=best["prior_t"], prior_weight=best["prior_weight"], anneal_from=best["anneal_from"])
            print(f"[config] from {path}: {cfg}")
        else:
            cfg = cls()
            print(f"[config] {path} not found, using provisional defaults")
        return replace(cfg, steps=QUICK_STEPS) if _quick else cfg

    def prior(self, weight=None):
        return ScorePrior(t=self.prior_t, weight=self.prior_weight if weight is None else weight)


@dataclass
class Case:
    obs: forward.Observation
    y: jnp.ndarray
    x_hat: jnp.ndarray
    truth: np.ndarray
    grad_norm: float


def run_case(source, cfg, mask=None, noise_scale=1.0, blur_cutoff=None, seed=0, prior=None):
    obs = forward.make_observation(source, mask=mask, noise_scale=noise_scale, blur_cutoff=blur_cutoff)
    truth = jnp.asarray(source.truth, jnp.float32)
    y = obs.simulate(truth, jax.random.PRNGKey(seed))
    prior = cfg.prior() if prior is None else prior
    x_hat, g = reconstruct.map_estimate(obs, prior, y, steps=cfg.steps, lr=cfg.lr, anneal_from=cfg.anneal_from)
    return Case(obs, y, x_hat, source.truth, g)


def save_json(obj, name):
    path = results_dir() / name
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin))
    return path


def _to_builtin(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))
