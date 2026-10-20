"""Shared configuration and the simulate -> reconstruct step used by all experiments."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import forward, reconstruct
from .prior import ScorePrior

RESULTS = Path(__file__).resolve().parents[2] / "results"


@dataclass(frozen=True)
class Config:
    # chosen on the dev split only (experiments/tune_prior.py)
    prior_t: float = 0.05
    prior_weight: float = 1.0
    anneal_from: float = 2.0
    steps: int = 1500
    lr: float = 1e-2

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
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
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
