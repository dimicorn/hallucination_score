# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research code: feature-level prior/data attribution for galaxy image reconstruction with a learned score prior, extending the pixel-level hallucination score of Sampson & Melchior (2023). The research plan (phases A–D, experiment design, success criteria) is in `docs/PLAN.md`; `README.md` records setup decisions and findings so far. Read both before changing methodology.

## Commands

```bash
uv sync                                   # CPU env; Python 3.12 only
scripts/fetch_data.sh                     # downloads data/hsc_9697_72_subset.pkl (checksum-verified; not committed)
uv run pytest                             # unit tests, ~5 s, no data file needed
uv run pytest tests/test_core.py::test_adjoint   # single test
```

No linter or formatter is configured.

Experiments (writes to `results/`, git-ignored). Every script takes `--quick` (1 source, 1 seed, 200-step fits), which writes to `results/quick/` and never touches real results. Use it to check that changes run end to end (~2 min on CPU); quick-mode numbers are not results.

```bash
for s in tune_prior phase_b_features phase_a_baseline; do uv run python experiments/$s.py --quick; done
for e in masking snr blur prior; do uv run python experiments/phase_d.py $e --quick; done
uv run python experiments/analyze.py --quick
```

Full runs go in the same order without `--quick`. The order matters because of file dependencies:
- `tune_prior.py` writes `tune_prior.json`; `Config.load()` reads it (otherwise falls back to provisional defaults in `pipeline.Config`).
- `phase_b_features.py` writes `feature_population.json`; `evaluate.population_std()` needs it, so `phase_d.py` fails without it.
- `analyze.py` reads whichever `phase_d_*.csv` exist.

GPU: follow the README's GPU steps (CUDA jaxlib + cuDNN 8.9 pin, then `scripts/clear_execstack.py` on glibc ≥ 2.41), then always use `uv run --no-sync ...` (or `export UV_NO_SYNC=1`) so uv doesn't revert jax; "a CUDA-enabled jaxlib is not installed. Falling back to cpu" means it was reverted. A `cannot enable executable stack` ImportError means jaxlib was reinstalled and the script needs rerunning.

## Dependency pins

`jax==0.4.18`, `equinox==0.10.7`, `galaxygrad==0.0.19`, `numpy<2` are deliberate: the galaxygrad checkpoints only deserialise with equinox 0.10.x, which needs jax from the same era. Don't upgrade them, and use APIs that exist in these versions (e.g. `eqx.static_field`, not `eqx.field(static=True)`).

## Architecture

Package `src/provenance/` (installed as `provenance`), experiment drivers in `experiments/`.

The pipeline for one case is: `data.Source` (SCARLET model as 32×32 peak-normalised truth + real HSC PSF and noise) → `forward.make_observation` (mask / noise scale / Fourier low-pass) → `Observation.simulate` → `reconstruct.map_estimate` → `attribution` at `x_hat`. `pipeline.run_case` does simulate+reconstruct; `evaluate.evaluate` wraps it and returns one row per feature, which `phase_d.py` collects into CSVs.

Shared conventions in `pipeline.py`: every experiment script calls `parse_args()` (adds `--quick`, sets a module-level flag), then `Config.load()`, and writes output via `results_dir()` / `save_json()` so quick mode is redirected automatically. Experiment scripts shrink their own module-level sweep constants in a `use_quick_settings()` function. Follow this pattern for new experiments.

Key points that are spread across files:
- **Curvatures are matvecs.** `F_D` (`Observation.curvature_matvec`, exact, linear model) and `F_P` (`ScorePrior.curvature_matvec`, one JVP through the score net) are functions `v -> Fv`. `curvature.py` turns any matvec into a Hutchinson diagonal, a dense 1024×1024 matrix, or a quadratic form `vᵀFv`. Pixel scores use diagonals; feature attribution uses quadratic forms along the normalised feature gradient (`features.gradients`, via autodiff).
- **Prior data space.** The score net works in `u = log(1+x)/0.1`; `ScorePrior.score` maps back to x-space including the log-Jacobian term (which scarlet2's `NNPrior` drops). `F_P` is a learned Jacobian, so it can be non-symmetric or have negative curvature; `attribution` flags rather than reports R in those cases.
- **Diffusion time t matters.** t = 0 is outside the trained regime, and by Tweedie's identity directional prior curvature is capped at about `1/s_t²` (`ScorePrior.curvature_cap_matvec`). That is why attribution reports `κ = I_P / I_cap` and absolute `I_D` next to the ratio `R`. `map_estimate` can anneal t coarse-to-fine (`anneal_from`) before holding it at `prior.t`.
- **Reconstruction** is projected Adam on the score (there is no prior log-density value), with `x ≥ 0` enforced by clipping, all inside one `eqx.filter_jit` + `lax.scan`.
- **Features** (`features.py`) are differentiable JAX functions of the image: moments use a fixed σ = 6 px Gaussian window, and asymmetry rotates about the unwindowed centroid. Flux, size and hf_power are strongly correlated.
- **Data split.** `data.split` gives 8 dev sources (used only by `tune_prior.py`) and the rest as eval. Keep tuning on dev only.
