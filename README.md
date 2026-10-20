# hallucination_score

Feature-level prior–data attribution in astronomical image reconstruction — which
morphological features of a reconstructed galaxy are constrained by the measurements,
and which are mostly supplied by a learned prior. Extends the pixel-level hallucination
score of Sampson & Melchior (2023), [arXiv:2306.13272](https://arxiv.org/abs/2306.13272).
The research plan is in [`docs/PLAN.md`](docs/PLAN.md).

## Setup

```bash
uv sync                      # CPU environment (jax 0.4.18, equinox 0.10.7, galaxygrad 0.0.19)
scripts/fetch_data.sh        # HSC tract 9697 / patch 72 source subset -> data/
uv run pytest                # fast unit tests (~5 s)
```

The pins are deliberate: the galaxygrad 0.0.19 checkpoints deserialise with
equinox 0.10.x, which needs jax of the same era.

**GPU.** After `uv sync`, swap in a CUDA build of the same jax version and run with
`--no-sync` so uv does not revert it:

```bash
uv pip install "jax[cuda12_pip]==0.4.18" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv run --no-sync python -c "import jax; print(jax.devices())"
```

## Running the experiments

Run in this order; everything writes to `results/` (git-ignored).

| Step | Command | What it does |
|---|---|---|
| 1 | `uv run python experiments/tune_prior.py` | Picks prior time `t`, weight `λ`, annealing on the **dev** split (21 settings × 16 fits). Later steps read `results/tune_prior.json`. |
| 2 | `uv run python experiments/phase_b_features.py` | Feature distributions on clean truths and autodiff-vs-finite-difference checks; writes the population scales used by step 4 (seconds). |
| 3 | `uv run python experiments/phase_a_baseline.py` | Pixel-level baseline: half-mask control (12 sources × 3 noise draws), exact-vs-Hutchinson convergence, F_P properties, t-sensitivity. |
| 4 | `uv run python experiments/phase_d.py {masking,snr,blur,prior}` | Experiments 1–3 and the prior-strength intervention (C2). |
| 5 | `uv run python experiments/analyze.py` | Figures and `results/analysis.json` from whichever step-4 outputs exist. |

On a 12-core CPU a fit takes ~2.5 s (1500 sequential Adam steps) and an exact 1024×1024
F_P another ~1.5 s, which puts steps 3–4 at roughly 1–2 h in total. On a GPU the exact
Hessians (1024 batched JVPs) speed up a lot; the MAP fits are sequential per case, so
the larger win would come from batching cases with `vmap` (not done yet).

## Layout

```
src/provenance/
  data.py         HSC SCARLET models as peak-normalised 32x32 truths, real PSF + noise level
  forward.py      y = M (k * x) [+ Fourier low-pass] + noise; exact F_D; mask generators
  prior.py        galaxygrad score net mapped to image space; F_P via JVP; smoothing cap
  curvature.py    Hutchinson diagonal (Algorithm 1 of the paper) and exact dense matrices
  reconstruct.py  projected-Adam MAP with coarse-to-fine annealing of the prior's t
  features.py     flux, size, e1, e2, concentration, asymmetry, high-frequency power
  attribution.py  pixel ΔF; feature-level C1 diagnostic (I_D, I_P, R, D, κ, flags)
  evaluate.py     per-case evaluation shared by the experiments
experiments/      drivers for the plan's phases (see table above)
tests/            unit tests: adjoint, exact F_D, null space, Hutchinson, features, prior
```

## Setup decisions and findings so far

These came out of building and checking the pipeline on CPU. The quantitative results
are still pending the full runs.

1. **Ground truth.** The `scar_*` columns of the HSC pickle are SCARLET's deblended,
   deconvolved single-galaxy models, the same kind of image the prior was trained on.
   They are used as truths and observed with the real HSC PSF and noise level, so
   feature errors can be measured. 255 sources fit a 32×32 stamp with S/N ≥ 20
   (8 dev / 247 eval). *Caveat:* the prior was trained on SCARLET models from three
   HSC tracts that include 9697, so these sources are probably **not held out**.
2. **The prior's data space.** The score net expects `u = log(1 + x) / 0.1` with `x`
   peak-normalised. Evidence: real galaxies get much smaller scores than noise only
   under this mapping, and probability-flow samples mapped back have background ≈ 0
   and peak ≈ 0.9. galaxygrad's own sampler assumes a linear standardisation instead.
   The vendored scarlet2 `NNPrior` uses the log transform, but evaluates the
   chain-rule factor at the already-transformed point and drops the log-Jacobian
   term; `prior.py` does both correctly.
3. **t = 0 is outside the trained regime.** The training loss weight `1 − e^{−t}`
   vanishes at t = 0, and at t = 0 pure noise gets as large a score as a real galaxy.
   On the dev split, t = 0 reconstructions are worst. The partial CPU tuning run
   (19/21 settings, median relative error, full / half-masked data) gave:
   t=0: 1.20 / 1.21; t=0.01: 0.56 / 0.68; t=0.05: 0.32 / 0.56; t=0.2: 0.32 / 0.51
   (λ=0.5 with annealing for t>0). The likelihood-only fit gives 1.44 / 2.41.
4. **The prior curvature is capped by t.** By Tweedie's second-order identity, for
   the VP score at time t, `−∇² log p_t = (I − a ∂D/∂u)/s_t²` with a PSD denoiser
   Jacobian, so every directional prior curvature lies in `[~0, 1/s_t²]`. In a smoke
   test, I_P ≈ 2000–2200 for almost every feature and condition, which is about the
   cap at t = 0.05. Consequences: the sign of the pixel score ΔF (and the feature
   ratio R) can be moved by the choice of t alone, and at t = 0 the cap is infinite.
   The code therefore also reports `κ = I_P / I_cap` (how much of the cap the prior
   uses along a direction) and the absolute data support `I_D`. Phase A measures the
   t-sensitivity directly. A unit test confirms the learned prior respects the cap.
5. **Windowed moments.** Unweighted second moments were dominated by faint
   reconstructed halos (size 4.4 px vs 1.7 px true for one source), so moments use a
   fixed σ = 6 px Gaussian window. Asymmetry rotates about the unwindowed centroid.
   Autodiff gradients match finite differences (median relative error ≤ 0.5%; 2% for
   asymmetry because of bilinear-interpolation kinks).
6. **Feature correlations.** On the reference set, flux and size have Spearman
   ρ = 0.91, and flux and high-frequency power ρ = −0.69. Feature-level attributions
   for these are not independent; this is reported with the results rather than
   corrected for.

## Reference

Sampson, M. L. & Melchior, P. (2023). *Spotting Hallucinations in Inverse Problems with
Data-Driven Priors.* https://arxiv.org/abs/2306.13272
