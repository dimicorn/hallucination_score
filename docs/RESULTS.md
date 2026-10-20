# Results: first full run

> These results use the original fixed σ = 6 px moment window. Because of the
> background-gradient problem described below, the features have since been switched to
> adaptive windows; Phase B and Phase D need to be rerun before comparing with this file.

Analysis of the first complete run of all phases (GPU, 2026-10-20/21, code at `f114e77`).
Numbers come from `results/` (`analysis.json`, `phase_a.json`, `phase_d_*.csv`) plus extra
checks on the raw CSVs. Question numbers Q1–Q5 refer to `docs/PLAN.md` §7.

**Setup.** Tuned on the 8 dev sources: prior time t = 0.2, weight λ = 0.5, annealed from
t = 2.0, 1500 Adam steps. Phase D uses 12 eval sources; masking and SNR use 3 noise draws,
blur uses 2. Row counts match the design: masking 2520, SNR 1512, blur 840, prior 2520.

## Summary

1. **The feature level carries information that the pixel map does not (Q4).** The pixel
   score labels the whole footprint prior-dominated in every masking condition for every
   source with S/N ≲ 90. Across sources it tracks S/N (Spearman −0.88), so it works as a
   per-source S/N threshold. Feature-level data support `I_D` spans five orders of
   magnitude and responds to masking, noise and blur. The mechanism is in §Q4.
2. **R is a monotone function of `I_D` and t.** `I_P` sits at the Tweedie cap (median
   κ ≈ 0.95) and barely varies (std of log10 `I_P` = 0.1, against 2.2 for `I_D`).
   Spearman(R, `I_D`) = −0.999 in every experiment. At t = 0.2 the learned prior acts
   locally like an isotropic penalty of strength λ/s_t². R therefore adds nothing beyond
   `I_D`, and its absolute level is set by t.
3. **Six of the seven features mostly measure background pixels.** For size, e1, e2,
   concentration, asymmetry and flux, 95–99.5 % of the gradient's weight ‖∇φ‖² falls on
   pixels where the truth is below 2 % of peak, at a median radius of about 10 px.
   Galaxy flux sits at a median radius of 1.5 px. These six features get nearly identical
   `I_D` and move together under every degradation. Only hf_power behaves differently.
4. **Curvature and the λ intervention agree under degradation but disagree when ranking
   features within a case (Q5).** The disagreement comes from size and concentration:
   both are biased, and both are sensitive to the prior weight while reading
   data-dominated.

## Pixel-level baseline (Phase A)

- **Half-mask control.** Reproduced qualitatively: ΔF is higher in the masked half (median
  202 vs 105), and 33/36 cases have a prior-dominated masked half. The sign saturates,
  though: only 9/36 cases have a data-dominated observed half, and the median fraction of
  prior-dominated pixels is 1.0 in both halves.
- **Why it saturates.** diag F_D is about 30 per pixel, while diag F_P is about 250 in the
  background (near the cap λ/(s_t²·0.01·(1+x)²) ≈ 276) and about 100 on the galaxy. The
  structure inside the footprint follows image brightness through the (1+x)⁻² factor of
  the log transform, not whether the pixel is observed.
- **The map flips only for bright sources.** Unmasked, the prior-dominated fraction is 1.0
  for all seven sources with S/N ≤ 69. It is 0.41 at S/N 94, 0.97 at S/N 126, and 0.02 or
  less at S/N ≥ 181.
- **Hutchinson estimator.** Relative error of diag F_P is 1.2 % at 16 probes, 0.15 % at
  1024 and 0.08 % at 4000. The sign of ΔF on the footprint agrees with the exact value
  100 % of the time, even at 16 probes.
- **F_P properties.** The asymmetry ‖F_P − F_Pᵀ‖/‖F_P‖ is about 3 %, and 0.1–0.3 % of the
  eigenvalues are negative.
- **t-sensitivity.** The size of ΔF falls steadily as t grows (masked half: 1230 at
  t = 0.01, 512 at 0.05, 169 at 0.2), while its sign does not change. Background κ exceeds the
  theoretical ceiling of 1 at small t: 1.49 at t = 0.01, 1.27 at 0.02, 1.09 at 0.05,
  0.99 at 0.2. So the learned score violates the Tweedie bound at small t, and
  `test_prior_curvature_below_smoothing_cap` passes at t = 0.05 only because of its 10 %
  margin.

## Where the feature gradients live

Feature gradients evaluated at the truth images of the 12 eval sources (medians over
sources):

| feature | median radius of ‖∇φ‖² (px) | fraction of ‖∇φ‖² where truth < 2 % of peak |
|---|---|---|
| (truth flux, for reference) | 1.5 | 0.006 |
| flux | 12.8 | 0.95 |
| size | 10.0 | 0.98 |
| e1 | 10.0 | 0.99 |
| e2 | 9.9 | 0.99 |
| concentration | 10.0 | 0.99 |
| asymmetry | 15.3 | 1.00 |
| hf_power | 2.2 | 0.33 |

The moment window weights each pixel by r²·exp(−r²/2σ²) with σ = 6 px, which peaks at
r = 8.5 px. That is far outside galaxies of size about 2.6 px. The aperture flux gradient
is uniform over a large aperture, so it is also dominated by area. These features are
sensitive to faint halo flux (see README finding 5), so the background pixels do matter
for them. Still, the directional diagnostic then mostly measures whether the annulus at
r ≈ 10 px is observed. As a result, it cannot tell size, shape and concentration apart.

Pixels removed by each mask (median over sources):

| mask | 25 % | 50 % | 75 % |
|---|---|---|---|
| half-plane | 25 % of pixels, 0 % of flux | 50 %, 63 % | 75 %, ~100 % |
| core (by flux) | 0.4 % of pixels | 0.9 % | 2.0 % |
| outer (by flux) | 98.1 % of pixels | 99.2 % | 99.7 % |

## Q1: Does data support fall when relevant measurements are removed?

Yes. Within a source, `I_D` falls steadily as more of the image is masked. Median
within-source Spearman between flux removed and `I_D` is −0.92 for half-plane masks, −0.97
for outer masks and −0.57 for core masks. `analyze.py` pools sources, which dilutes this to
−0.26 to −0.78 (`masking_response`).

Change in log10 `I_D` against the unmasked case (paired by source and seed, median):

| mask | asym | conc | e1 | e2 | flux | hf_power | size |
|---|---|---|---|---|---|---|---|
| core 0.75 | 0.00 | −0.01 | 0.00 | 0.00 | −0.01 | −0.19 | 0.00 |
| half-plane 0.25 | −0.25 | −0.08 | −0.06 | −0.08 | −0.12 | −0.03 | −0.07 |
| half-plane 0.75 | −0.62 | −0.84 | −1.18 | −0.82 | −0.63 | −1.21 | −0.75 |
| outer 0.25 | −3.52 | −3.42 | −3.06 | −3.12 | −1.67 | −0.44 | −2.90 |
| outer 0.75 | −4.69 | −4.23 | −5.04 | −5.10 | −2.42 | −1.02 | −4.24 |

- **Severity measure.** Removed flux is a poor measure of severity. Core masks remove 29–76 %
  of the flux and leave `I_D` unchanged, because they touch almost none of the pixels the
  gradients live on. The 25 % half-plane mask removes no flux, yet it lowers asymmetry's
  `I_D` by 0.25 dex. Likewise, the core-vs-outer comparison in `masking_feature_vs_pixel`
  (`rank_corr_I_D_core_vs_outer` = 0.14) is not matched in pixels.
- **Blur is the cleanest selective response.** Only hf_power responds: Δlog10 `I_D` is
  −0.53 at cutoff 0.2 and −0.72 at 0.07, while every other feature changes by at most
  0.07 dex. Most of the drop happens at the first cutoff, since the feature's own band
  starts at 0.25 cycles/px.
- **The SNR response holds by construction.** `I_D`·(noise scale)² is constant within a
  (source, seed, feature) to a coefficient of variation of 0.006, because F_D ∝ 1/σ². The
  pooled ρ ≈ −0.70 for every feature therefore says nothing about the method. Within a
  source it would be −1.

## Q2: Prior dominance vs low total support

- **The `weakly_constrained` flag behaves sensibly with noise.** For e1 it rises from 0 %
  at noise ×0.5 to 92 % at ×16, and for size from 0 % to 72 %. It never fires for flux or
  hf_power.
- **Asymmetry is flagged in 39 % of cases even at noise ×0.5,** which fits its gradient
  living in the image corners.
- **With `I_P` essentially constant, "prior-dominated" and "low `I_D`" are the same
  statement.** Telling them apart would need a prior whose curvature varies by direction
  (κ well below 1), as it does only for hf_power (κ ≈ 0.70) and on the footprint
  (κ ≈ 0.75).

## Q3: Stability

- **Noise draws.** The median spread over noise draws is 0.005 in log10 `I_D` and 0.002
  in R. Between sources it is 1.58 and 0.29. The spread of feature error over noise draws
  is 0.09σ, against 0.52σ between sources.
- **Hutchinson seeds.** Stable, see Phase A.
- **Changes to the prior.** Stable in sign, but magnitudes depend on t, and so does R's
  absolute level. At t = 0.05 the cap is about 3.7 times larger, so features that
  read R ≈ 0.3 here would read about 0.6. The tuning chose t = 0.2, the largest value in
  the grid, so the optimum may lie beyond the grid.

## Q4: What feature-level attribution adds beyond the pixel map

Across all masking conditions, the pixel score is 1.0 (fully prior-dominated) in at least
half of the cases. Feature-level R for the unmasked case:

| asym | conc | e1 | e2 | flux | hf_power | size |
|---|---|---|---|---|---|---|
| 0.35 | 0.30 | 0.32 | 0.30 | 0.28 | 0.92 | 0.27 |

**The mechanism.**
- The data constrain PSF-smooth combinations of pixels about 1/Σk² ≈ 28 times more
  strongly than a single pixel. A smooth direction v has ‖W^½Av‖² ≈ w‖v‖², while the
  diagonal of F_D is w·Σk².
- The prior's curvature, by contrast, is nearly isotropic.
- Smooth features are therefore data-dominated (`I_D` ≈ 10^2.7–10^2.8, against
  `I_P` ≈ 260), even though every individual pixel is prior-dominated.
- High-frequency directions are not: hf_power has `I_D` ≈ 10^1.2.

This is the main positive result. It does not depend on t for the ranking, though it does
for R's absolute level.

**Feature errors.** In predicting feature errors under masking, σ_local does about as well
as the pixel score or better. Spearman with |error| for σ_local versus the pixel score:
asymmetry 0.52 vs 0.39, concentration 0.41 vs 0.03, e1 0.42 vs 0.36, e2 0.42 vs 0.13.
Under the SNR sweep the two are comparable. Neither predicts size errors under masking
(ρ ≈ 0).

## Q5: Curvature vs prior-weight intervention

S is the change in a feature between λ = 0.5 and λ = 2, divided by twice the population
std. That gives the change per doubling of λ, in population-std units. `I_P` scales
exactly with λ (median 260 → 525 → 1060), as expected.

- **Agreement under degradation.** Under half-plane masking, lower `I_D` goes with higher
  S (Spearman −0.48 to −0.83 for asymmetry, concentration, e1, flux and size). Asymmetry
  shows all three diagnostics agreeing on a hallucination: at 75 % half-plane masking its
  error is 8.5σ, R rises from 0.35 to 0.71, 80 % of cases are flagged weakly constrained,
  and S reaches 1.1–2.3 at the 50–75 % levels.
- **Disagreement within cases.** Ranked across the seven features of one case, R and S
  anti-correlate (median ρ = −0.23 over 120 cases). Without size and concentration the
  median is +0.10. Those two read data-dominated (R ≈ 0.27–0.30) but are the most
  prior-sensitive (median S ≈ 0.31–0.36; the other features are at 0.05–0.07).
- **Size and concentration are biased even with full data.** The median signed error is
  +0.62σ for size and +0.73σ for concentration (flux +0.14σ). The bias shrinks under
  blur: size goes from +0.62 to +0.17 at cutoff 0.07, concentration from +0.73 to −0.28.
- **Hypothesis (not tested).** The positivity clip turns background noise into faint halo
  flux, and the low-pass removes part of the noise that drives it. A local curvature
  diagnostic cannot detect a bias of this kind. The prior-weight intervention can.
- **S predicts the actual error only weakly** (Spearman 0.46 for asymmetry, 0.39 for
  concentration, about 0 for the rest).

## Status against the BASP minimum (PLAN §8)

| item | status |
|---|---|
| Pixel-level baseline reproduced | Qualitatively; sign saturates for S/N ≲ 90 |
| Interpretable feature set | Defined, but six of seven are dominated by background gradients |
| Defensible feature-level diagnostic | `I_D` and σ_local yes; R is a t-dependent rescaling of `I_D` |
| Masking and S/N experiments | Done; the S/N response of `I_D` holds by construction |
| Quantitative response to information removal | Yes within sources; feature selectivity only for hf_power |
| Comparison with the pixel baseline | Yes, with a clear mechanism |
| Intervention diagnostic (nice to have) | Done; it disagrees for size and concentration |

**Go/no-go (PLAN §10).** Go, with fixes. The feature level gives stable information that
the pixel map lacks. But the feature definitions have to be fixed before the
feature-by-feature results can be read.

## Next steps

1. **Adaptive feature windows.** Scale the moment window and the flux aperture to each
   source's size (e.g. HSM-style adaptive moments), then rerun Phase B and Phase D. Check
   that the gradient statistics above move onto the galaxy.
2. **Masking design.** Add core vs annulus masks with equal pixel counts, and define
   severity as the fraction of ‖W^½Av‖² removed, not the fraction of flux.
3. **Reporting.** Lead with `I_D` and σ_local relative to the population spread. Report κ
   as the prior-specific quantity, and quote R only together with t. Present the SNR
   sweep as a test of errors and flags, not as validation.
4. **`analyze.py`.** Compute severity correlations within each source, and add the
   gradient-location statistic.
5. **Halo hypothesis.** Test whether the size and concentration bias comes from
   noise rectified by the positivity constraint, e.g. by comparing the bias against noise
   scale and against a reconstruction without the clip.
6. **Tuning grid.** Extend the t grid past 0.2, or report that the chosen value sits at
   the grid edge.
