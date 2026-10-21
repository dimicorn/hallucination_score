# Results

Three full runs so far. Run 3 is current. Run 1 used a fixed moment window whose
feature gradients sat on background pixels; run 2 fixed the features but used the
first tuning grid (t = 0.2); run 3 uses the wider grid (t = 0.5), adds the Laplace
feature posterior and the t-sensitivity run. Question numbers Q1–Q5 refer to
`docs/PLAN.md` §7. Phase D uses 12 eval sources; masking and SNR use 3 noise draws, blur
uses 2. Row counts match the design in every run: masking 2520, SNR 1512, blur 840,
prior 2520.

## Run 3: t = 0.5, Laplace diagnostics, t-sensitivity (current)

GPU, 2026-10-21, code at `6b718d4` (analysis rerun locally with `93703da`). Tuning, Phase
A, B and D and `t_sensitivity.py` were all rerun. Raw outputs are in `results/`; run 2's
are kept locally in `results/run2_adaptive_window/`, run 1's in
`results/run1_fixed_window/`.

**Setup.** The wider grid picked t = 0.5, λ = 0.5, no annealing (score 0.391; with
annealing 0.391 as well, t = 0.5 with λ = 1 0.398, t = 0.2 0.410; t = 1.0 is not in the
top six). Larger t trades a little full-data accuracy for better masked reconstruction:
from t = 0.1 to 0.5 the full-data error rises 0.306 → 0.329 and the half-masked error
falls 0.535 → 0.453. The tuning score averages the two.

### Summary

1. **The feature ordering is robust to t; R's level is not.** Over t = 0.02–1.0 the
   within-case ranking of features by R agrees with the tuned t at Spearman 0.93–1.00
   for fixed reconstructions, and 0.86–1.00 when reconstructing again at each t (0.76 at
   t = 0.02, where the reconstructions themselves change). R's level moves a lot
   (concentration, full data: 0.75 at t = 0.02, 0.08 at t = 1.0). The information gain
   moves little (flux 3.55 → 3.34 nats, concentration 2.20 → 1.98). **Claim the
   ordering, and quote information gain as the absolute number.**
2. **The Laplace σ_post fixes most of the overconfidence.** Fraction of errors within 1σ
   (68 % if calibrated), masking, directional σ_local → Laplace σ_post: flux 32 → 49 %,
   size 12 → 36 %, e1 46 → 60 %, e2 52 → 69 %, concentration 39 → 52 %, asymmetry
   10 → 20 %, HF power 7 → 13 %. Median |error|/σ_post is 0.6–1.0 for ellipticity,
   concentration and flux, but 2–4 for asymmetry and HF power: their remaining error is
   the prior's bias, which a local Gaussian cannot capture.
3. **The headline pattern repeats at t = 0.5.** Unmasked medians:

   | | flux | size | e1 | e2 | conc | asym | hf_power |
   |---|---|---|---|---|---|---|---|
   | log10 `I_D` | 2.63 | 2.34 | 2.18 | 2.15 | 2.48 | 1.72 | 1.20 |
   | κ | 0.09 | 0.16 | 0.17 | 0.20 | 0.39 | 0.47 | 0.80 |
   | R | 0.01 | 0.04 | 0.11 | 0.08 | 0.12 | 0.43 | 0.86 |
   | signed error (σ_pop) | +0.02 | +0.13 | 0.00 | −0.07 | −0.07 | −0.62 | −0.47 |
   | σ_post / σ_pop | 0.03 | 0.11 | 0.27 | 0.30 | 0.13 | 0.24 | 0.14 |
   | info gain (nats) | 3.41 | 2.25 | 1.33 | 1.23 | 2.04 | 1.42 | 1.96 |

   Spearman(R, |bias|) across features 0.61; joint information gain 13.0 nats.
   Information gain counts the prior's narrowing too: HF power is well determined
   (2.0 nats) because the prior pins it (κ = 0.80), not the data. Read it with R.
4. **Error prediction: comparable to the pixel score, not better.** At t = 0.5 the pixel
   score saturates less, so it carries more signal. Spearman with |error|, R vs pixel:
   masking flux 0.46 vs 0.30, e2 0.54 vs 0.45, concentration 0.37 vs 0.14, but e1 0.45
   vs 0.54, asymmetry 0.47 vs 0.54, HF power 0.34 vs 0.59 (size 0.30 vs 0.34); noise
   flux 0.86 vs 0.69, concentration 0.68 vs 0.58, size 0.74 vs 0.73, with the pixel
   score ahead for e1, asymmetry and HF power. The feature level adds selectivity, a
   per-feature data-only flag and calibrated widths rather than a better single error
   predictor.

### Details

- **Pixel score.** The footprint is ≥ 95 % prior-dominated for all sources with
  S/N ≤ 69 and largely data-dominated from S/N ≈ 94 (9 %); Spearman with S/N −0.95.
  Median footprint ΔF with full data +34. Half-mask control: 30/36 masked halves
  prior-dominated, 15/36 observed halves data-dominated (run 2: 33 and 9).
- **Selectivity.** Paired Δlog10 `I_D`: core mask (central rings holding 75 % of the
  reference flux), flux −0.26 (others ≥ −0.19); outer mask (outer rings, 75 %), concentration −2.46, e1/e2 −2.03/−2.08, flux
  −1.04. Core-vs-outer rank correlation of the feature orderings +0.18, −0.61, −0.68
  at 25/50/75 % (run 2: −0.50, −0.79, −0.50). Within-source Spearman of `I_D` with flux
  removed: half-plane −0.92, core −0.51, outer −0.97. Low-pass at 0.07 cycles/px:
  HF power −0.91, e1/e2 −0.77/−0.79, asymmetry −0.47, size −0.18, concentration −0.14,
  flux −0.01.
- **Fully masked galaxy (75 % half-plane).** `I_D` falls 5.7–7.8 dex and R = 1 for
  every feature except HF power (0.60; its `I_D` even rises by 0.51 dex). At t = 0.5
  without annealing, the prior-only reconstruction collapses to a near single-pixel
  spike (flux 1.1 against a true 12.4, size at the window floor), whose Fourier-based
  HF gradient spreads into observed pixels. A degenerate case.
- **Weak-support flag.** Median |error| flagged vs unflagged: flux 0.84 vs 0.04, size
  0.40 vs 0.26, e1 0.47 vs 0.18, e2 0.64 vs 0.19, concentration 1.41 vs 0.20, asymmetry
  1.00 vs 0.58, HF power 1.26 vs 0.54.
- **Posterior correlations (full data).** flux–size 0.73, size–concentration −0.54,
  asymmetry–HF power 0.41; ellipticity is nearly independent of the rest (|ρ| ≤ 0.1).
  Median `neg_mode_share` 0.10 (flux), 0.13 (size), < 0.05 otherwise.
- **Intervention.** Median S 0.01–0.09 σ_pop per doubling of λ; within-case
  Spearman(R, S) +0.09. Still a weak validator.
- **Errors against t** (reconstructed again at each t, full data) are stable except for
  asymmetry, whose bias grows from 0.48σ at t = 0.02 to 0.70σ at t = 0.2–1.0.

## Run 2: adaptive moment windows, t = 0.2 (superseded by run 3)

GPU, 2026-10-21, code at `2159018`, with t = 0.2, λ = 0.5, annealed from t = 2.0 (the
first tuning grid). Phase B and Phase D were rerun; tuning and Phase A were reused from
run 1. Raw outputs are kept locally in `results/run2_adaptive_window/`.

**What changed.** Moments use an HSM-style adaptive elliptical Gaussian window, flux is
the adaptive flux, and asymmetry is measured about the adaptive centre in a window twice
as wide. On the truth images, the background share of each feature's gradient fell from
95–99.5 % to 9–15 % (concentration 55 %, because of its flat outer aperture). Autodiff
gradients match finite differences to a median relative error of 0.06–1.2 % on the real
sources.

### Summary

1. **Features now get different attributions, and the prior's curvature depends on the
   direction.** Unmasked medians:

   | | flux | size | e1 | e2 | conc | asym | hf_power |
   |---|---|---|---|---|---|---|---|
   | log10 `I_D` | 2.63 | 2.34 | 2.17 | 2.14 | 2.48 | 1.73 | 1.18 |
   | κ = `I_P`/`I_cap` | 0.08 | 0.16 | 0.13 | 0.16 | 0.48 | 0.36 | 0.74 |
   | R | 0.03 | 0.09 | 0.15 | 0.14 | 0.25 | 0.56 | 0.92 |
   | signed error (σ_pop) | +0.02 | +0.10 | 0.00 | −0.12 | −0.03 | −0.62 | −0.48 |

   The learned prior barely constrains flux, size or ellipticity, as expected for a
   population that spans all brightnesses, scales and orientations. It constrains
   concentration and high-frequency structure much more. `I_P` is no longer pinned at the
   Tweedie cap (median κ 0.33, against 0.95 in run 1). Spearman(R, `I_D`) is −0.965,
   against −0.999, so R now carries some prior-specific information.
2. **The two most prior-dominated features are the two with systematic bias.** Asymmetry
   and hf_power are both underestimated: the reconstruction is smoother and more
   symmetric than the truth. All other features are within ±0.12σ. Across the seven
   features, Spearman(R, |bias|) = 0.57.
3. **Run 1's size and concentration biases came from the features.** With full data, the
   errors fell from +0.62σ to +0.10σ for size, from +0.73σ to −0.03σ for concentration,
   and from 0.14σ to 0.02σ for flux. The fixed window was largely measuring faint
   background flux, so the halo hypothesis from run 1 no longer needs testing.
4. **Feature-level R predicts feature errors better than the pixel score, for most
   features.** Spearman with |error|, R against pixel fraction:
   - masking: flux 0.44 vs 0.25, size 0.29 vs 0.16, e2 0.59 vs 0.42, concentration
     0.34 vs 0.05 (six of seven features; recomputed with each case counted once, the
     first version counted the unmasked cases three times)
   - SNR: flux 0.83 vs 0.64, size 0.78 vs 0.62, concentration 0.77 vs 0.57
   - Asymmetry is the exception: the pixel score does better (masking 0.35 vs 0.26,
     SNR 0.54 vs 0.41). For hf_power the two are comparable (R is better under masking,
     0.48 vs 0.36; the pixel score is slightly better under SNR, 0.57 vs 0.53).

### Q1: Response to removing information

Change in log10 `I_D` against the unmasked case (paired by source and seed, median):

| mask | flux | size | e1 | e2 | conc | asym | hf_power |
|---|---|---|---|---|---|---|---|
| core 0.75 | −0.31 | −0.09 | −0.15 | −0.15 | −0.05 | −0.02 | −0.19 |
| half-plane 0.50 | −0.32 | −0.32 | −0.38 | −0.34 | −0.32 | −0.28 | −0.22 |
| half-plane 0.75 | −7.20 | −6.73 | −6.66 | −8.00 | −5.74 | −7.69 | −1.21 |
| outer 0.25 | −0.25 | −0.71 | −0.53 | −0.54 | −0.87 | −0.71 | −0.44 |
| outer 0.75 | −0.92 | −1.82 | −2.03 | −2.03 | −2.13 | −1.36 | −1.02 |

- **Core and outer masks now hit the features in opposite orders.** Core masks hurt flux
  most, since flux is measured in the core. Outer masks hurt concentration and ellipticity
  most, and flux least. Median within-source Spearman between flux removed and `I_D`: core
  −0.78 (run 1: −0.57), half-plane −0.92, outer −0.97.
- **Removing the whole galaxy is now detected.** At 75 % half-plane masking (≥ 99.4 % of
  the true flux in masked rows), `I_D` falls 6–8 dex and every feature reads R = 1.00. In
  run 1 it fell only 0.6–1.2 dex, because the gradients leaked into observed background
  rows. The reconstruction there is a fainter, smaller galaxy supplied by the prior:
  median flux 3.2 against a true 12.4, size 1.1 against 2.0.
- **Blur separates features by how much fine detail they need.** Change in log10 `I_D`
  at the strongest low-pass (cutoff 0.07 cycles/px):

  | e1 | e2 | hf_power | asym | size | conc | flux |
  |---|---|---|---|---|---|---|
  | −0.95 | −1.00 | −0.72 | −0.59 | −0.22 | −0.14 | −0.07 |

  In run 1 only hf_power responded.
- **Features still co-vary with overall severity.** Across masking cases, the median
  Spearman between the six non-hf features' `I_D` changes is 0.94 (minimum 0.78, for
  asymmetry), against 0.96 in run 1. The selectivity shows up in which features each kind
  of mask hits, not in this correlation.
- **The SNR response still holds by construction** (F_D ∝ 1/σ²).

### Q2: Prior dominance vs low total support

R now separates the regimes: with no data (75 % half-plane), R = 1.00 while κ stays at
0.2–0.6. The `weakly_constrained` flag, however, never fires in any experiment. It
compares σ_local = ‖g‖/√(`I_P` + `I_D`) with the population spread, and the prior's
curvature alone keeps σ_local below that spread even with no data. With no data, flux
has σ_local = 0.04σ_pop while its actual error is about 0.6σ_pop. **When the prior
dominates, local curvature is badly overconfident.**

**Fix (applied after run 2).** The flag now uses the data alone: a feature is weakly
constrained when σ_data = ‖g‖/√`I_D` exceeds the population spread. `analyze.py`
recomputes it from the existing columns, so run 2 was reanalysed without a rerun.
- **What it flags.** With full data it flags almost nothing (asymmetry 3 %). With the
  galaxy fully masked, it flags 92–100 % of every feature. Under noise and blur, it flags
  ellipticity, asymmetry and hf_power first (e.g. e1: 28 % at noise ×2, 86 % at ×16).
- **Flagged rows have larger errors for every feature.** Median |error| across masking,
  SNR and blur, flagged vs unflagged (in σ_pop):

  | | flux | size | e1 | e2 | conc | asym | hf_power |
  |---|---|---|---|---|---|---|---|
  | flagged | 0.71 | 0.76 | 0.50 | 0.67 | 1.10 | 0.70 | 1.10 |
  | unflagged | 0.04 | 0.14 | 0.19 | 0.22 | 0.11 | 0.58 | 0.52 |

  The gap is smallest for asymmetry, whose prior-driven bias is present even with full
  data.

Negative prior curvature now occurs in 0.6–0.8 % of rows (20/2520 in masking). These are
flagged and excluded from R.

### Q3: Stability

Unchanged in kind from run 1. The pixel-level and Hutchinson results carry over.
Feature-level t-sensitivity has not been measured. Run 1's argument that R's level is
set by the cap applies less now that κ ≪ 1 for most features, but this needs measuring:
evaluate the attribution at fixed reconstructions for several t.

### Q4: Beyond the pixel map

The pixel score still labels the whole footprint prior-dominated for every source with
S/N ≲ 90. At the feature level, the same reconstructions range from strongly
data-dominated (flux, R = 0.03) to prior-dominated (hf_power, R = 0.92). The ranking
matches where the actual biases are, and the feature-level R predicts feature errors
better than the pixel score does (Summary point 4).

### Q5: Curvature vs prior-weight intervention

- **The run 1 disagreement is gone.** Size and concentration are no longer
  prior-sensitive: median S fell from 0.31/0.36 to 0.03/0.04. The median within-case
  Spearman(R, S) went from −0.23 to +0.11.
- **The λ intervention is a weak validator overall.** S is small for every feature
  (median 0.01–0.05σ_pop per doubling of λ). Per-feature Spearman(R, S) ranges from
  −0.09 to 0.36. Under noise, lower `I_D` goes with higher S for flux, size,
  concentration, asymmetry and hf_power (ρ −0.39 to −0.53); under half-plane masking
  there is no relation.
- **Why it fails where the prior dominates most.** When the data carry no information,
  the MAP estimate is the prior's mode for every λ > 0, so S → 0 in exactly the
  prior-dominated limit. At 75 % half-plane masking, median S is 0.001–0.012. A second
  prior, rather than a reweighted one, would be a more informative intervention
  (PLAN C2).

### Other changes

Feature correlations on the reference population increased: Spearman flux–size is 0.96
(run 1: 0.91), and size–concentration −0.79 (run 1: −0.48). Adaptive flux and size both
scale with source extent.

### Status against the BASP minimum (PLAN §8)

| item | status |
|---|---|
| Pixel-level baseline reproduced | Qualitatively; sign saturates for S/N ≲ 90 |
| Interpretable feature set | Yes; gradients sit on the source (except concentration's outer aperture) |
| Defensible feature-level diagnostic | R and κ yes; σ_local is overconfident when the prior dominates, so low support is now judged by σ_data |
| Masking and S/N experiments | Done |
| Quantitative response to information removal | Yes, and selective across features (core vs outer, blur) |
| Comparison with the pixel baseline | Yes: different rankings per feature, better error prediction |
| Intervention diagnostic (nice to have) | Done; weak, and blind to the fully prior-dominated limit |

### Next steps

1. ~~Redefine the weak-support flag on data curvature alone.~~ Done (Q2).
2. Feature-level t-sensitivity: attribution at fixed reconstructions for several t.
   Done in run 3 (`experiments/t_sensitivity.py`). The tuning grid now extends to
   t = 1.0 (run 1 chose t = 0.2 at the old grid's edge).
2b. Feature-space Laplace diagnostics (the abstract's "full local curvature" and
   "posterior information gain"): posterior std, cross-feature posterior correlations,
   information gain and KL against the population distribution of features, and calibration against
   the actual errors. Done in run 3 (`src/provenance/laplace.py`).
3. ~~`analyze.py`: within-source severity correlations, κ and signed bias per feature,
   error-prediction figure.~~ Done: `analysis.json` now has `baseline_full_data`,
   paired `I_D` changes, `masking_selectivity` (rank correlation of the core and outer
   feature orderings: −0.50, −0.79, −0.50 at 25/50/75 %) and `low_support_check`, plus
   `fig_error_prediction.png`.
4. A second prior for the intervention diagnostic, if one is available.
5. Optionally, give concentration a smooth outer weight so that its gradient also stays
   on the source.

---

## Run 1: fixed σ = 6 px moment window (superseded)

Analysis of the first complete run of all phases (GPU, 2026-10-20/21, code at `f114e77`).
Numbers come from what is now `results/run1_fixed_window/` (`analysis.json`, `phase_a.json`, `phase_d_*.csv`) plus extra
checks on the raw CSVs. Question numbers Q1–Q5 refer to `docs/PLAN.md` §7.

### Summary

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

### Pixel-level baseline (Phase A)

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

### Where the feature gradients live

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

### Q1: Does data support fall when relevant measurements are removed?

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

### Q2: Prior dominance vs low total support

- **The `weakly_constrained` flag behaves sensibly with noise.** For e1 it rises from 0 %
  at noise ×0.5 to 92 % at ×16, and for size from 0 % to 72 %. It never fires for flux or
  hf_power.
- **Asymmetry is flagged in 39 % of cases even at noise ×0.5,** which fits its gradient
  living in the image corners.
- **With `I_P` essentially constant, "prior-dominated" and "low `I_D`" are the same
  statement.** Telling them apart would need a prior whose curvature varies by direction
  (κ well below 1), as it does only for hf_power (κ ≈ 0.70) and on the footprint
  (κ ≈ 0.75).

### Q3: Stability

- **Noise draws.** The median spread over noise draws is 0.005 in log10 `I_D` and 0.002
  in R. Between sources it is 1.58 and 0.29. The spread of feature error over noise draws
  is 0.09σ, against 0.52σ between sources.
- **Hutchinson seeds.** Stable, see Phase A.
- **Changes to the prior.** Stable in sign, but magnitudes depend on t, and so does R's
  absolute level. At t = 0.05 the cap is about 3.7 times larger, so features that
  read R ≈ 0.3 here would read about 0.6. The tuning chose t = 0.2, the largest value in
  the grid, so the optimum may lie beyond the grid.

### Q4: What feature-level attribution adds beyond the pixel map

Across all masking conditions, the pixel score is 1.0 (fully prior-dominated) in at least
half of the cases. Feature-level R for the unmasked case:

| asym | conc | e1 | e2 | flux | hf_power | size |
|---|---|---|---|---|---|---|
| 0.35 | 0.30 | 0.32 | 0.30 | 0.28 | 0.92 | 0.27 |

**The mechanism.**
- The data constrain PSF-smooth combinations of pixels up to 1/Σk² ≈ 20 times more (the
  value for the HSC i-band PSF used; an earlier draft said 28, which was wrong)
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

### Q5: Curvature vs prior-weight intervention

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

### Status against the BASP minimum (PLAN §8)

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

### Next steps

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
