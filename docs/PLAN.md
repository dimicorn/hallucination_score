# Feature-Level Prior–Data Attribution in Astronomical Image Reconstruction

## 1. Objective

**Research question:** Given an astronomical image reconstructed using an observational likelihood and a learned generative prior, which *morphological features* are constrained by the measurements, and which are predominantly determined by the prior?

The starting point is Sampson & Melchior's [*Spotting Hallucinations in Inverse Problems with Data-Driven Priors*](https://arxiv.org/abs/2306.13272), which identifies prior-dominated **pixels** through separate likelihood- and prior-curvature estimates. This project investigates **feature-level provenance** instead of merely assigning a score to each pixel.

**First target:** A controlled, astronomy-specific proof of concept suitable for BASP.

**Longer-term target:** Use empirical findings to motivate a more general methodology and benchmark for an ICML main-track paper. Do not attempt the full general ML problem in the first study.

### Scope and terminology

- **Data-driven:** The observational likelihood constrains a feature near its reconstructed value.
- **Prior-driven:** The learned prior plays the larger role in constraining or selecting that feature.
- **Weakly constrained:** Neither source gives a strong constraint on the feature.
- **Hallucination:** A plausible but potentially incorrect reconstructed structure. *Prior dominance is not proof that a structure is false.*

The central goal is to quantify **information provenance**, not to claim that we can unambiguously identify the physical truth of every feature.

---

## 2. Mathematical setup

Let the measurements be

\[
y = f(x) + \epsilon,
\]

where \(x\) is the underlying astronomical image, \(f\) is the known forward model, and \(\epsilon\) is the measurement noise. Reconstruction uses

\[
p(x\mid y) \propto p(y\mid x)\,p(x),
\]

with an explicit likelihood and a learned prior (e.g. a score model).

For a linear Gaussian observation model,

\[
y=Ax+\epsilon,\qquad \epsilon\sim \mathcal N(0,\Sigma),
\]

the data curvature is

\[
F_D = -\nabla_x^2\log p(y\mid x)=A^\top\Sigma^{-1}A.
\]

The local prior curvature is

\[
F_P = -\nabla_x^2\log p(x).
\]

For a differentiable score model \(s_\theta(x)\approx\nabla_x\log p(x)\), estimate \(F_P\) using the negative Jacobian of \(s_\theta\).

**Important:** These are local negative Hessians (observed curvature), not necessarily positive-semidefinite Fisher information matrices. Learned-prior curvature can be negative in some directions. Do not turn arbitrary signed curvatures into percentages of information.

---

## 3. Phase A — Reproduce the pixel-level baseline

**Purpose:** Establish a trustworthy baseline before introducing feature-level attribution.

### Tasks

- [ ] Select an initial galaxy dataset and small image size (e.g. isolated HSC galaxy models; begin at 32–64 pixels per side if training cost matters).
- [ ] Obtain a working reconstruction with a differentiable likelihood and learned prior; prioritize using existing code/checkpoints if available.
- [ ] Implement or reproduce the Sampson–Melchior diagonal curvature estimator using Hessian-vector products (HVPs) and Hutchinson probes.
- [ ] Compute separately:
  - [ ] the data-curvature diagonal;
  - [ ] the prior-curvature diagonal;
  - [ ] their difference \(\Delta F=\operatorname{diag}(F_P)-\operatorname{diag}(F_D)\).
- [ ] Reproduce a known controlled case: remove measurements from half the image and test whether the missing region becomes prior-dominated.
- [ ] Repeat over multiple sources and noise realizations.

### Deliverables

- Reproducible reconstruction and attribution code.
- A baseline figure: **observation → reconstruction → data curvature → prior curvature → pixel hallucination score**.
- Sanity checks on curvature estimates, including probe-count stability and comparison to an exact Hessian on tiny images.

**Exit criterion:** The missing-data test behaves as expected, and its result is reasonably stable across probe seeds.

---

## 4. Phase B — Choose an interpretable feature space

Start with **5–10 features** rather than a large neural feature map.

| Feature | Why it is useful | Example degradation |
|---|---|---|
| Total flux | Overall source strength | Reduced S/N, masking |
| Half-light radius | Source size | Outer-region masking, blur |
| Ellipticity | Global shape | Directional loss of information |
| Position angle | Orientation | Anisotropic measurement coverage |
| Concentration | Central vs outer light | Outer-region masking |
| Asymmetry | Non-axisymmetric structure | S/N loss, selective masking |
| High-frequency power | Fine-scale morphology | Blur, Fourier-mode removal |

Prefer smooth/differentiable definitions or approximations for curvature computations. Keep conventional morphology estimators for evaluation even when they are not differentiable.

### Two distinct mathematical routes

**Route 1 — Features extracted from an image (lowest implementation cost).** Define

\[
z=\phi(x)\in\mathbb R^m.
\]

For feature \(\phi_k\), study image-space directions related to its gradient, such as

\[
v_k=\frac{\nabla_x\phi_k(\hat x)}{\|\nabla_x\phi_k(\hat x)\|}.
\]

A directional curvature comparison \(v_k^\top F_Dv_k\) vs \(v_k^\top F_Pv_k\) is an initial **local sensitivity diagnostic**, *not* automatically an exact Fisher information decomposition of the marginal feature \(z_k\). Correlated features and nuisance directions can change the interpretation.

**Route 2 — Explicit low-dimensional reconstruction model (more powerful, but stronger assumptions).** Parameterize image variations as \(x=g(z)\) with \(m\ll N_{\rm pixels}\). Then evaluate the likelihood \(p(y\mid g(z))\), and define an appropriate **prior density on \(z\)**. If \(z\) is a true generative latent, its prior must be specified in latent coordinates; \(p(x=g(z))\) alone is not generally the normalized latent prior. With a consistent density, direct full \(m\times m\) Hessians or posterior inference become feasible.

**Decision:** Begin with Route 1; introduce Route 2 only if a credible decoder/parameterization and feature-space prior can be constructed.

### Deliverables

- A tested feature extractor with units and definitions.
- Feature distributions on clean/reference images.
- A documented choice of which feature-space formulation is mathematically valid for each proposed diagnostic.

---

## 5. Phase C — Quantify prior and data dominance

Begin with **one primary diagnostic** and **one independent validation diagnostic**; avoid implementing every candidate at once.

### C1. Local directional curvature — primary baseline extension

For a feature direction \(v_k\), compute

\[
I_{D,k}=v_k^\top F_Dv_k,\qquad I_{P,k}=v_k^\top F_Pv_k.
\]

When both terms are nonnegative and their sum is sufficiently large, use

\[
R_k=\frac{I_{P,k}}{I_{P,k}+I_{D,k}},\qquad
D_k=\frac{I_{P,k}-I_{D,k}}{I_{P,k}+I_{D,k}}.
\]

- \(R_k\to0\): locally data-dominated.
- \(R_k\to1\): locally prior-dominated.
- Report \(I_{P,k}+I_{D,k}\) as **total local curvature/support** so that an almost unconstrained feature is not mistaken for a confidently determined one.
- Flag negative-curvature, near-zero-denominator, and numerically unstable cases instead of forcing a fraction into \([0,1]\).

For explicit low-dimensional models, evaluate **full** feature-space curvature matrices and their cross-feature correlations; optionally study generalized eigenvalues if the matrices are positive definite on the selected subspace.

### C2. Observation/prior interventions — independent test

Reconstruct the same source under controlled modifications:

1. Hold the prior fixed; alter the observation operator, noise, or mask.
2. Hold the observation fixed; alter prior strength or switch to a second plausible prior (if available).
3. Measure how each reconstructed feature changes.

Interpret strong changes under prior interventions as **prior sensitivity**, not as a literal probability that the feature was invented. Perturbation magnitudes must be standardized or otherwise compared on a justified scale.

### C3. Optional global diagnostics (only if tractable)

- **Feature posterior/prior information gain:**
  \[
  D_{\rm KL}\!\left[p(z_k\mid y)\,\Vert\,p(z_k)\right].
  \]
  Here the distributions should be induced consistently from the image posterior/prior (or from a correctly specified latent model). KL indicates how much the observation changes the feature distribution; it is **not** a percent split between data and prior.
- **Profile likelihood:** Study how the best possible data fit changes as a particular feature is fixed to different values, optimizing over other image variables.
- **Posterior sampling:** Push image-posterior samples through \(\phi\), or sample a valid small-dimensional posterior directly. Use this as a check on local-Hessian interpretations.

### Deliverables

- Per-source table of feature scores and stability estimates.
- At least one comparison between curvature attribution and an independent intervention-based diagnostic.

---

## 6. Phase D — Controlled astronomical experiments

### Experiment 1: Progressive masking (priority 1)

Mask a source at several levels (e.g. 0%, 25%, 50%, 75%), using both contiguous and feature-targeted masks. Reconstruct at each level and track every feature's attribution. Different features should lose support at different rates.

### Experiment 2: Noise / SNR sweep (priority 2)

Increase measurement noise with the same underlying source. Examine the relationship between S/N and feature-level attribution; test stability over repeated noise draws.

### Experiment 3: Spatial-frequency degradation (priority 3)

Remove selected Fourier modes or blur the observation. Compare large-scale and small-scale feature provenance. This is an especially natural bridge to future radio-interferometric experiments.

### Grounded identifiability checks

For linear \(A\), exploit whether a test direction \(v\) is visible to the measurements:

\[
\|\Sigma^{-1/2}Av\|^2=v^\top F_Dv.
\]

A direction with \(Av=0\) is unobservable by the data. This gives a useful reference for controlled experiments. An entire nonlinear morphological feature may mix visible and invisible directions, so degradation fraction alone is **not** ground-truth feature provenance.

### Dataset and split safeguards

- Use a held-out set of sources for evaluation.
- Record masking/noise generation seeds and noise covariance.
- When a clean reference is available, separately evaluate **feature reconstruction error** versus **feature provenance**.
- Do not equate visual plausibility with observational support.

---

## 7. Evaluation and success criteria

### Main evaluation questions

1. Does estimated data support decrease when a feature-relevant measurement direction is removed?
2. Can the method distinguish prior dominance from simply having low total support?
3. Are the estimates stable across noise draws, Hutchinson seeds, and reasonable prior changes?
4. What does feature attribution reveal beyond the original pixel map?
5. Do local curvature and interventions tell a consistent story? Where do they disagree?

### Candidate quantitative measures

- Rank correlation between known degradation severity and inferred data support.
- Direction-level discrimination of visible vs null-space test directions (where possible).
- Stability / confidence intervals across noise realizations.
- Agreement of diagnostic rankings across methods.
- Feature reconstruction error relative to clean reference images.

Avoid claiming calibrated probabilities of being “data-generated” or “prior-generated” without a probabilistic definition and validation.

---

## 8. Minimum publishable BASP proof of concept

**Must have:**

- [ ] Validated reproduction of the pixel-level Sampson–Melchior baseline.
- [ ] A small, interpretable set of galaxy morphology features.
- [ ] A defensible feature-level attribution diagnostic.
- [ ] Controlled masking and at least one S/N experiment.
- [ ] A quantitative test that feature-level attribution responds to controlled removal of relevant information.
- [ ] A direct comparison showing what feature-level analysis adds to the pixel-level baseline.

**Nice to have:** A second diagnostic (interventions), uncertainty intervals, and a Fourier-mode removal experiment.

### Proposed figures

1. **Baseline:** galaxy observation, reconstruction, pixel-level data/prior dominance map.
2. **Main result:** curves of feature-level prior dominance vs masking or S/N for several morphological features, with uncertainty bands.
3. **Interpretability:** two sources with similar-looking reconstructions but contrasting observational support for a selected feature.
4. *(Optional)* Comparison of curvature and intervention-based rankings.

### BASP positioning

- **Title:** *Feature-Level Prior–Data Attribution in Astronomical Image Reconstruction*.
- **Session:** *Physics-informed Machine Learning for Astronomy and Astrophysics*.
- **Core message:** The original score asks *where* learned priors dominate. We ask *which astronomical features* are actually constrained by the measurements.
- **Abstract:** State the study as an investigation unless quantitative results are available; add measured findings when they exist.

---

## 9. Suggested execution sequence

| Sprint | Focus | Checkpoint |
|---|---|---|
| 1 | Dataset, forward model, reconstruct with learned prior | A working reference reconstruction |
| 2 | Pixel hallucination-score reproduction | Controlled masked/unmasked baseline figure |
| 3 | 5–10 feature definitions and differentiability tests | Feature measurements validated on reference images |
| 4 | Local curvature and first feature-level dominance plots | Masking response for at least three features |
| 5 | Noise sweep and intervention validation | Quantitative evaluation and stability tests |
| 6 | Consolidate figures, limitations, BASP material | Reproducible proof-of-concept package |

These are **relative sprints**, not fixed dates. If the learned prior is expensive to reproduce, simplify the initial imaging task instead of allowing model training to consume the entire project.

---

## 10. Research risks and decision points

| Risk | Mitigation |
|---|---|
| Learned prior cannot be reproduced easily | Start with public weights/code or a simple analytic / lightweight learned prior; keep the mathematical validation separate |
| A feature extractor is nondifferentiable | Use smooth proxies for local curvature and conventional estimators for evaluation |
| Feature-space Hessian is misinterpreted | Distinguish directional image curvature from a genuine feature-space density; use an explicit latent model for the latter |
| Prior curvature is indefinite | Report signed curvature or use an explicitly justified PSD approximation; do not present misleading percentages |
| Reduced features lose important spatial information | Compare morphological features to a small spatial basis / shapelets later |
| Perturbation scores depend on arbitrary intervention strength | Predefine and normalize intervention scales; conduct sensitivity analyses |
| All features move together under degradation | Design selective corruptions and report cross-feature correlations |

**Go/no-go question after the first experiments:** Does a feature-level measure provide stable, interpretable information about observational support that the pixel map does not? If not, investigate the reason before adding more complex diagnostics.

---

## 11. Future ICML expansion — out of scope for BASP

Potential contributions for a subsequent general-methodology paper:

- A formal definition of **feature provenance / identifiability** in generative inverse problems.
- Basis-aware or invariant attribution under latent reparameterization.
- Full-matrix and generalized-eigenvector analyses in learned representations.
- Nonlocal methods based on posterior sampling, profile likelihoods, and information gain.
- A validated benchmark with controlled null spaces and known measurement identifiability.
- Tests across different inverse problems and priors (inpainting, deblurring, super-resolution, radio interferometry).
- Calibration and robustness studies that distinguish prior sensitivity from factual hallucination.

The astronomy proof of concept should identify which of these questions is empirically consequential, rather than attempting all of them immediately.

---

## Reference

Sampson, M. L. & Melchior, P. (2023). *Spotting Hallucinations in Inverse Problems with Data-Driven Priors.* https://arxiv.org/abs/2306.13272
