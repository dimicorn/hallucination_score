# BASP abstract

Generative priors can improve ill-posed astronomical image reconstruction, but they may also introduce plausible structures that are only weakly constrained by the observations. Existing hallucination diagnostics compare the information contributed by the likelihood and the learned prior at the pixel level, identifying regions that are data- or prior-dominated.

We investigate a feature-level generalization of this idea. Instead of asking only where a reconstruction is prior-dominated, we aim to determine whether individual morphological properties of a reconstructed source are supported primarily by the data or by the prior. By projecting reconstructions into a low-dimensional feature space, we can use richer diagnostics than pixel-wise curvature alone, including full local curvature, posterior information gain, and sensitivity to controlled perturbations of the observations.

As a proof of concept, we study galaxy reconstruction under masking and varying noise levels, where the available observational information can be systematically controlled. We test whether inferred prior dominance increases when information relevant to specific morphological features is removed. This provides a path toward interpretable feature-level provenance and more trustworthy use of generative priors in astronomical imaging.
