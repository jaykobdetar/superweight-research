# Independent pooled gate-covariance review

Passed. An independent explicit layer/head coordinate fill and FP64 covariance eigendecomposition reproduce104 saved comparisons, including both spaces, segment covariances/eigenvalues, and leave-one-passage-out PC1 fractions/principal-angle cosines. Maximum discrepancy9.24e-14. Peak reviewer RSS514836KiB; no model/GPU load.

The layout is8 passages ×510 aligned queries ×96 coordinates, ordered by layers4–11 and heads0–11. Query indices1–510 match the calibration/scored-predictor range; early1–126 and later127–510 correspond to target ranges2–127 and128–511. The “raw b” result still centers every coordinate before covariance; it simply omits subtraction of log(q). Detrended r subtracts log(q) and then centers. Neither calculation standardizes each coordinate's variance.

PC1 explains30.611425% for centered raw b and32.703286% for centered detrended r. Detrended cumulative variance is54.798177% at rank3 and69.196709% at rank6; participation ratio6.987588 and entropy effective rank16.690953 agree. These effective-rank measures have different definitions and are not numbers of mechanisms. Eight separately fitted per-layer PC1 directions are not one global scalar factor. The pooled covariance is not close to rank1 by its own descriptive variance criterion, but that neither proves nor disproves one causal mechanism, nonlinear latent structure, or a successful intervention family.

All vectors remain correlated calibration observations from the original eight passages, not4080 independent samples or heldout confirmation. Same-query concatenation also combines different sequential layer computations. Leave-one-passage-out analysis describes sensitivity of this fixed calibration panel and does not supply population uncertainty. This analysis changes no experimental directions or maps and should not be used to select a new rank/layer on the current evaluation outcomes.

Evidence: `global_gate_covariance_audit.py/.json`; source `analyze_global_gate_covariance.py` SHA256 `d8148dc49009ced01c2237ff323862e9fad56269a09b9794e82183b9ffae9598`.
