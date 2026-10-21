# Independent gate-subspace review

CPU-only audit completed; no model or GPU loaded and no experiment files edited. `gate_subspaces_audit.py` and its JSON record the replay. Peak RSS 549,720 KiB. No blocking implementation or numerical defect found.

All 544 raw records, 68 summary groups, 68 paired contrasts, and 56 held-out spectrum records reproduce. The 16 development windows supply 8,176 query vectors per layer; centered covariance reproduces exactly. Stored PCA bases satisfy orthonormality within 6.06e-8 and the eigen equation within 1.01e-8. Both seeded nested Gaussian QR controls reproduce exactly. Development fitting precedes evaluation; the reused evaluation panel remains exploratory with respect to this ongoing investigation.

Native-clean, mean-KV, and rank-16 results exactly match the corresponding prior global-gate controls. Rank-4 clean/deleted loss, predictions, and KL are identical. Scoring uniformly excludes predictor 0 and uses 510 targets/window. Rank 16 explicitly bypasses projection and uses the current-recipient oracle; it is not a model-level full-rank projection test. Separate CPU full-rank projection reconstruction error is at most 2.98e-7.

| PCA rank | Wiki PPL ratio | Hume | Plato | Python |
|---|---:|---:|---:|---:|
| 0 | 1.91743 | 1.17411 | 1.15589 | 1.23974 |
| 1 | 1.53511 | 1.04348 | 1.04428 | 1.08168 |
| 2 | 1.31820 | 1.02791 | 1.02738 | 1.04315 |
| 4 | 1.16827 | 1.01246 | 1.01365 | 1.02277 |
| 8 | 1.05370 | 1.00550 | 1.00753 | 1.00657 |
| 12 | 1.01947 | 1.00152 | 1.00370 | 1.00064 |

Ratios use the native-clean baseline. Wiki rank-4 agreement is 80.51%, rank-8 87.30%, and rank-12 91.91%; rank-12 excess NLL is 0.01929 nats [0.00610, 0.03670]. Thus near-baseline mean loss outside Wiki does not imply exact recovery or a universally low-dimensional mechanism. All 16 comparisons of the two random bases at ranks 4/8 against matched PCA have positive paired NLL intervals. This supports the importance of the learned directions within this approximation family, not an optimal or minimum causal rank.

Clamping is a real nonlinear step and can move gates outside the nominal affine subspace. In Wiki, 1.29% of rank-4 projected scalar gates lie outside [0,1] before clamping. Unclamped-minus-clamped excess loss is 0.002344/0.001562/0.000694 nats for ranks 2/4/8, with positive paired intervals. Most analogous Hume/Plato differences cross zero. Clamping improves some results slightly but does not explain the substantial rank trend. Dynamic recipient gate vectors were not retained: bounds are structurally checked and aggregated, not independently reconstructed per causal forward.

Held-out descriptive spectra concern **native mean-KV-world** gates, not the gates arising dynamically during altered forward passes. Both reported centering conventions replay: own-mean centering measures within-panel covariance; development-mean centering additionally includes mean shifts. Wiki own-centered PC1 shares range 29.1–66.8% across layers and 90%-energy ranks range 6–10; rank-4 development directions retain 67.3–84.5% of own-centered energy. These spectra are distinct from the earlier static 14-by-2048 source-template SVD and do not establish a global causal dimensionality.

Intervals are paired window bootstraps over eight windows/corpus, not independent-token intervals or broad model/corpus generalization guarantees. Recipient conditional attention mixtures remain query-dependent even when gate projections use frozen bases.
