# Scoped review of Report11 draft methods, attribution and figures

This review covered `report11_methods.md`, `report11_prior_findings.md`, the three initial figures and their plotting sources. It used local frozen protocols, saved numerical-review evidence and primary-paper pages. No pretrained model or GPU was used. The V2 failure arrived during review; this is not approval of a final report or of a completed gate experiment.

## Scientific wording and changes applied

The draft correctly limits the study to an analogous GPT-2 allocation circuit after the selected single-weight transfer screen was negative. It separates current local mixture preservation from a fixed donor trajectory, makes source-query identity relative to zero-V P rather than native C, treats covariance as descriptive, and does not infer one residual-stream mechanism from low rank. The exposed Wiki passage and conditional n3/n4 bootstrap interpretation are appropriately explicit.

I applied only these draft prose clarifications, then returned ownership to root:

- Define `b` directly from score log-sum-exp minus source score; `a=sigmoid(b)` is the exact-arithmetic identity, not a numerically evaluated logit of rounded mass.
- State centering by calibration alpha and restoration of alpha+log(q); distinguish fixed-seed QR controls from fitted PCA.
- Name saved principal-angle cosines accurately and clarify disjoint calibration passages without implying all sources are unrelated corpora.
- Identify the numerical-drift figure as pre-final-normalization block-output residuals, with a short-reference norm denominator. It is not the V2 symmetric final-hidden guard plot.
- Separate established literature from this project's conditional-regrowth/KV work, and qualify the prior native sink-output reduction by zero or negligible source value.

No frozen experiment file was changed. The draft's revision section remains stale following the new V2 failure and is explicitly returned to root for rewriting. It must not imply V2 completed or passed.

## Figure assessment

All three PNGs were visually inspected: text is legible, axes and layer indices are visible, and there is no clipping or overlap. Plotting sources read the preserved calibration spectra and native-diagnostic comparison fields. The spectrum title correctly avoids causal-necessity claims. The numerical plot explicitly marks one native input and separates block-output absolute and relative error; no irrelevant final-hidden threshold line is present.

The gate schematic needs a small mathematical correction before release: show centering of `r=b-log(q)` and restoring `alpha+log(q)` after retain/remove projection before the sigmoid. Its calibration box should separately label fitted means/PCA and fixed-seed random QR, rather than implying QR was learned from those texts. Root owns the diagram source and figures; I sent these concrete corrections without editing them. A useful caption addition is that the spectrum comes from zero-V calibration, not unmodified native C. No figure is evidence that the gate experiment completed.

## Primary attribution checks

Fu et al. explicitly present the implicit gate and sigmoid score expression in §3.1/Appendix B.2; their Table3 also distinguishes all-head from selective source-value zeroing. The report appropriately avoids treating this algebra or zeroing as new. [Attention Sink Forges Native MoE](https://arxiv.org/html/2602.01203v1).

Qiu et al. study head-specific/query-dependent post-attention sigmoid gating. HyGA separately introduces low-rank learned gates and a gate using concatenated head outputs. The draft accurately treats those as precedents rather than as experiments reproduced here. [Gated Attention](https://arxiv.org/html/2505.06708v1), [Hybrid Gated Attention](https://arxiv.org/html/2608.11805v1).

The massive-activation paper decomposes source-related attention contributions and tests a zero-key/zero-value alternative; the GPT-2 paper explicitly studies query-bias/key alignment and targeted interventions. Kernel-PCA work concerns a different key-derived feature space. Those distinctions support the cautious comparison but cannot establish priority. The previously inaccessible SSRN full text remains unresolved; this review did not claim to inspect it. [Massive Activations](https://arxiv.org/html/2402.17762v2), [GPT-2 circuit account](https://arxiv.org/html/2604.14722v1), [Kernel-PCA analysis](https://arxiv.org/html/2406.13762v1).

## Current disposition

The methods/attribution edits above are complete. Diagram corrections and the failed-V2/fixed-shape rewrite belong to root. A final integrated report review is still required after new scientific results, if any, have passed independent audit. The narrower fixed-shape proposal has a separate author policy review; it is not evidence that either failed protocol has passed.
