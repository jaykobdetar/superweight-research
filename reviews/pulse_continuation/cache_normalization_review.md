# Cache bypass and cross-layer normalization review

CPU-only audits completed with peak memory below 525MiB. No model loading, GPU
use, or experiment edits. All recorded raw metrics and paired bootstrap summaries
replayed: cache 2,592 observations/324 conditions; normalization 816/102. Model
hashes and parameter versions are unchanged.

## Actual prefix-cache bypass

The model receives tokens 1: only, with cache positions 1..n−1 and a one-token
cache at every layer. Cached keys/values are clipped and packed correctly as
[batch 1, heads 16, sequence 1, features 128]. Position-zero RoPE is identity. The
full-donor cache is taken from the same full forward; the independent dictionary
cache is obtained from a separate single-token forward without suffix input.
The source-deletion hook remains clean throughout this batch.

The 128/512/1024 sequences are nested crops with checked input hashes. Primary
scoring excludes predictor 0 uniformly; full scoring includes its logits from the
exact-token dictionary. Every bypass variant reuses that identical first prediction,
including mean and zero-cache variants. Therefore the full panel is a hybrid using
exact identity logits for its first prediction, not an entirely mean-cache model.
All primary arrays equal the corresponding full arrays with their first element
removed. On `The` inputs, fixed-`The` cache equals exact-single-token cache exactly.

Full-donor bypass matches clean primary losses and predictions exactly in all 18
corpus/identity/length cells. Independently generated single-token caches produce
small nonzero differences, providing a matched cached-path numerical control.
Use that reference when interpreting tiny mean-cache changes. Scored clean is
full-input `use_cache=False`; `full_cache` appears in timing only, so predictive
equivalence of the latter was not directly saved or checked.

Development means are fitted before evaluation from clipped native caches.
Hybrid caches retain exact identity layers 0/1 and use means for layers 2–15.
This differs from averaging preclip projections and clipping afterward in the
mediation batch. Actual cache results closely preserve behavior, but neither
mean-cache nor hybrid replacement is exactly distribution-equivalent. Controls
include a fixed-token cache and zero cache, with strongly corpus-dependent effects.

Timing interleaves four modes for 27 rounds, discards three warmup rounds, and
includes CPU→GPU cache copies/construction. Mean measured times are 50.061 ms for
full standard caching, 52.599 ms for exact bypass, and 52.656 ms for mean bypass.
Thus bypass is 5.07% and5.18% slower than full caching in this implementation.
Dictionary creation, calibration, and loss/KL scoring are excluded. This is one
prompt length on one device; no speedup or broader practical acceleration claim
is supported. KV storage is 128 KiB per identity, excluding cached logits and
dictionary overhead.

## Cross-layer normalization

All 816 observations score 510 predictors 1–510, matching the mediation baseline.
Full normalized-source donor swaps at layers 2–15 reproduce donor loss, prediction,
and KL exactly in both directions. Layer 2 full-vector swaps exactly match the
earlier all-head layer 2 KV0 cuts. Source sigma recomputation, synthetic hook
localization, raw scoring, and all 102 summaries/204 contrasts passed.

Coordinate/complement/full swaps act on the already-normalized source vector.
Scale swaps retain the current recipient's centered source input but divide by
the donor population-standard-deviation term, including epsilon. Only attention
input norms at source 0 are changed; residual skips, MLP norms, later positions,
and final output normalization retain recipient computation. The full attention
input cut is sufficient because it controls all outgoing source K/V paths.

All-layer complement rescue leaves clean-relative mean NLL +0.04114/+0.01647/
+0.01361 nats for WikiText/Hume/Plato. However, argmax agreement is only 89.83%/
94.31%/93.85%; small likelihood differences must not be described as exact output
restoration. Coordinate-only rescue worsens deletion. Scale-only rescue is much
less complete on WikiText (+1.59279 nats) than Hume (+0.07821) or Plato (+0.07271).
These support a substantial normalization-scale route, not sufficiency of scale
alone or elimination of coordinate-dependent content effects.

Development-mean full normalized-source replacement can largely repair deletion
without a held-out donor, reaching approximately 99.39–99.78% argmax agreement for
Wiki-fitted constants across these corpora. This remains a narrowly scoped source 0
replacement at attention inputs and does not replace general content processing.

No blocking defect found. Preserve exploratory/reused-evaluation scope, conditional
pointwise uncertainty, numerical-control comparisons, and the absence of a speedup.
