# Analytic template and learned-residual review

CPU-only audit passed for all 672 observations, 84 summaries, and 168 paired contrast
intervals. No model/GPU use or experiment edits; peak RSS approximately 506MiB.
Raw losses, KL, agreement, baseline means, hashes, score ranges, and model integrity
passed. Every template produces identical saved loss/prediction/KL when started
from clean versus position-zero-deleted source states, on all 24 evaluation windows.

## Fitting geometry

The analytic axis is the zero-epsilon LayerNorm limit of one dominating negative
coordinate: coordinate 1764 is −√2047, all others are 1/√2047. It has zero sum and
unit feature variance within float precision. The matched axis scales that vector
separately at each layer to the development mean's coordinate 1764. Subtracting it
defines a learned residual. All stored means, axes, residuals, rank truncations,
shared-residual and wrong-axis controls, and final FP16 templates replay exactly.

The SVD matrix is 14×2048: one static development-mean residual vector for each of
layers 2–15. It is not a token-by-feature downstream perturbation covariance.
The primary rank approximations use the uncentered residual SVD; the separately
saved centered-across-layer SVD subtracts the mean across 14 layer rows. Both
factorizations reconstruct their stated matrices. Uncentered residual energy
retained at ranks 1/2/4/8 is 88.456%/96.629%/99.277%/99.936%.

“Rank2 + axis” therefore means a rank-two residual with layer-specific coefficients
plus a separately fitted rank-at-most-one axis component. It can have total matrix
rank three and is not a two-dimensional description of the network's dynamic
computation. Its fit includes 14 axis coefficients, residual basis vectors, and
their per-layer coefficients.

Both seeded random controls have coordinate 1764 exactly zero, zero mean across
remaining coordinates, and per-layer norm matched to the full learned residual
before FP16 casting (relative norm mismatch below 1.8e−7). They share the coordinate-
matched axis base. They do not specifically match the smaller rank-two residual's
norm; compare them with that common base and state their actual matching target.

## Data and interpretation

Every declared prior exclusion span was independently reconstructed and checked.
New inputs are length 512, mutually nonoverlapping within corpus, disjoint from
prior calibration/development/evaluation and continuation spans, and match their
saved token hashes. Plato's original source hash and interval exclusions match.
Some spans immediately adjoin earlier spans; they are unused tokens, not new
independent documents. Calibration remains the prior development split and rank
choices were fixed in advance. Intervals remain conditional and pointwise.

WikiText perplexity ratios are 1.38502 for the pure ideal axis, 1.38012 for the
coordinate-matched axis, 1.02433 for rank1+axis, 1.00517 for rank2+axis, and 0.99936
for the full Wiki mean. Rank2 argmax agreement is 97.35% with KL 0.00327; the full
mean reaches 99.14% with KL0.000492. Thus low-rank static residual structure helps
substantially, but rank2 is not exact recovery. Full means themselves are also
approximate. Hume/Plato axis-only loss penalties are much smaller, motivating the
planned fixed-source-identity comparison.

Both random additions increase mean NLL relative to their coordinate-matched axis
base in all three corpora. One Hume seed's paired interval crosses zero, so report
“neither random draw improved average loss,” not statistically significant harm
for every random control. Two draws do not characterize arbitrary random subspaces.

Clean/deleted equality follows from fixing every outgoing source attention-input
vector at layers 2–15 and omitting predictor 0. It is a successful causal-cut control,
not evidence of general invariance to deleting the scalar elsewhere or to changing
other positions. The previous downstream covariance findings remain compatible
with this compact static source template.

No blocking correctness issue found. Preserve the static/dynamic SVD distinction,
total-template rank qualification, nonexact agreement, and calibration/selection
scope in subsequent claims.
