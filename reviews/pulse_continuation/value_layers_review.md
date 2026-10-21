# Independent CPU review: layerwise source-value dependence with fixed keys

The implementation and raw metrics pass. The audit verifies 1,504 observations in 188 groups, 1,344 forwards, unchanged model hashes/version counters, calibration/token/script hashes, all 44 named prespecified sets and their 39 distinct layer configurations. Memoized aliases have identical losses, predictions and KL. Native, full-mean and full-zero-value anchors reproduce the quiet-anchor run exactly. CPU peak RSS was about 517 MiB; no model/GPU was loaded and no experiment artifacts were modified.

Every layer 2–15 receives fixed mean source keys and explicitly fixed source values (mean or zero). Earlier value changes therefore cannot drift later source K/V interfaces. Recipient queries and other token states remain free to respond, which is part of the intended causal effect. Layers 0–1 remain native. The experiment addresses layer dependence inside this imposed source-interface world, not single-layer necessity or sufficiency in the unmodified scalar-deletion circuit.

The two random controls reproduce exactly from their saved seeds. Each layer/head vector has the corresponding mean-value norm before FP16 rounding; maximum relative norm error after rounding is 7.75e−5. Maximum absolute random component is 1.61133, well below the clipping threshold of 8, so clipping does not confound the norm match. These are two fixed directions, not a well-sampled distribution over alternatives.

## Results and limits

| Corpus | All values zero | Restore only L6 | Restore middle L6–11 | Remove middle L6–11 from all-mean | Keep values L3–15 only |
|---|---:|---:|---:|---:|---:|
| Wiki | 218.921571 | 26.991306 | 3.625340 | 36.949735 | 1.051356 |
| Hume | 1.286599 | 1.179299 | 1.069759 | 1.129305 | 1.004715 |
| Plato | 1.268648 | 1.166400 | 1.055590 | 1.111894 | 1.007807 |
| Python | 3.800147 | 1.878826 | 1.144668 | 1.646916 | 1.004091 |

Entries are exp(mean paired NLL difference from clean), using 510 matched predictions per window. Restoring only L6 has the largest observed single-layer mean benefit across these four fixed panels, but it is far from sufficient. Its apparent ranking is exploratory across the tested layers and should not be described as independently validated localization of an optimal layer.

On Wiki, every single-layer removal from the mean world is much milder than the complete zero-value intervention; the largest mean NLL increase relative to all-mean is about 0.09682 nats (L7), whereas removing the middle band yields roughly 3.61 nats relative to all-mean. Restoring early L2–5 alone gives 84.26×, middle L6–11 alone 3.625× and late L12–15 alone 157.43×. This supports distributed, interacting value contributions with substantial middle-layer influence in these clamped worlds. It does not establish additive attribution, an exact compensation mechanism or a sufficient sparse set.

The two norm-matched random-value worlds yield Wiki ratios 210.60× and 178.12×. They fail to preserve clean performance despite retaining vector magnitudes, supporting directional specificity beyond magnitude alone. Do not claim neither improves over zero: the second seed improves NLL versus zero by 0.206256 nats (paired 95% interval 0.122832–0.289140 improvement). The first seed's small improvement interval crosses zero. These results also do not establish semantic content in the useful value directions.

Near-clean mean loss after retaining most layers is not exact output recovery. For example, keeping L3–15 values on Wiki has 1.05136× loss ratio but only 90.37% prediction agreement. All comparisons reuse eight fixed windows per corpus with no new fitting; pointwise bootstrap intervals do not adjust for the many examined layer sets.

All 188 summary rows and 188 contrasts, including paired bootstrap intervals, now replay against the saved token metrics. The final status is recorded in `value_layers_audit.json`. No implementation or numerical discrepancy remains.
