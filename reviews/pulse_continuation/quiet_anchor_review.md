# Independent CPU review: global zero-value anchor and source removal

All 832 observations, 104 summary rows and 208 contrasts replay from saved token metrics, including paired intervals. The 832 forwards, hashes, model integrity, mean-cache unpacking and source-mask localization pass. Prior residual-path clean/deleted anchors reproduce exactly, and explicit versus native masks agree exactly. CPU peak RSS was about 537 MiB; no model/GPU was loaded and no experiment artifact was modified.

## Controlled interpretation

The key/value factorial patches only source-position K/V blocks, all heads, at layers 2–15. Fixed means are the previously calibrated means of clipped position-zero native caches. Unpacking preserves head layout and ordinary clipping leaves these bounded constants unchanged; RoPE at position zero is identity. `Native` means recomputed under the current intervention, not clamped to a clean donor at every layer. In particular, zeroing an earlier source value can alter later native source keys through source self-attention and its residual stream.

The source-column mask removes key zero only for queries 1 onward in selected layers; source zero retains self-attention, other causal entries are untouched, and layers 0–1 remain unchanged. Removing a source column renormalizes attention over the remaining keys. Setting its value to zero instead retains its softmax mass, reducing the weighted contribution of other values. These are deliberately different interventions, not equivalent ways to delete the same additive vector.

Both fixed-K/fixed-V conditions and source removal at every layer 2–15 exactly erase dependence on the original scalar-deletion state for saved losses, predictions and KL after predictor zero. This validates their complete source-path cut, without making either condition equivalent to clean performance.

## Verified results

Perplexity ratios relative to matched clean, using 510 scored predictions per window:

| Corpus | Native K, zero V | Mean K, zero V | Zero K, zero V | Source removed at L2–15 | Mean K, mean V |
|---|---:|---:|---:|---:|---:|
| Wiki | 203.939735 | 218.921571 | 16.241574 | 2.526719 | 1.000240 |
| Hume | 1.255903 | 1.286599 | 1.046411 | 1.030665 | 0.999811 |
| Plato | 1.241110 | 1.268648 | 1.040433 | 1.034712 | 0.999842 |
| Python | 3.654477 | 3.800147 | 1.511876 | 1.159391 | 0.999705 |

These are exp(mean paired NLL difference). The zero-V conditions are worse than source removal on every corpus, with the corresponding paired bootstrap intervals above zero. The frozen mean-K comparison is particularly useful: the drastic Wiki difference between zero V and mean V remains when source keys are fixed across both interventions, so it cannot be attributed solely to later native source-key changes.

The tested global zero-value replacements are insufficient to preserve the network's behavior. Nonzero value vectors supply an important contribution beyond pure zero-valued routing in these interventions. This does **not** establish semantic information in those constants: they may provide a bias, calibration signal or other value contribution. It also does not exclude useful zero-value behavior in particular heads/layers, or establish a universally necessary value direction.

Native K with mean V and mean K with native V both preserve clean loss closely, whereas zero K with nonzero V is imperfect (Wiki about 1.447×). This supports a role for both key-dependent weighting and nonzero source values, with interactions. Mean K/mean V is still approximate: Wiki prediction agreement is 99.2892%, not 100%.

Layer removal is nonadditive. On Wiki, removing only L2 gives 1.08043×, removing layers 3–15 gives 5.45436×, and removing all layers 2–15 gives 2.52672×. Thus a larger removal can hurt less; effects must not be summed or converted into additive mechanism fractions. Layer-specific value and compensation tests would help distinguish these possibilities, but this panel does not resolve them.

Eight reused fixed windows per corpus support descriptive paired intervals, not independent-document or universal claims. Python results have wide intervals and should not be treated as equally precise as Wiki. No numerical discrepancy or review blocker remains.
