# Independent CPU review: fixed source-attention gates

The completed run passes: all 640 observations, 80 summaries and 80 contrasts, including paired bootstrap intervals, replay from saved token metrics. Script/calibration/token hashes and unchanged model integrity match the manifest's 656 forwards. CPU peak RSS was about 680 MiB; no model/GPU was loaded and experiment artifacts were not modified.

## Reconstruction and calibration

Hook ordering captures the patched source keys. The head layout, dimension-128 scaling, causal masking, clipping and final FP16 cast after RoPE match native attention. Conditional `u` is computed by native SDPA with source zero excluded for queries 1 onward, while the oracle source mass is reconstructed with full causal source eligibility. Source values are fixed Wiki means. Position zero is untouched by the output recombination.

Eight old Wiki and eight old Hume development windows are exactly recovered from saved tokens and do not overlap the evaluation windows. Their masses come from the native fixed-mean-KV world, before evaluation. Saved global means and four position-bin means reproduce exactly. Reconstructing the first full Q/K snapshot per corpus on CPU gives a maximum source-mass discrepancy of 6.56e−7 versus the saved GPU values. The remaining calibration masses are retained and their aggregation is independently verified, but full Q/K snapshots exist only for the first calibration window per corpus.

Synthetic CPU tests verify all fixed-gate formulas, bin coverage and position-zero localization. Zero gate exactly leaves the conditional attention output unchanged. The bias control deliberately computes `u + a*v`, without the `(1−a)` attenuation. All evaluated clean/deleted states produce identical saved later losses, predictions and KL for each gate variant. Native clean, mean-KV and source-removal anchors match the previous quiet-anchor run exactly.

The oracle is close rather than bitwise exact to native mean-KV attention. Across corpus groups, absolute mean NLL drift is at most 8.85e−5 nats; the worst individual-window drift is 0.000327906 nats. Maximum token loss difference is 0.0434399 nats, with 18 changed argmax predictions among 16,320 scored tokens. These numerical errors are much smaller than the fixed-gate effects, while setting a practical limit on tiny differences.

## Verified results and interpretation

| Corpus | Native mean K/V | Wiki global gate | Wiki four-bin gate | Hume global gate | Hume four-bin gate | Source removed |
|---|---:|---:|---:|---:|---:|---:|
| Wiki | 1.000240 | 1.980134 | 1.942158 | 1.979634 | 2.128530 | 2.526719 |
| Hume | 0.999811 | 1.575024 | 1.543296 | 1.065032 | 1.053223 | 1.030665 |
| Plato | 0.999842 | 1.535610 | 1.501147 | 1.060087 | 1.049126 | 1.034712 |
| Python | 0.999705 | 1.672484 | 1.615230 | 1.137059 | 1.117826 | 1.159391 |

Entries are exp(mean paired NLL difference from native clean), scoring 510 predictions per window. Oracle recombination remains near the native mean-KV baseline; the tested arithmetic-mean gates do not preserve that fidelity. Four coarse query-position bins provide limited improvement and are not uniformly better (Hume-calibrated bins worsen Wiki relative to Hume global gates).

Gate calibration transfers unevenly: Hume-calibrated gates outperform Wiki-calibrated gates on Hume, Plato and Python, despite all conditions retaining the same Wiki mean values. These outcomes support sensitivity to the distribution of source weights and their coupling to recipient context. They do not prove that every possible optimized constant-gate family must fail. Only empirical arithmetic means and four prespecified bins were tested.

“Query-independent” describes the replaced source gate `a`, not the conditional mixture `u`, which still depends on recipient queries, nonzero keys and values. The complete computation remains context-dependent. The observed losses do not uniquely identify a query feature, a minimal gate subspace or a natural gate/value compensation mechanism.

Bias-only performance is not uniformly worse than attenuation-plus-bias: with Wiki calibration it is worse on Wiki (3.57006× versus 1.98013×), but better on Hume, Plato and Python than the corresponding Wiki global gate. Thus do not infer a universal benefit of attenuation from these comparisons. Gate one replaces all later attention output with the constant values and is catastrophically poor; it is an extreme control, not a competitive simplification.

The panel reuses eight fixed evaluation windows per corpus, has no new fitting/selection on outcomes, and conditions its pointwise intervals on the frozen calibrations. No performance-speed or broad generality claim is warranted. No implementation or numerical discrepancy remains.
