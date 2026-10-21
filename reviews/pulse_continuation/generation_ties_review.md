# Independent CPU review: targeted greedy divergence replay

The diagnostic passes. All four cases match the declared selection: every first divergence in the original exact-full-donor or hybrid mean-cache conditions. The audit verifies 892 forwards, 24 result rows, 48 full saved logit vectors, unchanged model hashes/version counters and the original generation-artifact hash. Repeated logits are bitwise equal in every condition; all original step predictions reproduce. CPU recomputation of clean-to-condition KL differs from the GPU result by at most 3.03e−9. Audit RSS was about 494 MiB and no model/GPU was loaded.

The fixed clean histories, cache positions and selected step alignment are correct. These are exploratory diagnostics selected from observed divergences, not a new independent evaluation or an estimate of the prevalence of ties.

| Triggering case | Clean margin | Condition margin | KL | Interpretation |
|---|---:|---:|---:|---|
| Wiki 3, hybrid, step 29 | 0 | 0.03125 | 0.000155667 | Clean tie between “ five” and “ six” changes choice |
| Plato 0, exact full donor, step 6 | 0 | 0.0078125 | 0.000006733 | Clean tie between “ been” and “ said” changes choice |
| Plato 2, hybrid, step 7 | 0.1640625 | 0.046875 | 0.002239862 | Non-tied clean preference changes from “ plurality” to “ concord” |
| Python 3, hybrid, step 28 | 0.03125 | 0 | 0.000120232 | Condition creates a tie between “self” and newline |

For the exact full-donor control, the maximum absolute logit difference is 0.0078125. Repeated determinism and the zero clean margin support numerical sensitivity of greedy decoding in this case; they do not establish that every implementation or precision would choose the same token.

The hybrid-cache errors cannot all be dismissed as FP16 ties. The Plato 2 case has a substantial clean margin, a maximum logit change of 0.208984375 and a larger distribution shift. The approximate cache can change meaningful model preferences even while average matched-history KL remains small. Neither different wording nor this preference shift alone measures output quality.

Unlike the original generation aggregate audit, these selected steps retain full logits, permitting independent distribution-level KL verification. This resolves that audit limitation for these four diagnostics only. No implementation or numerical discrepancy was found.
