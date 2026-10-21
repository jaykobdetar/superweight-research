# Independent native-prefix diagnostic review

**Approved for the fixed18-forward diagnostic at the hashes below.** This does not approve a gate-study amendment or change the earlier failed run’s status. No pretrained model or GPU computation was used in this review.

The panel is coherent:2 hook-free default references,7 cases each for default and forced-math SDPA, and2 original frozen-study C hook paths. All inputs share tokens0–127. Rotated/constant suffix tests preserve512 shape and native masking; padded512 deliberately changes the mask and is analyzed separately. Observer hooks returnNone; hook-free and frozen-study comparisons test their equivalence rather than assuming it. Original inputs, source hashes, selection and saved native C token identities are checked before model load.

Explicit-mask safety now checks every future key j>q for the first128 queries, including keys within that prefix. Prefix-only FP64 Q/K/V reconstruction is valid because these queries cannot legally use later keys. Actual context is captured before the attention output projection. Native implicit masking is additionally tested by fixed-shape changed-suffix comparisons. The code preserves precision settings, records them inside each backend context and asserts restoration. Backend-specific profiler operators are distinguished from generic SDPA dispatch; missing specific events remain unobserved instead of being guessed.

Scoring preserves the original FP32 chunk32 arithmetic and126 shared targets2–127. Full128 logits/hidden states are retained; sourceq0 and unscored query127 are separately compared. FP64 KL direction is reference||trial; top-two margins and mismatch positions are recorded. The25 comparisons,276 layer-pair score records and192 local context reconstructions match the18-case design. Failed native C alignment uses `failure_state.prefix_pairs.C`, never the P512 `current_hidden` holder.

Cases are saved separately and analysis loads only bounded case pairs. Completed outputs and active partial capture survive runtime failure. No drift threshold is used to hide outcomes or automatically trigger extra inference. Parameter hashes/version counters, frozen source identities and precision settings must close successfully. Timings include profiling/observation and cannot support speed claims.

Independent evidence: `native_prefix_diagnostic_synthetic.py/.json` ran all18 cases on a newly initialized CPU Transformers GPT-2 model (12 layers,12 heads,width768,inner16,vocab97; nonzero Q/output biases). All56 checks passed, including observer/hook-free/study equality, repeat equality, same-shape suffix invariance, mask rejection, FP64 reconstruction, count/shape/scoring checks and hook cleanup. Maximum padded-context FP64 discrepancy was1.00143e−6; peak RSS1259168KiB. Random-model exactness does not predict pretrained error magnitudes.

Corrections before freeze: within-prefix future-mask checking, backend-specific availability labeling, separateq0/q127 metrics and top-two margins, early failed-input/selection hash checks, and an unambiguous `count_abs_gt_1e_minus_3` field. The original gate runner/protocol/data were not edited.

| Frozen file | SHA256 |
|---|---|
| `native_prefix_diagnostic.py` | `561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d` |
| `GPT2_PREFIX_DIAGNOSTIC.md` | `c0959ccc6315a2f59f8d2862b1d2d0462b6a244d101418b934b22b64f214ccd7` |

No launch blocker remains. Interpret the future results as a one-input numerical diagnostic. Do not select a backend or amend tolerances solely because it passes the old threshold; evaluate repeat/suffix/hook equivalence and the location/scale of divergence first. Any gate-study numerical policy remains a separate prospective decision after those results.
