# GPT-2 query bias and source allocation: design review

Independent prospective review, 21 October 2026. Reviewed `GPT2_QUERY_SOURCE.md`; no implementation/evaluation yet. CPU-only/read-only, no model loaded. **No causal-design blocker found.** The following diagnostic definitions should be frozen before execution; implementation and numerical review remain required.

Reviewed protocol SHA256: `fabb87c17196403c439fdf22982216861245c73e4d2e3eb0d7d358f5bf748922`.

## What the factorial establishes

Within C/E/B, the 2×2 learned-query-bias-removal × source-access factorial is well defined because every main arm sets source0 V to zero, uses the same explicit causal-mask implementation, and changes Q only at later queries. Subtracting the learned bias from the native combined projection is a finite intervention on the Q vector; it avoids an arbitrary gauge-dependent division of softmax logits into source and non-source effects.

The proposed signed NLL interaction `(removed−present)_blocked−(removed−present)_open` is a valid **effect-modification contrast**. It can show that the effect of removing learned query bias depends on source accessibility. It does not establish that the learned bias acts only through source mass, identify an additive mediated fraction, or separate direct changes to non-source mixing from changes in source allocation. Bias removal affects scores for every key, and later Q/K/V states remain endogenous.

Comparing learned-bias subtraction with one fixed, per-head norm/multiset-matched randomized subtraction is appropriate as a control for the chosen intervention direction. It is neither a randomization significance test nor proof of unique bias specificity. Preserve results for the fixed seed even if they fail the proposed mechanism.

The preceding zero-V versus blocked-source observations came from another panel. This batch correctly repeats both arms on matched fresh inputs. Cross-panel numerical comparisons should remain explicitly separated.

## Hook and closure logic

Native GPT-2 Q/K/V ordering and the learned Q-bias slice must be checked in the implementation. Subtract only at positions1..end, after the native `c_attn` projection; preserve source0 Q, all K and non-source V at the hook operation. Source V becomes literal zero in the specified layers. The copied learned/random vectors must be frozen and stored, and never assigned back into model parameters.

Source0 has no access to later positions. Across query/mask arms within a fixed zero-V world, its own Q/K/V trajectory should therefore agree exactly if source0 self-attention and source0 Q are unchanged. This is a useful architectural control. It does **not** imply source0 states equal the corresponding native nonzero-V world, or that later-position K/V remain equal across arms after intervention. Restrict any whole-projection equality assertion to immediately before/after each hook.

Blocked later queries must have source mass exactly zero. Query0 must still have source mass one and retain self-attention. Explicit/native-mask zero-V references should report loss/prediction drift rather than requiring unsupported kernel-level bitwise equality. The learned-bias factorial and random controls should all use the same explicit-mask dispatch.

## Diagnostic definition and independent auditability

Specify the weighted-value comparison point as the actual per-head attention context **before `c_proj`**, with no output-projection bias or residual addition included. Comparing against the captured model SDPA context is stronger than only calling an additional stand-alone SDPA routine. It also avoids redundant compute. Use the pinned checkpoint's actual scale/configuration, dropout0 in evaluation, post-intervention Q/K/V, and the precise applied mask.

Under literal zero source V, the source vector `a0*V0` is identically zero. State whether the reported non-source vector is the unnormalized contribution `sum(j>0) a_j V_j` or the conditional mixture normalized by non-source mass. They are different quantities: with zero V, the first equals the full attention context, while the second separates mixing from attenuation by source allocation. If retaining both, define the conditional mixture directly over permitted non-source logits, with query0 marked undefined rather than dividing by zero. This is a diagnostic clarification, not a new causal arm.

The fixed maximum absolute discrepancy threshold1e-4 is an intelligible FP32 implementation guard. Apply it to all finite output elements, record the maximum and relevant output scale per arm/layer, and reject NaN/Inf explicitly. If it fails, preserve the failed snapshot and stop as declared; do not enlarge the tolerance or switch attention kernels while calling it the same experiment. Numerical agreement validates the computation, not the causal interpretation.

For an independently replayable diagnostic, save a small **predeclared raw snapshot panel**, not just masses and errors. One economical choice is Wiki natural window0, C zero-V world, both L4 and L11, all six learned-present/removed/random × source-open/blocked arms. Store post-intervention Q/K/V, applied mask and actual pre-projection context. This is roughly100MB in FP32, requires no extra forwards, and allows an independent CPU reviewer to verify full softmax/context reconstruction on those cases. Keep all-arm masses/norms and maximum-error counters; do not imply the sampled raw replay independently verifies every unretained tensor.

## Counts, separation and resources

Counts agree: 3 native + 3 native-mask zero-V references + 12 learned factorial + 6 random controls =24 per input; 32 input/start pairs +8 anchors gives **776 forwards/records**. Native/ref/duplicate arms are controls, not extra independent passages. The seventh-panel fractions and used-token-span checks cover24 prior spans per corpus. Preserve the frozen panel and abort on overlap rather than silently replacing passages after outcomes.

Full attention score/probability matrices should be temporary, never accumulated for every layer/forward. Per-layer/head/query masses and vector norms are feasible, but write incremental checkpoints and record CPU/GPU peaks. A [12,512,512] FP32 matrix alone is12MiB; a persistent all-run stack would be unnecessary. The small raw snapshot panel above keeps independent review within3GiB. Use one GPU model, no parameter mutation, and preserve failures/partial outputs.

With these definitions, the protocol is ready for implementation. Its interpretation remains a conditional functional extension of the existing query-bias/sink literature, not discovery of query biases, universal attention-sink necessity, a unique repair circuit, or faithful generation.
