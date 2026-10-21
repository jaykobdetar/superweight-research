# T19 independent prospective design review

**Decision: the causal question and full eight-boundary design are justified. Conditional approval to refine the protocol and prepare inputs; not approval for implementation freeze or pretrained execution.** The numerical policy, exact evidence layout and measured resource limits below remain launch prerequisites. This review reads saved T18 evidence and performs metadata-only interval arithmetic. It runs no tokenizer, selection, calibration, model or GPU computation and edits no frozen study files.

Reviewed proposal SHA256: `c6248525eb2b201e8024a729ddf94d0d5797d16760a8161e283523cc1cb150ef`. Supporting identities and interval calculations are in [gate_native_value_design_evidence.json](gate_native_value_design_evidence.json).

## Why this is the next informative test

T18's fresh-Wiki FF L10 removal had natural/BOS PPL/P of 0.980979/0.980339 on Ethics, but PPL/C of 1.000043/1.000783, with both C-relative intervals including one. The improvement under zero source values therefore largely restores loss toward the intact reference there. The old/fresh L10 direction cosine was 0.996647, and the small calibration-center changes did not erase the conditional sign pattern. Testing native values now separates an unresolved boundary condition from another minor calibration perturbation.

The proposed eight cells are necessary to distinguish local source-value dependence from upstream state changes and downstream processing. Keep the chosen L10, fresh F center/basis and single orthogonal control fixed. Do not add native recalibration, other layers or a new rank search. The 34-condition/2,184-forward panel is a reasonable minimal *uniformly controlled* design for that question: eight boundaries × baseline/current/removal/orthogonal, two arithmetic bridges, 32 paired inputs, all suffix controls and eight anchors. It is not necessary to reduce the causal factorial merely to save storage; the evidence can be represented compactly without discarding per-target outcomes.

## Required causal and naming precisions

1. Define every bit explicitly as **0 = retain native source V, 1 = zero source V**. The current table's “zero setting/native setting” column order is ambiguous beside 000=intact and 111=P. Define the changed object as the combined `c_attn` projection's V slice at q0, before attention, for all heads. U covers L4–L9 together, L covers L10 and D covers L11; L0–L3 remain native.
2. “Native V” means the value computed on that boundary's live trajectory. It is not a clean-C donor value. U can change L10 Q/K, non-source V and the source value itself. L cannot change its own incoming L10 Q/K/non-source V; D cannot change any L10 state. A D interaction concerns the immediate final attention block and subsequent computation, not an unspecified long feedback pathway.
3. Gate-trial source q0 states must equal their own boundary baseline throughout the network. Source q0 states need not match across value boundaries. Source zeroing itself also changes the source residual trajectory, including the zero-value attention module's remaining output bias; it is not equivalent to deleting an attention head or all source computation.
4. Use explicit estimands: for operation g and boundary w, `Delta_g(w) = L_g(w) - L_base(w)` per passage. The local contrast is `Delta_g(U,1,D)-Delta_g(U,0,D)`, and analogously for U and D. Report removal-minus-orthogonal effects and the 000-versus-111 difference. These are controlled finite intervention interactions, not mediation fractions or transmission/dependence decompositions. The reused F map remains fitted in P, even when applied at 000.
5. Keep full-target loss primary. Early/later views are secondary views of the same 512-position execution. Use the same four passage bootstrap draws across paired arms and paired start variants; no token bootstrap or pooling the two starts into eight independent passages. If input capacity forces another sample count, revise prospectively and bootstrap its actual n.

## Stable native-value computation

The ideal identity is correct. For each head/query q>0, define diagnostic scores using exactly the model scaling and effective causal mask. Let `a_diag` be the sum of the explicitly reconstructed softmax's non-source probabilities; compute the conditional mixture `u` with a separate non-source-only softmax. The delivered native-value operation is

`c_delta = c_SDPA + (sigmoid(b_target) - a_diag) * (u - v0)`.

Call this mass **diagnostic/reconstructed mass**, not an exactly observed fused-SDPA probability. SDPA exposes its actual context, not its internal probability array. Separately compare `sigmoid(b)`, `a_diag`, direct source probability, the manual full attention context and `a_diag*u + (1-a_diag)*v0` against their appropriate references. The subtraction `1-a_diag` and the explicit source probability can differ by rounding; that difference should be measured, not silently identified as exact.

The stable conditional mixture avoids dividing a tiny context by mass in native conditions. A saturated finite mass of 0 or 1 need not invalidate the native delta rule if scores, b, mixture and target remain finite; record both endpoints and do not clip or use epsilon replacement. The legacy scaling bridge separately requires its original strictly positive mass, finite gain and original reconstruction checks. An underflow failure there is a bridge failure, not evidence that the native formula is undefined.

The current/no-change arm must return the original actual context exactly, with zero delta. Do not recompute `sigmoid(b)-a_diag` and call it zero. Also specify the exact behavior when a scientific map happens to return b unchanged; an exact no-change bypass is defensible, but it must be fixed before inference rather than introduced after seeing drift. No heuristic “small delta” bypass is acceptable. q0 is copied exactly; unaffected layers/positions return their original values.

For common U and operation, capture both pre-boundary projection identity and post-boundary identity. L variants must have identical incoming Q/K/non-source V, b, a, coefficient and target b/mass; only V0 differs. D variants must have bit-identical L10 current and delivered contexts. At this shared state, a further useful numerical identity is

`(delta_c_native - delta_c_zero) = -(a_target-a_diag) * v0_native`.

This compares changes relative to each world's own local context. The full ideal context difference is `(1-a_target)*v0_native`; actual SDPA context differences have an independently measured reconstruction residual. Neither identity implies equality of downstream losses.

## Numerical bridge policy to freeze before launch

There are two distinct questions: whether the local formula is correctly implemented, and whether the two rounding paths are interchangeable for this scientific comparison. CPU tests establish arithmetic and hooks; they cannot prove the pretrained downstream function has a fixed sensitivity to rounding.

For the zero-value bridge, with `e = c_SDPA - a_diag*u` and `g = a_target/a_diag`, ideal arithmetic gives

`c_scale - c_delta = (g-1)*e`.

Thus a large gain can amplify a reconstruction residual. T18's whole-panel maximum gain was 3.16e11, outside the saved L10 snapshot subset. This does not establish such a gain at L10, but it makes “algebraically equivalent” insufficient evidence of numerical equivalence. Test tiny masses, large gains, cancellation and saturation on synthetic arrays before freeze.

Recommended prospective policy, to be accepted or replaced explicitly **before any T19 pretrained outputs**:

- Preserve the existing 1e-4 max-absolute manual-context and conditional-reconstruction guards where they apply. Require finite operands/results, correct masks and unchanged source states. Evaluate decomposition, mapped-gate and shared-state identities in FP64 diagnostics; record max absolute, RMS and symmetric relative-L2 errors. Do not normalize by a zero reference norm.
- On each actual bridge pair, require local delivered-context max error ≤1e-4. Compare all scored target losses: max absolute difference ≤1e-3 and absolute per-passage mean difference ≤1e-5 nats. These are proposed scientific-continuity thresholds, not derived error guarantees; the mean threshold is materially smaller than T18's smallest full-view L10 effects.
- Report both directional bridge KLs, final-hidden/logit absolute/RMS/relative errors and argmax mismatches. Do not require exact argmax: record mismatch margins and whether each is compatible with the observed logit difference. A near-tie argmax difference is not by itself a failed formula. A separate hard KL threshold should only be added prospectively with an explicit rationale, not fitted to observed T19 values.
- Exact native observer identity, current identity, fixed-shape suffix identity and causal-order identities remain exact. Approximate bridge thresholds cannot excuse failures in these controls.

Run the two counted bridge arms early in each input's fixed order. If a hard bridge criterion fails, preserve both sides and halt the planned accepted batch; do not relax it on the same exposed panel. A later, separately reviewed version could study the new delta operator on its own terms, but must not claim a reproduced T18 arithmetic endpoint. No bridge failure should silently discard an input or select a favorable arithmetic implementation.

## Fresh-input feasibility and exposure

Metadata-only union calculations add all 16 T18 used spans to the inherited inventory. Remaining characters/largest remaining gap are: Wiki 897,492/32,905; Utilitarianism 151,974/35,997; Ethics 493,810/121,614; optimizer components 47,159/12,759. These indicate plausible room and **do not prove four eligible 512-token windows**. I did not tokenize or select a new window.

Use the exact pinned processed sources and matching GPT-2 tokenizer. Freeze the existing 1,001-grid distance-then-numerator rule and preferred fractions before selection. Preserve the inherited mapped OLMo/original-report spans, full historical calibration/development buffers, T16/T17 spans and all T18 used spans plus their token hashes. Append current historical metadata coverage; do not discard the known 561-hash union merely because some records lie outside the newest manifest scan. Code must stay within one source component and the same component offsets must be verified.

The earlier “no prior usage” assertion for the then-new books/code is no longer appropriate: T18 is now expected prior usage and must become an exclusion, while unexpected matches require resolution. Evaluation excludes historical used spans under the approved rule; do not accidentally require every unused extraction buffer to be disjoint, and do not weaken the full-buffer exclusions for actual development/calibration. Keep any failed selection and revise input scope explicitly if capacity fails. Hold the F map, directions, tensor layout and their semantic digests fixed without refitting. Freshness refers to this recorded research exposure, not model pretraining or independent research history.

## Resource design and audit evidence

At the recorded disk check, 2,340,335,616 bytes were free, leaving 192,851,968 bytes above the 2 GiB floor. T18's old 176+48 MiB plan would require 2,382,364,672 free bytes and **does not fit**. Publication/export changes may consume more, so this is not a reservation.

A provisional **128 MiB normal +32 MiB bounded failure** layout requires at least 2,315,255,808 free bytes, leaving only about24 MiB of external headroom at this check. This is a design target, not measured approval. Keep all 1,096 main and 1,088 suffix per-target numeric records, coefficient arrays, scalar/vector norm summaries, reference identities and checks. Do not save all full contexts or all full-vocabulary logits. Process one boundary at a time, retaining only native-C and the current boundary baseline logits for metrics, rather than eight complete baseline logits unnecessarily.

An alternative is reviewed storage recovery that preserves all logical bytes of completed prior evidence. [The separate assessment](gate_native_value_storage_recovery_review.md) prefers measured zero-range sparsification, then conditionally allows exact post-audit hardlink deduplication with explicit shared-storage provenance. No recovery has been performed or budgeted as available here. Do not reduce scientific controls to fit disk; remeasure any wider evidence layout only after actual free space is verified.

An auditable candidate layout is:

- Immutable per-input outcome shards; suffix outcomes retain only the unchanged prefix with its correct target indices. All operation/source/mask/current/suffix fingerprints include shape and dtype. Baseline-relative and C-relative KL/agreement have distinct field names.
- Common L10 gates and projection states are stored once only after exact equality assertions across matching L/D cells; derived aliases preserve membership and are independently rejected if corrupted. All affected-mode coefficients and norm arrays remain available. Store successful scalar maxima as maxima, not as a claim to have saved every underlying vector.
- One predetermined natural Wiki input has eight full L10 states: baseline/removal/orthogonal at 000 and111, plus both111 scaling bridges. At these boundaries, common Q/K/non-source V, actual context and stable mixture may be losslessly deduplicated only after exact checks. Save both native and zero V0. Save final post-LN hidden states and four full-vocabulary readout rows (1,126,127,510) for these eight arms. Other boundary identities remain runtime checks plus fingerprints, explicitly not a full offline reconstruction.
- Measure the complete final serializer schema with finite high-entropy arrays, all shards, analysis/metadata/source allowances and exact fixed counts. Broad estimates suggest a layout around110–120 MiB may be possible, but no such measurement has been made here. Do not claim the provisional128 MiB cap is met until the actual writer demonstrates it.
- A failure capsule retains the exact current input, failure stage, attempted/completed forward identities, both failed local-array sides, and selected worst/fixed logit rows plus per-query error maxima. It must not dump two128-row full-vocabulary tensors, which alone exceed32 MiB. Measure each feasible stop path separately; keep any already completed partial shard in its normal allocation. No silent tensor truncation or deletion of old raw runs.

Require normal-cap/failure-cap/floor checks before loading weights, before each input group and before writes, with a measured per-group projection. Include prospective output still owed by the T18 package in the free-space check. Recheck an idle GPU and use a single model worker. Independent CPU review remains below3 GiB; production memory should be measured and separately reported.

## Stop gates and readiness

Before a pretrained launch: resolve bit encoding and mass terminology; pin F artifacts; approve exact numerical thresholds; pass fresh tokenizer-only preparation and independent span/hash review; pass random CPU forward fixtures for all34 modes and hostile mask/order/identity cases; pass full writer/failure sizing within currently available space; and obtain a separate implementation review with exact source hashes.

Hard stops include nonfinite or invalid mass/mixture states, unintended projection changes, exact no-op/source/suffix/order failures, failed bridge acceptance, input-exclusion failure, mismatched map digests or insufficient projected disk. Record zero forwards for setup failures and actual attempted/completed counts for later failures. Preserve every failure without retrying on modified thresholds or selected inputs. No efficacy-based early stopping, no interim corpus/sign-based arm changes and no publication action are part of this review.

The recommended next executable action is a **new tokenizer-only T19 input preflight**, after the revised proposal and helper receive static review. The full causal design is worth pursuing; its current status is a well-motivated prospective study with unresolved execution prerequisites, not a result.
