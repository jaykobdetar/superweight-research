# Independent Llama replacement protocol review

Reviewed `LLAMA_REPLACEMENTS.md` before implementation/evaluation. CPU-only document review; no model/GPU load. No conceptual blocker found. No original experiment files changed.

The proposed test follows the preceding source K/V boundary result without assuming that a fixed source replacement is faithful. Fresh calibration is separate from evaluation, means are frozen before test outcomes, and Wiki/Hume pooled and single-corpus fits provide an informative transfer comparison. First-token causal dependence is correctly explicit: eight calibration passages can represent fewer than eight distinct source states. The raw source0 construction, BF16 cast, model revision, no-chat-template evaluation and target2..511 scoring are appropriately scoped.

The C/R/N/D panel is a proper scalar-versus-complement factorial when defined as native copied-matrix counterfactuals. Restoring the selected weight in the otherwise zeroed column preserves that output's native dot product. The selected-coordinate equality assertion is valuable. Derive the paired NLL interaction `N − R − D + C`; it is not expected to vanish merely because the underlying real-valued column contributions partition. No additional forwards are needed.

Pooled mean K-only and V-only conditions explicitly retain endogenous complementary projections. Full fixed K/V in C and N should produce identical scored output; exact full-clean-donor rescue and native copy/repeat controls are appropriate architectural checks. A single sign/permutation control preserves each vector's absolute-value multiset, giving a useful norm-matched direction control, but does not exhaust possible random directions or prove mean-subspace optimality. The protocol appropriately avoids these stronger claims.

Material implementation clarifications sent to parent:

- State that the randomized full-K/V control and the two mean-intermediate replacements use C recipients; those recipients were implicit in this draft.
- Specify the signed arithmetic mean of the selected calibration activation, then one BF16 cast. The selection score used absolute magnitudes, but that is not the replacement estimand.
- Report natural evaluation first-token coverage by seen/unseen calibration IDs, with BOS separate. Distinct calibration count alone does not tell whether evaluation tests a new source state. Duplicate starts create a passage-frequency-weighted mean; do not describe them as independent contextual source states.
- Prespecify the C/R/N/D NLL interaction above.

With these intended C controls, expected count is eight calibration forwards plus32 inputs ×16 evaluation conditions plus eight anchors:528 forwards and520 evaluation records. Implementation review must confirm exact condition count, hashes, pre-RoPE native8-head K/V placement, source-only replacement, source position in full native F.linear recomputation, and fresh used-span separation.

Interpretation limits remain important: source0 constants need not encode semantic content; near-native average loss is not fidelity; natural/BOS tests use raw token streams on an instruction-tuned model; and n=4 bootstrap intervals are conditional on this small corpus panel. One checkpoint and length512 cannot support model-wide or long-context universality. The designed negative cases, argmax/KL metrics and reporting of calibration composition address these limits appropriately.
