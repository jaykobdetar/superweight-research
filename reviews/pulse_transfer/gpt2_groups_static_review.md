# GPT-2 grouped-contribution static review

No launch-blocking defect found in `GPT2_GROUPS.md`, `fresh_inputs.py`, or `gpt2_groups.py` as inspected. No GPU/model was loaded for this review.

Selection uses original development scores and frozen layer/output row. Top-three matching columns have deterministic score/column ordering. The strongest different output coordinate derives from the original top-ten ranking; its strongest contributor is in L1, while the direct residual control is applied after L2. Describe it as a coordinate control, not an L2 neuron selected by this rule.

The copied native Conv1D addmm includes bias and full token shape. Hooks replace only position0's selected output coordinate or full output vector; separate post-block hooks zero explicitly specified source residual coordinates. All later token outputs are asserted unchanged at the intervention site. Group activations are asserted unchanged versus native. Distinct MLP-output and post-block-residual interventions must remain distinct in interpretation.

Fresh input preparation checks source and processed-text hashes, retokenizes original extraction passages to verify old512-token hashes, and computes used character spans from token offsets. Each new used span is required to be disjoint from all old and earlier-new used spans in the same corpus. No new fitting occurs. Unused8192-character buffer overlap is correctly distinguished from evaluated-token overlap.

Count:32 inputs ×21 conditions plus8 repeat/oracle controls =680 forwards. Scoring starts predictor1, targets2..511. Full three-column factorial interactions should use paired NLL, not additive PPL ratios. The confirmation is restricted to this predeclared family; later hypotheses selected from its outcomes need another fresh panel.

This is static sign-off; runtime exact controls, span checks, parameter hashes, finiteness, and saved-output replay remain required.
