# M0 donor and symmetric-decomposition audit

Reviewed 2026-10-23. This continuation preserves earlier frozen audits. No learned forward, model download, or parent-file edit was performed. Checks used compact states and selected existing tensor slices, sequentially with two BLAS threads. Actual donor passes were executed by the parent, not this auditor.

## Findings

The frozen reciprocal donor design and algebra are correct for their stated local, controlled-prefix question. The model-validation wrapper resolves the earlier execution-gate concern when used as the entry point. Two labeling/interpretation cautions were reported immediately and accepted by the parent:

1. Treated result `m0_activations` records the native, pre-hook activation vector, not the effective swapped activation vector. The forward captures its original activation array; the hook modifies only the projected output. This does not invalidate the intervention or contaminate donor selection. The parent will document the field and derive effective vectors separately without changing frozen artifacts.
2. Full-M0 output donation is a module-level reference, not an opposite-world restoration guarantee. The recipient L0 pre-MLP residual remains. Its E-minus-R norm is 0.241–0.264 on discovery panels, and L1 attention reacts endogenously. Calling it a positive control cannot mean it must reproduce the opposite world exactly.

## Prefix traces and edge hook

All 16 archived-normalized-state comparisons were recomputed. Eight native U/R states are bitwise equal to the older cache; edge-state maximum absolute difference is 2.64645e-5. The recorded earlier-position hashes match their native R hashes for both edge interventions in every panel. The compact traces contain only q=128 states, so this audit checks hash consistency, not a second learned replay of those earlier states.

An isolated test of the actual edge-hook function confirms that it changes only head H14's context slice at query 128, zeros the source-64 probability and renormalizes the remaining probability vector, preserves all other coordinates and input arrays exactly, and returns the original object at other layers. It matches an independent FP64 formula within 2.98e-8 on the synthetic test. In real arithmetic this is equivalent to deleting that one attention-score edge. FP32 renormalization of an already rounded probability vector need not be bitwise identical to recomputing the masked softmax; the archived endpoint checks measure that difference here.

## Symmetric feature decomposition

For a centered feature row a, residual x, numerator n=a·x and population-LN scale s=sqrt(mean(center(x)^2)+1e-5), the code uses

Delta(n/s) = Delta(n) mean(1/s) + mean(n) Delta(1/s).

This symmetric product identity is exact in real arithmetic. The first term is split into the observed A0, M0 and A1 residual changes. Because R and E share input token IDs, the embedding cancels in the A0 difference. The second term is the symmetric normalization allocation, not a separately identified causal mediator. The name `normalization_interaction` denotes this accounting convention, not a distinct higher-order causal effect.

Independent recomputation agrees with every saved component within 1.42e-14. The pure product identity closes within 7.11e-15. The actual stored component closure is at most 1.271e-6, and M0 neuron-sum closure at most 4.922e-6, consistent with independently rounded FP32 residual arithmetic/projections. Thus the title's “exact” describes the algebra, not literal bitwise closure of saved traces.

The neuron formula is Delta(activation_j) times a·down_column_j times mean(1/s). Six selected existing down columns reproduce saved coefficients within 8.33e-17. The frozen primary-gate top-four selection [1091,3419,6535,4798] is reproduced from mean absolute contributions over the four discovery panels.

Neuron 1091 ranks first individually on every discovery panel for gate1710, up3950 and up4786. Its signed contributions are respectively +43.348 to +46.483, -2.516 to -2.346, and +1.463 to +1.569. These are signed feature-change allocations, not nonnegative percentages, necessity, sufficiency or loss effects. Cancellation allows an individual contribution to exceed the net change. All 16 separate M0-1091 gate/up factorial entries independently reproduce within 1.12e-16 from the two existing weight rows and saved normalized inputs.

## Reciprocal donor protocol

All frozen token hashes are correct. Each panel has identical R and E input IDs of length 129. The E world differs by the H14 query128-source64 edge intervention. Discovery donors are the newly validated actual prefix traces; controlled context holdouts erase 187/535 to 209 before forcing positions 0=510 and 64=128=187. They are new controlled contexts, not naturally occurring unmodified examples or a population sample.

For each recipient world, the donor is its opposite world. At L0 only, the hook copies the output and replaces only q=128. For a selected group it copies the recipient post-SwiGLU activations, assigns the donor values for those indices, and freshly projects through the same down matrix. Other activations, earlier positions and layers are untouched. L1 attention/normalization remain endogenous. A synthetic test of the actual extracted hook verifies all five treatment branches, including unchanged input arrays and other-layer identity.

Sham, 1091, rest-three and their union form a valid 2×2 within each direction. The full-M0 output donor is an additional module reference. Counts are 8 panels × 2 worlds × 5 interventions = 80 intervention passes plus 8 native holdout passes, total 88. Conditional claims should specify the recipient world, donor world, targeted group and measured L1 endpoint. They do not establish unrestricted necessity/sufficiency, isolated direct mediation, natural-prefix reachability or downstream behavior. No loss/head stage is run here.

## Completed output checks

While this read-only audit was in progress, the parent's permitted donor batch completed. All 88 result NPZ SHA receipts and run identities pass. The 80 intervention result labels match the corresponding panel/world/treatment. The full-output donor outputs equal the opposite-world outputs bitwise. For selected groups, the actual output equals the saved sham output plus the four-column donor-activation delta within 2.103e-6; no full projection or learned forward was run to check this. The sham normalized-state error is at most 2.599e-5 against its matching native donor state, below the frozen 5e-4 criterion.

Every treated `m0_activations` array is indeed the native pre-hook vector. Deriving an effective vector from native values plus the opposite-world group values is unambiguous. Full-output donors should instead be labeled as output patches unless a complete donor activation vector is explicitly attached as a reconstruction reference.

Operational hardening, not observed failures: the donor runner checks global complete/all_pass and row count but not each row's pass/FP32 fields; those current ten rows were independently verified and all pass. Discovery trace model-code identity matches current validated code but is not separately asserted by the donor runner. Resume handling is less strict than the earlier 24-forward runner for orphan JSON-only files; no orphan/identity failure occurred in this batch. Preserve the already frozen runner and report these as future hardening, not after-the-fact protocol changes.

## Original 24-forward execution wrapper

`execute_validated_study.py` checks complete/all_pass, exactly ten individual passing FP32 comparisons, current model code SHA matching validation, and complete acquisition at the exact pinned revision. It writes an execution-validation receipt tied to the frozen protocol and wrapper hashes, then calls the unchanged runner. All current dependencies meet those checks. The original frozen protocol remains SHA 14e31dc8822fec9bae98a6cd26ae6df0af445e569b7258df8c94216edb524f4c.

The legacy runner's direct --execute entry remains ungated; this conclusion applies to the documented wrapper command. Passing scientific validation never independently grants a CPU slot. Slot coordination remains external to this audit.

See MANIFEST.json for immutable reviewed source/result identities and the three check scripts/JSON outputs for reproducible compact checks.
