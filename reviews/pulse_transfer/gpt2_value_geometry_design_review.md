# GPT-2 source-value geometry: prospective design review

Reviewed `GPT2_VALUE_GEOMETRY.md`, 21 October 2026. CPU/read-only review; no model, GPU or experiment execution. **No design blocker found.** Implementation/synthetic and numerical audits remain required. The prior regrowth-K/V numerical audit is assigned separately; its preliminary values are not independently certified by this review.

Reviewed protocol SHA256: `c128dc5d6d4668706cca94b81461cf30e59be391ca5c31821ef12308bbf473df`.

## Estimand and causal controls

The four cells form a valid conditional 2×2 factorial. At every source0 V head in L4..11, one factor supplies the E/B unit direction and the other supplies the E/B Euclidean norm. Source0 keys are fixed to E in all four cells; the recipient remains B. The two simple effects and the stated signed NLL interaction therefore compare the intended controlled factors.

“Magnitude” here means the complete **per-layer/per-head norm profile**, changed jointly across the selected layers and heads. It is not a single global scalar or uniform inflation: the recorded B profile can be smaller in some components, including L11. Likewise, the direction factor jointly changes all tested head directions. Results cannot by themselves locate a decisive layer/head, establish a monotone norm-dose relationship, or attribute every effect to excessive norm.

E and B differ only at source0 after L3. Clamping their complete later source K/V removes that difference from later-token computation. Thus E keys plus the E-direction/E-norm value cell in B must reproduce E's scored outputs exactly. Copying a given K/V cell into E and B recipients must agree on scored later-token outputs. However, the **B-direction/B-norm cell with E keys is the mixed E-key/B-value world, not native B**. Its exact diagonal property concerns the reconstructed V vector; do not assert restoration of B's native output under E keys.

C-source K/V copied back into C is an appropriate identity control. The C/E/B current-key zero-V controls and the B fixed-E-key zero-V condition distinguish the native-key worlds from the common-key factorial. Zero V retains attention mass on source0 and can change later source keys when those keys are endogenous; it is not source-column blocking. Fixed source keys also do not fix later queries or attention gates. The protocol states these limits correctly.

## Permutation and numerical controls

The single predeclared signed permutation per layer/head is useful. Use the **same frozen transformation for E and B and every input**, generated once before evaluation. A shared permutation preserves the angle between the original E/B vectors within each head as well as their respective absolute-value multisets; it tests the placement/sign of that pair relative to the learned downstream maps. It is not an independent resampling of two directions.

Permuted E and permuted B have different original directions and their own norm profiles. They therefore provide two matched controls, **not one common random direction tested at both norm profiles**. The present controls are sufficient for the stated experiment; no extra arm is requested. A single seed supports a comparison with this frozen control only, not a distribution-wide conclusion about random directions or unique directional specificity.

Exact diagonal donor copies correctly avoid normalization/reconstruction roundoff. For mixed vectors, independently verify finite norms, finite scaling ratios and outputs, intended alignment, and norm agreement using an explicit FP32 tolerance. A norm's floating-point reduction can differ slightly after a permutation even when the absolute-value multiset is exact. Require multiset equality and exact assigned patch vectors; do not confuse mathematical norm preservation with bitwise equality of separately reduced norms.

## Panel, counts and analysis

The fixed sixth panel and actual-used-token-span checks preserve prospective evaluation separation. Within-input E/B donors are intentional oracle counterfactuals; they are not development-fitted replacements. No claim of deployment-ready calibration or a held-out donor predictor should follow.

Counts are correct: 3 native + 4 factorial + 2 recipient closure + 3 current-key zero + 1 fixed-key zero + 1 C identity + 2 signed permutations = 16 per input. Thirty-two inputs plus eight anchors yield **520 forwards/records**. Three native forwards also generate donors, so no additional calibration/model passes are needed. Duplicate recipient/identity cells are controls, not independent samples.

The predeclared interaction sign is coherent. Preserve all four cells, simple effects at both partner levels, native C and conditional E/B baselines, KL/agreement, corpus/start separation, and all four passage-level observations. With n=4 and many comparisons, descriptive paired intervals cannot establish population-wide effects or unique mediation. Near-native loss under a direction or norm swap is not native representation recovery, semantic content, faithful generation or a speedup.

## Required implementation review

- Native GPT-2 combined `c_attn` output must be split into Q/K/V correctly, then V into 12 contiguous 64-dimensional heads. Patch only source0 K/V; leave Q and later positions unchanged at the patch operation.
- Capture unpatched C/E/B donors before patched recipients, retain the E erasure/B L3 write block in recipient worlds, and reset all per-forward hook state.
- Generate the per-head permutation/sign maps once, save them, and reuse them across inputs and both donors. Avoid accidentally advancing the random generator inside the input loop.
- Check first-boundary later-token equality and all declared identity/cross-recipient closures, while allowing downstream computations to change endogenously after intervention.
- Record full raw donor vectors and transformed vectors or sufficient deterministic reconstruction data, parameter hashes/version counters, operation-locality checks, and all failed checks.

The scientific interpretation should remain a conditional projected-value geometry decomposition. This is not a priority claim or a new general theory of normalization, coordinate regulation, or attention sinks.
