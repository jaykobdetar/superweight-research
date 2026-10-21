# Independent upstream MLP static review

Reviewed `UPSTREAM_MLP.md`, `upstream_mlp.py`, and completed outputs; no model/GPU loaded. Extracted CPU hook tests are `upstream_mlp_static_test.py/.json`; numerical replay is `upstream_mlp_audit.py/.json`.

Hook order and localization are appropriate. The established causal-mask hook runs before the extra H12-only boundary self-mask. Current projected H12 values are clipped to the model's native bounds and remain unrotated, as values should. The valuep context replacement is the correct one-key self-attention comparison; value0 tests a different current source value vector. The ordinary/isolated H12 donor conditions remain separate from those current-value controls.

Gate/up patches alter only neuron1710 at the boundary, after linear projection and before SiLU/product. Pre/post MLP normalization and effective scalar factors are saved; source0 remains intact. Both-factor patch closure should match the donor's scalar intermediate, not its whole MLP vector or model output. The changed scalar multiplies the entire down-projection column; separately deleting the [1764,1710] contribution isolates a different intervention and must remain labeled accordingly.

Initial code hard-asserted that CPU-half SiLU/product equals the GPU-computed scalar. This is not guaranteed across device kernels. The parent changed it before execution to retain CPU equality/discrepancy diagnostics, while preserving exact both-factor donor scalar closure assertions. This resolves the identified pre-run blocker.

H12-absent single-head controls (L1H0/H3/H15) and complement31 are explicit follow-up controls on reused evaluation data. Complement31 includes L0 changes, so unpatched H12 can respond endogenously; it does not hold H12's original context fixed. No new selection or fit occurs. Expected run size is 26 conditions × 64 cases = 1,664 forwards, with the same 254 scored targets per case.

## Completed numerical replay

All 1,664 raw records, 416 score summaries, 1,216 contrasts, 416 factor summaries, and 192 reconstruction/routing checks replay. Native ordinary/isolated and both H12 donor conditions exactly match the preceding upstream run. Both-factor donor scalar closure is exact, source0 quantities remain unchanged, and all factor patch localization checks pass: normalization/head contexts stay recipient-native and all other MLP intermediates remain unchanged. The boundary-deletion condition leaves all captured pre-projection quantities unchanged. CPU FP16 SiLU/product happened to reconstruct all 1,664 saved intermediates exactly; this was tested, not assumed. Peak RSS 650,800 KiB (about 636 MiB).

Three pairs of conditions have exactly equal retained loss, predictions, KL, and all saved factor/normalization/value/context tensors in every case:

- Current H12 value0 replacement equals isolated-world H12 donor induction.
- Current H12 valuep replacement equals H12-only selfmask.
- Isolated recipient with ordinary complement31 equals ordinary recipient with current valuep replacement.

The last equality is consistent with the complement restoring all L0 contexts, changing the endogenous unpatched H12 value while its isolated attention mask remains self-only. It is not evidence that H12 was held fixed. The first equality confirms the isolated donor H12 context is exactly the ordinary current source0 value vector for these same-ID inputs; it does not measure natural H12 attention probabilities.

Across sixteen corpus/identity/prefix groups, isolated-H12 induction gives mean potential pulse −503.09 to −328.65, while current boundary value/selfmask gives only −9e-6 to +0.29022. Zero context and the H12-absent single-head patches also fail to induce a comparable pulse. Routing to a boundary's current value is therefore not equivalent to using the isolated/source0 value features.

For an illustrative Wiki/The/prefix256 group, ordinary mean gate/up are −9.52/−1.19 with pulse +0.000664; isolated gate/up are 14.77/−27.69 with pulse −258.62. H12 donor induction produces mean gate/up 32.50/−23.27 with pulse −478.04. Restoring only the ordinary gate into the induced state almost removes the pulse (+0.0112); restoring only ordinary up leaves −24.22; restoring both recovers the original scalar intermediate exactly. These are means of measured outputs, not a product-of-means reconstruction.

Despite large pulse suppression, scalar-factor swaps and restorations give near-ordinary PPL: isolated gate-only/up-only/both swaps span approximately 0.9822–1.0001×, and induced ordinary-factor restorations approximately 0.9865–1.0002× across groups. By contrast, deleting only row1764's contribution in the induced H12 condition gives 77.98–421.36× PPL. This is not a contradiction: scalar-factor interventions alter the entire down-projection column, while the row deletion changes one coordinate. The records' pulse field for deletion conditions is the potential contribution computed from the unchanged pre-projection intermediate; the intervention removes that contribution downstream. A direct target-row/complement factorial is required before attributing the difference to compensation or a unique causal route.

Explicit absent-head controls provide useful negative interventions but remain a follow-up on the same four-window panels. They do not establish natural H12 necessity, exhaustive uniqueness, or the irrelevance of all other heads. Exact local factor closure does not imply full-model donor reconstruction.
