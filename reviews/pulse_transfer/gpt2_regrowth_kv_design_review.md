# GPT-2 regrowth K/V protocol review

Reviewed `GPT2_REGROWTH_KV.md`, 2026-10-21, before implementation/evaluation. No model or GPU loaded. **Design passes with the interpretation constraints below; implementation and numerical review remain required.**

## Causal boundary and panel

E and B share the source0 erasure after L2. They differ only in source0 coordinates 138/447 of the L3 MLP output. Consequently, their L3 attention and non-source post-L3 states must be identical. Source0 K/V at L4 is the first possible path into the scored later tokens. Native later-token L4 Q equality is a valid boundary check; equality of later queries at every subsequent layer is neither expected nor required.

Clamping all source0 K/V from layers4..11 to a full E or B donor fixes all remaining source-to-later-token edges. It should therefore reproduce the donor's later-token outputs in either recipient, even though source0 residuals and its own query path differ. Mixed E/B source K/V must agree across the two recipient worlds for the same reason. These are architectural controls, not evidence that a new repair pathway has been discovered.

Full source blocking at layers4..11 likewise removes the E/B difference from later-token outputs while preserving source0 self-attention. Zero source values with current keys is a different intervention: it retains source attention mass and permits later source keys to change endogenously. The additional fixed-E-key zero-value condition is essential and correctly included. Blocking versus value-zero must be compared within the matched explicit-mask execution reference, with native-mask drift reported separately.

Counts are coherent: 3 native worlds + 8 donor-factorial runs + 8 zero/mask runs = 19 per input; 32 natural/BOS inputs plus 8 anchors gives 616 forwards/records. Duplicate native/diagonal or cross-recipient conditions supply controls and do not add independent statistical observations. Bootstrap the four distinct passages per corpus/start, paired throughout.

The fifth fresh panel is frozen before outcomes, checks actual used token spans against all four earlier panels, and discloses front matter/headers. A failed span check should abort rather than silently choose outcome-dependent replacements. Final source0 identity and scoring exclusions should be retained in the saved manifest.

## Material interpretation constraints

1. This factorial tests **all downstream source K/V paths jointly**, conditioned on the two worlds. It does not identify L4 alone, one head, or a single neuron as the mediator. The original layer3 intervention stays recipient-specific, and all other later computations remain endogenous.
2. Predeclare the signed NLL interaction for the 2×2 donor panel, `NLL(EE) - NLL(EB) - NLL(BE) + NLL(BB)`, alongside the simple effects at each fixed partner. This avoids turning mixed-state outcomes into additive mediation percentages. Either equivalent sign convention is fine if stated before interpretation.
3. Full-donor rescue can close exactly by construction. The informative outcomes are asymmetric mixed K/V effects and zero-value versus source-block behavior, against both native C and conditional E baselines. Low native loss does not imply native raw trajectories, faithful generation or semantic value content.
4. Keep the literature attribution in this protocol: this is a source-local transmission extension of previously studied GPT-2 coordinate regulation. A sink/query-bias explanation requires an actual sink/gate readout or additional causal comparison; NLL and K/V dependence alone do not prove it.

## Implementation checks to require

- Patch only source0 K/V slices of GPT-2's combined `c_attn` output, preserving Q and all later-position projections bitwise at the patch operation. Remember that this is a forward projection-output intervention, not a weight edit.
- Capture native E/B donors before recipient patches. Reinitialize per-forward state and masks; keep recipient source erasure and L3 blocking active for every patch condition.
- For masks, preserve native causality, source0 self-attention, and all non-source columns. Apply the source-column block only in layers4..11 and only for query indices greater than zero. Explicit/native references must use the same dispatch as their blocked counterparts.
- Save first-boundary Q equality, complete diagonal/cross-recipient and blocked closures, and full vector operation checks. Later model recomputation may change naturally after the first boundary; local patch equality must not be confused with unchanged downstream representations.

No additional experimental arm is needed for launch readiness. The NLL interaction definition is a small analysis clarification; it does not change the 616-forward design.
