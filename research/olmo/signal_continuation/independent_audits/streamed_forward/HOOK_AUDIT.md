# Alternate-writer hook audit

Audit time: 2026-10-23 05:43 UTC. This continuation does not change or supersede the earlier frozen L1 replay numbers.

Verdict: no substantive intervention-algebra defect found in the fixed group_factorial treatments. Source/position locality, partial axis/complement semantics, carry-only/group-only semantics, and same-world add-back comparator are correct. The callback is an activation/output intervention at L1, not a global parameter ablation.

Files reviewed:
- continuation/alternate_writer_hooks.py SHA256 f02bce6e277f4b93bea33cf8cafb48ffa226241b5b6735f59ee6fe6052db6e8b.
- continuation/analyze_controls.py SHA256 5b04a897b196e8574908b055d56bc0255a3ad21a009d95d8d7586f241dfbf632.
- continuation/results/hook_validation.json SHA256 32bc6a23adb1b86df42dc582166d53e6d355696deac973de68a54b8196561640.

## Correct semantics

The callback receives post-SwiGLU neuron activations. It copies selected rows and zeroes dropped activation coordinates before a fresh down-projection product. Whole-column deletion therefore avoids subtracting the dominant1710 contribution from a rounded native output. The caller subsequently adds the modified output to the unchanged incoming residual.

Only the selected token rows in the output copy are overwritten. Input activations, normalized inputs, native outputs and down weights are not changed. Layers other than1 return the original output object. Keeping a group zeros all other MLP activations but retains incoming carry; keep=() is carry-only, not a zero residual stream.

Let u be the unit centered coordinate1764 axis and G the group's raw down-projected write. Axis removal subtracts (G·u)u. Complement removal subtracts G−(G·u)u. Since mean(u)=0, the raw mean/DC component belongs to the complement. Their sum is exactly G in real arithmetic. analyze_controls.py instead centers G before constructing its complement; this gives identical centered/LayerNorm geometry, even though the raw mean differs. The code correctly calls these vector-component interventions, not neuron ablations.

Same-world add-back starts with1710 and the group absent, then restores the group's native raw write. Its algebraic comparator is drop1710, not completely native. The group activations at L1 are unchanged by an L1-output intervention, so same-world values are appropriate.

## Exactness caveat and needed controls

Selected-position recomputation can change matrix-multiplication shape/reduction; add-back also uses FP64 arithmetic before assignment into native output dtype. Consequently reconstruct_native and same-world add-back are not guaranteed bitwise equal to their comparators. The existing full-tensor validation checks small numerical errors, not exact reconstructed output identity. Run both untouched-native versus reconstruct_native and drop1710 versus same-world-add-back through the full model, and report discrepancies, rather than assuming zero downstream effect from small local error.

Nonselected rows and non-L1 behavior are bitwise/exactly preserved. These claims are separate from arithmetic reconstruction exactness.

The hooks use the supplied down matrix without automatic storage rounding. They are appropriate for the scientific FP32 run. Supplying unrounded FP32 down to a half_round forward would mix precision; either keep these hooks FP32-only or supply the identical rounded matrix for that diagnostic.

The currently generated treatments are valid. For future arbitrary specs, keep and partial_group should not be combined unless the partial group is actually retained; current assertions do not enforce that additional relationship. No generated group_factorial treatment uses this unsupported combination.

## Independent tests without full-weight reads

check_hooks_compact.py used the nine saved down columns and zero for all other columns, with nonzero raw means and two simultaneous selected positions. This is a sparse compact test model, not a new complete-model run. All27 treatment checks passed; max difference versus an independent FP64 formula was3.41e-5. Nonselected rows, input arrays and other-layer identity checks were exact. Axis/complement orthogonality, raw-mean placement and additive closure passed. Add-back versus drop1710 differed by up to9.54e-7, illustrating the precision caveat above. Results are in hook_compact_checks.json.

All180 gate/up factorial rows were independently recomputed from cached gate/up values and compact columns. Maximum difference was1.78e-15.

For pair3950/4786, aggregated axis-write totals across the four H14 panels are:
- R gate/R up: −0.01652 to −0.00961.
- R gate/E up: +0.04169 to +0.05372.
- E gate/R up: +3.26333 to +3.84696.
- E gate/E up: −8.28883 to −7.12659.
- Difference-of-differences EE−ER−RE+RR: −11.97668 to −11.03052.

Thus each one-factor donor total is positive while the joint donor total is negative. This supports a fixed-pair arithmetic interaction. It does not establish per-neuron necessity, natural realizability of independent factors, held-out discovery, or downstream rescue.
