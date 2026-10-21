# Frozen two-unit factorial after column-component result

The first-unit column factorial shows that deleting row1764's contribution is
far more damaging than deleting its whole neuron1710 output, including at native
source0. Test whether the second known source weight L2[1764,8041] is required for
that milder whole-column counterfactual. This is a candidate backup pathway,
not yet an established explanation.

Use the same16 heldout512-token windows (first4/corpus) with natural/The/period
source identities. For each unit independently choose clean, row-only deletion,
or whole-column deletion. Full3x3 factorial at position0 only, plus a separate
all-position application using identical input IDs and scoredtargets2..511.
No new weight values or fitting. The global row deletions are reference controls;
the new question concerns entire-neuron interventions and their interactions.

At each MLP save source0's original computed intermediate activation, scalar
weight, actual applied state, and source0 down-projection output before/after.
An activation times original weight in a deleted condition is its potential
unremoved contribution, not the applied pulse; explicitly distinguish them.
Weights remain unchanged. Native F.linear uses copies with respective columns
zeroed. First-unit source-only conditions must replay column_factorial raw
metrics exactly; clean both scopes share the same native baseline.

Interpret the paired effect of second-unit row/column removal conditional on
first-unit whole-column removal. Increased reliance does not establish a unique
backup or prove the second unit solely explains recovery. All-position tests
also remove later computations, so changes versus source-only are scope effects,
not causal position comparisons. Reused panels and smalln remain exploratory.
