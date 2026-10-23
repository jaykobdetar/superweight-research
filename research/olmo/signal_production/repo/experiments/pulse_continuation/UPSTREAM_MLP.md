# Frozen follow-up: the production gate and L1/H12's value route

After the upstream development screen selected L1/H12, test how it changes the
MLP neuron1710 whose down-projection contains the source weight. Do not retune
selected heads or sweep the numerical weight. H12 patching changes many features;
this experiment distinguishes its effect on the two scalar MLP factors.

Use the SAME four held-out windows per corpus, The/period identities, and real
prefix lengths128/256 as upstream. This is a mechanistic follow-up on reused
held-out text, not a new confirmation set. Same254 scored target IDs. Original
source0 stays intact; boundary-only source deletion is separately measured.

Capture pre/post MLP normalization at source0 and boundary, and gate_proj/up_proj
neuron1710 before SiLU/product. In ordinary and isolated worlds, swap each scalar
or both from the other world. Patch only this neuron's projection output, leaving
other neurons unchanged. Factorial endpoints must reconstruct the donor's scalar
intermediate exactly; they need not reproduce the donor's full MLP vector/output.
Each changed neuron contributes through its entire down-projection column, not
only through the special row1764; disclose this distinction.

Reproduce H12 isolated-donor induction and ordinary-donor suppression, then patch
ordinary gate/up scalars back into the induced world. This tests the proximate
scalar route without pretending a correlation is a causal account. For the
induced head condition separately delete only the super-weight contribution at
boundary to assess downstream conditional dependence on that pulse.

Routing alternatives in an ordinary recipient: replace only L1/H12 boundary head
context with (a) its current clipped value at boundary, (b) current clipped value
at source0, (c) zero, or (d) native H12 attention with only boundary query restricted
to itself. The self-context and head-mask controls should agree in the pulse
intermediate (minor SDPA numerical effects recorded, not silently ignored).
Contrast current boundary value versus isolated-world H12 donor to separate L0
history's effect on value features from the L1 routing choice. Save raw contexts,
projected values, scalar factors and reconstruction checks. No fits.

Before this follow-up starts, add explicit H12-absent controls: individual L1
heads0,3,15 and the other31 early head/layer channels, both donor directions.
These are follow-up controls on reused heldout data, not a repair or relabeling
of the prior random sets (which all happened to include H12). Remaining
unpatched contexts are endogenous and may change after earlier patches.
