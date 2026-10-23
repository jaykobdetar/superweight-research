# Distinguish the downstream routes from H14 to neuron1710

Report6 shows H14 edge removal induces q pulse but reciprocal H12 context swaps
do not transfer the qualitative state. Test three additive block outputs atq:
A0=layer0 attention output, M0=layer0 MLP output, A1=layer1 attention output.
Native residual order is h0+A0, then +M0, then +A1 before the layer1 MLP's norm.

Use16 constructed R windows from newline_heads and12 unedited eligible windows
from newline_natural; no new fitting/selection. Freeze H14 from development.
Capture cleanC and edge-removedE native donors. For each of8binary combinations,
set A0,M0,A1 atq independently to the saved C orE outputs (full2048-vectors),
running otherwise in C. This freezes the other components while testing each
component combination; it is not an additive attribution of the natural network.
Also insert each individual E component into C and each individual C component
into E, leaving downstream computations endogenous (six dynamic single-branch
tests). A0 alone in this dynamic mode must transfer the complete edge intervention
exactly because the only original change is one L0 attention context atq.

All-C/all-E frozen combos must match corresponding full native outputs exactly.
These are architectural implementation controls, not discoveries. Record q
pre/post-MLP1-normalization vectors, gate/up factors, scalar pulse, all three
native/applied component vectors and original/per-token output metrics. Use same
targets as each parent panel; baseline always C, with separate paired E-relative
contrasts. Expected448forwards (16/case;28cases). Evaluate main/pair/three-way
scalar interactions explicitly rather than declaring one common low-rank cause.

Scope: repeated exploratory texts, four windows/group; natural first positions
include two p0 DuBois cases. Retain natural per-window heterogeneity. No weight
mutation, no independent GPU/model loads, and CPU review before execution.
