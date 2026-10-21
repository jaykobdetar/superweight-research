# GPT-2 query bias and retained source attention with zero values

Prospective design, 21 October 2026. The previous source-K/V and value-geometry
experiments support conditional damage via projected source values. Literal
zero source values preserve much better loss than blocking source access, even
though output fidelity is incomplete. This next experiment asks whether learned
query biases contribute to that remaining source-allocation dependence. Query
bias mechanisms are prior art (Ran-Milo et al., 2026); this is a scoped functional
test under the present causal lesions, not discovery of query biases or sinks.

Use native C, E=erase source coordinates447/138 after L2, and B=E+block their L3
MLP writes. Keep the fixed GPT-2 FP32 checkpoint, SDPA, seed20261021, 512tokens,
natural/BOS starts, score targets2..511. All query/value/mask changes apply only
in L4..11. No weight is modified and no head/layer is selected from evaluation.
Use the seventh fresh panel, fractions.025,.225,.425,.625, four passages each
from Wiki validation, Du Bois, Hobbes and optimizer code. Verify actual used
token spans against all24 earlier spans/corpus and each other. Preflight passed;
the run must independently repeat it. No model-weight value sweep is involved.

Every main factorial arm sets the source V projection to exactly zero. Cross
query-bias removal (no/yes) with source blocking (no/yes). Query removal subtracts
the native learned c_attn Q bias from Q at positions1..end, all12heads. Q at
position0, every K, and every non-source V are unchanged by this hook. Source
blocking uses an explicit causal attention mask that additionally forbids
queries1..end from attending source0; query0 retains self-attention. The unblocked
factorial arms use an explicit causal mask of the same form. Query-bias removal
affects all attended keys, so a main effect alone cannot identify a source-only
mechanism; the interaction with source blocking is the relevant conditional
test. Source-specific logit splitting is avoided because its assignment would
depend on a softmax logit offset convention.

Matched controls subtract a fixed signed permutation of the learned Q bias,
separately in each layer/head, with source blocked and unblocked. CPU seed20261024,
no seed selection; absolute-value multisets/norms match the actual bias per head.
This is a randomized-direction subtraction control, not a trained alternative
bias. Native C/E/B anchors and zero-V native-mask references establish baseline
and explicit-mask drift. Natural window0 of each corpus gets repeat/source
identity anchors. Total per input:3native+3native-mask-zeroV+12factorial+
6random-controls =24conditions; 32inputs plus8anchors =776forwards/records.

Record source attention allocation using the actual post-intervention Q/K,
scale1/sqrt64, and applied mask. Recompute softmax in FP32, retaining per-layer/
head source mass for every query and source/non-source attention-vector norms.
This reconstruction is a diagnostic calculation, not the implementation of
model attention. Validate its complete weighted-value output against native
SDPA on every layer/arm with a predeclared max absolute tolerance1e-4; report
maximum drift and any failures rather than silently switching kernels. Source
mass at blocked later queries must be exactly zero; query0 remains causal.
The source vector is its unnormalized weighted contribution p0*V0 and must be
exactly zero in zero-V arms. The non-source vector is the unnormalized weighted
sum over positions1..end, not a mixture renormalized after removing source mass.
Preserve raw post-intervention Q/K/V, applied mask and actual preprojection
attention context for a frozen audit subset: Wiki natural window0, L4 and L11,
C zero-V arms with learned bias present/removed/random and source open/blocked.
This adds no forwards; approximately100MB of local snapshots permit independent
CPU reconstruction. Keep masses/norms for every other arm.

For NLL, report the learned-bias removal effect with source open and blocked,
and I=(removed−present)_blocked−(removed−present)_open. Compare corresponding
random-direction effects without calling one random seed a significance test.
Group corpus/start, n=4 paired bootstrap2000 seed20261021. Report native NLL/KL/
argmax agreement and source-allocation distributions. Neither high source mass
nor correlation with loss proves mediation. Query effects can alter downstream
non-source computations; no unique architecture or universal-circuit claim.
The bias-by-blocking interaction establishes effect modification by source
access; it cannot prove the learned bias acts only through source attention mass.

Validate exact native repeat/identity closures, zero-V native-mask versus
explicit-mask references (report drift), source K/V/query0 equality across
query/mask arms within each zero-V world, hook locality, copied bias/permutation
identities, immutable parameter hashes/versions and source-blocked source mass.
Preserve failures/partial outputs; one GPU model, independent implementation
and numerical reviews. If attention-reconstruction tolerance fails, stop this
batch, preserve evidence and diagnose before any new preregistered attempt.
