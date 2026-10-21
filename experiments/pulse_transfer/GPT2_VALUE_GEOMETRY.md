# GPT-2 source-value magnitude and direction factorial

Prospective follow-up, 21 October 2026. The preceding regrowth K/V batch
suggested that restoring E values helps much more than restoring E keys, and
that zeroing source values can prevent much of B's loss. These preliminary
numbers require independent audit before established interpretation. Saved
donors show larger B value norms in L4..10 but smaller norms in L11; norms alone
are not causal evidence. This test separates projected-value magnitude and
direction, with all source keys clamped to the same per-input E donor.

C, E and B retain the prior definitions: native; source0 coordinates447/138
erased after L2; E plus those L3 MLP output-coordinate writes zeroed. All other
computations remain native. Source K/V patches span every head of L4..11 only.
Use the pinned FP32 GPT-2/SDPA checkpoint, seed20261021, natural and BOS starts,
512tokens, score targets2..511, no training or weight mutation.

Use a sixth panel, four passages per Wiki validation, Du Bois, Hobbes and
optimizer code at fractions.175,.375,.575,.775. Check actual512-token spans
against all five prior panels and each other before the run; if any overlap
exists, do not execute this panel. No component selection or calibration uses
evaluation outputs. Freeze this panel even if results are negative.

For each input obtain native C/E/B source K/V donors. Split each projected V
into12 contiguous64-dimensional heads. For each layer/head define unit
directions uE=VE/||VE|| and uB=VB/||VB||; require both norms nonzero. Four main
conditions are V=uD*nN for D,N independently E/B. Exact diagonal cells copy
the original donor vectors (avoid normalization roundoff). Mixed cells scale
the donor direction by the other donor's per-head norm in FP32. They are an
explicit headwise magnitude/direction intervention, not an inferred semantic
direction. Use B recipient and E keys for all four cells. Also copy both
diagonals into E recipient as duplicate closure controls, not new samples.
“Magnitude” denotes the entire E-versus-B layer/head norm profile; B is not
uniformly larger. The B-direction/B-norm cell with E keys is the mixed
E-key/B-value state, not native B. Only E/E with E keys fully restores E donors.

Controls: C/E/B native; repeat and source identity anchors on natural window0;
zero V in C/E/B with current keys; zero V with fixed E keys in B; C keys/values
jointly in C as an identity check; E keys with a fixed signed permutation of
E values or B values in B. The signed permutation is drawn before execution
with CPU seed20261023 independently for each layer/head, preserves each head's
absolute-value multiset and norm, and is identical across inputs. Both original
E/B norm scales are retained; no favorable seed search. This is16conditions
per input (3native+4factorial+2recipientchecks+3current-keyzero+1fixed-keyzero+
1Cidentity+2permutation), plus8anchors =520 forwards/records. There are no
hidden calibration passes.
The signed-permuted E and B controls retain their respective input directions
and norm profiles under the same random map; they are not one common random
direction tested at two scales.

Report paired NLL simple magnitude effects at each direction, direction effects
at each magnitude, and interaction
I=NLL(DE,NE)-NLL(DE,NB)-NLL(DB,NE)+NLL(DB,NB).
Group by corpus/start, n=4, bootstrap2000 seed20261021; no multiple-comparison
correction or general-population coverage claim. Report native KL and argmax
agreement as well as PPL; low loss alone is not faithful output reproduction.
The changing downstream queries are part of the intervention response; fixed
source keys do not mean fixed attention gates. Avoid additive mediation claims.

Save per-token losses/KL/argmax, raw donors, complete source trajectories,
norm/cosine/absolute-value controls and locality checks. Verify C identity,
E diagonal restoration, duplicate recipient closures, exact copied-patch
values, all non-source projection coordinates and queries unchanged by hooks,
and unchanged parameter hashes/versions. Preserve failed checkpoints. One
GPU model only; independent hook and numerical reviews before interpretation.
