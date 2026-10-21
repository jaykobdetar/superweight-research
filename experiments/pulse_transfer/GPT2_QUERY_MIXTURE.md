# GPT-2 query-bias allocation versus conditional mixture

Prospective protocol, 21 October 2026. The reviewed query/source factorial
shows that learned-Q-bias removal reduces source attention, while its NLL
effects vary across corpus, lesion and start. It does not resemble blocking
the source. This follow-up separates two changes at a defined attention-output
boundary; it does not infer mediation from attention correlations.

Use the same immutable GPT-2 FP32 SDPA checkpoint and seed20261021. Native C
has no coordinate lesion. At L4..11, set V0 exactly zero. Define donor P with
learned Q bias present and donor R with that bias subtracted from Q at positions
1..end. All heads/layers are included without evaluation-based selection.
Use a fresh eighth panel at fractions .125,.325,.525,.725 in Wiki validation,
Du Bois, Hobbes and optimizer code; repeat exact actual-token-span exclusion
against all28 prior passages/corpus. Natural/BOS replace only token0. Length512,
score targets2..511. These are32 paired inputs, not independent starts.

For each layer/head/query q>=1, save actual native SDPA preprojection context
c and non-source attention mass a=sum_{j>=1}p_j. Source V is zero, hence c is
an implementation-level approximation to a*u, where u is the conditional
non-source value mixture. Do not compute a as1-p0. Use all four pairs of mass
donor G in{P,R} and mixture donor D in{P,R}. At every treated c_proj input,
replace the context for q>=1 with c_D*(a_G/a_D). Copy c_D exactly on diagonal
G=D. Leave q0 untouched. Repeat the full factorial in P and R recipients;
all four donor pairs must give exact identical outputs across recipients.
These duplicates are closures, not extra samples. Fixed donor states are
collected in their own unpatched forward trajectories; mixed trajectories
need not equal any native trajectory. No additive mediation fraction is valid.

Because dividing by a small mass can amplify diagnostic error, also compute
u_stable directly by softmax over non-source logits only, with the same causal
mask, for q>=1. Require finite positive a with no clipping. Record min/max a,
maximum gain a_G/a_D, finite outputs, c_D/a_D versus u_stable max absolute
error and mixed context versus a_G*u_stable error. Require each max absolute
error <=1e-4, as well as the existing full native SDPA/manual diagnostic
<=1e-4. Save errors, donors and failure state and stop on failure; do not
silently change arithmetic/kernel/threshold. Report maxima, including native
reconstruction c versus a*u_stable. Exact diagonal copies intentionally avoid
numerical differences from needless divide/multiply operations.

Per input record C native, zero-V native-mask P, explicit-mask P/R donors,
P/R source-blocked controls, P/R current-context identity controls, and eight
recipient×mass×mixture factorial arms =16conditions. Natural window0 per
corpus adds native repeat and source-coordinate identity anchors:520forwards.
Require donor diagonal loss/prediction/KL closures, current-context closures,
cross-recipient closures, source Q0/K/V equality across all zero-V arms and
unchanged q0 output. Record explicit/native-mask drift. Validate every model
parameter version/hash. All hooks operate on copied activations, not weights.
Hash the complete final normalized hidden tensor on every forward and require
identical hashes for each exact output closure, in addition to saved metrics.

Preserve per-token losses/KL/predictions, tokens, actual donor contexts, masses,
stable mixtures, source states, operation checks and paired aggregates.
Store donors as separate input files to permit reviews below3GiB. Frozen raw
audit subset: Wiki natural window0, all factorial arms plus P/R donors, L4/L11
post-intervention Q/K/V/mask/native context and delivered context. Donor files
retain all eight layers. This allows independent FP64 attention reconstruction
and exact patch arithmetic checks. Other arms retain summary error evidence.

Primary factorial uses recipient P (R is a closure). Report mass effect at each
mixture, mixture effect at each mass, and interaction
I=Y_RR-Y_RP-Y_PR+Y_PP, with first index mass, second mixture. Report four
group means and paired n=4 bootstrap2000 seed20261021 by corpus/start;
uncertainty is conditional on the fixed intervention and passage selection.
Also compare both mixed cells with P and R donors, native C loss/KL/argmax
fidelity, blocked-source controls, and mass/gain/error distributions. No
multiple-comparison or population-generalization claim. Known query-bias/sink
mechanisms are primary prior art; this is a scoped functional decomposition.

One GPU model/process, independent CPU implementation and numerical reviews.
No claim of faithful replacement, computational savings, universal mechanism,
or a unique direct/indirect causal path. Stable-mixture computation is a
diagnostic; native SDPA remains the model kernel.
