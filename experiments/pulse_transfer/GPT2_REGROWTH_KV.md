# GPT-2 source transmission after the layer3 write

Declared before this batch,21 October2026. The layer-local test found a strong
conditional role for the L3 MLP's writes to source coordinates447 and138 after
their erasure at the L2 boundary. These coordinates and related compensatory
circuits overlap prior GPT-2 literature; this is a targeted causal extension,
not discovery of massive dimensions or self-repair. This next test asks whether
the observed benefit is transmitted through source keys, values, or their joint
state, without identifying a neuron or assuming a query-bias explanation.

Frozen worlds: C is native. E erases both residual coordinates at source0 after
L2. B performs E and additionally zeros the two coordinates in the L3 MLP output
at source0 only. No later writer is blocked. Attention and all other coordinates
remain native. E/B first differ after the L3 MLP: L3 attention and all non-source
L3 outputs must match. Therefore the first possible transmission boundary is
source0 K/V in L4, and later-token L4 queries must initially be identical.

Use a fifth fresh held-out panel, fractions0,.05,.9,.95, four passages each from
Wiki validation,Du Bois,Hobbes and optimizer code,512tokens,natural/BOS starts.
Check actual used token-character spans against all four previous GPT-2 panels
and each other. These edge-of-source fractions may include front matter/code
headers and should be described as such, not silently curated after results.
No selection or calibration uses this panel. All indices zero-based; fixed
FP32 checkpoint/SDPA/seed20261021; score targets2..511.

Retain native C,E,B results and all source0 K/V projection donors for E and B
in layers4..11. In each of E and B recipients, independently choose allsource
keys from E/B and values from E/B, giving a full2×2factorial in each recipient.
Patch only K/V slices of the native combined QKV projection, keeping query
outputs and every later-position projection unchanged. Require each diagonal
donor to reproduce the corresponding donor world's scored losses/argmax IDs
exactly, and same mixed K/V in E/B recipients to agree. These are architectural
closure checks rather than discoveries.

In each recipient also test zero source values with endogenous keys; zero source
values with E keys clamped; an explicit causal-mask reference; and a mask that
blocks queries1..end from attending source0 in all layers4..11. Query0 keeps its
self attention. Report reference drift and require exact scored-mask closure
between E and B under full source blocking. Native repeat and source identity
copy anchors run for natural window0 in each corpus. Total19conditions/input
(3worlds+8factorial+8zero/maskconditions) plus8anchors =616forwards/records.

Preserve per-token losses/KL/argmax, all source residual and later writer vectors,
projection donors, first-boundary query equality, patch-locality counters and
mask closures. All parameters stay immutable and hashes/version counters must
match. Report conditional contrasts against E as well as C, all corps/start
groups, n=4paired bootstrap intervals2000draws with fixed seed, and native-to-
intervention KL/argmax agreement. Mixed-state damage and value-zero versus
source-block contrasts are not additive mediation fractions or evidence that
values carry linguistic content. No faithful generation or speedup claim.

For the 2×2 NLL factorial define I = NLL(KE,VE) - NLL(KE,VB) -
NLL(KB,VE) + NLL(KB,VB), and report both simple key effects and both
simple value effects. Use the E recipient as the unique analysis panel;
duplicate recipient/diagonal records are closure controls, not additional
independent samples. Compare source blocking against its explicit causal-mask
reference and value zeroing against its native reference, and report both
the raw contrast and the reference-drift-adjusted contrast.

Before execution independently inspect hooks and run CPU synthetic checks.
One local GPU model at a time; preserve failed checks. Independent numerical
review is required before interpreting the results as established.
