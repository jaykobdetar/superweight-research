# GPT-2 layer-local conditional coordinate writes

Declared before this batch,21 October2026. Previous fresh E/A/M experiments
showed that both erased source coordinates reappear after layer3 and later MLP
write blocking is consequential only after erasure. They did not establish
whether layer3 alone suffices or later layers can independently recover useful
source behavior. This follow-up tests all later layers without selecting neurons.

Keep the frozen coordinates[447,138], model revision, FP32/SDPA, seed20261021,
source0 scoring and native attention unchanged. E erases the two residual
coordinates at position0 after layer2. A layer-set block zeros those coordinates
only in the selected layers' MLP down-projection outputs at source0, before
residual addition. It does not block attention writes or reset the residual
again. All other MLP coordinates and positions are untouched; later computations
remain endogenous.

Use a fourth fresh panel at fixed fractions .25,.45,.65,.85, natural/BOS starts,
512tokens, four passages each from Wiki validation,Du Bois,Hobbes,optimizer code.
Check actual used token-character spans against all three prior GPT-2 panels
and each other, aborting if overlap occurs. No calibration or new candidate
selection is needed for this finite prespecified layer family.

First run E×block-layer3×block-layers4..11 (eight combinations). Then under E,
allow exactly one MLP layer to write those coordinates, for each layer3..11;
deduplicate the layer3-only condition already present. Also block successively
longer prefixes3..k, k=3..11; deduplicate existing sets including the only-layer11
condition. Finally, two frozen random-coordinate controls[588,104] use E with no
MLP block and E with all later MLP writes blocked. This produces24 unique
conditions per input, plus eight clean repeat/identity anchors:776 forwards and
records. Save the complete label/set map before evaluation, including aliases.

Use the same-input native baseline, scoring targets2..511. The only-layer11
writer condition must match the all-later-writers-blocked condition exactly on
scored per-token losses and argmax IDs: a final-layer source0 MLP output cannot
influence later positions. Treat that as an architectural negative control.
Native repeat and source residual identity-copy anchors must be exact. Assert
modified MLP coordinates are zero and all other coordinates/positions are
unchanged. Preserve all source residual/attention/MLP vectors and reconstruct
the residual additions independently: residuals after all12 blocks, attention
and MLP outputs only in instrumented layers3..11. Attention remains native, so
blocking all MLP writes does not keep the residual coordinates identically zero.
Model parameters remain immutable.

Report each layer's conditional loss/KL/agreement and coordinate trajectories,
not just the best layer. Compare block-layer3 versus E-only, block-later-layers
versus E-only, and their NLL difference-in-differences. A later unblocked layer
that improves outcomes is not automatically the same mechanism as the original
writer, and approximate loss preservation is not native trajectory restoration.
Any single-layer sufficiency claim is conditional on this coordinate-write
family with attention and the rest of the network still active.
Bootstrap paired windows n=4,2000draws,seed20261021, without selection-adjusted or
population-wide claims. The condition family is exploratory localization on
fresh held-out inputs; any neuron selection it motivates requires separate
perturbed development inputs and new evaluation.

Record exact input/model/code/protocol identities, raw outputs, all closures,
forward count and GPU/host/wall accounting. Run only after the six-batch portable
validation releases the GPU. Independent CPU review precedes scientific claims.
