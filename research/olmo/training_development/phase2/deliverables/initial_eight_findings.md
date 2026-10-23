# Initial eight-checkpoint study: separately frozen results

Completed 22 October 2026. This record excludes the subsequently selected four-checkpoint follow-up.

## What was tested

Official allenai/OLMo-1B-0724-hf revisions at 2k, 4k, 8k, 56k, 80k, 88k, 120k and pinned main d7cbab742d80589e714b1a2d7f838dcd21cbe143. Four fixed synthetic prose probes, 13 conditions each, five fixed single tokens, and four selected MLP neurons (1710, 4820, 5755, 8041). CPU NumPy prefix only: embeddings, all of layer 0, layer-1 attention, and the selected layer-1 gate/up rows. Scalar contributions are measured through output row 1764, not full model loss or importance.

The initial checkpoint/prompt protocol SHA256 is 03b2692ddf8c39a4cb13e9cf089757d1a3cd099440fc3420e812c033f345b899. FP16 rounding sensitivity was added before any earlier-checkpoint inference and is explicitly not a CUDA SDPA equivalence claim. The pinned-endpoint validation passed numerical source-pulse references, causal/self-only invariants and qualitative H14/H12 effects on the new probes.

## Main observations

1. Early activity is not confined to the final primary. At 8k, the single-token “The” contributions are -1.52 for neuron 1710, -15.65 for 4820 and -6.51 for 5755.
2. The intermediate division of activity is token-dependent. At 56k, “The” is carried mainly by 1710 (-18.67), but a period is carried by 4820 (-449.16) and 5755 (-112.48); 1710 contributes only +0.0010 to the period. Their SwiGLU gates differ accordingly.
3. A large down-projection coefficient does not date the mature mechanism. The primary coefficient is largest in the row at the sampled 88k point and reaches its sampled trajectory maximum at 120k. Yet the final-identified H14/newline and H12 empty-history interventions do not have their mature effects on these probes through 120k.
4. At 120k the primary source contribution is -152.26 for “The,” but -0.106 for a period. Finally it is -258.52 and -253.87 respectively. The final primary coefficient is smaller, not larger, than its 120k value.
5. In the final model, inserting an earlier newline reduces the later-newline primary contribution from a median -299.55 to +0.00055. Removing the relevant H14 attention edge restores -305.01, while the H0 edge control does not. At 120k the corresponding three medians are +0.00216, -0.00041 and -0.00040. This study therefore has an important unmeasured later interval, which motivated the separate adaptive extension.
6. At the final model, the H12 donor patch restores a median 1.842 times the isolated-versus-ordinary empty-history gap; its H0 control does not. At 120k the H12 fraction is -0.00636. These fractions are not used when the denominator has magnitude <=1.

## Exploratory cross-time compatibility test

This was added after the eight-checkpoint results, with no new prompts, checkpoints or downloads. For the single-token “The,” the 120k prefix plus its own selected-neuron readout gives -152.26 and the final prefix/readout gives -258.52. Final readout on the 120k normalized prefix state gives +0.465; 120k readout on the final state gives +1.744. Both hybrids turn the gate negative. Gate-row and up-row cosine similarities between 120k and final are about 0.075 and 0.144; the normalized prefix-state cosine is about 0.507.

This rejects a description in which a fixed native-coordinate upstream feature is merely amplified by the scalar. It demonstrates cross-time incompatibility/coordinated geometry change. It does not distinguish functional coadaptation from a broader change of representation basis, and it is not a training intervention.

## Analytic bound on the paper-listed second weight

The pinned implementation uses non-affine LayerNorm with epsilon 1e-5, hidden dimension 2048, SiLU activation, and bias-free gate/up/down projections. For normalized input h, sum(h)=0 and ||h||^2 <= 2048. Centering the gate and up rows g and u leaves their dot products with h unchanged. Because |SiLU(a)| <= |a|,

|w * SiLU(g.h) * (u.h)| <= |w| * 1024 * (||g|| ||u|| + |g.u|).

The final W[1764,8041] bound is 0.1603298493, before floating-point roundoff. It is a prompt-independent upper bound on this one scalar contribution in this exact checkpoint. It rules out a hidden primary-sized pulse through this scalar, but not every possible downstream effect or a different model revision. The bound for neuron 1710 is 1262.37, so it is consistent with the observed approximately 258 pulse. This derivation is not valid unchanged for affine LayerNorm/RMSNorm, biases, or other activations.

## Audit and limitations

See study_audit.json and precision_audit.json. All eight checkpoints and 2,096 downloaded tensor slices were rehashed. Counted transfer: 2,715,887,454 bytes, below the 2.75 GB cap; peak inference RSS: 405,008,384 bytes. All causal-prefix checks passed. For the stored probe contributions, the maximum precision-scaled difference abs(FP32-FP16approx)/max(1,abs(FP32)) is 0.00777; all contributions with FP32 magnitude >1 retain their sign. These checks are not bit-exact full-Transformers validation.

The study does not measure end-to-end loss, whole output-column geometry, optimizer moments, real historical training batches, or training-time counterfactuals. The four prose probes and five source tokens are not representative corpus samples. Tracking final-identified H14/H12 does not rule out analogous earlier functions in other heads. Absence here means absence of the mature measured effect on this fixed panel, not absence of all possible precursor functions. Official HF checkpoint provenance does not supply the missing certified native optimizer-state lineage or repair the published training-config mismatch.
