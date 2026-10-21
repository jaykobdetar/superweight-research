# Continuation factorial: independent CPU review

Reviewed `HYPOTHESES.md`, `factorial.py`, `worker.py`, `analyze.py`, and completed
factorial outputs. No experiment edits, model loading, or GPU use. CPU synthetic
and numerical checks are preserved in this directory.

## Implementation and scoring

- Source ablation substitutes only L1 output coordinate 1764 at the requested
  position, using the temporary projection with weight[1764,1710] zeroed. Model
  hashes and parameter versions match after the run. The worker executes jobs
  sequentially and experiment hooks are removed in `finally`.
- Q/K/V patches correctly intercept preclip projections: Q at all positions,
  K/V at position 0, selected head only. Captured clones survive native clipping.
  All later Q/K/V values are bitwise identical between clean and source-deleted
  runs before block 2 attention. Q-only interventions, and adding Q to corresponding
  K/V conditions, have exactly zero effect on saved losses, predictions, and KL.
- Normalization coordinate/complement/full modes replace the requested coordinates
  of the normalized source at block 2; all later positions stay unchanged. Donor
  scale uses the donor population variance plus epsilon, keeping the recipient's
  centered raw input. The formula matches the intended sigma intervention. These
  are normalized-vector interventions; a coordinate swap after normalization is
  not an isolated intervention on that raw coordinate before normalization.
- Full normalized-vector replacement affects every attention head at block 2 but
  preserves the recipient residual skip path. Incomplete rescue is therefore
  compatible with a large attention-path effect; it does not refute the patch.
- KL is correctly oriented clean||condition, using full-vocabulary float32
  log-softmax. Baseline clean values are zero. Minimum saved token KL is
  −4.31e−7, ordinary finite-precision error, not negative information divergence.
  Do not infer distributional equality from NLL or argmax agreement alone.
- Prefix scores use predictors[p:p+511], targeting exactly suffix tokens 1–511
  for every prefix length. Rebuilt token hashes and target equality passed. Both
  first source and suffix boundary tokens are `The`. Prefix length changes history
  and distance, and the negative-control deletion has a context-dependent source
  activation; it does not isolate absolute position as the only changed factor.
- New spans are explicitly checked against conservative prior-tranche spans.
  Windows are the inferential units; the fixed-seed 2,000-resample paired bootstrap
  in `analyze.py` uses correct matched-window differences and token-hash checks.

## Attention diagnostic precision issue

`diagnostics()` originally omits native FP16 rounding after applying RoPE.
Native OLMo returns rotated Q/K in their original dtype before SDPA. CPU replay
using the saved caches reproduces original diagnostic tensors exactly and also
computes the corrected reconstruction. Across 64 head/state/window combinations,
maximum relative context error falls from 0.0008414 to 0.0003092 after restoring
native rounding. Maximum mean key0 mass change is 0.00002204. This is a diagnostic
fidelity issue with negligible impact on the present interpretation; factorial
model outputs are unaffected. Keep original provenance and use corrected CPU
diagnostics for future precise reporting. Full normalized-intervention caches
were not saved, so their diagnostic summaries cannot be identically recomputed.

The decomposition `context = p0*v0 + (1−p0)*conditional_nonzero_mixture` passed
numerically for every saved natural attention condition. Native SDPA context
itself remains slightly different from FP32 explicit softmax reconstruction.

## Completed measurements

All 1,488 observations, 186 condition groups, and 1,424 forwards are present. Saved
NLL/KL means, baseline means, argmax agreements, token hashes, source localization,
prefix target alignment, and Q-null effects passed. Peak numerical review RSS is
approximately 579MiB, below the 3 GiB review budget.

- Hume independently replicates interacting K0/V0 damage: K-only +0.08584 nats,
  V-only +0.07254, joint +3.88627. WikiText gives +0.04660, +0.61294, +5.49233.
  Q cannot explain the initial broadcast under this source-localized intervention.
- Swapping normalized coordinate 1764 alone causes little damage but makes rescue
  worse. Complement and donor-scale interventions have much larger effects.
  Hume full/complement/donor-scale rescue leaves roughly 0.150/0.182/0.188 nats
  above clean, while coordinate-only rescue adds 1.004 nats to deletion. This is
  strong evidence for a broad normalization-mediated route, not a unique or
  complete causal circuit. The source coordinate still matters upstream through
  normalization; weak post-normalization coordinate swap effects do not contradict
  that role.
- Original-source deletion and L2/H15 joint damage remain large when scoring the
  same suffix after longer prefixes. Deleting the same scalar at the later `The`
  boundary while keeping the original source intact changes mean NLL by at most
  about 4.1e−5 nats in these cells. This supports first-source/context specificity,
  without proving literal-position dependence or equivalence on other prompts.

No blocking defect was found in factorial inference or scoring. The concrete
outstanding correction is attention-diagnostic post-RoPE rounding; it is quantified
above and does not require redoing model inference.
