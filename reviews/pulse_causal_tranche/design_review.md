# Independent CPU review — implementation and design

Reviewed `DESIGN.md`, `run_tranche.py`, supporting intervention code, the pinned
model configuration, and installed Transformers OLMo source. No model was loaded,
no GPU was used, and the runner was not edited. Synthetic tests used approximately
493 MiB peak RSS. The tested runner hash is recorded in `cpu_hook_checks.json`.

## Findings

1. **K/V intervention placement is correct.** Linear forward hooks intercept
   projection outputs before native Q/K/V clipping, reshaping, and RoPE. Captures
   use detached clones, avoiding the subsequent in-place clipping alias issue.
   The pinned configuration has 16 attention heads, 16 KV heads, and 128 features
   per head, matching slices. The pre-`o_proj` contexts have contiguous head
   blocks in the installed implementation. Decoder outputs are tensors here.
2. **Direction, scope, and sign are correct.** Clean donor into deleted recipient
   is rescue; deleted donor into clean recipient is damage. K-only/V-only/combined,
   zero/all positions, full head contexts, and pulse sabotage are separate. The
   residual delta is deleted minus clean; rescue subtracts it. Positions below 9
   remain untouched by residual patches. Same-state K/V controls cover both
   recipient states. CPU tests passed these localization and identity properties.
3. **Initial FP16 subtraction concern was resolved before launch.** Current code
   converts both operands to FP32 before subtracting. Typical synthetic full-delta
   patches reconstructed donors exactly. However, additive reconstruction is not
   universally bitwise equivalent to assigning a donor tensor: with clean=65504
   and donor approximately 0.0003, FP32 subtraction/addition reconstructs 0. This is
   a numerical caveat, not evidence it affects actual measurements. Call this a
   full-delta patch unless actual donor reconstruction has been measured. At final
   block 15, the full rescue should approximately recover clean `later9` losses,
   and full damage should approximately recover deleted `later9` losses. Earlier
   blocks need not match because unpatched positions 0–8 can affect later tokens.
4. **Centered covariance is implemented correctly.** It pools development tokens 9+
   as observations, subtracts the feature mean, divides by n−1, and uses descending
   eigenvectors/eigenvalues. Squared singular values equal eigenvalues times n−1.
   The stored mean energy is separate. PC interventions project the *raw* delta
   onto directions fitted from centered covariance; held-out retention projects
   *centered* delta. These are legitimate, distinct estimands and must be labeled
   explicitly. `mean_pc8` instead adds the mean and centered top8 reconstruction.
   Random8 uses one fixed random subspace per layer and per-token norms matched
   to raw-delta PC8. It does not characterize variation over random subspaces.
5. **Selection does not use held-out outcomes.** Nominees are prespecified;
   context damage/rescue thresholds and constants are fitted on eight development
   windows. Twelve causal evaluation windows are separate. Calibration, development,
   and evaluation token spans are disjoint under the enforced corpus length and
   start-spacing assertions. Old anchor provenance identifies WikiText validation;
   new WikiText uses test. The 8-start generalization panels reuse the first eight
   causal evaluation starts; they are paired panels, not independent replications.
6. **Start-token comparisons preserve suffix IDs.** ID0 is replaced after
   tokenization, same starting offsets are used for all identities and lengths,
   and fresh amplitudes are fitted on separate calibration starts. At position 0,
   causal attention cannot use later context. Repeated artificial first-token IDs
   therefore do not constitute16 independent amplitude observations; dispersion
   mostly probes determinism for these conditions. Natural start variability is
   token-distribution variability, not within-token context variability.
7. **Boundary scoring is aligned correctly.** A continued sequence's losses at
   predictors 512–1022 score the same target IDs 513–1023 as a reset suffix's
   predictors 0–510. But ablation at global position 0 in the continued run and
   local position 0 in the reset run changes source identity and source distance as
   well as available context. This is a compound boundary diagnostic, not an
   isolated effect of reset/context. Uniform RoPE offsets are a useful numerical
   invariance control; interpret treatment effects relative to offset-matched
   clean scores, with zero-offset clean reported separately.

## Required numerical/reporting checks when results arrive

- Verify replication error, unchanged model hashes/parameter versions, complete
  condition/window counts, finite losses, and all identity controls.
- Pair by window and token hash within each condition family. Compute window mean
  differences before bootstrapping; never bootstrap individual tokens as IID.
- For rescue, compare against deleted; for damage, against clean; for pulse
  sabotage, against pulse. NLLs recorded in `windows.jsonl` use clean as baseline,
  so they are not already the rescue or sabotage contrast.
- For each start identity/length, compare old/fresh pulse with the matched clean
  and deletion states. Length comparisons should use shared first 127 losses and
  preserve window pairing. The later9 range overlaps first 127; do not present
  these as disjoint bins. The suffix512 range is predictor indices 512 onward.
- Bootstrap paired windows with the frozen seed and 2000 draws. Transform mean
  NLL differences and interval endpoints with exp for perplexity ratios. State
  these are conditional, pointwise intervals with n=8 or12; many comparisons,
  shared corpora and repeated starts limit broad significance claims.
- Verify covariance symmetry, trace/eigenvalue agreement, mean decomposition,
  saved raw-delta dimensions, orthonormal vectors, retention bounds, selected-set
  threshold replay, and equal-size layer-matched replacement controls.
- Preserve PC raw-delta versus centered-retention labeling. Do not infer necessity
  from large activation changes, add head effects into mediation percentages,
  or describe replacements as measured speedups.

The runner collects raw observations; it does not itself implement the promised
bootstrap summaries. That remains an analysis-stage requirement. Raw development
clean/deleted per-token baselines are not separately saved, although selection
effects and deltas are saved; avoid claiming those baseline arrays are available.
