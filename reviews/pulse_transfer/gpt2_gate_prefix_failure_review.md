# Independent audit of the failed native-prefix guard

**The frozen gate study failed and remains failed.** It stopped after33 forwards (8 calibration plus25 evaluation forwards), with24 evaluation records. There is no completed gate-subspace result or full-panel causal conclusion. The frozen source hashes still match. The failed manifest has no completed after-run model-hash/version closure, so this audit does not certify that unexecuted completion check. This audit loaded only saved CPU tensors, with538644KiB peak RSS; no model or GPU inference occurred.

The failure occurred while recording native C at512 tokens against native C at128 tokens on Wiki validation window0, natural start. No gate replacement was active in either member of that comparison. The1e−3 maximum hidden-entry guard failed, while the1e−3 shared-loss guard passed. The error is not an attention reconstruction or gate-formula failure: those failure holders are null. The prior128-token current/fullrank controls are exact in saved metrics and final-hidden hashes.

## Independent quantification

All hidden statistics below use final `ln_f` states and FP64 subtraction of the saved FP32 tensors. Positions are zero-based. Source query0 is excluded from scoring; shared scored queries1–126 predict targets2–127. Query127 has no target in the short input and is reported separately.

| Region | Maximum absolute drift | RMS absolute drift | Relative L2 drift |
|---|---:|---:|---:|
| All128 shared hidden states |0.002273560|1.376497e−5|1.623847e−6|
| Source query0 |2.670288e−5|1.348826e−6|4.922452e−7|
| Scored queries1–126 |0.002273560|1.387126e−5|1.626904e−6|
| Unscored shared query127 |4.577637e−5|2.647432e−6|4.804833e−7|

The largest entry is query62, coordinate430: short118.330078125 versus long118.327804565, a signed long-minus-short difference−0.002273560, or1.921371e−5 relative to that short entry. The next largest entries are coordinates496 and36 at the same query. The five entries above1e−3 occur at queries62 and110;25 entries exceed1e−4. The whole-prefix median absolute difference is3.576279e−7 and its99th percentile is3.453684e−6. Maximum per-position relative L2 drift is1.915251e−5, at query110. Small relative/RMS values describe the scale; they do not override the failed preregistered absolute test.

Across126 shared scored targets, argmax agreement is100%. Maximum absolute NLL drift is0.0001621246, mean absolute drift2.473161e−5, and mean signed long-minus-short drift+7.545043e−7 nats. Mean NLL is4.020248158 short versus4.020248912 long, giving a paired perplexity ratio1.000000755. Full logits were not saved for this pair, so the failure data cannot supply cross-length logit error, KL, or argmax margin diagnostics. Native C's stored zero KL is its own reference, not evidence of zero cross-length KL.

A failure-holder detail matters: `active_input`, `current_hidden`, `current_gates` and current source/snapshot fields refer to **P512**, which had already been evaluated before `record(C)` performed the prefix assertion. The native comparison is correctly preserved in `prefix_pairs['C']`; using `current_hidden` as native C would compare different interventions. The audit used the former and verified its short hidden hash against the native short record.

## Interpretation and bounded next diagnostic

This is observed native length-dependent drift, before any gate replacement in the failing comparison. Small relative errors, concentration in large coordinates, near-identical losses, and exact shorter within-length controls are consistent with numerical shape sensitivity. They do **not** establish that explanation or exclude suffix leakage. The saved data lack a fixed-shape changed-suffix counterfactual, hook-free native reference, and matched attention-backend comparison.

Before any protocol amendment, use a separate native-only diagnostic on the exact failed tokens, preserving the failed directory and frozen protocol. A bounded16-forward panel is sufficient for the first question:

1. For default SDPA and forced SDPA math separately, run short128 and original512 twice each, plus three512 variants: a deterministically rotated suffix, a constant suffix, and prefix-valid padding/mask. Keep positions0–127 identical and use the same native model with no coordinate/gate intervention. This is14 forwards; make the padding arm explicit rather than optional in the declared count.
2. Add hook-free native128/512 references for the default backend, checking that observer-only captures do not change the model's outputs. This brings the total to16; alternatively establish the same equivalence with an explicitly declared existing reference.
3. Save prefix logits and final hidden states, plus per-layer prefix Q/K/V, attention context and block outputs. Shard by condition. Record backend/TF32/determinism settings, exact masks/position IDs, seeds, checkpoint identity and before/after parameter hashes. Do not silently change precision flags or calibrate maps.
4. Report within-shape repeat drift; fixed-shape suffix-swap drift; short/long drift; hook-free/observer drift; and each backend separately. Use max/RMS/relative hidden and logit errors, NLL, cross-length KL, argmax agreement and minimum/mismatch margins. Locate the first layer where differences appear. Preserve all outcomes without selecting the best backend as the declared main experiment.

Exact fixed-shape suffix invariance would support causal isolation in these tested configurations. Reduction under forced math would implicate backend arithmetic, but would not uniquely identify all sources of shape dependence. Persistent differences could arise in other matrix operations, normalization or dispatch; fixed-shape suffix influence would require investigating masks/hooks before resuming. These controls are diagnostic, not a universal proof of causal independence.

Do not relabel this run passed or merely raise the threshold to exceed0.00227356. Any revised tolerance/metric must be a separately versioned, justified prospective amendment, preserve this negative result and disclose that the first evaluation input was already exposed. The saved calibration may remain frozen only if its identity and unchanged use are explicit; no rank/basis selection should depend on these partial results.

Evidence: `gpt2_gate_prefix_failure_audit.py` and `gpt2_gate_prefix_failure_audit.json` contain streaming file hashes, full per-position statistics, the top20 entries, source/scored separation and raw control checks. This is a completed audit of a failed run, not approval of a modified experiment.
