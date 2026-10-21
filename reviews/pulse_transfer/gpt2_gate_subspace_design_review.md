# Prospective review: calibrated causal gate subspaces

**Recommendation: pursue the detrended gate-logit design, retaining the paired128/512 context-length test.** It addresses the user's requested dimensionality, causal-direction and simpler-replacement questions more directly than another donor-factorial sweep. It replaces evaluation-input donor trajectories with a frozen calibration map applied to current endogenous states. No code or model execution is approved by this document alone: freeze the protocol and conditions, pass data preflight, then obtain the usual independent implementation review.

This assessment read the query/mixture protocol, implementation and completed-run audit metadata, plus the existing hypothesis ledger and calibration-input preparation. The completed mixture results were still undergoing independent numerical review when the design request arrived; provisional group patterns are motivation, not findings established by this review. No model, GPU, external action or GitHub operation was used. This note is a prospective design artifact, not an implementation.

## Why this is informative, and what it does not answer

For source0 with literal zero value and a later query q, define

`b_h(q) = logsumexp(j=1..q, score_h(q,j)) − score_h(q,0)`.

In exact arithmetic, the non-source mass is `a_h(q)=sigmoid(b_h(q))`. A calibrated map acting on the12 current head logits at a given layer/query can change their allocation profile while preserving that step's conditional non-source mixture. It needs neither future layers nor an independently run evaluation donor. However, it still computes the ordinary Q/K/V, scores and conditional mixture, so success alone demonstrates no inference speedup or FLOP reduction.

The principal competing explanation is the number of available keys. Even if non-source scores are homogeneous, `b` contains a `log(q)` term shared across heads. Globally centered covariance of b can therefore be nearly rank one for a position/count reason, without a shared learned circuit. Use the proposed primary coordinate

`r_h(q) = b_h(q) − log(q)`.

The subtraction is a prespecified change of coordinates, not a claim that all position effects are removed. Real score distributions, positional embeddings and heads may leave substantial position dependence. Report globally centered spectra of raw mass a, b, r, and b after centering separately at each query position. The last spectrum is a diagnostic of variation across calibration passages at the same position. None of these descriptive spectra establishes causal importance; the interventions test that separately.

This is an analysis of **12-dimensional gate profiles per layer**, not a measurement of the rank of the complete downstream residual stream or of every changed coordinate. Applying the replacement in all eight layers gives a joint functional test. A favorable result does not locate the most important layer/head, prove one shared mechanism, establish native-trajectory restoration or resolve all earlier super-weight questions.

## Freeze calibration and intervention definitions

Use four fresh WikiTest and four fresh Hume passages, natural starts only, with512 tokens and native learned query biases. During calibration, apply the same literal source-V0 zeroing in L4–11 used by the evaluation baseline P; no gate replacement is active. Record original model C metrics if desired, but do not fit a gate map from a different causal world accidentally. Explicitly identify WikiTest as a calibration source rather than presenting the result as a standard benchmark test-set score.

At each layer separately, fit in FP64 using equal passage weights and a predeclared set of query positions. A natural choice is scored queries1–510, giving4080 rows from8 passages. These rows are correlated observations, not4080 independent texts. The final unscored query511 may still receive the frozen map but contributes neither a target nor an independent observation. Report the precise choice.

Let mu be the calibration mean of r, and U_k its leading centered covariance eigenvectors. With the current endogenous r at evaluation time:

| Arm | Supplied logit/mass |
|---|---|
| Retain leading k | `b' = log(q) + mu + U_k U_k^T (r−mu)` |
| Remove leading k | `b' = log(q) + mu + (I−U_k U_k^T)(r−mu)` |
| Position-only | `b' = log(q) + mu` |
| Constant mean logit | `b' = mean_calibration(b)` |
| Constant mean mass | `a' = mean_calibration(a)` |
| Global-b rank1 nuisance comparison | `b' = mean(b) + U_b,1 U_b,1^T (b−mean(b))` |

All means/bases are per-layer/per-head quantities frozen from calibration. Constant mean logit and constant mean mass differ because sigmoid is nonlinear; retain both as separately defined simple replacements. For logit arms, `a'=sigmoid(b')`. Centering must use calibration means, never means recomputed from an evaluation window or current query batch.

Use ranks1,3,6 without later rank selection. Full-rank projection should additionally be reconstructed diagnostically, but the full-rank causal control must copy current actual context exactly. A separate current-context copy checks the hook. For retained and removed subspaces, fit one frozen random orthonormal12-dimensional basis per layer using a declared seed; use nested columns at ranks1/3/6 and the same calibration mean. Compare retained k with random-retained k, and removed k with random-removed k. Removing k retains12−k dimensions, so retaining k and removing k are not matched retained-rank controls except at k=6.

One QR seed is adequate as a bounded matched comparison, but it does not yield a randomization significance test or prove specificity to the leading learned basis. If apparent PCA superiority becomes central, use prespecified additional seeds on a later fresh panel. Record QR sign convention, covariance normalization, eigenvalue order, basis orthogonality and full reconstruction residual. No sign choice should depend on evaluation results. Leave-one-calibration-passage-out spectrum/subspace stability summaries are inexpensive CPU diagnostics and preferable to treating every token as an independent bootstrap unit; they do not select ranks.

At each layer/query, compute b/r from that arm's **current** post-intervention Q/K state. Earlier gate changes can alter later states and mixtures. Thus this test permits endogenous network response; it is a different intervention from clamping every later mixture to a frozen donor. State this difference when relating it to the preceding factorial.

## Numerical and causal controls

Retain literal source V0=0 and query0 untouched in all gate arms. Compute actual non-source mass as a direct probability sum, not `1-p0`. Derive the stable conditional non-source mixture from non-source logits. The operational context replacement can retain the established arithmetic `c_current*(a'/a_current)`; specify the exact dtype/order, then require the same finite-value and `1e-4` context/normalized-mixture/rescaled-context guards as the reviewed mixture runner. Neither a covariance fit nor bounded a' makes division by tiny current mass automatically safe.

The completed prior mixture audit records minimum current mass about`3.31e-12` and donor gains above6.7million, so conditioning diagnostics remain material. The prior batch passed its guards; these numbers are not evidence that a new projection must pass. Preserve all failures, current Q/K/V/mask, calibration identity, current/proposed logits/masses and delivered context. Do not clip logits/gains/masses or silently refit a failing projection.

Freeze the policy for FP32 sigmoid saturation before execution. The denominator must be strictly positive and finite. A projected numerator may round to0 or1; either allow these finite endpoints explicitly and report their counts, or preregister failure at saturation. Do not introduce a floor or ceiling after observing evaluation behavior. Report current/projected logit ranges, gain extrema and calibration-range extrapolation descriptively without adapting the map.

All main arms should share the same causal-mask dispatch. Include C, the P zero-V reference, explicit/native-mask reference drift, current-context copy, full-rank copy and the stated repeated/identity anchors. Compare complete final normalized hidden hashes and per-token metrics for identity closures. Save a frozen raw snapshot subset and all calibration tensors; runtime errors/operation counters cover all remaining rows. No evaluation outcomes may influence calibration windows, ranks, QR seed, centering or tolerances.

## Calibration separation and two lengths

Proposed calibration fractions `.25,.45,.65,.85` and evaluation fractions `.075,.275,.475,.675` are prospective choices, conditional on preflight. For calibration, exclude the complete8192-character extraction buffers of earlier GPT-2 development passages and the earlier Llama fresh-calibration passages at`.2,.4,.6,.8`, as proposed. Validate against their actual saved manifests rather than assuming those fractions alone guarantee separation. For evaluation, verify actual used-token spans against all32 previously used spans per corpus and each other. Abort and document an overlap before inference; choose a replacement panel only as a new frozen design, not after outcomes.

Construct128-token inputs by taking the exact first128 tokens of each saved512-token input. Natural/BOS must differ only at token0 at both lengths. Fit calibration once at512; do not refit for BOS, shorter length or individual corpus. This makes those evaluations real transfer of a fixed calibration. Keep context-length means separate because128 scores targets2–127 while512 scores targets2–511. The shorter run and the matching prefix of512 share126 scored targets; compare those directly for causal-prefix drift, including gate values and hidden/output diagnostics where retained.

The map must operate across heads of one query only: no evaluation centering across positions, suffix-dependent normalization, or per-window PCA. This structurally preserves causality. Do not require cross-length bitwise equality blindly because GPU kernel shapes can change rounding. Freeze a numerical prefix tolerance and report loss/hidden/logit drift separately from argmax agreement, since near-tied argmaxes may change under small drift. Exact identity controls within each length still must be exact. Do not count lengths or natural/BOS starts as new independent passages; retain n=4 passage-level paired summaries within corpus/start/length.

## Counts, resources and decision rule

The complete stated control set gives **21 main conditions**, rather than19:

- C, P_native, P, current-copy, fullrank-copy:5.
- Mean logit, mean mass, position-only, global-b rank1:4.
- Detrended-PCA retain/remove ranks1/3/6:6.
- QR retain/remove ranks1/3/6:6.

With64 input/start/length pairs,16 repeat/identity anchors and8 calibration forwards, that is **1368 forwards**. A19-arm design is possible only by moving specified controls to a declared anchor subset or dropping them explicitly. The roughly10% difference from the proposed1240-forward estimate is not a scientific reason to remove the informative paired-length test. Prefer the clear21-arm accounting unless actual resource constraints require a smaller control subset. Freeze expected record/check counts separately from calibration forwards.

One sequential GPT-2 worker remains tractable based on the prior local batches; do not extrapolate a precise runtime before measurement. Covariance matrices are tiny12×12 objects. Retain/shard calibration b/a/state data and a limited raw-QKV snapshot panel; do not accumulate all attention score matrices. Independent CPU audit should remain under3GiB. Avoid concurrent model loads and report calibration, evaluation and analysis resources separately.

Use NLL change from **P zero-V** to measure the added cost/benefit of gate replacement, while still reporting total fidelity to native C. Save KL/argmax agreement to P as well as C; otherwise the preexisting source-value-zeroing drift is confounded with the new map's output deviation. Define retained/removed-vs-matched-random contrasts prospectively and show all fixed ranks and simple controls. Four-passage bootstrap intervals are conditional, small-sample summaries, not multiplicity-adjusted evidence of population-wide behavior.

A useful outcome is not necessarily low rank. If the position-only or constant control performs comparably, that is evidence favoring a simpler gate description in this restricted setting. If leading-subspace retention preserves loss but random retention does too, do not claim a specific leading direction is necessary. If leading removal harms loss more than random removal, it supports importance of that calibrated gate direction for this joint intervention; differing downstream feedback still prevents additive causal-attribution percentages. If all replacements fail, preserve that result and investigate whether corpus/start shift, tail directions, or numerical conditioning explains failure without choosing a successful rank after the fact.

**Disposition:** this is a high-value, bounded next tranche after the Report10 milestone. It does not require another parameter-value sweep or UIT work. Save the explicit21-arm or amended19-arm protocol, preflight the new spans, and then review the implementation before GPU execution. No access blocker was encountered; the previously denied GitHub action is unrelated and must not be retried.
