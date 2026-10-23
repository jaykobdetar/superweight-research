# Content-donor numerical and causal audit

Reviewed 2026-10-23. No learned forward, download, parent-file edit or model-weight computation was performed by this auditor. Checks are independent compact token decoding, synthetic-array algebra, extracted actual function bodies and read-only completed-result verification.

## Conclusion

No donor-direction, query-index, routing-replay or value-replacement mathematical error was found in the reviewed execution version. The worker fixed the initially flagged validation/cache/run-identity gaps before any outputs and retained the immutable runner once execution began.

Final reviewed content runner SHA256: 29a76fc2e8635b9d5e8b94bfde52107cf936152a089f56d22dcfb3f408749501.
Algebra SHA256: 72a56fb4385c0f56dd1bb963384690e471ddccbf409ad520d0f6a175b33826be.
Input manifest SHA256: 7c7afdea710322f8bfe287b0146d370637d5dce3baf6fcd8c86ff5a813f95ff1.
Pre-output amendment SHA256: 3fb81976e864b9731fb269d074f0d15208cd80e33a619ea0a5aafef409a2ba8b.

## Inputs and scoring

All forty texts (two recipients, two paraphrase donors and one irrelevant donor per pair) independently decode exactly from the pinned tokenizer vocabulary bytes. All have the declared 17–20 tokens, token510 at source0, exactly one newline187 at the declared p=10–12, and a common post-p suffix. Recipient pairs and paraphrase pairs each differ at exactly one fact-label token, p−2. Token-list hashes use ASCII Python-list serialization; runtime sequence hashes use little-endian int64 bytes. These are distinct named identity conventions, not equal digest values.

Every query is q=n−2, so its next token is the common appended A-label token. The scored hidden state cannot attend to that future token. B recipients are nevertheless scored for label B by selecting their correct head row, not by treating the common observed A target as correct. The two-logit margin equals the full-vocabulary correct-minus-foil log-probability difference; correct-label NLL uses the shared full-vocabulary normalizer. A synthetic test of the actual score function verifies both orientations and the distinction between observed-target and correct-label loss. Final LN is applied before the head, and the two-row manual A logit is checked against the streamed full-vocabulary target logit.

A positive margin is preference over the specified foil, not necessarily a full-vocabulary argmax or a claim of calibrated factual competence. The separate full-vocabulary argmax is saved. There is no future-label leakage from appending A.

## Value algebra and routing scope

At a patched source s and later query q, the context correction is P[h,q,s] times (Vdonor[h]−Vcurrent[h,s]), concatenated over heads. Queries ≤p receive zero correction. Independent loops over head/query/source agree with production replay and correction within 9.89e-8 and 2.59e-8 on synthetic FP32 arrays.

The primary frozen_all branch replays each recipient world's baseline P at all sixteen layers, multiplied by CURRENT recipient V. It does not replay the old complete context or freeze suffix values. The value correction applies only in layers2–15 and queries>p. Thus Q/K/routing changes induced by earlier suffix perturbations cannot change the P used by later attention, but suffix V/MLP pathways remain endogenous. Native and post-boundary-source0-masked baselines are separate routing worlds.

The secondary branch freezes P only at the preselected layer8, asserts that P still equals baseline there, and leaves later routing endogenous. The actual extracted closure was tested at layers0,1,2,8,15 in both scopes. Both branches match their formulas exactly on synthetic arrays. They estimate different causal effects and must not be pooled.

Same-label donors use the corresponding independent paraphrase; opposite-label donors use the reciprocal paraphrase. Both have the same position p and donor suffix. Native donor-prefix V is valid for both recipient worlds because source0 masking begins after p and cannot alter any donor/recipient prefix state ≤p.

Identity uses recipient p V at p. Cross-content source0 uses the irrelevant donor's source0 V at source0. All sequences begin with the same first token, so this is an exact causal null in real arithmetic; in the mask0 world it is additionally nulled by zero routing mass. In contrast, p_from_source0 replaces V at p using recipient source0 V while preserving p routing. It is substantive, not a null control. It removes later-fact information from the supplied value vector only; K/P and other content paths remain intact, so it is not complete information deletion.

## Ordinary-source controls

The raw control substitutes the opposite-label paraphrase's fact-token value at recipient fact position p−2. It is a known-carrier reference, not content-free. It is two tokens closer/farther in source distance than p for the same query, and the documentation correctly disclaims exact distance matching.

The norm-matched control scales each query's aggregate context correction so its post-W_O norm matches that of the mismatched newline-p correction at the same layer/query. A scalar commutes with linear W_O, including the contributions of all heads. Independent FP64 output norms agree within 3.87e-7 in the synthetic test; an impossible zero-to-nonzero direction match is rejected. Prefix corrections remain exactly zero.

Because the scale varies by query, this is a query-dependent message perturbation, not a single fixed donor V vector. It also need not remain inside a naturally produced/clipped value range. Matching norms does not match direction, semantics, routing mass or source distance. The pre-output amendment and updated runbook make this distinction explicit.

## Safeguards and reviewed changes

Initial review found that cache resumes lacked NPZ checksums and run_meta did not bind the content-runner/algebra code. Those issues were corrected before outputs. Current execution requires ten individual passing FP32 validation rows, complete/all_pass, and a matching core-model SHA. Both RLIMIT_AS and model RSS cap are 4,000,000,000 bytes; BLAS/OpenMP/MKL are set to two threads before NumPy import. The <1 GB target is a target, not the hard cap.

Run metadata binds the unchanged input manifest, contrast protocol, amendment, content runner, algebra, core model, and validation JSON/NPZ hashes. Result identities derive from that metadata. Every resumed baseline/donor cache is checksum-verified, and resumed interventions verify identity and path stem. Running code remained SHA 29a76f… during review. The earlier default protocol had 32 recipient baselines and seven conditions; the separately recorded pre-output amendment adds maskp/mask0p and ordinary_raw, making 64 recipient baselines +24 donor baselines +256 primary interventions =344 total. A single-pair baseline smoke is eight forwards. The updated runbook agrees.

Minor non-blocking reporting concerns were conveyed: two baseline display-summary reads in the frozen runner do not independently call verified_result, although the baseline load path has just verified them; the separate donor summary originally needed identity checks for every input JSON. The worker subsequently hardened the separate donor summary to check identity/manifest/path stem and labeled baseline prefix-control counts explicitly; those analysis-only changes were rechecked, with the forward runner unchanged. The input manifest's all_baselines_archived=true was a prospective protocol declaration at input freeze, not proof that baseline execution had already occurred; actual result receipts establish completion.

## Completed baseline results independently verified

The worker's authorized baseline run completed during this audit. All64 result identities and cache SHA256 values verify. All16 recipient-label groups have exact sixteen-layer prefix-hash equality across native, mask0, maskp and mask0p. Every expected masked attention-probability entry is zero in layers2–15 at queries>p. Cached dimensions and every correct/foil score orientation are consistent. Maximum manual observed-target logit discrepancy is 7.6294e-6, below the frozen 1e-3 tolerance.

The predeclared usage cost is correct-label NLL(maskp)−NLL(native), or NLL(mask0p)−NLL(mask0), not the common appended-A target loss for B recipients. Its indexing is correct. The >0.01-nat reciprocal-pair mean usage gate passes3/8 pairs in native routing and2/8 in mask0 routing. Across all eight, mean costs are −0.00584360 and −0.06281794 nats respectively. Signs are heterogeneous, and the usage gate is separate from binary discrimination. This substantially limits interpretation of a later all-panel donor null as evidence against payload.

No donor-intervention effect is established by this audit. Retain all pairs, show any baseline-qualified or usage-qualified subset separately, keep the routing worlds/scopes separate, and do not infer abstract semantics, natural-prefix reachability, long-corpus sink behavior or population prevalence from these eight hand-built pairs.

## Reproducibility

check_compact.py: token bytes, reciprocal positions and independent value algebra.
check_runner_logic.py: actual score/kwargs/hook body tests with synthetic arrays.
check_baseline_receipts.py: all64 completed baseline identities, cache hashes, prefix invariants, masking zeros and score consistency.
MANIFEST.json records reviewed immutable source snapshots and audit outputs.
