# Independent V2 implementation review

Status: ready for the frozen 2,032-forward V2 run. Static policy review, independent CPU integration and adversarial analyzer fixtures passed. No launch blocker found. No experiment code was edited by this reviewer. No pretrained model or GPU was loaded.

Reviewed runner `8a1b6b08b5d1e0d379247455fda610ebb98a4c3102161d37cc6531c8ba807dda`, analyzer `c96d43612206d652b5171efa2ecb028740c04e43d79bc92f257775bad7a51f66`, protocol `40d8410fa20cd806699abb023c3cad17deba0695acfd2ad26ca97f47343eb48c`. The final runner delta adds failure-phase/resource metadata without changing inference or guards.

## Policy and causal scope

The implementation matches the approved post-failure amendment. It leaves V1 unchanged and failed; fixes the original map/raw byte hashes and tensor digest before loading a model; copies and verifies input/selection provenance; and performs zero new calibration forwards. No fitting, rank selection, precision switch or failure-dependent substitution occurs in V2. The primary source can be the original calibration or a deliberately supplied byte-identical rebuild. Nonidentical rebuilt maps are rejected.

The scaled hidden check uses symmetric FP64 per-query and whole-prefix L2 denominators, rejects zero/nonfinite denominators, and enforces the frozen 1e-4 local and 1e-5 global bounds. It retains the 1e-3 maximum shared loss guard and 1e-4 attention/context guards, while recording old absolute-threshold exceedances and top-two margin/mismatch diagnostics. No epsilon denominator, clipping or automatic relaxation is introduced.

The 21 conditions retain the original GateSubspace arithmetic. Query-wise projected gates do not pool future positions. Source Q/K/V and query0 identity assertions are relative to the zero-source-V P world, not native C: setting source V to zero already affects query0. Active gate edits preserve P's query0 context. The analyzer retains both native-C and P loss/KL/agreement comparisons; small loss alone does not establish functional fidelity.

Every longer input receives all 21 suffix rotations with the same length, absolute positions and mask representation: 672 controls, separate from 1,360 scientific records. The scorer computes exactly 128 chunk32 log-softmax rows, gathers only unchanged targets1..127, and uses targets2..127 for primary comparisons. It never gathers changed target128 or scores altered suffix outcomes. Query127 logits/distribution distances remain available. Runtime equality checks cover prefix logits, hidden states, current/target gates, source Q/K/V, masks and current/delivered contexts. Full original/control arrays are preserved on equality failure; pending originals and available current state survive earlier errors.

Successful logits and contexts are mostly represented by separately computed hashes and flags, with fixed raw snapshots. A saved-artifact auditor can recompute the saved hidden/gate/source/metric comparisons, but cannot reconstruct all absent successful logits/contexts from hashes. This limitation is explicit in protocol/analyzer output. Exact controls test this prescribed suffix change family, not universal invariance. Main/control accounting is 2,032 returned forwards and 2,032 attempts on success, 1,488 main checks, 680 cross-length pairs and 15,360 gate checks. Attempts and consumed control forwards remain distinguished on failure.

All n=4 tables remain, alongside complete unexposed tables excluding Wiki window0 across both identities and lengths. Both bootstrap functions use actual n=3 for that Wiki subset, with fixed seed and 2,000 resamples. Controls never enter scientific summaries; the subset is not an independent dataset or a multiple-comparison correction.

## Independent CPU evidence

`gpt2_gate_v2_independent_tests.py/.json` passed at the final hashes. It uses a randomly initialized 12-layer/12-head/width768 GPT-2, inner width16 and vocabulary89, with explicitly nonzero attention QKV/output biases. Twenty-one model forwards cover ten representative condition pairs plus a short P prefix: C/P/P_native/current/fullrank, retained and removed PCA, QR removal, position and global controls. Prefix logits, hidden states, contexts, source K/V and dual-reference metrics are exact under the rotated suffix; current/fullrank are exact P controls. Independent FP64 KL(P) differs by at most 3.54e-7. All twenty gate mappings separately pass a future-query perturbation locality fixture. Hook cleanup is complete.

Additional fixtures verify the symmetric norm formula, old-bound counts, zero/NaN/Inf rejection, actual-n3 bootstrap and both reference summaries. A fake scorer with out-of-vocabulary labels at every changed target128 onward completes, demonstrating those labels are never gathered. An injected nonfinite KL failure retains logits and counts the consumed forward. Peak reviewer integration RSS was 1,026,680 KiB. One early reviewer fixture supplied a 128-row reference to a 127-row short prediction array; the fixture was corrected to the native scoring shape. This was not a runner defect.

The author's separate all-21-mode CPU integration and complete fabricated analyzer smoke provide supplementary evidence; they are not represented as independently authored tests. The independent analyzer guard extension explicitly credits its author-supplied full-panel generator.

## Limits

CPU random-weight tests do not establish pretrained numerical pass rates, scientific effects or GPU resources. The planned pretrained run may still fail a frozen guard; such failure must remain visible. Parameter/model integrity in an eventual numerical audit will rely on saved runner checks unless separately re-executed. Independent numerical review is still required before interpreting any rank or gate result. No GitHub action occurred.

## Final analyzer guard results

`gpt2_gate_v2_independent_analyzer_guards.py/.json` passed the full fabricated 1,360-record/672-control panel and all count/aggregation paths. Independent added corruptions were rejected: mismatched independently recorded logit fingerprint; missing suffix control; modified saved control hidden state despite refreshing its file hash; and failed manifest despite a completion timestamp. Existing scalar and raw-closure corruption checks also rejected and stale passed audit status was cleared. Actual n=3 bootstrap was checked numerically. Peak child RSS was 897,340 KiB (parent 876,120 KiB). This fixture reuses the credited author generator, while the added adversarial cases and separate random-weight integration are independent reviewer work.
