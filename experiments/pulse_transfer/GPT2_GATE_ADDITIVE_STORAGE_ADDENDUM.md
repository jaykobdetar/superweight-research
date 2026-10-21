# Additive-logit study: prospective storage addendum

Status: proposed for independent review before any additive-study model forward. This document supplements the original [proposal](GPT2_GATE_ADDITIVE_PROPOSAL.md); it does not alter or replace that file or the completed input preflight. The original proposal SHA256 remains `9542fa49a39564d5c6835f95a7ead774552f84da071d69033e86d11eff314d68`. The preflight remains `d0fa737595f63ff7c48507925a4ddb500013f5301a245878f85ee72beabee66b`, with zero model forwards. Final implementation approval is a separate prerequisite.

## Reason and scope

The actual production writer was exercised through all 1,992 orchestration steps with synthetic tensors, stubbed forward/readout functions, real serialization, copied frozen dependencies, and active disk/counter checks. No neural model or tokenizer was loaded, no GPU was used, and no new scientific outcomes were observed. The eight-row layout occupied 2,074,040,465 bytes. Reserving its declared 64 MiB failure capsule left only 6,334,319 bytes under the 2 GiB total cap. That margin is insufficiently comfortable for longer numerical JSON values and subsequent analysis files.

Change only the directly saved sampled full-vocabulary logit rows from `(0,1,31,63,126,127,254,510)` to **`(0,1,126,127,254,510)`**. Keep the same four window-0 natural inputs, all 31 actual conditions, and all three FP64 additive readouts. Query 0 remains an unscored boundary observation; query 1 starts the primary score; query 126 ends the early view; query 127 starts the later view; query 254 is an interior later query; query 510 is the last scored query. Queries 31 and 63 are no longer directly saved GPU-readout validation rows. The post-LayerNorm inputs to `lm_head` remain saved at all 512 queries for every one of the 29 distinct hidden trajectories per input. The all-query loss decomposition and output metrics remain unchanged.

All 16 passages, 32 start-token inputs, 31 conditions, 1,992 forwards, maps, directions, seeds, source/suffix controls, four full attention snapshots, estimands, score ranges, tolerances and uncertainty procedures remain unchanged. This is a prospective evidence-storage reduction. It is not an efficacy selection or recalibration. The all-query CPU readout remains an independently reconstructed numerical approximation; six sampled raw rows validate its numerical relation to actual model readout at those rows only. No stronger all-query GPU-readout claim follows.

## Exact sizing and explicit allowances

The four original eight-row files were re-serialized independently and each exactly reproduced the byte size from the full-writer fixture. Only these four files were then re-serialized at six rows; the complete writer was not rerun.

| Component | Bytes |
|---|---:|
| Measured eight-row full writer | 2,074,040,465 |
| Exact reduction in four row files | −59,503,104 |
| Projected six-row full writer | 2,014,537,361 |
| Additional JSON numerical-digit/growth allowance, 16 MiB | 16,777,216 |
| Additional analysis-output allowance, 32 MiB | 33,554,432 |
| Bounded failure reserve, 64 MiB | 67,108,864 |
| Combined prospective bound | 2,131,977,873 |
| Total cap, 2 GiB | 2,147,483,648 |
| Remaining margin | 15,505,775 |

The projection is measured exact row-file replacement applied to the actual full-writer layout. The extra JSON and analysis amounts are prospective allowances, not measurements of real results. Zero tensors do not reduce the size of the uncompressed tensor storage. The fixture's operation metadata uses the full field structure and representative real prior gate-check float lengths; new context-summary digits and newly added guard metadata can differ. Preserve these allowances and enforce cumulative caps; do not infer a production RSS measurement from a stubbed writer.

To reserve both later analysis and first-failure evidence, the implementation should enforce a normal experiment-output limit of **2 GiB − 64 MiB − 32 MiB = 2,046,820,352 bytes**. The extra 16 MiB JSON allowance and 15,505,775-byte margin fit within that normal-output limit. The analysis writer should enforce its separate 32 MiB new-output allowance and the overall 2 GiB cap. The 64 MiB failure budget remains available even if analysis has been written. Do not remove prior work or discard declared raw evidence to meet a limit.

Before loading a model require at least 4 GiB free: the entire 2 GiB new-output cap plus the 2 GiB disk reserve. Before each input group, record current serialized bytes and remaining disk, and require the next group's frozen projected upper bound to fit inside the normal-output limit and leave at least 2 GiB disk free. Include checkpoint and cumulative metadata growth in the group increment. A resource failure before loading records zero attempts/forwards; a later failure preserves completed partial shards and resource/identity metadata. Retain post-checkpoint actual-byte, free-space and production-RSS guards.

Measured increments and proposed rounded upper bounds are:

| Group class | Measured six-row increment maximum | Prospective next-group upper bound |
|---|---:|---:|
| Wiki window-0 natural: rows and four attention states | 142,875,507 | 146,000,000 |
| Other window-0 natural: sampled rows | 100,543,076 | 104,000,000 |
| Every other input | 55,848,483 | 58,000,000 |

These are resource projections rather than proofs that future JSON cannot grow. Post-checkpoint guards remain authoritative. Any enlarged evidence structure requires review and re-sizing before inference; no automatic reduction or clipping of evidence follows a failure.

## Failure evidence and memory

The failure helper's independent full-size serialization tests retain all production fields and conservatively use independent array storage. Six reachable classes, each with the maximum accumulated family ledger and pending prefix reference when applicable, occupied 13,781,377–31,777,463 bytes. The largest was suffix-comparison failure with an active additive accumulator. Gate, reconstruction, singleton and suffix failures were tested separately because each raises immediately; their impossible simultaneous union was not used. Oversized serialization was rejected before writing. Selected maximum-error and first-nonfinite rows and the full per-query error vectors were verified after serialization.

Keep the helper's original eight diagnostic failure rows, plus its maximum-error/first-nonfinite selection, unchanged. Those are failure diagnostics rather than the successful run's six sampled readout rows. Keep both actual/reference selected arrays, full-array fingerprints, one full failing-layer QKV/context capture when available, and the exact last-forward/record/check identity fields. Completed current-input hidden/gate/suffix/readout/additive data are first saved once to their normal partial shards; the capsule does not duplicate full pending bundles or full accumulators. The explicit 64 MiB limit is measured before writing. If an unforeseen capture exceeds it, preserve partial shards and the capture-error metadata and stop; do not silently truncate or rerun.

The full-writer fixture peaked at 1,300,268 KiB RSS; failure sizing peaked at 1,349,636 KiB. These are CPU-only resource tests, not production-memory forecasts. The prospective production RSS ceiling stays 4 GiB; independent review jobs stay below 3 GiB. No additional model memory, checkpoint, corpus extraction or calibration is introduced.

## Audit artifacts

- [Full actual-writer fixture](review/gate_additive_full_layout.py) and [measured evidence](review/gate_additive_full_layout.json): 1,000 main records, 992 suffix records, 1,800 runtime-check records, 928 distinct hidden tensors, 1,792 main and 1,792 suffix compact gate states, and four attention snapshots.
- [Recorded checkpoints](review/gate_additive_full_layout_checkpoints.json) and [group increments](review/gate_additive_full_layout_increments.json).
- [Six-row exact serialization](review/gate_additive_six_row_budget.py) and [budget evidence](review/gate_additive_six_row_budget.json).
- [Reachable failure layouts](review/gate_additive_failure_layout.py) and [capsule evidence](review/gate_additive_failure_layout.json).

The complete writer measurement used runner SHA256 `0a3f712199169973f414a430ca9211c894a731e603b10bfe848a4b9939838204` and failure helper `5136d71bffc2fb3c47d801fa2546212fe23d7d41b6940377236260b3edb95313`. Subsequent guard and six-row implementation changes need final independent review against this addendum. Temporary synthetic output directories were removed after recording sizes; no old raw output was deleted. All original reports, failed runs, successful input preflight, and frozen calibration/directions remain intact.
