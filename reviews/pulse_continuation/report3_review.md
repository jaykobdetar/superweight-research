# Independent factual and scope review of periodic report 3

Reviewed `build_report3.py` and the rendered text of `report3.html` against the eight run manifests, saved numerical audits, raw generation records and targeted logits. No headline numerical error or substantive causal overclaim was found. Value-layer outcomes are explicitly excluded from this frozen snapshot, as intended.

## Verified claims

- The eight named runs total 22,168 forwards and 773.9935965538025 seconds of their recorded execution timers. Maximum allocated GPU memory is 4.195065021514893 GiB; the reported process RSS high-water value is 5.1447296142578125 GiB. Rounded displayed numbers are correct.
- The static residual energy, rank-plus-axis distinction, template numbers and source-identity results agree with their audits. Static across-layer rank is distinguished from downstream dynamic rank.
- Selected top-eight Wiki repair is approximately 1.293× versus 1.412× axis-only; top 32 is approximately 1.337×. Other corpora's selected top-eight repair intervals relative to axis cross zero. The report properly separates template-world screening from the original deletion circuit, and notes one matched random draw per set size and lack of joint optimization.
- Boundary L0/L1 isolation exactly reproduces the full intermediate in 64 cases. The 4.91–6.88 and 6.96–8.28-nat ranges refer to later-only scoring in the correct matched-mask worlds. The report does not claim a natural reset or interchangeable pulses.
- Extended context source deletion ratios, approximate cache agreements and exact full-donor controls agree with raw data. Length-dependent numerical differences, reused text and absence of a demonstrated speedup are acknowledged.
- Generation sequence counts and matched-history agreements match all 96 raw records. All EOS counts are zero. Exact-control tie sensitivity is distinguished from the non-tied hybrid discrepancy; fidelity is not equated with quality.
- Quiet-anchor ratios and causal closure assertions match raw metrics. The report explicitly distinguishes native-key feedback, zero-value mass retention, source-removal renormalization and semantic-content uncertainty.
- Small window counts, reused panels, pointwise conditional bootstrap intervals, limited corpora/checkpoint/precision and incomplete distribution-level reconstruction outside retained logits are disclosed.

## Requested clarifications before freezing

1. The sentence “All replacements use actual cached execution and omit source-token processing” needs to specify **the suffix execution pass**. The full-donor control first processes the complete source-containing context to obtain its donor; independent dictionaries and frozen templates have separate preparation costs. This qualification avoids implying that the end-to-end full-donor procedure skips that computation.
2. Label the host RSS number as the resident worker process's **lifetime high-water mark**. It is the same cumulative `ru_maxrss` value across these batch manifests, not independently measured residency for each batch. The present numeric value is correct.
3. At review time, `README.txt`, `replay.py` and `bundle_identity.json` still describe report 2. Before distributing the report 3 archive, include snapshot-specific dependency/replay instructions consistent with the report's promise, retain the audited eight runs, and identify report 3 separately from the previous immutable archive. This is a packaging check, not a defect in the scientific results.

No experiment changes are requested. Review performed on CPU without a model load. The separate value-layer audit has also finalized all 188 summaries and 188 contrasts with no discrepancy, but remains outside report 3 as declared.

## Resolution verification before archive creation

The builder and rendered report now explicitly limit omitted source-token processing to the suffix pass, disclose donor/dictionary/template preparation, and label host RSS as the reused worker's lifetime high-water mark. Both wording clarifications are resolved.

`README_report3.txt`, `replay_report3.py` and `dependencies_report3.json` now describe the eight-run snapshot and its frozen-original-dependency replay semantics. The four separately listed prior dependency hashes and all three recovered source-text hashes verify. The recorded Library identity is `[private Library archive identifier retained locally]`; the local report size matches that upload record. Remote uploaded bytes were not independently fetched in this review.

Packaging inspection found and the parent corrected two additional defects before archive creation: the archive file list now includes `review/audit_common.py`, and the replay wrapper detects worker failure markers plus invalid manifest integrity fields instead of trusting only a completed timestamp.

The corrected driver passes isolated CPU-only stub-worker tests: successful completion exits zero; failure after a completed manifest exits nonzero; a `JOB_FAILED` marker split across writes exits nonzero; a false parameter-integrity flag exits nonzero; and an existing destination is refused. No actual worker/model replay was executed. The test and results are preserved as `report3_packaging_audit.py/.json`.

No remaining report or replay-wrapper blocker was found. Archive-content hashes/CRC still require validation after the archive is constructed; this review does not claim to have inspected an archive that did not yet exist.
