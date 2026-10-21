# T18 standalone preparation: author checks

Status: **author checks passed; root review of the layout-auditor integration pending**. No pretrained replay, model loading, GPU work, numerical reference comparison or external publication occurred. The latest correction did not rerun preparation or the numerical audit. The primary runner, its imported experiment dependencies, canonical inputs and raw outputs were not edited.

New deliverables are `prepare_wiki_calibration_replay_inputs.py`, `reproduce_wiki_calibration.py`, `GPT2_WIKI_CALIBRATION_REPLAY.md`, and `wiki_replay_metadata/`. The complete hash inventory and limitations are recorded in [wiki_replay_author_readiness.json](wiki_replay_author_readiness.json). The helper SHA is `b7e69361a4ac2f5d6a41f20426806d0fdd7dd4f98d3d1000e0dee91a9d27b585`; the wrapper SHA is `11705e68a54769ae5ff75c06623a8d4422fcc05a01541717574717b1828bb5ed`.

The metadata identity is `7e3a43808ee51ab8a44526d3cb9e9f545e12ac771a1723ca2d8ca9f360a8a0b6`:14 files,1,197,246bytes. Export checks the pinned canonical14 inputs, source-contract identity and capacity audit. Exact member and privacy validation reject undeclared files, symlinks, absolute locators and raw token IDs. Historical JSON pointers are permitted only at their documented fields and with the observed vocabulary. The package retains all561 historical duplicate-hash commitments,185 provenance hashes, both source scans and the full recorded exclusions. The Hume capacity check recomputes the union arithmetic from the preserved intervals; it does not claim a new historical-token mapping audit.

## Actual zero-forward preparation

The end-to-end default wrapper completed at `.reproduce-t18-author` using the existing reviewed eight-forward calibration reference. Its model/forward/GPU counters were zero. The helper verified seven source files and the pinned GPT-2 tokenizer; reconstructed all four retained Wiki calibration windows and16 evaluation windows; exactly matched all seven calibration and58 evaluation rejections; and required all eight canonical file hashes. Natural/BOS and current first IDs are rebuilt locally. Six metadata files intentionally differ; canonical hashes and actual portable hashes are both recorded. Prior T16/T17 first-ID membership is an explicitly inherited attestation, not a fabricated reconstruction.

| Measurement | Actual result |
|---|---:|
| Total staged workspace | 8,688,514 bytes |
| Wrapper wall time | 4.129 seconds |
| Input reconstruction wall time | 3.230 seconds |
| Peak reconstruction RSS | 762,424 KiB |
| Peak wrapper RSS | 26,188 KiB |
| Model forwards / model loads / GPU | 0 / 0 / none |

The original and staged runner are both preserved. AST normalization proves that only the literal `INPUT_HASHES` assignment changed. This preparation measurement preceded the adapter correction and used36 source files; the current inventory covers37, including the approved adapter. The original completed fixed512 reference was also checked separately against all seven calibration files, its completed manifest and all20 declared source hashes; a second end-to-end wrapper staging with that reference was not run. A caller's source inventory must include the selected reference's declared source dependencies.

The preparation used helper SHA `36fbbf48fffc695deb268419d8a1dfe0b3fc47c15f125a3548b16c077dd51490`. Subsequent changes only hardened the export CLI against overwriting an existing failed attempt and tightened allowed metadata fields. The `reconstruct` function's AST is unchanged, and the current stricter validator passed against the preserved actual outputs. No unnecessary tokenization repeat was made. These differing source identities are explicit in the readiness JSON.

## Bounded fixtures

wiki_replay_author_tests.py (private-only file `wiki_replay_author_tests.py`, SHA256 `305240e06d48f8ac39d411052df910e8e94d787c43ea25ebba51c9cc3bec9761`; see the [omission manifest](../../manifests/wiki_calibration_review_export.json)) and its JSON/log record58 passing checks. They include deterministic byte-identical metadata re-export, actual-source inventory checks, current/old reference validation, missing/extra metadata, wrong pointer context, absolute locators, altered token bytes, altered BOS/starting IDs with all output hashes rebound, unauthorized stripped-ID fields, changed score length, exclusion changes, missing history hashes, a seventh metadata substitution, wrong direction binding, duplicate/chained assignments, and missing runtime dependencies.

Low-disk preparation fails with an explicit zero-forward/false-model/false-GPU manifest. An existing failed export cannot be overwritten by trying the same directory again. Child-state tests use explicitly synthetic completed and partial manifests: they verify exact1548/4 counters on a valid completed fixture, preserve known14-of15 attempted-forward facts on a partial fixture, reject incomplete completion, and keep a missing manifest unknown. These are bookkeeping fixtures, **not actual model runs**. The latest tests used69,364KiB peak RSS and0.897seconds; their small temporary directory was removed only after successful completion. The low-disk failure record is retained separately.

The added checks reject a missing or changed adapter, a changed frozen auditor, missing layout attestation, changed map values, nonexact directions, changed tolerance policy, an integer masquerading as the required false flag, incorrect matrix counts and incorrect strides. They also validate the primary run's existing manifest, analysis and saved-state audit against the wrapper's completion contract. That last check reads saved JSON only; it does not repeat the numerical audit.

The first export attempt failed because its privacy validator omitted valid historical pointer components (`prefix`, `suffix`, `natural`, `calibration`, `evaluation`). The partial attempt and an explicit failure note remain in `wiki_replay_metadata_draft/`. The final validator uses exact schema locations and a bounded component vocabulary. The earlier safe export before private-ID hash commitments remains in `wiki_replay_metadata_before_identity_commitments/`; it is not the final package. Neither attempt changed canonical study data.

## Frozen auditor and approved layout adapter

The original direct auditor failure is preserved. The wrapper now stages and invokes `audit_gate_wiki_results_layout.py` at SHA `c837ab07243683e08dcfd0288fcf76a7ca8372dfdad06d605a01a80fe23f3974`. It requires the unchanged frozen auditor at SHA `ebae29f3c04c937aa8abf7dcbb608af91ee0a957f3ffface128e914e6b5f1686`. The adapter restores only the original column-major strides of24 O/F/M L4–L11 QR bases after NPZ deserialization. The completion contract verifies both source hashes, exact attestation membership, unchanged tensor values and map digests, exact directions and geometry, and `tolerances_changed is False`. This fixes the replay route without modifying frozen scientific code or relaxing its numerical checks.

## Remaining review and execution scope

The new scripts remain drafts until root review of this correction. The guide now uses paths from the repository root: `experiments/pulse_transfer` and `results/gate_wiki_calibration_reproduction/{metadata,parent_metadata,source_inventory.json}`. The package must supply the approved inventory and both auditor files. The wrapper refuses edited or missing dependencies.

The optional eight-forward calibration bootstrap remains the existing documented route. Its preparation needs the pinned Llama tokenizer/config, but no Llama weights or old model run. A wholly GPT-2-only bootstrap is deferred. The new T18 wrapper itself needs only GPT-2 and the seven T18 source files when a valid M8 reference is supplied.

No execution branch was exercised with a pretrained model. It is intended to run1548 forwards plus CPU analysis/saved-state audit only after approval, with the unchanged176MiB normal/48MiB failure allowance and2GiB free reserve. A numeric comparator and separate readout audit remain future work before claiming a matched scientific replay. The current deliverable is a tested, bounded preparation route, not a completed independent replication.
