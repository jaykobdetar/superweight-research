# T18 runner-shaped storage and failure evidence

The complete memory-only serialization fixture passed at runner SHA256 `80559661cad5cc7205b4ce0041090e8037c97238709da3febdff089a84519b46`, codec `83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06`, and context implementation `62a4719b9c0d50b72d23263b98f32b4dca7ef7172b36df8f5deff6b0a0463619`. This is a resource/serialization test, not model execution or scientific evidence. The draft remains blocked by empty input hashes, unset measured group bounds and `STORAGE_READY=False`; independent final review is still required.

The [fixture](gate_wiki_full_layout_fixture.py) uses the runner's actual `pack_input_payload` function, the actual lossless no-pickle NPZ codec and the actual failure-payload dictionary extracted from the runner AST. It serializes all32 input shards and four calibration shards in memory, along with all map/direction/spectrum/geometry payloads. No large payload file was written to disk. Finite high-entropy FP32/FP64 arrays stress compression; they are explicitly artificial and do not satisfy model equations. Existing real small map/spectrum objects supply the exact fitting schema. Independent hook/codec/numerical tests establish correctness separately.

The complete cardinalities are776 main per-target records,1,128 checks,776 operation records,768 suffix checks/operations/per-target metric sets,768 mode-norm records,32 P-source references,11 fixed L10 snapshots and11 full-hidden/four-row readouts. Suffix loss has127 entries; prediction/KL/P-reference arrays have128. Comparisons preserve every available unchanged target, including q0 and q127 where defined; the scoring subset remains126 targets2..127. Changed suffix targets are never scored.

| Budget component | Bytes |
| --- | ---: |
| Encoded payloads plus current source/input files |156,663,206|
| Explicit remaining JSON/source/analysis allowance |16,777,216|
| Total normal allocation in fixture |173,440,422|
| Proposed176MiB normal cap |184,549,376|
| Remaining margin |11,108,954|
| Largest measured snapshot input shard |55,098,870|
| Largest measured ordinary input shard |3,157,322|
| Largest measured calibration shard |414,505|
| Largest measured failure capsule |23,862,122|
| Proposed48MiB failure cap |50,331,648|

The16MiB allowance is a prospective bounded allocation, not a claim to have serialized future analysis results. Actual current copied source/input file sizes are counted separately. Proposed pre-group projections are58MiB for the single snapshot input,4MiB for other evaluation inputs and1MiB for a calibration input. Their slack covers checkpoint/index growth. The runner must reject a measured payload before writing if it exceeds remaining normal capacity, while keeping the2GiB reserve and separate48MiB failure allowance. A startup requirement of176MiB+48MiB+2GiB is2,382,364,672 bytes; recheck actual free space at launch. These limits need to be frozen in the final protocol rather than silently replacing the earlier64MiB prospective failure allowance.

Eight feasible, mutually exclusive immediate-failure scenarios were measured: gate diagnostics, reconstruction, singleton incoming-state mismatch, suffix mismatch, identity closure, snapshot deduplication mismatch, metric failure and calibration failure. They retain actual failing gate/context sides, exact current512-token input, attempted/completed forward identities and selected failing/full-vocabulary rows with full per-query error summaries. The largest case is snapshot deduplication mismatch. The fixture does not combine stop paths that cannot coexist. The runner never truncates a capsule to fit: an unexpected size violation is recorded as a preservation error. That path still needs independent review; it is not a guarantee that every conceivable error can be fully preserved under any external disk exhaustion.

Final evidence: [JSON](gate_wiki_full_layout_fixture.json), SHA256 `8ccb6681e183ba2d78030190411525cfa43ac1b0808b6965061ba7b21edb91ea`; [log](gate_wiki_full_layout_fixture.log). Peak RSS1,217,696 KiB,17.38 seconds, zero pretrained or random-model forwards and zero GPU use. Earlier successful layouts remain under `gate_wiki_full_layout_fixture_before_suffix_metrics.*` and `..._before_failure_provenance.*`; their missing suffix arrays/older provenance are not presented as final coverage.

The runner now persists the verified initial model hash immediately after loading, records calibration-save/checkpoint and fitting stages explicitly, separates last attempted from last completed forwards, and counts completed model returns even if later scoring fails. The root-owned `audit_gate_wiki_results.py` compiles at SHA256 `ceaa55091ab4efc4a77d621e882dc9a3cba41e8007b9d576c9d140aa46412f65`. Its earlier missing-space syntax error was reported and repaired by the owner. This author review does not replace the independent implementation signoff or authorize a model launch.
