# Independent gate replay implementation review

Approved for the parent to launch the explicit sequential replay at the hashes below. This is implementation approval, not a claim that the real replay has completed.

- `run_gate_study.py`: `15942f06e04c4df3bfbff504f2e69aacd79ed0a4b2ceee03db8f75abdc1950f7`
- `compare_gate_study.py`: `76d80425cb73bcccf9b5ce8262a7f4e0ea9b9d2dc429cfac5adf0063b7d6391c`

Static inspection and 20 CPU-only fixtures passed. Fixtures reuse the credited author data generator and add independent stage-order and calibration-stop tests. They cover tokenizer/seed provenance rejection, cached zero-model evidence, sequential 8-calibration + 1352-study forwards followed by analysis, refusal to resume an existing run automatically, and preserving mismatched calibration artifacts without substituting reference maps. Corrupted raw metrics, archived arrays, source/gate tensors, duplicates, failed runs, NaNs and dtype changes were rejected. Comparator failure clears stale passed output. Peak test RSS was 529136 KiB; no model or GPU was loaded.

The numeric comparator checks 17 deterministic result files and 1,734,000 archived per-token values against newly read raw tensors. Private comparison additionally reads 105 tensor files/shards and independently reduces 9,520 scalar metrics across both runs. Metadata equality alone is insufficient. Runtime, memory, calibration-reference provenance and platform versions are not required to be identical; scientific identities and source hashes are checked. Successful full logits and full contexts outside fixed snapshots were not retained, so their runtime hashes cannot be independently reconstructed from exported data. Exact comparison does not itself prove independent execution; actual replay logs and distinct output files still require review.

Use a fresh replay workspace after a source-ledger update: changed staged files intentionally fail exact-copy checks. The separate NumPy export audit and Report11 review remain separate work items and do not block this wrapper launch. No publication action is authorized by this review.

Evidence: `gate_replay_independent_tests.py` and `gate_replay_independent_tests.json`.
