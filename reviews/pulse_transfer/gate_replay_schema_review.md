# Independent calibration-schema compatibility review

Approved comparator SHA256 `c4f1db286b8025e9080635b3062ea845e50828e7a5e2191866589a3020f6ca25` for CPU comparison of the completed replay. No model rerun is needed.

The diff is confined to calibration-operation schema normalization and its disclosure. It accepts exactly the original12-field schema or the rebuild4-field schema, validates original metadata and operation counts against frozen constants, and compares all four shared fields recursively with exact equality. Eight layer gate-check records are required per passage. This does not relax tensor, scalar, source-hash, or main-run checks, and it does not invent missing rebuild counters.

Eighteen independent CPU tests passed, including all eight actual paired calibration records, both same-schema cases, and rejection of each bad original constant, missing/shared/extra fields, partial schemas, missing gate layers, nested numeric corruption and changed input identity. Peak RSS17876KiB; no model or GPU loaded. Evidence: `gate_replay_schema_review.py/.json`.

The initial failed comparison JSON is preserved, and `compare_gate_study_initial.py` still has original SHA256 `76d80425cb73bcccf9b5ce8262a7f4e0ea9b9d2dc429cfac5adf0063b7d6391c`. The failure occurred before raw-result comparison and remains a comparator-schema failure, not an inference failure. Actual replay numerical equality must be established by the subsequent CPU comparisons; it is not asserted by this compatibility review.
