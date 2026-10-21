# Additive-study storage author checkpoint

The full serialization task is complete. No pretrained model, tokenizer or GPU was run, and the successful input preflight was not repeated. All temporary synthetic outputs were removed. No old output or frozen research source was deleted or changed.

The actual writer's 1,992-step stubbed orchestration completed with 1,000 main records, 992 suffix records, 1,800 checks, 928 saved hidden arrays, 1,792 main and 1,792 suffix compact gate states, and four full attention states. It occupied **2,074,040,465 bytes** and peaked at **1,300,268 KiB RSS**. This validates serialized shapes/counters/resource accounting only; fake values, readout arithmetic and in-memory hashes are not scientific evidence. The exact measured source versions are preserved in [gate_additive_layout_measured_sources](gate_additive_layout_measured_sources/manifest.json).

The eight-row layout leaves just 6,334,319 bytes after reserving a 64 MiB failure capsule. This is a resource concern before inference, not a failed scientific hypothesis. The [storage addendum](../GPT2_GATE_ADDITIVE_STORAGE_ADDENDUM.md) therefore proposes six successful-run rows `(0,1,126,127,254,510)`. Exact reserialization of only four row shards saves **59,503,104 bytes**. The projected six-row writer is **2,014,537,361 bytes**; with 16 MiB extra JSON growth, 32 MiB analysis outputs and 64 MiB failure reserve, the combined bound is **2,131,977,873 bytes**, leaving **15,505,775 bytes** below 2 GiB. All hidden inputs and all runtime scientific metrics/conditions remain; direct sampled GPU-readout validation loses rows31/63, as explicitly documented.

Six reachable failure-layout envelopes were measured using the actual bounded helper. Their exact sizes range from **13,781,377 to 31,777,463 bytes**; the largest is suffix comparison with an additive accumulator. An intentionally oversized capsule was rejected before file creation. Maximum-error/first-nonfinite query evidence and retained array dtypes were checked after loading. Peak RSS was **1,349,636 KiB**. The failure helper's eight diagnostic rows remain a separate policy from the six successful readout rows.

Evidence:

- [Full writer test](gate_additive_full_layout.py), [result](gate_additive_full_layout.json), [checkpoints](gate_additive_full_layout_checkpoints.json), [group increments](gate_additive_full_layout_increments.json).
- [Six-row sizing](gate_additive_six_row_budget.py), [result](gate_additive_six_row_budget.json).
- [Failure layouts](gate_additive_failure_layout.py), [result](gate_additive_failure_layout.json).

Pending before GPU execution: independent review of the new storage addendum; implementation of the six-row and cumulative/projected resource guards; final independent implementation/guard signoff on the revised bytes. The original proposal and preflight remain immutable. Root owns the runner/analyzer edits and launch. This checkpoint is author sizing evidence, not independent launch approval.
