# T19 recovery after an unexplained worker interruption

Prospective recovery design, 21 October 2026. No partial efficacy results have
been inspected to select this design. The original scientific question,
16-passage panel, 34 conditions, calibration F, controls and numerical criteria
remain unchanged. Independent recovery implementation review is required
before another pretrained forward.

## What happened and what is available

The original sole worker started at 19:58:34 UTC. Its terminal session later
became unavailable. The host process and GPU compute queries confirm that it
is no longer running. The last journal write is at 20:05:45 UTC. There is no
final manifest, Python failure record or failure capsule. A kernel OOM query
for the interval returned no entries. The cause is unknown; absence of a
failure record does not prove that every unfinished check would have passed.

The hash-linked journal verifies 1,110 intents, attempts and model returns,
1,109 fully processed forward records, and 16 committed shards. The latest
checkpoint contains 1,092 forwards: 544 main, 544 suffix and four anchors.
The committed shards are the complete Wiki and Utilitarianism panel. Their
233,391,179 bytes match the checkpoint inventory exactly. The original 20
frozen sources still match their recorded hashes. The subsequent 18 model
returns belong to uncommitted Ethics window zero, natural start. Their full
raw arrays were in process memory and are unavailable; available journal
fingerprints are preserved. No partial numerical efficacy result from that
group will be used.

The original final model hash/parameter-version, precision, source, input and
calibration checks did not run. Current source/input bytes can be checked
retrospectively, but the interrupted process's final parameter and precision
state cannot be recovered or retroactively claimed. Preserve this limitation
even if continuity checks pass.

Original directory and log remain unchanged. The forensic inventory is
`review/gate_native_value_interruption_20261021.json`, with a separate retained
byte verification in `review/gate_native_value_retained_prefix_bytes.json`.
Do not overwrite, delete, relabel as complete, or silently resume this directory.

## Recovery and continuity gate

Use two bounded, sequential execution segments with exactly the same pinned
checkpoint, source code, inputs, frozen calibration and numerical settings.
Each segment loads one model, checks no competing GPU computation, and runs
all applicable guards. There are no concurrent model loads.

Segment A first replays the already preselected Wiki window-zero natural
input, including all 34 main, 34 suffix and two anchor conditions: 70 forwards.
Before any new evaluation, require exact recursive equality with the original
committed shard for every array, dtype, shape, scalar, diagnostic and
fingerprint. A hard mismatch halts recovery and preserves both compared
artifacts. No tolerance relaxation or alternative anchor selection is allowed.
This is a same-machine compatibility check; it does not prove every missing
final guard of the interrupted process.

After this gate, segment A executes all eight Ethics passage/start groups:
272 main, 272 suffix and two anchors, totaling 546 new scientific forwards.
Including continuity, segment A executes 616 forwards. Segment B executes
the eight optimizer-code groups, also 546 forwards. Each segment rechecks
parameters/versions, precision, sources, inputs and calibration before its
own successful finalization. The segmentation is a resilience measure after
the unexplained terminal loss, not a changed scientific estimand.

The original 16 complete shards are reused byte-for-byte. New outputs and the
continuity replay are stored in new immutable directories. A separate merger
may assemble a new, explicitly labeled composite study only after both
segments and the exact continuity gate pass. Copy or reference recorded
bytes without mutating the original. The merger must validate source and
input provenance, count/identity rosters, every shard digest, and preserve
links to the original and recovery journals/manifests. The composite must
never be described as one uninterrupted execution.

## Counts and inference

The completed scientific panel, if recovery succeeds, still has 2,184 retained
scheduled forwards: original 1,092 plus new 1,092. Actual cumulative model
returns will be 2,272: those 2,184, 70 continuity validation forwards and 18
discarded uncommitted returns. All counts are reported separately. Any
additional failed attempt must be accounted for rather than folded into these
prospective totals.

Run the unchanged frozen numerical analyzer on the explicitly complete
composite, then independently audit raw numerical results, controls,
contrasts, uncertainty and saved readouts. Interpret outcomes only after this
complete-panel audit. The original source freeze remains available alongside
any narrowly adapted recovery wrapper, with an exact code-difference record.
No new calibration, map choice, input selection or scientific threshold is
permitted by this recovery protocol.

Maintain the existing 2 GiB disk reserve and per-segment measured output and
failure allowances. Current root free space exceeds 28 GB after a separate,
user-approved cache cleanup. No further deletion, cleanup, deduplication or
storage replacement is authorized. Preserve the interrupted run and every
failed recovery or preparation artifact.
