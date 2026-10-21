# Report9 extension: independent export and portable replay readiness

**Verdict: the reviewed local extension is ready for inclusion in the local commit.** No remaining scientific export, reproduction or hygiene blocker was found. Final file inventory must be refreshed after this review/evidence is copied. This verdict does not authorize or attempt the denied remote repository action.

Evidence is in `report9_package_replay_audit.json` and `report9_guard_tests.json`; their CPU-only scripts remain in the private raw bundle. This review loaded saved per-token arrays, never a model, and used no GPU. Peak package/replay audit RSS was about **0.50 GiB**; guard fixtures used about **36 MiB**.

## Numeric export and hygiene

Independently compared all **6,480 records across nine batches** to the original private per-token tensors. Every exported loss/KL array and native-prediction equality mask is exact. The new extension accounts for **1,912 records**. All NPZ arrays are one-dimensional float32 loss/KL or boolean agreement masks; no input token IDs, prediction IDs, donor/activation tensors or model-weight tensors are exported. Source/token hashes and intentionally documented tokenizer literal IDs are provenance, not corpus sequences.

All **113 transfer result files** pass the expected JSON/JSONL/NPZ extension and recursive schema checks. No corpus text/snippets, token/prediction fields, private home/temp paths or Library IDs occur in result JSON. Current Git commit candidates were checked for restricted raw-file extensions, local configuration files, ignored-workspace leakage, symlinks, credential patterns and Library IDs. No restricted file or credential-pattern match was found. The ignored raw workspace and local configurations were explicitly checked with Git's ignore rules. This is a current-tree hygiene review, not a forensic audit of all historical Git objects.

The 30 exported source/protocol files match the export manifest. They also match their originals, except for the documented intentional ledger mapping: `HYPOTHESES.md` is the frozen six-batch snapshot, while `HYPOTHESES_FOLLOWUP.md` matches the current live ledger. Their distinct roles are explicit in the source manifest. The report copy equals the reviewed Report9 Markdown with only figure-link relocation; all four figure bytes match the reviewed originals.

Four broken links to deliberately omitted private review scripts/evidence were found initially. The exporter was corrected to label omitted scripts as private-bundle filenames, export safe CPU-test JSON and include the Report9 scientific review/hash evidence. All **17 local links** in the new report/reproduction/review scope now resolve. Private-data audit scripts remain outside the repository as documented; the public-to-the-repository numerical audit and full regeneration route are the supported portable entry points.

## Independent comparison of the completed portable replay

The three follow-up stages completed through the repository driver in a fresh ignored workspace. All **1,912 record fields** match the originals, as do check records, operation records, summaries, paired contrasts, condition files and data manifests. All **12 raw tensor files** are SHA256 byte-identical to the originals: tokens, per-token arrays, traces, K/V donors and the frozen random-control tensor where applicable. Every new output has a different inode from its original, consistent with regeneration rather than hardlinking the experimental outputs. Completed manifests match model revision/precision/seed, source-code identities, counts, parameter digests and version invariants.

| Replayed stage | Forwards/records | Worker seconds |
|---|---:|---:|
| Layer-specific writes | 776 | 42.1513 |
| Source K/V transmission | 616 | 40.0830 |
| Value norm/direction | 520 | 36.9962 |
| Total | 1,912 | 119.2305 |

The validation manifest separately records **162.3012 seconds** including driver input hashing/overhead. It should not be confused with the original experiments' **120.54 measured worker seconds**.

Six earlier dependency runs were **not rerun** in this follow-up validation. Their prior independently validated outputs were explicitly seeded via hardlinks. The dependency seed lists their manifest and per-token hashes; all match the old and new dependency locations. All seeded files match the existing prior-validation files and share their inodes. The new stages only read those dependency outputs. This preserves provenance while avoiding a false claim of a fresh nine-batch inference run. The seed manifest hash is `751acfcc34674b82fb425928a855ad73f251e7df17bb5abe3809a721239697cb`.

The earlier six-stage replay and this three-stage continuation establish successful same-machine regeneration of the documented stages. An independent audit of their saved arrays is not an independent researcher/model replication or evidence of hardware-independent bitwise equivalence.

## Driver, comparator and instructions

The driver lists all nine stages in dependency order and executes each experiment and analysis sequentially in separate processes. It verifies source and checkpoint identities, stages exact scripts/protocols, refuses changed staged files/configurations and refuses existing output directories. New stage dependency reads occur before model loading. No overwrite, silent resume, credential modification or remote model retrieval was introduced.

The comparator requires the nine-run export manifest, rejects missing/duplicate records or mismatched inputs/model invariants, and defaults to exact scalar equality. Its `six` and `followup` subset modes use only the selected local rerun outputs while retaining the complete archive identity. Its stated scope remains scalar comparison; the independent replay audit adds raw-file and full-record equality.

**42 CPU guard fixtures passed**, adapting the prior guards to nine stages and adding subset-specific cases. They cover empty/duplicate/missing data, invalid model identities/completion/counts, changed inputs/metrics, invalid tolerances, malformed numeric arrays, source/checkpoint/staged-code/config changes and existing outputs. The nine-stage launcher fixture intercepted all **18 experiment/analyzer subprocess calls**; no model process executed. Seven additional subset fixtures validate valid six/follow-up comparisons and rejection of missing, duplicate, changed or incomplete follow-up records and an incomplete archive manifest.

The actual portable NumPy-only numerical audit also passed all **6,480 records with zero scalar error**. The actual follow-up comparator passed all **1,912 records at tolerance zero**. Reproduction documentation correctly states nine batches/6,504 forwards/6,480 records, distinguishes the separate OLMo core workflow, gives pinned inputs/dependencies, explains private outputs and the original-machine/resource limits, and documents sequential regeneration without UIT prerequisites.

The principal reviewed driver hashes are:

| File | SHA256 |
|---|---|
| `scripts/run_transfer.py` | `1c40def01ffd19872fb1a8a520909565ba6226fb46b85842b85040aabeac178c` |
| `scripts/compare_transfer.py` | `8e4cb34a38f177551ba860b072dd682b5c669418350ac7acdae1e4ef76b24d4b` |
| `scripts/audit_transfer.py` | `b6aa4e8be89e2d6e787785225e850a8dbd26742a88f72b8b241637de623b37ad` |

No external action was performed. Private GitHub creation remains denied and unexecuted; this review neither retries that action nor changes credentials or permissions. Remaining local completion work is to copy the safe review/evidence, refresh the file inventory and record the resulting local commit.
