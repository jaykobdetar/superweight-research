# Report10 repository extension: independent CPU package review

**The corrected numeric-only extension passes export, driver-guard, dependency, and hygiene review. The completed two-stage query replay also passes independent CPU verification.** This reviewer used no model/GPU or GitHub action. Evidence: [package audit](report10_package_audit.json), [guard fixtures](report10_guard_tests.json), [portable numeric audit](report10_portable_numeric_audit.json), and [completed replay audit](report10_replay_audit.json). Their private CPU scripts remain with the raw bundle.

## Export fidelity and scope

Independently compared **7,776 records across eleven batches**, including the new **1,296 query records**, against original private per-token tensors. Every exported loss/KL vector and native-prediction equality mask is exact. Declared forward totals are **7,800**; the difference from records is the earlier 24 development/calibration forwards. The NumPy-only portable audit also passed all 7,776 records with zero scalar error, using 50,000 KiB peak RSS.

All 140 transfer result files have the expected JSON/JSONL/NPZ formats. NPZ data is limited to one-dimensional float32 loss/KL and boolean agreement masks. No input/prediction ID sequences, source/donor/activation arrays, model weights, corpus excerpts, private home/temp paths or Library IDs were found in result JSON. Token/source hashes, scalar tokenizer literal IDs, donor-file hashes and final-hidden hashes are provenance, not raw inputs or hidden vectors. All 44 newly referenced private raw-file hashes were independently verified without exporting those files.

The 41 source-manifest files match their recorded hashes. All experiment/dependency bytes referenced by the two completed manifests match the exported files; Python experiment sources match the originals. Historical ledger aliases remain explicit: the six-batch `HYPOTHESES.md`, follow-up snapshot and query snapshot are distinct frozen documents. Three report-input Markdown files have documented link-only transformations; original and exported hashes are retained, and each transformation was independently reproduced. Report10's builder now has its prior-art input beside it, avoiding an unstaged private review-directory dependency.

The report figures match the reviewed original PNG bytes. Scientific prose/figure interpretation is covered by the separate report review; this review checks their packaging and references.

## Corrected issues and guard coverage

Nine initial broken links were corrected. Safe review links now resolve within the repository; omitted raw-data audit scripts are identified as private-bundle filenames. All **29 local links** checked in the report, guide, README, query reviews and report-source inputs resolve. The `gpt2_message_mass_design_*` files are retained as the historical pre-run design of the completed mixture batch, explicitly labeled; their earlier message-first notation is superseded by the frozen mass-first protocol. No subsequent unexecuted gate-subspace review is included.

Two independent adversarial fixtures initially exposed permissive validation: an empty zero-record numeric run was accepted, and a manifest marked failed was accepted when it retained a stale completion timestamp. The numeric audit now requires positive record counts; the comparator rejects explicit failed status. Both fixtures now correctly reject. These were artifact-validation gaps, not defects in the actual completed runs.

All **51 CPU fixtures** pass. They cover empty/duplicate/missing records and manifests, malformed arrays/indexes, nonfinite data, changed inputs/metrics, invalid model hashes/completion/counts, negative tolerance, changed sources/checkpoints/staged code/configuration, and existing output directories. The eleven-stage launcher fixture intercepted all **22 experiment/analyzer subprocess calls**; no model process ran. Query-subset fixtures test valid selection and rejection of missing/duplicate/changed/failed query outputs or incomplete archive coverage. Six/follow-up subset guards remain valid.

## Reproduction and hygiene

The driver dispatches all eleven stages in dependency order and runs them sequentially. It verifies checkpoint/source identities, stages immutable code and a private configuration, and refuses changed staged content or pre-existing outputs. Query stages reference the preceding GPT-2 data manifests needed for span exclusion. `compare_transfer.py --series query` checks the two new outputs while requiring the complete eleven-batch archive identity. Its scope is exact scalar comparison by default, not full tensor replay; runtime protocol controls and a later independent raw-file comparison supply additional evidence.

The reproduction guide correctly gives eleven batches/7,800 forwards/7,776 records, pinned checkpoint/input retrieval and hashes, dependency order, private-output exclusions, original timing/resource limits, and subset comparison commands. Its query addition is about130.05 worker seconds and approximately3 GiB of private diagnostics/donors. This is separate from input hashing, environment installation, audits and packaging. The documented private raw-data path is necessary for full hook/donor audits; numeric-only auditing cannot regenerate logits or certify interventions.

Scanned **771 current Git-eligible files** for prohibited raw extensions, ignored-workspace/local-config leakage, symlinks, credential patterns and Library IDs; no prohibited artifact or credential-pattern match was found. Git ignore checks confirm private reproduction workspaces and `*.local.json` are excluded. This is a current-tree review, not a forensic scan of every historical Git object. Package-review peak RSS was606,324 KiB; guard fixtures used about36 MiB.

Reviewed driver SHA256 identities:

| File | SHA256 |
|---|---|
| `scripts/run_transfer.py` | `1665cdac2b474ace189b3c40bc5f5525c3a9272c093ccf888a0f156196e43004` |
| `scripts/compare_transfer.py` | `61ad7750ab3e073e903c2c4392691adf65dd9b48df817488a0a7acda2ed1c0ab` |
| `scripts/audit_transfer.py` | `98fd341a01fa9b090b7935fc7055f151c8462d37e8fb6f430c2fd8c71f524008` |

## Completed query replay

Independently verified the completed replay of **1,296 new records/forwards**: query-source 776 and query-mixture 520. All **44 recursive raw `.pt` files** match the original bytes under streamed SHA256 hashing and have different inodes from their originals. This includes all 32 chunked mixture donor files. Scalar/window fields and 21 JSON/record files also match, excluding audit-process RSS where applicable. Re-executing the CPU query comparator reproduced the saved comparison exactly.

The nine preceding runs are explicitly seeded dependencies, not regenerated inference in this replay. Their recursive file coverage, hashes and hardlinked inodes match the two previously validated workspaces; their per-token files also match the original runs. The seed manifest SHA256 is `15eac154f1f7c6c963753046dffca011351685cf02b2c200f066565c789fa57e`. Frozen source/staged bytes, checkpoint digest, model revision, dtype, seed, library versions and unchanged-parameter checks agree. Completion timestamps and the driver log establish sequential execution of the two new stages. Private workspaces, local configuration and logs remain ignored.

| Replayed stage | Records/forwards | Worker seconds | Peak GPU allocated bytes | Process peak RSS KiB |
|---|---:|---:|---:|---:|
| Query-source |776|72.054070|768,470,528|2,632,648|
| Query-mixture |520|60.080559|768,470,528|2,313,864|

The two worker intervals total **132.134629 seconds**; the full driver interval, including input hashing, is **176.711148 seconds**. These differ from the original experiments’ 130.053809 worker seconds. The independent streaming audit used **25,976 KiB peak RSS** and loaded no tensor/model arrays. This is exact reproduction on the original machine with nine validated dependency runs reused; it is neither a fresh eleven-stage inference run nor independent model/researcher replication.

No remaining package/replay correctness blocker was identified. Refresh the final file inventory after copying these review artifacts and record the resulting local commit/archive identities. The earlier export file/link counts above describe that audit's snapshot; the final evidence additions require their own inventory. Private GitHub remains denied and unexecuted; this review neither retries nor works around that denial.
