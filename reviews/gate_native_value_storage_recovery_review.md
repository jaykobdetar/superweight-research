# Prospective storage recovery review for T19

**Conditional approval of a preservation method, not approval of an executable mutation manifest. No files were sparsified, replaced, linked or deleted by this review.** Do not reduce the T19 scientific factorial or remove prior raw evidence just to fit the present disk budget.

The two scoped roots are `pulse_transfer/gpt2_gate_fixed512_run` and `superweight-causal-circuits/.reproduce-gate-validation/pulse_transfer/gpt2_gate_fixed512_run`. A read-only stat inventory finds105 common regular `.pt` path names, 2,224,005,975 logical bytes per copy, the same filesystem device, and distinct inode pairs for all105. Primary allocated blocks total2,224,599,040bytes; replay blocks total2,224,594,944bytes. These block counts are observations, not a guarantee of subsequently recovered free space.

The existing independent provenance audit SHA256 `6bc75f47714b46a6bb491e3b2ee09a564476ee8467bd9846dec006135cb6be4b` records completed8+1,352-forward replay,105 byte-exact file pairs and105 distinct inodes at audit time. The review here has not reread all4.4GB of content; a new mutation inventory must freshly verify every candidate against both hashes recorded in that audit. Completion manifests, logs, comparison results and scientific records are not deduplication targets.

## Preferred first option: verified sparse-hole reclamation

The installed `fallocate` advertises `--dig-holes`, which detects zero ranges and replaces them with holes. Logical size and read bytes can be retained while reducing allocated blocks, and separate inodes remain separate. Check the filesystem supports it on a small task-owned synthetic file first; measure potential aligned zero extents read-only before touching evidence. Do not use `--collapse-range`, `--insert-range`, `--zero-range` or arbitrary `--punch-hole` ranges on raw evidence.

The operation must have no concurrent writer. Freeze a finite allowlist with logical sizes, fresh whole-file hashes, device/inode/link count and allocated blocks, and pin the existing independent-replay evidence. After each eligible file, verify unchanged size and full content hash, then record allocated-block/free-space changes. Retain original mode/owner and record timestamps; sparse operations can change ctime even when bytes stay exact. Unsupported operation or hash mismatch is a stop, not a reason to substitute another mutation silently. This option's savings have not been measured and may be small for mostly nonzero tensors.

## Conditional alternative: exact hardlink deduplication

Hardlinking only already-proved byte-identical immutable outputs preserves the two logical paths and their complete read bytes. It is scientifically defensible **after** the independent replay is completed and audited. It does not turn a rerun into a copied result retroactively. However, it removes physical redundancy and introduces shared mutability: a future in-place write through either path changes both. The provenance documentation must state this plainly and retain the earlier distinct-inode evidence as a historical observation.

It also cannot preserve two different raw-file mtimes on one shared inode. Save both original stat records before mutation, and explicitly state that post-dedup filesystem timestamps/inodes are no longer independent-execution evidence. Keep the actual independent-run manifests/logs and their original hashes untouched. No new credential, filesystem immutability or security-setting changes are needed or authorized by this proposal.

The executable manifest needs an exact105-entry allowlist, not a glob executed against whatever files exist later. Per pair record:

- Both task-relative roots/paths, original raw-audit entry, logical size, both newly computed SHA256 hashes and expected immutable value.
- Both pre-operation `st_dev`, `st_ino`, link counts, allocated blocks, mode, UID/GID, mtime_ns and ctime_ns; reject symlinks, unapproved external links, nonregular files, wrong devices or unexpected owner/mode differences.
- Chosen canonical primary path, an unoccupied temporary path in the replay file's own directory, operation identity and an explicit planned/in-progress/completed state.
- Post-operation actual hashes/sizes and shared inode/link count, updated block/free-space accounting and before/after hashes for untouched run manifests/audits.

Use a one-file transaction: reread both hashes and identities; create the temporary hardlink to the canonical primary; verify its inode, size and hash; then atomically replace only that entry's replay path. Flush the transaction record and directory metadata as appropriate. Verify both final path hashes immediately. If a process stops after replacement but before the completion record, inspect the inode/hash state and reconcile the existing transaction; do not blindly repeat or treat an uncertain operation as unexecuted. Any mismatch stops further replacements. Never remove or replace a primary path, unique content, run manifest, failed-run artifact or unlisted unrelated file.

There is no need to rerun the model or numerical audit for unchanged bytes. A final logical inventory must prove all105 original hashes still appear at both exact paths, all non-target study files/manifests are unchanged, and the expected number of pairs now share inodes. Report recovered *allocated* bytes and actual free space separately from the unchanged4,448,011,950logical bytes represented by both path trees.

## Restoring physical independence

Before any future write to a shared raw file, materialize an independent copy. With sufficient free space above the research reserve, stream the chosen path to a new temporary regular file in that directory, preserve the desired original metadata from the transaction inventory, flush and verify the full expected hash, then atomically replace only that path. Verify different inodes and identical bytes afterward. Do not use another hardlink, and do not claim to restore the old inode number or ctime. This operation needs enough temporary space for the largest restored file; at the current inventory `attention_snapshots.pt` alone is423,261,199bytes. A copy-restore may be deferred until adequate free space exists, but the shared-storage annotation must remain until then.

## Recommendation

Prepare and independently review the read-only candidate manifest and sparse savings estimate first. If sparsification suffices, retain distinct inodes. If not, the reviewed exact hardlink transaction is preferable to dropping scientific arms or deleting old evidence, provided the shared-storage and timestamp limitations are accepted and documented. Parent owns any execution decision. This review performs no recovery action and makes no claim of newly available disk.
