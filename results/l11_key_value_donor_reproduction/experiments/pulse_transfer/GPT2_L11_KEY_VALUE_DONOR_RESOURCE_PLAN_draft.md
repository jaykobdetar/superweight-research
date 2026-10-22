# T22 symbolic resource and complete-failure plan — prospective draft

**READY for independent design/resource review only. Conditional feasibility; no implementation, CPU model fixture, selection, checkpoint access or model/GPU run is approved by this document.** The proposal v1 and its scientific clarification v2 are preserved. This plan fixes the producer ownership and branch lifetimes that an implementation and a separately authorized CPU witness would have to demonstrate. No T21 cap or measured layout is treated as a T22 measurement.

The machine-readable [symbolic inventory](review/t22_symbolic_resource_inventory.json) is the authoritative path/shape/dtype/array-occurrence inventory. It contains descriptors and integer byte counts, not private tensors or experimental results. It binds proposal v1 `7b1dd7287a69ab895cc071b7a8f050295ba37b31b266fde3add88b158ab1adc0`; the operative scientific clarifications are [proposal v2](GPT2_L11_KEY_VALUE_DONOR_PROPOSAL_v2.md), `6238ea83f978705b0069e3b476f3e867ba95f08fbae2864d6e92827b88e2f159`. Resource inventory SHA256: `bce2948f8a4e5d6a67c22f89e8d91b7fd34c133084208e9ce8e88e7aff9dcea7`.

## Fixed scope and method

The proposed study has 12 arms, main then suffix per arm, 32 input identities, 768 scientific calls plus 8 anchors = **776** calls, planned A/B 388 each. The five L10 recipe states, nine L11 projection/transplant states, twelve final-hidden states/48 readout rows and twelve five-row timing states have distinct purposes. They are not twelve duplicated full L10 snapshots. The 19 contrast definitions, three metric families and three scoring views remain unchanged by this resource plan.

Only metadata and shape inventories from the preserved T21 whole-layout witness were read for this calculation. Its report SHA is `752ec44e982eb7cfac52389692a56468a49beee3bf8ec35d0931d0a1aa272dba`; it is evidence for the frozen codec's existing array universe, not proof that a new producer emits the proposed tree. The frozen storage codec SHA is `83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06`. Existing duplicates actually emitted by that layout are counted. New arrays and prospective removals of incompatible branch unions are explicit inventory entries, not empirical storage observations.

The codec writes each tensor occurrence as a separate numeric NPY member inside an NPZ archive; it does not deduplicate aliases. The inventory therefore counts serialized occurrences. Raw bytes below are sums of shape products times dtype size, with no assumed compression ratio. Complete scalar dictionaries, source/inventory records and metadata are covered by explicit caps, to be checked against the eventual source and witness.

## Normal evidence

| Shard class | Count | Array occurrences per shard | Raw array bytes per shard |
|---|---:|---:|---:|
| Ordinary | 28 | 1,040 | 4,385,576 |
| Anchor | 3 | 1,085 | 4,637,188 |
| Selected raw-panel | 1 | 1,365 | 252,604,128 |

All 32 shards total **389,311,820 raw array bytes**. The conditional serialized upper bound is **512,134,989 bytes**, including 64 MiB for preparation/source/finalization artifacts, metadata allowances, NPY/ZIP overhead and zlib worst-case expansion. It is below 512 MiB (536,870,912 bytes) by **24,735,923 bytes**. This is not a measured production size. It requires the eventual normal files and metadata to be members of the declared inventory and caps; no accidentally retained full closure-reference logits are allowed in normal shards.

## Ownership and bounded lifetimes

1. Per input, retain variant-specific baseline, full proposals and the once-computed matched recipe; retain only the early-support delivery required by this study. The new state machine does not execute unused T21 support arms. Full L10 frozen arithmetic and all byte branches remain unchanged.
2. Keep baseline full L11 QKV and required baseline references for both variants. Keep original full donor QKV for both directions and variants through the four single-path arms. Never borrow a main reference for suffix or a PC reference for orthogonal control.
3. The hook owns one direction's **two actual donor later-reference sets** at a time: L11 context/write/block, final hidden and all later vocabulary logits. Each set covers q127–511. Both sets together occupy 164,253,320 bytes, including 154,791,560 bytes of logits. PC donor main/suffix precede PC-KV main/suffix; then orthogonal donor main/suffix precede orthogonal-KV main/suffix. A successful variant retires its reference only after all five closure checks pass; a failed variant retains every compared operand. The inventory conservatively allows both variant references during closure failure.
4. After both closures of a direction, retain only its donor QKV and small closure receipts. All four direction/variant closure keys must exist before any single-path arm. The full closure-reference pair is absent during those later arms; the larger twelve-state partial normal payload cannot coexist with that pair under the prospective ordering.
5. For the selected raw-panel input, a PC-KV main failure can have at most five completed main snapshots; an orthogonal-KV main failure at most seven. A post-closure failure may have all twelve. Those exact prefix trees, rather than a union of all snapshots and all live closures, are counted. For an interrupted save, attach the still-owned uncommitted raw payload once. If a partial save committed successfully, retain its immutable path/hash receipt rather than serialize that full payload a second time.
6. `timing_current` and `timing_snapshot` are one full set at mutually exclusive stages. Native-pending, native-failure, interpolation-failure, suffix and identity comparisons are distinct branches; include the branch that occurred and its complete actual/expected operands. Do not manufacture missing future fields or union failures from mutually exclusive instructions.
7. On failure, freeze the current producer state against reset/forward/mutation. A single capsule owns the existing CPU trees; it does **not** deep-clone complete already-CPU caches again. Actual GPU operands needed for the failed comparison are copied once. Metadata names the owning paths for the comparison. No new alias-aware codec, dynamic deduplication or post-hoc omission is proposed. Frozen native-stage copies and duplicated fields already emitted by frozen components remain counted. If composing frozen `clone_tree` behavior forces another full cache clone, this envelope does not apply: count that term and obtain a prospective amendment before proceeding.
8. After a committed input, retain only scalar/inventory records for that input, not its raw array trees. Serialization must not retain an old encoded blob or traceback frame containing one while constructing a second large failure blob. Preserve the textual traceback and complete owned comparison evidence; do not accumulate retry buffers. These are producer-lifetime requirements, not permission to remove saved files.

## Complete branch inventories and conditional bounds

Every listed branch includes the applicable frozen native/write/current caches, baseline L11 extras, current original/expected/delivered L11 arrays, computed metrics/counters and the allowed uncommitted snapshot prefix. The JSON lists each component and every array path. A suffix failure includes its actually compared expected prefix fields, not unrelated whole left/right bundles. Identity/current/anchor failures retain both full vocabulary operands. Closure failures retain the actual later operand and full donor reference; the known-ABI invalid-output branch instead retains the entire actual 512-row output. No q511 evidence is dropped.

| Failure branch | Array occurrences, upper | Raw array bytes | NPZ bytes, conditional upper | CPU RSS bytes, conditional upper |
|---|---:|---:|---:|---:|
| `identity_current_or_anchor` | 709 | 449,607,280 | 458,596,394 | 2,521,636,559 |
| `PC_KV_closure` | 1,078 | 565,129,874 | 574,395,072 | 2,897,706,178 |
| `orth_KV_closure` | 1,315 | 627,088,490 | 636,527,282 | 3,099,462,267 |
| `orth_closure_knownABI_invalid_model_output` | 1,315 | 652,619,046 | 662,065,631 | 3,182,454,108 |
| `postclosure_transplant_or_cache_or_source` | 1,779 | 478,140,060 | 487,836,234 | 2,615,958,979 |
| `postclosure_suffix` | 1,879 | 537,053,604 | 546,833,001 | 2,807,615,249 |
| `postclosure_native_pending` | 1,788 | 487,123,468 | 496,828,258 | 2,645,174,441 |
| `postclosure_native_failure` | 1,792 | 490,315,180 | 500,023,554 | 2,655,555,569 |
| `donor_interpolation_early_failure` | 1,326 | 394,946,718 | 404,321,850 | 2,344,858,273 |

The largest conditional failure NPZ is **662,065,631 bytes**, below 1 GiB by 411,676,193 bytes. The largest raw tree has 652,619,046 bytes. These bounds apply to the fixed FP32, batch1, 512×768, vocabulary50257 ABI and bounded primitive metadata, not arbitrary malformed external tensor sizes. Completeness means every operand defined for the failed branch; it does not mean speculative arrays from every other possible failure.

## Serialization and RSS assumptions

For an array's raw byte count `n`, bound its NPY header by 256 bytes and compressed length by the default-zlib bound applied to `n+256`: `x + (x>>12) + (x>>14) + (x>>25) + 13`. Bound ZIP container overhead by 384 bytes per member plus 256 bytes for the archive end; numeric array names are bounded and archives are below 4 GiB. Normal metadata's compressed member is at most 1 MiB per shard; failure metadata's compressed member at most 8 MiB. Raw codec metadata is at most 8 MiB normal and 16 MiB failure. The future CPU witness must verify the exact codec headers, member names and metadata caps. No high-entropy compression discount is used.

The conditional encoding-stage RSS formula is:

`896 MiB non-evidence CPU + owned raw arrays + ceil(2.25 × NPZ upper bound) + 64 MiB metadata/compression workspace + 32 MiB normalized-NumPy copies`.

The 2.25 multiplier allows the in-memory BytesIO capacity, returned bytes and reallocation under the actual runtime. It is a proposed measurable allowance, not a proof for arbitrary allocator behavior. The 64 MiB workspace must cover bounded JSON temporaries and the compression chunk; the 32 MiB copy allowance covers normalization of the retained NumPy recipe trees by the existing codec adapter. A separate header/identity stage bounds the largest `tensor_identity` `tobytes()` temporary (102,926,336 bytes) before the compressed archive buffer exists; it is lower than the encoding-stage maximum. The 896 MiB non-evidence budget includes Python/scalar objects, CPU model/runtime residuals and allocator retention not already counted as owned evidence. An earlier T21 setup measurement does not establish this post-load T22 budget.

The resulting maximum is **3,182,454,108 bytes** versus the 3 GiB limit of 3,221,225,472 bytes: only **38,771,364 bytes of conditional headroom**. This narrow margin does not justify a model launch. The actual producer may exceed it through cloning, allocator retention, metadata growth or buffers not represented here.

The implementation must therefore add a minimal **pre-encode current-RSS admission check**, using the actual owned-tree size and the declared encoding/header-phase increments, plus explicit checks of the 896 MiB non-evidence and metadata/copy/buffer assumptions. Record the measured terms, stage and reason before admission; do not start an encode whose prospective RSS exceeds 3 GiB. This check complements the existing resource guards and does not turn a record-only function into a hidden throwing cleanup path. A guard rejection must preserve the already-computed fields/counters and owned failure operands without a second full cache clone. Its failure/finalization path needs a small source/CPU witness; an admission rejection is not proof that complete evidence was saved. If the implementation cannot preserve the complete branch within these limits, stop before a model run and amend the plan. The witness, failure and old versions remain saved.

## API integration and review boundary

The hook owns end-state/closure references. Add a narrow **model-output-observed** event when the actual LM output reaches the bound end observer, before any closure checks that may throw. The runner separately increments **model-returned** only after the model call returns. Never relabel an output-observed callback as a successful return. Normal fingerprint/copy capture precedes `finish`; `finish` requires the direction/variant closure receipt and all protected-domain checks. Failure serialization retains the actual captured fields and counters from the reached stage. Baseline and donor full QKV survive until singles finish; full direction-specific closure references do not.

Independent review must accept the branch completeness, scientific v2 scope and source/API ownership before implementation. Only then may a separately authorized smallest changed-ABI fixture and one complete whole-layout/failure witness test the exact producer/codec with high-entropy data, source hashes, dtype/shape/member/byte roundtrip, cloning behavior and peak RSS. Do not start that witness now. Any field or lifetime deviation requires recalculation before a model run; no threshold relaxation, field dropping, outcome-based selection or historical model rerun is permitted.

This work executed integer arithmetic over descriptors only: **zero fake/pretrained forwards, zero checkpoint/weight accesses, zero input selection, zero scientific source edits and zero deletions**. The initial two-buffer envelope and first syntax-error attempt are preserved in adjacent review files. All 30 frozen T21 source hashes were checked unchanged. The resource plan is ready for review, not approval to launch.
