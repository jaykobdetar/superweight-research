# T21 prospective resource-only amendment — draft for independent review

This amendment proposes changing only the T21 failure-evidence cap from
384MiB (402,653,184 bytes) to **1GiB (1,073,741,824 bytes)**. It changes no
scientific arithmetic, support sets, thresholds, arms, order, controls, scoring,
input selection, calibration, snapshot membership or model-forward counts in
`GPT2_QUERY_WRITE_TIMING_PROPOSAL_v4.md` (SHA256
`a17e3e7e8150935950d58df1d2b266dac2bf91a6c65c4a8539d7e28fcf3e0819`).
The normal cap remains512MiB, CPU peak limit3GiB, pre-model MemAvailable minimum
4GiB and physical disk reserve2GiB. Production sources still use the original
384MiB failure cap at the time this draft is written. Independent review and
literal source binding must precede any cap change, selection or model load.

## Why the initial cap fails

The single authorized new-ABI fixture executed22 synthetic CPU forwards:
ten arms, main then suffix, plus two natural-window0 native anchors immediately
after baseline main. It loaded no tokenizer, pretrained checkpoint or GPU model.
Its immutable report is
`review/t21_writer_layout_fixture_1792624844525884190/report.json`, SHA256
`752ec44e982eb7cfac52389692a56468a49beee3bf8ec35d0931d0a1aa272dba`.
Executed fixture source SHA256 is
`2475864fbbe9dd1a5b909e6827f2fd813fb6cafbe56d58af0cee85d6789c54b0`.

All32 high-entropy normal shards passed lossless member/container roundtrips.
There were328 main metric records including eight anchors and776 explicit
control dictionaries. Measured normal bytes were276,735,757; charging the entire
prospective32MiB prepared-input budget AND another32MiB for final metadata gives
343,844,621 bytes, below512MiB by193,026,291 bytes. The selected ten-state panel
was168,903,264 bytes; the largest ordinary shard was3,652,682 bytes.

The complete finite conditional failure union has1996 array members with
896,033,778 raw array bytes. It includes the full main/suffix caches, fixed
matching recipe and six support-delivery states; four-node timing references,
current arrays and snapshots; native pending/failure state and full compared
wrapper records; both compared bundles and current511-target metrics; full512
maxima plus up to ten vocabulary rows; and the complete uncommitted ten-state
snapshot/readout/timing panel including both anchors. The readout-operand
failure branch includes **two full512×50257 FP32 operands**, rather than only
prefixes or sampled rows. Extra full observer output and prefix operands are
also included conservatively across mutually exclusive failure branches.

Encoded size was831,398,069 bytes, so the original384MiB production criterion
rejects it. With separate prior authorization, those exact oversized fixture-only
bytes were preserved at
`review/t21_writer_layout_fixture_1792624844525884190/oversized_failure_blob.npz`,
SHA256 `8be79139820b03a1f883344f73c3899542907452b3056cfc7291b09f290a141f`.
Every member name/order, logical scalar/container, dtype, shape and actual array
byte round-tripped. This artifact is explicitly **a rejected production-cap
fixture**, not a successful384MiB production write. No payload was truncated,
deduplicated, deleted or silently omitted.

Measured peak RSS was2,504,524KiB =2,564,632,576 bytes, below3GiB by656,592,896
bytes. Wall time was83.6654 seconds. Disk free after the retained fixture was
23,371,415,552 bytes, above the2GiB reserve. The full source closure was checked
unchanged at the end. This is a finite known-ABI failure-union certificate, not
a theorem about arbitrary externally supplied shapes or unlimited error text.

## Conservative NPY, DEFLATE, metadata and ZIP bound

A read-only follow-up inspected all1997 ZIP members (1996 arrays plus the
metadata array), their headers and the exact archive container. It did not
rerun forwards or decode the large array payloads. Evidence:
`review/t21_failure_resource_bound_1792625137744003236/report.json`; bound source:
`review/t21_failure_resource_bound.py`.

- Raw arrays:896,033,778 bytes.
- Raw encoded logical metadata:3,379,358 bytes.
- NPY headers:255,616 bytes, exactly128 per member.
- Total uncompressed NPY members:899,668,752 bytes.
- Exact ZIP overhead:225,466 bytes, including local ZIP64 extras, central
  directory and end record. Member names/order remain fixed; no new member
  schema is proposed.

For each uncompressed NPY member of length n, use the zlib default-parameter
`compressBound(n) = n + (n>>12) + (n>>14) + (n>>25) + 13`. The exact runtime1.3
C implementation, obtained from this interpreter's exported symbols, matched
that expression at every member length and every charged length. The ZIP codec
uses default-parameter raw DEFLATE; the bound includes zlib framing and therefore
conservatively bounds that raw output. Summing the per-member bounds and all
ZIP overhead gives900,193,843 bytes for the measured member universe, without
relying on its observed compression ratio.

Charge a further **32MiB of raw metadata growth**, beyond the measured3.38MB,
for bounded runtime failure messages and source/resource provenance changes.
Recompute the metadata member's DEFLATE bound at36,933,918 NPY bytes; retain all
other members, shapes and headers. The resulting bound is **933,758,516 bytes**,
below1GiB by139,983,308 bytes. Any added array/member, changed shape/dtype,
compression settings, or metadata growth beyond that reservation requires a
new compatibility/bound review; it is not covered by this certificate. Runtime
hard caps still reject any unforeseen overflow without truncation.

## Sequential storage reservation

Keep the reviewed remaining-cap accounting and immutable output policy. The
new local writer explicitly supplies the T21 caps; the frozen T20 writer must
remain unchanged.

- At A admission, reserve2GiB physical free,1GiB for the active failure,
  512MiB remaining A normal cap, and two future512MiB normal caps for B and the
  composite:4,831,838,208 bytes initially (4.5GiB).
- B may release A's unused future cap only after complete A validation and
  predecessor binding. Its initial reservation is2GiB free +1GiB failure +
  512MiB B +512MiB composite =4,294,967,296 bytes (4GiB). Retained A files are
  already charged to physical disk usage; they are never deleted or counted as
  free space.
- The composite's conservative admission reservation remains2GiB free +1GiB
  failure +512MiB normal =3,758,096,384 bytes (3.5GiB).
- During writing, actual used bytes consume the remaining normal allowance;
  failure writes reserve their projected bytes plus untouched future caps and
  the2GiB floor. Only one segment/fixture process runs at a time. Earlier
  failures and all other artifacts remain physical disk usage for every later
  admission. No automatic retry or assumed cleanup creates free space.

The one normal cap per segment and composite, individual final guards, CPU
peak3GiB and pre-model4GiB host headroom are unchanged.

## Minimal code delta after approval, and no-new-forward verification plan

1. Preserve the measured T21 source versions and hashes. Add a T21-local
   `EvidenceWriter` subclass in `query_write_timing_storage.py` that delegates
   every operation to the immutable T20 class but has explicit constructor
   defaults512MiB/1GiB/2GiB. Merely rebinding a module constant is insufficient:
   imported T20 default arguments would otherwise remain384MiB. No frozen T20
   file is edited and no serialization or guard arithmetic is rewritten.
2. Set the new T21 failure-limit constant, pin/archive the approved addendum in
   the new runner source closure before weights, and expose the actual cap in
   segment/composite metadata. A/B equality/predecessor guards continue checking
   exact cap fields. Other scientific bodies and payload membership remain
   AST-identical to the measured version.
3. Without model forwards, prove the wrapper delegates to the unchanged T20
   save/checkpoint/guard methods and that its effective defaults are the new
   values. Exercise initial384MiB rejection of the measured encoded length and
   new1GiB admission using the actual inherited guards, plus a small production
   codec signed-zero/scalar/full-shape-header witness and immutable-output tests.
4. Reuse the already retained blob and its exact roundtrip certificate. A
   streaming recheck may independently verify all metadata and array member
   hashes without a second large decoded copy. Do not rerun the22 forwards,
   duplicate the831MB blob, or call it newly serialized under different
   scientific semantics. Recompute the member/header/DEFLATE bound and hash
   the unchanged preserved artifact. Source/addendum metadata changes fit the
   explicit32MiB extra metadata reservation; any larger layout change instead
   needs a new bound review.

This draft grants no launch or publication approval. Input-chain attestation,
preparation/source literals, independent code/audit policy review and final
resource gates remain separate requirements.
