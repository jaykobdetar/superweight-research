# T20 prospective resource amendment: complete failure evidence

This addendum changes only the failure-artifact storage allowance in the
frozen T20 protocol, SHA256
`e562fd57097228fb3348d8a7d564010456651c9dd6bde4540496a11bae338aff`.
It is written before fresh T20 passage selection, pretrained model loading,
or T20 model outcomes. It requires independent review and a passing complete
layout fixture before the separate source/input/resource launch freeze.

## Preserved sizing failure

The immutable first complete-layout attempt is
`review/t20_writer_fixture_1792618876598339573/report.json`, with its measured
sources archived in the adjacent `measured_sources/` directory. It completed
26 synthetic CPU forwards using the production hook and capture/packing
helpers, then serialized and byte-round-tripped all 64 high-entropy normal
shards. Normal storage plus the 32 MiB metadata allowance was 360,061,900 bytes,
within the unchanged 536,870,912-byte normal cap. Peak RSS was 1,893,676 KiB.

Its complete high-entropy failure tree contained 327,881,682 bytes of arrays;
serialization required 304,642,136 bytes, exceeding the original
268,435,456-byte failure allowance. The writer rejected the blob before
writing it. The report, shapes/dtypes/byte counts, normal artifacts, source
snapshots, and traceback are preserved. The unsaved failure arrays and blob
were in RAM only and are not described as recovered evidence. Recreating
them requires a new deterministic synthetic fixture, not a model experiment.

## Resource change and unchanged requirements

Raise the failure-artifact cap from 256 MiB to **384 MiB (402,653,184 bytes)**.
The complete raw-array universe has 74,771,502 bytes of headroom under the new
cap, before container metadata and compression; this arithmetic is a sizing
motivation, not a claim of a universal worst-case proof. The next fixture must
save and round-trip the complete capsule, account for every required failure
field and compared side, and establish the implemented finite shape bound.
No omitted array, reference/deduplication codec, lossy compression, summarizing
replacement, or evidence cleanup is authorized by this amendment.

Keep the normal-artifact cap at 512 MiB per segment/composite, the independent
2 GiB free-disk reserve, and the CPU peak RSS limit of 3 GiB. With all planned
outputs still outstanding, reserve three normal caps plus one failure cap and
the reserve: **4,160,749,568 bytes (3.875 GiB)**. Continue subtracting only
verified materialized output from outstanding allowances and checking actual
free space before and during execution. Current free space on
`/dev/nvme0n1p2` was 26,409,594,880 bytes when this addendum was prepared;
prelaunch must remeasure it. No files are deleted or replaced.

The two 388-forward sequential segments, 776 scheduled returns, inputs,
calibration, intervention arithmetic, endpoints, controls, snapshots,
statistical rules, numerical tolerances, and scientific stopping rules are
unchanged. The original failed fixture remains a failed attempt. A new
passing fixture does not retroactively relabel it.

## Finalization ordering correction

Separately bind the narrowly scoped writer correction that performs all cap,
reserve, and bounded metadata checks before writing the canonical completed
manifest as the last fallible operation. Preserve the measured source first.
Hostile cap/reserve fixtures must demonstrate that rejected finalization leaves
no completed canonical manifest acceptable to the merger. This is evidence
bookkeeping; it changes no experiment or numerical acceptance condition.

Independent review must inspect the amended cap, complete failure membership,
finite shape assumptions, round-trip result, memory peak, remaining-space
accounting, and finalization behavior before this addendum can be used for a
pretrained launch. This document alone is not launch approval.
