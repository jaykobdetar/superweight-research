# T22 resource addendum v6: measured allocator retention

Prospective source-only amendment; execution requires separate review/GO.
Change only non-evidence CPU admission from896MiB to1152MiB. The3GiB RSS,
512MiB normal,1GiB failure and2GiB disk-reserve limits remain unchanged, as
do scientific arithmetic, controls, schedule, codec and array membership.

The first full stress fixture remains FAILED. It durably saved the largest
normal shard235656934B and completed its member/expected-entropy verification
before the next ordinary admission. That admission observed1077941464B of
non-evidence RSS, above939524096B, despite a total process peak1323085824B
below3GiB. This is observed post-encode allocator retention. The rejected
ordinary4385576B entropy tree was not saved and remains explicitly missing.

The new allowance is1207959552B. Its conservative known-header sum is:
`1207959552 +760300838 +760300838 +102926336 +102926336 +67108864 +33554432 =3035077196`,
below3221225472 by186148276B. V5 retained-copy/second-copy/byte-string terms
are unchanged. Conditional selected-normal buffered upper becomes2388350602B;
failure header upper2880843158B, buffer upper4258120184B (reject), and bounded
stream upper1970161982B. These calculations do not promise arbitrary allocator
behavior; actual per-route admission/RSS remains mandatory.

The fresh tail source reuses the existing large shard by immutable reference,
hash-binds the original FAILED report/source/journal, and re-verifies its
members without regenerating entropy. Only31 normal shards are generated,
using independent seed20261023:378 MAIN and372 SUFFIX records. Combined with
the inherited14/12 records, totals are392/384 across32 shards. Combined normal
charge conservatively includes the entire original artifact tree, new normal
files,32MiB prepared-input allowance and32MiB metadata allowance. Nothing is
copied, deleted, replaced or retroactively relabeled successful.

The same fresh failure entropy tree is streamed through both existing header
routes. Actual admission is recomputed immediately before each route; bounded
headroom is required for both, and frozen-header headroom for that route.
Forced routing remains distinct from observed buffer refusal. No completed
large shard, fake schedule, tokenizer or pretrained/model run is repeated.
