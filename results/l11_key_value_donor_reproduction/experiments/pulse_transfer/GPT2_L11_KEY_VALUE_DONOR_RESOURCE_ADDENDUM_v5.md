# T22 resource addendum v5: noncontiguous CPU normalization

Prospective source accounting only. This does not approve a model, selection,
or large fixture execution. All v4 evidence and both failed small-fixture
histories remain preserved. Scientific arms, order, arithmetic, controls,
scores, thresholds, array membership, shapes, dtypes and caps do not change.

The unchanged frozen `_encode` retains a separate contiguous CPU array for
each noncontiguous Torch occurrence. Its subsequent `tensor_identity(obj)`
independently normalizes the same operand again, then allocates its C-order
byte string. Earlier admission charged NumPy normalization and an identity
byte string, but omitted these Torch normalization allocations. This source
amendment closes that specific accounting gap; it does not change the codec.

## Admission terms

Let R be current measured RSS, T the sum of bytes of CPU Torch occurrences
whose `is_contiguous()` is false, U the maximum byte size of such an
occurrence, I the largest serialized array byte size, A the existing buffered
NPZ upper bound, and P=max(32MiB, actual NumPy normalization bytes).

- Header upper: `R + T + U + I + 64MiB + P`.
- Buffered encoding upper: `R + T + ceil(2.25*A) + 64MiB + P`.
- The encoding upper is the maximum of these phases; buffered admission
  requires both to fit. NumPy's separate32MiB allowance is unchanged.
- Frozen-header direct streaming uses the revised header admission.
- Bounded-header direct streaming remains `R + 64MiB + 1MiB`; its SHA/NPY
  traversal attaches existing CPU views and does not call `.contiguous()`.

T is summed per serialized occurrence, including repeated references, because
the frozen codec preserves every occurrence. This is RAM accounting, not an
alias codec or file deduplication. U is transient; it is not added to the
retained buffered phase after metadata creation has finished.

The pre-model/post-load/pre-forward guard now explicitly covers original raw
evidence, retained normalization copies, the second normalization temporary,
and the identity byte string. Its maximum under unchanged896MiB non-evidence
admission is:

`939524096 +760300838 +760300838 +102926336 +102926336 +67108864 +33554432 =2766641740`.

This is below3221225472 by454583732 bytes. Each term is separately named in
the `known_header_admission` receipt. It remains a conditional bound for the
declared finite ABI and controlled ordinary failures, not a proof about an
arbitrary allocator or unknown external shape.

## Whole-layout stride stress, without more model calls

The new unexecuted layout source uses the same actual normal/failure
containers and reviewed per-occurrence shapes. For Torch occurrences with
two or more nontrivial dimensions, it allocates independent finite entropy in
the reversed shape, then reverses axis order as a view. This produces a dense
noncontiguous tensor with the original shape/dtype/nbytes and exactly the same
amount of backing storage. There are no holes, extra backing arrays, aliases,
or dropped members. NumPy occurrences retain their existing category and
ordinary allocation behavior.

These are intentionally stressed strides, not a claim to reconstruct the
actual model tensor strides. Scalar/vector occurrences remain contiguous.
The streamed verifier still checks actual C-order bytes, dtype, shape,
fingerprint, member order, and the complete scalar/container tree. No fake
schedule or successful small-case replay is added.

JSON-only analysis of the complete concrete maps gives:

| Universe | Raw bytes | Stressed Torch occurrences | Retained-copy bytes | Largest second-copy bytes |
|---|---:|---:|---:|---:|
| Selected normal input | 253652704 | 327 | 250815648 | 4718592 |
| Actual reached orth donor SUFFIX failure | 694044990 | 465 | 672322648 | 102926336 |

At the unchanged896MiB conditional baseline, selected-normal header and
buffered uppers are1554092928 and2119915146 bytes. The actual-map failure
header upper is2612407702 bytes, and its buffered upper is3989684728 bytes,
so the latter route is rejected at that conservative upper. Bounded streaming
is1701726526 bytes. These are calculations, not measured runtime peaks. The
whole fixture must record actual admission/RSS and must not claim an observed
buffer refusal solely from a forced route control.

All serialized raw-byte and NPZ bounds from v4 remain unchanged. The global
normal upper remains513184538 under512MiB. The reached-stage conservative
stream upper remains712217304 under1GiB; the disk projection remains712413912
including196608 bytes for possible journal-refusal sidecars. The exact mapped
failure694044990 remains below the694250482 branch raw upper. Metadata-only
source/receipt additions remain under the existing32MiB metadata allowance;
the complete32MiB prepared-input allowance remains separately charged.

## Compatibility and remaining gates

The old95a storage source, old7b layout source and prior HOLD map/contract are
preserved. The accompanying v5 source-closure report binds exact current
source hashes and verifies that restoring only the two accounting functions
and removing the two new copy helpers recovers the old storage AST. Codec,
failure preservation, disk/NPZ caps and scientific methods are unchanged.
The stress source differs only in entropy allocation/view layout and its
explicit scope label. Complete map entries remain identical; only source
bindings/provenance are adapted in a fresh HOLD version.

Required next gates are independent source/resource/N-receipt review and a
separate explicit exclusive CPU whole-layout GO. No pretrained load, GPU,
additional fake forward, or large entropy allocation is performed here.
