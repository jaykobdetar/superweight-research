# T22 resource addendum v2 — actual L11 mask and source-specific failed operands

Prospective source-only delta. Preserves the original symbolic inventory `bce2948f8a4e5d6a67c22f89e8d91b7fd34c133084208e9ce8e88e7aff9dcea7` and resource plan `ebc7f75070826d7f20bd8e1d0c5faca40241d781cadc2dd557b4a48a7c6b779e` unchanged. This addendum neither approves a fixture/model nor establishes measured allocator behavior. Root/peer review and separately authorized CPU witness remain required.

The exact additive path/shape/dtype/byte inventory is [t22_symbolic_resource_inventory_v2_additions.json](review/t22_symbolic_resource_inventory_v2_additions.json), SHA256 `73bb998a3014de4a8d5a61622916e25f573ef302ec82cb1fa86285e78aaa52cb`. Hook source at assessment is `gate_l11_key_value_donor.py`, SHA256 `1a4b9ed45d3ae621e76751f5c2f7639bd3db2a4eacc5e9f4b0003a90dec8cf28`. No new metric, arm, numerical tolerance, codec, cap or frozen source changes are introduced.

- Save one actual selected baseline MAIN **L11** diagnostic effective mask, FP32[1,1,512,512], under `l11_mask_reference`:1,048,576bytes. All nine selected L11 panel states bind actual-mask fingerprints and exact baseline-reference guards. No L10 mask assumption and no nine duplicated raw mask arrays. The hook exposes `l11_effective_mask = native.effective_masks[11]`; model mask arguments remain unchanged.
- Runtime baseline references retain each variant's actual L11 effective mask. Failure handoff copies those two GPU tensors into the complete owned CPU tree:2,097,152additional bytes. Current native masks were already included in the original core. The optional uncommitted selected normal payload adds the single normal mask occurrence as well. These copies and per-member codec overhead are counted, even if some objects happen to share underlying GPU storage.
- Count the nine current L11 FP64[512] norm arrays explicitly in the hook failure tree. Bound the runner's failed suffix expected seven L11 prefixes (13width units at128rows), nine prefix norm arrays and extra actual norm/metric occurrences. Suffix metrics retain actual127/128 shapes;511-row metric descriptors are only conservative storage upper bounds.
- Count the external identity comparison's actual/expected full hidden/write/metrics and the source comparison's actual/expected48source arrays as separate serialized occurrences when supplied. This is complete defined comparison evidence, not a second entire bundle/cache.
- Add explicit post-closure known-ABI invalid-output/callback and source-readout-q0 branches. An invalid output or failing observation callback saves its complete actual512-row output. A failed combined later-logit comparison keeps its actual385-row operand; it does not also serialize an unrelated full copy. The wrapper binds the actual projected output before inherited cache checks, and binds the actual raw readout before any observation callback can throw.

Conditional revised bounds (unchanged896MiB non-evidence CPU,2.25archive-buffer and96MiB workspace/normalization assumptions):

| Quantity | Conditional upper bytes | Cap bytes | Headroom bytes |
|---|---:|---:|---:|
| Complete normal evidence | 513,184,538 | 536,870,912 | 23,686,374 |
| Largest complete failure NPZ | 665,257,028 | 1,073,741,824 | 408,484,796 |
| Worst process RSS | 3,192,817,343 | 3,221,225,472 | 28,408,129 |

The largest RSS branch remains orthogonal combined-KV with a known-ABI invalid output. **Only28,408,129bytes of conditional RSS headroom remain.** This supports source review and a later authorized witness only. The descriptor additions deliberately overbound optional reached-stage occurrences; they are not an actual encoded archive/member count. Before any model run, the source-specific whole-tree witness must bind the exact real array universe and metadata/NPY/ZIP sizes, prove no extra whole-cache clone, measure buffers/RSS and validate current-RSS admission. Any additional field/copy or unsupported assumption must be preserved and prospectively amended; no failed evidence may be dropped to fit.

Zero fake/pretrained forwards, checkpoint/weight access, input selection or deletion occurred in making this addendum. All30 frozen T21 source hashes remain unchanged. All earlier drafts and the source-edit context failure remain preserved.
