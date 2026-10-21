# Restricted T19 export: independent audit passed

The existing v1 export has exactly 83 files (244,621,310 bytes). All 41,632 allowed arrays match the current raw composite in shape, dtype and values; all 32 scalar/control ledgers match an independent projection. The 1,096 main records include eight anchors; all 1,088 suffix records are present. Exact membership and every external inventory hash/byte count pass.

Public numerical/readout audit views match explicit expected fields. All 32 saved prediction IDs are omitted; no forbidden token-ID fields, private locators or unexplained dense JSON vectors were found. No QKV, hidden states, contexts, full readouts, tokens or weights are exported. Frozen analysis, original missing guards, journal commitments and the first failed/readout-review chronology remain intact.

The actual export used archived adapter `56718edef3dbbab45fadd5da2f4bacde1bc50c731035ced66c803ffa2950388d`; its `export.json` is `13b21a7c807447bd6b6a831dd75a8847f75ddef14ea35866d6fc2d0bb7b41e13`. The adjacent JSON binds the raw manifest, inventories and independent auditor source. Audit: 29.811 seconds, 244,899,840-byte peak RSS, one CPU thread, zero model forwards, weights reads, GPU operations, deletions or publication. This approves the reviewed existing export's contents, not arbitrary future exporter inputs or repository integration.
