# T18 preserved input-failure audit

The failure is valid and cannot be resolved by searching a finer Hume grid under the frozen exclusions. The 212 Hume exclusion records merge into 20 intervals covering 302,793 of 347,236 processed characters. The remaining 44,443 characters occur in 19 gaps; the largest is 5,492 characters at `[318520,324012)`. Every gap is shorter than 8,192. Thus there are **zero eligible full 8,192-character buffers at any integer start**, not merely zero on the 1,001-point grid. This conclusion is conditional on the unchanged exclusion policy, which includes historical conservative buffers; it does not claim every excluded character was processed by a model.

The largest remaining Hume gaps are:

| Half-open interval | Characters |
| --- | ---: |
| `[318520,324012)` | 5,492 |
| `[76000,81454)` | 5,454 |
| `[310955,316314)` | 5,359 |
| `[127862,132274)` | 4,412 |
| `[119103,123395)` | 4,292 |

All 1,001 recorded Hume rejections independently match historical overlap and the specified distance/numerator order. The full union, all gaps and exact source identities are retained in audit evidence (private-only file `gate_calibration_factorial_failure_audit.json`, SHA256 `ba7efb46473e2f99d76b230abfebe3056274d0c7613f77a97a1af470fd8fb632`; see the [omission manifest](../../manifests/wiki_calibration_review_export.json)).

## Mapping and preservation checks

The independent audit script (private-only file `gate_calibration_factorial_failure_audit.py`, SHA256 `60fb677cc4f620543c93f6363989edb67e5c44f8d96829ea2276feafcdcf2a07`; see the [omission manifest](../../manifests/wiki_calibration_review_export.json)) does not import or execute the selection helper. It verified 185 provenance hashes and nine processed source identities; re-tokenized the original Wiki-test/Hume streams with the pinned OLMo tokenizer; matched all 404 saved range records, including 272 exact saved input vectors and 132 original/declaratively conservative ranges; and checked the complete 28 saved-token-file universe plus four explicit input aliases. The actual lengths are 128, 512, 1,024 and 4,096 tokens. It also checked 48 inherited transfer calibration/development buffers, the 102 original-report Wiki-validation ranges and the supplemental audit's zero overlaps with all eight T16/T17 Wiki passages. A file with zero Wiki-test/Hume vectors has other source branches; their offsets are not transferred between corpora.

The four retained Wiki-test passages exactly match the saved partial arrays, passage hashes, GPT-2 tokenization, end offsets and first-token identities. Their grid numerators are **128, 375, 625 and 874**. Every full 8,192-character extraction buffer is disjoint from all historical exclusions and the other three new buffers. All seven recorded earlier Wiki candidate rejections match the frozen order and overlap policy. Before these new choices, Wiki has 504 grid starts allowed by the historical interval geometry; this is descriptive capacity, not another selection.

The preflight remains failed at `calibration_selection`, with four partial calibration passages, no evaluation panel and zero model forwards. Every file in its directory was hash-checked before and after the audit and left unchanged. The audit used 983,216 KiB peak RSS and 1.43 seconds; no model, GPU, new passage selection or calibration fitting occurred. An initial auditor-only assertion transposed the raw gate shape. Its script, JSON and log remain under `gate_calibration_factorial_failure_audit_initial_shape_fixture.*`; after checking the actual `[12,511]` layout, the corrected audit passed. That assertion was not a study failure or a change to its evidence.

## Recommended prospective revision

Use the **original four Wiki-test calibration passages versus the exact four fresh Wiki passages already retained**. Both fits then use 2,040 vectors per layer (four passages, q1..510), preserving source, sample count, query weighting, calibration world and fitting algorithm. The original raw file contains all required old Wiki tensors, so deriving that four-passage fit requires no new old-calibration forwards. The fresh fit would require four new calibration forwards after the revised inputs, protocol, implementation and storage layout pass review. Natural start identities need not match and must remain explicit.

Keep the original eight-passage Wiki/Hume mixture, M8, as a separately labeled frozen anchor if the measured storage/forward budget permits. Comparing M8 directly with fresh Wiki4 confounds source mixture and sample count; it must not substitute for the matched Wiki4-by-Wiki4 center/basis factorial. The center/basis estimands remain finite interventions conditional on these two fitted maps, not population calibration variability or feedback mediation. The original M8 maps/directions stay unchanged. L10 and all L4–L11 remain preselected from prior development evidence.

The 16 evaluation passages have not been selected. Their original source/grid/exclusion policy should be preserved in a new explicit revision, and all inputs frozen before any new model output. No Hume replacement, shorter calibration buffer, relaxed exclusion, resampling after geometry or silent corpus change is warranted. With 20 primary modes the revision would require 1,292 forwards (four calibration + 640 primary + eight anchors + 640 suffix controls). Adding four M8 remove/orthogonal-by-layer-set anchors makes 24 primary modes and 1,548 forwards. The parent should freeze the actual choice before inference.

The other unresolved prerequisite is disk capacity. At this audit boundary the filesystem had 2,403,385,344 bytes free, less than the proposed 2 GiB reserve plus 256 MiB normal and 64 MiB failure allowances. A measured information-preserving layout refinement is required before launch; old raw outputs must not be removed to meet it. Input preparation can proceed with its separate small allowance after static review. This audit authorizes no model launch or publication.

Frozen evidence: script SHA256 `60fb677cc4f620543c93f6363989edb67e5c44f8d96829ea2276feafcdcf2a07`; passing JSON SHA256 `ba7efb46473e2f99d76b230abfebe3056274d0c7613f77a97a1af470fd8fb632`.
