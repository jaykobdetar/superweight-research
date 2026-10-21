# Independent CPU review: static-template residual paths

Completed 2026-10-21. No model was loaded and no GPU was used. The audit peaked at 543,712 KiB RSS. Experiment artifacts were not modified.

## Validation

`residual_paths_audit.py` independently reconstructs all 3,000 window rows, 375 summary groups and 523 contrasts from saved token metrics, including paired bootstrap intervals. The manifest records 3,008 forwards, matching 3,000 scored interventions plus eight Python calibration forwards. Runner/template hashes and unchanged model hashes/parameter versions pass.

All 224 development ranking entries and their eight per-window improvements replay exactly. Ranking is frozen before evaluation. Every seeded random set reproduces exactly, has the same per-layer counts as its selected set, and has zero overlap with that selected set. The selected top eight are L3H7, L2H2, L12H6, L2H9, L4H9, L2H6, L7H1 and L3H11.

Evaluation spans are mutually disjoint within each corpus and outside declared previous-use exclusions. Wiki/Hume saved evaluation tokens match their original corpus slices. Wiki development tokens match the original development slices. Python calibration and evaluation spans are separate, source-file hashes match, and the Python mean is computed before evaluation without entering head ranking. Plato's source hash matches. Source documents and adjacent windows are not independent document samples; these remain small fixed-corpus tests.

The edge hook changes only the chosen position-zero K/V head blocks at the linear projection output, before native clipping/RoPE. Synthetic AST-extracted hook tests verify localization and untouched Q/other layers. Full 224-channel cuts reproduce the donor template world's saved loss, prediction and clean-to-condition KL **exactly in both directions**, for all 32 evaluation windows. Every condition scores 510 targets, excluding predictor zero uniformly.

## Findings and interpretation

This screen isolates channels carrying the difference between two imposed source templates: a coordinate-matched analytic axis and the Wiki development full mean. It does not rank the original scalar-deletion circuit, identify an optimal joint head set, or estimate downstream mechanism rank. Other source normalized states remain clamped to the recipient template throughout each intervention.

| Held-out corpus | Axis / clean perplexity | Rank-two residual template / clean | Full Wiki mean agreement with clean |
|---|---:|---:|---:|
| Wiki | 1.412462 | 1.002895 | 99.2647% |
| Hume | 1.011680 | 1.000068 | 99.7549% |
| Plato | 1.009617 | 1.000224 | 99.8529% |
| Python | 1.023740 | 1.000180 | 99.8284% |

Ratios are exponentials of the mean paired NLL difference. Near-baseline loss is not exact output preservation. The Python full mean also transfers well to these four panels; the tiny improvements do not establish a generally superior template. Its calibration and evaluation come from the same two local Transformers source files, so this is a narrow code transfer test.

On Wiki, selected top-eight repair improves the axis world by 0.088460 nats (95% paired bootstrap interval 0.059484–0.120169), closing about 25.6% of the 0.345228-nat axis-to-full-mean gap. Top 32 closes less, 0.054796 nats. Repair is therefore partial and not monotonic in set size. Top-eight repair against the axis baseline has intervals crossing zero on Hume, Plato and Python, despite sometimes outperforming their fixed random controls.

Reverse interventions are much stronger: Wiki top-eight damage increases NLL by 0.978217 nats (0.875638–1.077341), exceeding the complete axis-versus-full-mean gap. Top 32 adds 1.636695 nats. This asymmetric, nonadditive behavior is consistent with incompatibility between mixed channel states; it must not be presented as an additive percentage of a mechanism or as sufficient localization of the whole residual effect. All-224 replacement returns exactly to the coherent donor world.

For Wiki top eight, K-only damage adds 0.899655 nats, V-only adds 0.009897, and joint K/V adds 0.978217. K-only repair improves by 0.085009 nats versus joint 0.088460 and V-only 0.020233. This supports K-dominant effects for this selected set and these template worlds, with interactions; it does not show that values are generally irrelevant.

All selected Wiki sets outperform their corresponding random repair controls. The selected damage sets also have larger effects than those controls across all four corpora. These are comparisons against **one frozen random set at each size**, with uncertainty over eight windows, not uncertainty over random head selections. The many exploratory comparisons have no multiplicity correction. Do not turn the best observed held-out set size into a newly validated optimum.

No implementation blocker or numerical discrepancy was found. The key remaining limitation is scientific scope: this is useful evidence about static-template channel interactions, while sparse sufficiency, broad corpus generalization and a complete explanation of the original deletion circuit remain unestablished.
