# Independent later-use static review

No execution blocker identified in LATER_USES.md, later_uses.py or analyze_later_uses.py. CPU-only extracted hook tests pass all clean/row/column × source/later/all combinations; the hook preserves input/output tensors, uses the copied zero-column linear projection, and replaces only intended token rows or coordinate1764. Layer1 gate/up and target activation captures precede the output intervention, while layer2 activation remains a downstream outcome. The runtime exact checks properly test the unchanged first-neuron inputs. Existing Run.Source is clean, so it does not double-delete.

All sixteen proposed +768 windows are disjoint from every original confirmation calibration/evaluation span and from one another. They are new windows in the same texts, not new corpora, and this review does not assert exclusion from checkpoint training. Preparation repeats the original tokenizer/text joins and verifies source hashes/token counts. Duplicate-window checks occur at preparation. Natural/The/period identity edits affect source0 only and leave suffix target IDs unchanged. All metrics exclude predictor0 and score targets2..511 (510 targets).

The seven conditions correctly implement each component's source×later factorial using a common clean reference. Analyzer contrast all−source−later+clean has the stated interaction sign; column−row negative means lower NLL for column removal. Paired inference remains only four windows/group with reused identities. Prespecified thresholds1/10/100 summarize amplitudes without selecting interventions by loss; top-eight amplitude snippets are descriptive. Potential pulse is computed before output removal; applied first target contribution is zero in the deleted scope, subject to floating-point projection rounding if inferred from output differences.

Evidence: later_uses_static_test.py/.json. No model/GPU loaded; peak RSS521428KiB. Numerical audit awaits completed output.

## Completed numerical audit

All336 raw records,84 summary groups,84 baseline contrasts and84 scope contrasts replay exactly from saved per-token metrics. All288 unchanged first-neuron checks and48 amplitude observations replay; source hashes, sixteen unique windows and frozen disjoint schedules match. CPU half-SiLU factor reconstruction is exact here, as a diagnostic only. Potential/applied pulse labels are correct. Source0's second-neuron activation is unchanged by later-only interventions, and source/all interventions agree at source0. All comparisons use510 matching targets. Peak review RSS was approximately502MiB; no model/GPU was loaded.

Natural-ID PPL ratios (source row / source column / later row / later column / all column):

- Wiki:1418.47 /1.60013 /0.999861 /0.999913 /1.59965.
- Du Bois:592.68 /1.83430 /16.4206 /1.01126 /2.19376.
- Hobbes:744.02 /1.02156 /60.0221 /1.01063 /1.28745.
- Optimizer code:588.19 /1.69439 /20.4496 /1.00812 /2.13256.

Later-row NLL increases are2.79854[0.92022,3.81776],4.09471[3.81757,4.43460],3.01796[0.95950,4.26919] nats in Du Bois/Hobbes/code. Later-column increases are0.011195[0.003735,0.017239],0.010578[0.004379,0.016247],0.008085[0.002444,0.014211]. These whole-column effects are modest but not zero; argmax agreement is97.70%,95.64%,97.55% respectively. Wiki later-only intervals straddle zero and predictions are near, not exactly, identical.

The first-unit row-versus-column distinction therefore reproduces on fresh source0 windows and naturally occurring later uses. A Wiki-only later-use null must not be generalized across these texts. The scope interactions are nonadditive: natural row source×later NLL contrasts are−3.37947,−4.81968,−3.37157nats with intervals strictly negative in Du Bois/Hobbes/code; corresponding column interactions are+0.16776,+0.22076,+0.22192nats with strictly positive intervals. Whole-column later removal under an intact source0 has a smaller effect than its conditional effect after source0 column removal. These are factorial interaction effects, not additive attribution percentages or proof of a unique compensation path.

Descriptive amplitudes: natural Wiki later absolute pulse maximum is0.05314, with no values>1. Natural Du Bois/Hobbes/code contain3/4/3 later values>100 across their four windows (max325.01/310.63/321.85). All ten natural-ID large later values occur at token187/newline, at positions1..30. Across all48 identity-window cases there are35 values>1:34 at token187 and one at token64 at position1 in optimizer-code period/window1; that case also has a newline pulse at5. Do not claim universal newline specificity. Token correlations and snippets motivate targeted tests, but these current all-later interventions do not isolate individual large-amplitude tokens, distinguish token identity from history, or establish a natural reset mechanism.

Evidence: later_uses_audit.py/.json. Four windows/group and reused identity variants remain the uncertainty unit; pointwise bootstrap intervals do not supply population-wide or multiple-comparison guarantees.
