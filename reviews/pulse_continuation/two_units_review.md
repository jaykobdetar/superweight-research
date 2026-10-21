# Independent two-unit static review

Reviewed `TWO_UNITS.md`, `two_units.py`, `analyze_two_units.py`, and completed outputs without model/GPU loading. No blocking issue found. Evidence is `two_units_audit.py/.json`.

The two independent output hooks target L1 column1710 and L2 column8041, each with row1764 or whole-column replacement from its copied zero-column matrix. Source scope changes only position0; all scope changes every position. Established Run.Source remains clean, so no hidden extra deletion is applied. Clean predictions are legitimately reused across scopes with updated scope metadata. Model weights stay untouched.

The L2 activation is computed after the first-unit intervention, so it is recipient-current rather than a frozen donor. The trace distinguishes activation × original weight (potential unremoved contribution) from the applied source0 target contribution, which is logically zero for row/column deletion. These are not the same as the FP16 difference between entire down-projection outputs. Before/after outputs retain a way to check row localization.

All nine factor combinations are evaluated in both scopes on exactly matched inputs and 510 targets. Source-only first-unit controls assert exact agreement with the established column-factorial panel. Conditional second-unit effects and difference-in-differences use the correct signed loss contrasts. All-minus-source comparisons are intervention-scope effects; they do not isolate position from the other computations removed.

Expected totals: 816 forwards, 864 recorded rows, 216 groups, 144 source-only implementation checks. Increased second-unit dependence after first-unit column removal would support conditional reliance in that counterfactual, not prove a unique backup mechanism or sole explanation of the milder whole-column result. Reused four-window panels remain exploratory.

## Completed numerical review

All expected counts match. All 864 raw records, 216 summaries, 216 generic contrasts, 192 conditional/interaction contrasts, and 108 scope contrasts independently replay. All 144 source-only first-unit anchors match the earlier column panel exactly, including retained KL. The L1 scalar is +0.63232421875 and L2 scalar −0.59228515625.

Potential/applied labels are correct: the potential is the saved recipient-current activation times original scalar; the applied target contribution is zero in row/column-deleted conditions. Source0 traces are bitwise equal across source/all scopes for each factorial condition, while later-token outcomes can differ. First-unit inputs are unchanged by every intervention; second-unit inputs are fixed conditional on the first-unit state and unchanged by their own subsequent output hook. Row-deleted outputs retain every complementary coordinate. The largest discrepancy between the measured FP16 target-output difference and the negative potential product is 0.106323 output units, reflecting native rounded linear outputs; the saved potential is not incorrectly presented as an exact finite output subtraction.

| Natural source0 | First column removed | Also second row | Also second column |
|---|---:|---:|---:|
| Wiki validation | 1.44506 | 1.70011 | 1.82145 |
| Du Bois | 1.73149 | 1.84070 | 2.47024 |
| Hobbes | 1.03320 | 1.03591 | 1.03766 |
| Optimizers | 1.01114 | 1.04687 | 1.02681 |

Entries are ratios to matched native clean. The conditional cost of removing the second column after the first is removed is Wiki 0.23148 nats [0.02017, 0.58618], Du Bois 0.35534 [0.000097, 1.06535], Hobbes 0.00431 [0.00060, 0.01114], and optimizers 0.01538 [0.00247, 0.02829]. These support some remaining conditional contribution of the second unit. However, the difference-in-differences relative to second-column removal in the intact-first-unit state includes zero in Du Bois and Hobbes; do not call increased reliance universal. Four-window intervals are especially wide for Du Bois.

Removing both columns is much milder than first-row removal but is not faithful recovery in Wiki/Du Bois. The result rejects the second unit as the sole simple explanation of the milder first-column counterfactual; it does not establish the absence of all backup pathways or a unique alternative circuit.

The second unit's mean source0 potential contribution is approximately −40.7 to −42.0 natively, becomes −201.5 to −211.1 after first-row deletion, but is only −0.87 to −9.01 after first-column deletion in the four natural-ID groups. Thus a larger second pulse accompanies the catastrophic row-only state, not restoration by itself. Pulse amplitude cannot be treated as an independent measure of restored function.

All-position scope is materially different. With both columns removed, natural-ID PPL ratios are 1.82316/3.07981/1.59961/1.23785 for Wiki/Du Bois/Hobbes/optimizers; the latter three exceed source-only results. For some row-removal combinations, all-position removal instead improves on source-only (e.g. Hobbes both-row all-minus-source loss −0.94066 nats). These are nonmonotonic scope effects, not evidence that later positions are uniformly irrelevant or protective. Peak CPU review RSS 541,880 KiB (about 529 MiB).
