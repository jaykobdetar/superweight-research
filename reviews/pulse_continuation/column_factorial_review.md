# Independent column-factorial static review

Reviewed `COLUMN_FACTORIAL.md`, `column_factorial.py`, and completed outputs without loading a model or GPU. No blocking issue found. Numerical replay is `column_factorial_audit.py/.json`.

Source deletion registers before the new Column hook. Every evaluation resets Column.mode; the reference check sets it clean while invoking established Source deletion, avoiding double application. The copied matrix zeros column1710, and output assembly selects only the target token and named coordinate subset. Row1764 and its 2,047-coordinate complement partition the full column contribution. No model parameter mutation occurs. The exact comparison against established row-only deletion tests potential numerical differences between the two F.linear matrix layouts.

Each boundary world has its own native logits/baseline, so KL and loss differences are matched within ordinary, isolated, or induced conditions. Native source0 panels score 510 targets; matched later-boundary panels score the same 254 suffix targets across prefix lengths. These are different target/history panels and must not be treated as a paired position-only contrast. At boundary, source0 remains intact. Captured pre-projection intermediate equality is asserted for every deletion and independent row-control run.

L2 normalized inputs/outputs and clipped keys/values are useful descriptive diagnostics. The stored keys are **pre-RoPE**, especially relevant when comparing boundary and source positions. At source0 the trace indexes [0,0] duplicate the same position; do not interpret its two rows as two distinct tokens.

Expected count is 1,200 forwards, 960 recorded rows, 240 condition groups with four windows each, and 240 implementation-equality checks. The factorial addresses the difference between whole-neuron-column removal and special-row removal; whole-column removal is not removal of the whole MLP. Loss interactions can be tested, but a unique compensation mechanism or a semantic interpretation would require additional evidence.

## Completed numerical review

All expected counts match. Independent replay passes 960 raw records, 240 summaries, 240 generic contrasts and 180 component-interaction contrasts. The 240 runtime row-implementation checks pass; source0 natural/The clean and row-deleted retained outputs exactly match the prior confirmation panel's first four windows. Boundary clean/row-deleted retained outputs match the corresponding upstream/MLP worlds exactly. Boundary KL is correctly measured against each world’s own baseline, so its numerical ratios can differ slightly from preceding ordinary-reference tables.

Pre-down-projection intermediates remain unchanged under all component deletions. At L2 input, row deletion changes only coordinate1764, complement deletion leaves that coordinate unchanged, and column deletion exactly combines their disjoint coordinate changes. This independently verifies the intended local factorial from saved traces. Boundary source0 traces remain unchanged. At source0, row deletion and whole-column deletion give exactly the same raw L2 target coordinate but different normalized target coordinates because the complementary inputs affect normalization.

| Natural source0 corpus | Row deletion PPL | Complement deletion PPL | Column deletion PPL | Column agreement |
|---|---:|---:|---:|---:|
| Wiki validation | 1933.9217 | 1.19041 | 1.44506 | 76.47% |
| Du Bois | 537.4264 | 1.03487 | 1.73149 | 78.19% |
| Hobbes | 647.9975 | 1.02146 | 1.03320 | 92.40% |
| Optimizers | 596.2062 | 1.01308 | 1.01114 | 96.67% |

These ratios use native source0 clean references and 510 targets/window. Full-column removal is much milder than row deletion but is not faithful recovery on every corpus. All twelve source0 corpus/identity groups have a negative joint-minus-individual removal contrast with a 95% paired interval below zero. Natural Wiki is −7.37345 nats [−8.03693, −6.63354]. This is a finite loss interaction; it does not imply that both deleted pieces were individually unnecessary.

At induced boundaries, whole-column ratios span 0.99798–1.00432 against each induced-world clean baseline, while row-only ratios span 78.465–422.960. At isolated boundaries they span 0.99735–1.00506 versus 74.691–402.483. Whole-column argmax agreement still ranges approximately 96.1–99.1%, so small mean loss changes are not exact model equivalence. Ordinary-boundary deletions are negligible, consistent with their very small neuron1710 activation. All 32 isolated/induced boundary groups have negative interaction intervals, whereas ordinary-boundary effects are near the numerical scale.

The saved down-column target coordinate accounts for 97.9053% of its squared norm (column norm0.63905). Nevertheless, its complementary 2.0947% matters strongly when the target component is removed. For natural Wiki, mean L2 normalized target is −10.39 after row-only removal and −37.09 after full-column removal, despite identical unnormalized target coordinates. This directly demonstrates a normalization consequence of the complement intervention; it does not yet identify which subsequent unit/circuit carries any remaining function.

The result materially limits interpreting catastrophic scalar weight deletion as evidence that the entire neuron or its large pulse is uniquely indispensable. Row-only deletion creates a different internal state from neuron-column deletion. The proposed second-unit factorial is a separate test of conditional reliance, not an established backup explanation. Panels remain reused, four windows each, and no checkpoint-general conclusion follows. Peak CPU review RSS 627,228 KiB (about 613 MiB).
