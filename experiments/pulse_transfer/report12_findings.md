**The loss effects are spread across layers and change sign across corpora.**
No tested singleton reproduces the full pattern of joint effects. On fresh
passages, removing PC1 at all eight layers raises Wiki perplexity about 5%,
while lowering the mean on books and code. Several singleton effects oppose
one another, and the joint result is consistently less favorable than the
sum of singleton loss changes. These are causal effects of the specified
patches, not proof of one underlying mechanism.

## Findings supported by the reviewed outputs

**The Wiki penalty replicates; the book improvement needs qualification.**
Joint PC1 removal gives perplexity/P of 1.0514 naturally and 1.0533 with BOS on
Wiki. Their paired intervals are [1.0315, 1.0749] and [1.0343, 1.0776]. Du Bois
means are 0.9488 and 0.9476, but the new intervals [0.8999, 1.0057] and
[0.8986, 1.0035] cross one. The direction of its earlier result recurs, while
its interval evidence does not. Hobbes ratios 0.9668 and 0.9734 have intervals
below one. Code means 0.9859 and 0.9795 have intervals crossing one. All eight
joint-removal groups change substantial numbers of predictions: agreement
with P is only 87.60–91.11%; with native C, 86.57–90.39%.

**Individual layers have different roles in this intervention family.**
L9 and L10 PC1 removal reduce loss on both books at both starts; all eight
paired intervals against P are below zero. L10 removal raises Wiki loss at
both starts, and its removal-minus-orthogonal-control contrast is positive
on Wiki but negative on both books, with all six intervals excluding zero.
For L10, the natural-start changes in mean NLL versus P are +0.00593 on Wiki,
−0.02475 on Du Bois and −0.01505 on Hobbes; BOS changes are +0.00574,
−0.02449 and −0.01585. This is a useful direction-specific, corpus-dependent
candidate for new confirmation, not a universally beneficial removal.

L5 removal also lowers book loss, but its benefit is not uniquely associated
with PC1: the matched orthogonal control improves Du Bois more, with both
paired removal-minus-control intervals positive; the Hobbes differences are
uncertain. L7 removal worsens Du Bois while improving code, and L11 removal
has positive loss-change means in all eight groups. Reporting only a best
layer would hide these opposing effects.

**Prominent calibration geometry does not rank functional importance.**
At L4, 98.12% of the PC1 direction's squared loading lies on head 11; later
layers' largest single-head shares are only 15.86–23.95%. Yet L4 PC1 removal
has perplexity/P between 0.9951 and 1.0023 and 98.82–99.66% agreement with P.
This does not make that head dispensable: the patch removes one frozen
logit direction under the zero-value boundary, not the head or source state.

The norm control also has a material limitation. On natural Wiki at L4, both
arms change the logit vector by a mean norm of 4.3842, but their mean context
changes are 0.2748 for PC1 removal versus 1.6801 for the orthogonal control.
At L10 the corresponding context changes are 2.5574 and 2.3140. Sigmoid
sensitivity and head-specific value mixtures can therefore make equal logit
perturbations unequal delivered-context perturbations. Those explanations
remain hypotheses; the norm data alone do not isolate their contributions.

**Layer effects do not add in NLL.**
Every one of the 24 corpus/start/family joint-minus-sum means is positive,
and 22 paired intervals exclude zero. The two exceptions are Du Bois PC1
removal. For Wiki PC1 removal, the joint NLL changes are +0.05014/+0.05196,
whereas the residual above the summed singleton changes is
+0.02443/+0.02592. For Hobbes, this positive residual offsets part of the
singleton improvements. Position replacement has still larger residuals on
Wiki and code. This rejects a simple additive description of these finite
loss effects. It does **not** identify feedback or establish a mediation
percentage: nonlinear model composition and nonlinear loss/readout both
remain possible contributors.

**The simple position replacement still fails to preserve computation.**
Replacing all eight layers' gates by their calibrated mean plus log(position)
gives perplexity/P of 1.0948–1.3739 and agreement with P of 74.51–79.87%.
The new panel provides no support for using that map as a faithful larger
replacement. Even the unpatched zero-value reference P agrees with native C
only 95.05–96.41%, despite perplexity/C of 0.9952–1.0127.

## What the covariance result does and does not add

A separate CPU-only calculation concatenated the eight layers' 12 gate logits
at each calibration query, then centered the 96 coordinates. After subtracting
log(q), the first global principal component explains 32.70% of variance;
ranks 3 and 6 explain 54.80% and 69.20%. Independent covariance-eigenvalue
recomputation matched 104 comparisons within 9.24e−14. These 4,080 vectors
come from eight correlated calibration passages. Eight separate per-layer
PC1 directions are not one global scalar, but this covariance calculation
neither establishes nor excludes one nonlinear causal mechanism. It changed
none of the frozen experimental maps.

## Numerical scope

The independent audit passed all 938 main records, 930 suffix controls,
1,688 main checks, 14,400 gate checks, 248 full and 496 segment summaries,
810 paired contrasts, and all 90 nonadditivity records. It directly checked
720 singleton incoming logit/mass states and 480 directional coefficients.
The 52 fixed FP64 snapshot reconstructions stayed within the unchanged
1e−4 guard; maximum saved delivered-context discrepancy was 2.23e−5.

The smallest current mass was 9.2813e−12 and the largest gain was
1.0774e11, with no explicit clipping. The worst gain occurred at Du Bois
window 0/BOS, position replacement at L4, head 11, query 1, where the target
mass rounded to one. That input is outside the fixed raw snapshot subset.
Its runtime conditional/reconstruction check passed at 2.61e−5; the report
does not substitute another input's FP64 snapshot for that missing check.

The first independent audit stopped on an exact CPU/GPU FP64 norm equality
comparison, with discrepancy 5.33e−15. Its original source and failure log are
preserved. The audit-only comparison was changed to an explicit 2e−12 bound;
production thresholds and outputs were untouched, and no model was rerun.
