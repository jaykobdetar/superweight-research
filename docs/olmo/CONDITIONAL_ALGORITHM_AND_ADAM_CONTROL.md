# Conditional pulse amplification and an adaptive Adam control

This supplement records two completed OLMo-1B studies after the earlier
[October 23 synthesis](RESEARCH_UPDATE_2026_10_23.md). It describes a
checkpoint-specific intervention response and a late optimizer snapshot, not a
complete language-model algorithm or a causal account of training.

The inference studies use `allenai/OLMo-1B-0724-hf` revision
`d7cbab742d80589e714b1a2d7f838dcd21cbe143`. Their primary writer is
`layers[1].mlp.down_proj.weight[1764,1710]`; indices are zero-based. The
optimizer supplement instead uses the native checkpoint labeled step 954000,
whose saved step counter is 954001. Do not conflate that checkpoint with the
released HF endpoint. The separate [coordinate erratum](../COORDINATE_ERRATUM.md)
also remains important: an earlier secondary-coordinate experiment used L2,
whereas the published secondary coordinate was at L1.

## Carrier-to-H10 amplification

Four selected layer-0 carrier neurons (1091, 3419, 6535 and 4798) change the
query, self-key and self-value of layer-1 H10. In two controlled backgrounds,
holding H10 at its native response removed **39.65% and 48.14%** of the
carrier-induced pulse magnitude. Allowing only H10 to respond, with the other
heads held native, preserved that pulse within **2.19% and 3.59%**. An exact
same-world sham and unchanged-prefix checks passed in this scope.

The full pulse remains substantial with H10 held native. These interventions
therefore support an amplification branch *conditional on the four-carrier
edit*, not a claim that H10 is necessary for any pulse or that the entire
native computation has been reconstructed. The measured reduction in residual
norm is relative to the carrier-edited, attention-clamped comparison, not to
the original native residual. The causal [protocol](../../research/olmo/algorithm_update/h10-protocol.json)
and [saved audit](../../research/olmo/algorithm_update/h10-causal-audit.json)
record 16 two-layer prefix passes and their controls.

## Where the two-source reduction fails

A parameter-defined local predictor uses the observed recipient baseline and a
donor carrier change. It recomputes the query's self-key/value and mixes them
with an earlier source; other heads remain at their native baseline. It is an
intervention-response model, not a token-to-behavior algorithm. The
[prospective scope protocol](../../research/olmo/algorithm_update/scope-protocol.json)
used 24 additional two-layer prefix passes and froze predictions before the
remaining outcomes. The [freeze audit](../../research/olmo/algorithm_update/scope-freeze-audit.json)
and [outcome audit](../../research/olmo/algorithm_update/scope-outcome-audit.json)
record:

| Condition | Joint criterion |
| --- | ---: |
| Strong carrier, two backgrounds × two earlier-source positions | 4/4 pass; pulse relative errors 0.56–2.07%. |
| Near threshold, both gate sides | 0/8 pass, although all gate signs agree; the up-read criterion fails in all eight. |
| No-earlier-source self-only controls | 2/2 pass. |

In the near-threshold cases, omitted attention mass and other-head responses
matter; restoring all H10 sources alone does not close the transition error.
An earlier fitted surrogate also failed its frozen 5% pulse criterion in both
fresh cases (5.0315% and 9.5804%); it was not rescued by rounding or refitting.
The strong-regime reduction is useful within its measured conditions, but it
does not establish general switching behavior, a unique circuit, a semantic
purpose or downstream language-model equivalence.

## An adaptive control for the late Adam snapshot

The existing 1710 study sampled 128 other residual axes using seed
`202610230619`, frozen before acquisition. At axis 1764, its second moment was
below all 128 sampled axes within neuron 1710; the inverse-denominator gain was
2.075 times their median. A **separate, adaptively selected** control used the
same axes for neuron 8041. At the focal axis its second moment was above 127 of
128 comparisons and its gain was 0.888 times their median. Raw `v` and
epsilon-adjusted gain are distinct: epsilon dominated 127/128 sampled control
axes and none of the sampled primary axes. The paired cross-column `v`
Spearman correlation was about -0.096.

The [control protocol](../../research/olmo/optimizer_control_8041/protocol.json),
[comparison](../../research/olmo/optimizer_control_8041/comparison.json),
[results](../../research/olmo/optimizer_control_8041/results.json) and
[one-page report](../../research/olmo/optimizer_control_8041/adaptive-8041-summary.pdf)
preserve the comparison and its scope. The control was selected after the
primary result, so it is descriptive rather than independent population
validation. It is one late checkpoint, not a measured full-column ranking,
current gradient, early optimizer history or causal training experiment. The
row-wide and within-neuron comparisons in the earlier synthesis have different
denominators and are not contradictory.

## Reproduction and boundaries

The included scalar reports and frozen protocols can be checked without a
model or downloads:

```bash
python research/olmo/algorithm_update/check_summary.py
```

This checks packaged hashes and selected reductions in saved JSON. It does not
rerun inference, reconstruct original model hooks or independently validate
the omitted hidden states, input token arrays or optimizer tensors. The
[algorithm source manifest](../../research/olmo/algorithm_update/SOURCE-MANIFEST.json)
distinguishes packaged paths and bytes from original execution hashes. The
[control source manifest](../../research/olmo/optimizer_control_8041/SOURCE-MANIFEST.json)
identifies its selected protocol and aggregate files. Original corpora,
checkpoint weights, optimizer slices and full activation archives are not
included. A full model reproduction needs the separately pinned assets and
the original, independently authorized computational environment.

The closest next research questions concern the missing transition, an
independent prediction of the carrier from source content and competition,
and downstream use of the whole output column. Earlier optimizer states and
matched training interventions would be needed to distinguish formation from
late maintenance. Those are proposals, not completed tests or authorization
to run them.

Two candidate inference designs were discussed separately: an eight-forward
finite K/V interaction test on two suffixes, and a 16-forward endogenous-routing
test conditioned on removing source-0 access for later queries in layers 2–15.
They were proposals, not outcomes, and the two suffixes in the first design
share a source state. The earlier [Q1 compact audit](../../reviews/2026-10-23-integration/q1_continuation_compact.json)
checks already saved continuation records; it is not evidence that either
proposed test ran.
