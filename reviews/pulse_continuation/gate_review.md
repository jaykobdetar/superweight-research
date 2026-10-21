# Gate decomposition and replacement audit

CPU-only review of `gate.py` and completed `gate_run`. No experiment edits,
model loading, or GPU use. Maximum audit RSS approximately 528MiB.

All 1,344 observations, 168 condition groups, and 1,360 forwards are present.
Raw NLL/KL means, baseline means, argmax agreement, hashes, and model/parameter
integrity passed. Residualized identity controls are bitwise equal in saved loss,
prediction, and KL for both heads, both states, and all three corpora.

## Implementation and isolation

The decomposition uses native clipping and post-RoPE FP16 rounding. Nonzero-key
mixtures use their own conditional softmax, avoiding division by a saturated
`1−a0`. Query0 is treated correctly with mass 1 and no nonzero-key mixture.
Parenthesized `(proposed−reconstructed)` makes identity corrections exactly zero.
All saved conditional mixtures are bitwise unchanged between clean and deleted
states, consistent with identical later Q/K/V at the first broadcast layer.

WikiText/Hume calibration uses eight reused development windows per corpus.
Every evaluation span is disjoint from declared previous-tranche and factorial
spans and from these calibration spans. Evaluation hashes match saved tokens.
Plato has no calibration inputs; both calibration sources transfer to it. Gate
constants have valid ranges and positionwise shapes. Wiki K0/V0 means for H0/H15
exactly match independently saved prior-tranche development constants after FP16
cast. Raw gate calibration activations and Hume calibration vectors were not saved
individually, so their arithmetic means cannot be independently replayed without
inference; code inspection confirms clean calibration only, before evaluation.

## Approximation limits

- Direct reconstruction changes mean NLL by at most 0.0001021 nats across the
  head/state/corpus controls.
- Oracle donor-gate/value/both operations differ from corresponding native
  K0/V0/K0+V0 patches by at most 0.0003076 nats in mean NLL. This supports their
  use for aggregate causal comparisons. It is not exact equivalence: maximum
  individual token loss discrepancy is 0.4609 nats, and up to 15 of 4,088 argmax
  predictions differ in a condition pair.
- Very small residualized replacement effects, especially constant-value mean
  NLL changes around 1e−4 or less, should not be interpreted as meaningful
  improvements over native execution merely because a point estimate is negative.
- `zero_head` computes native context plus negative reconstructed context. It
  leaves the small native-minus-reconstructed residual, rather than literally
  setting the head to zero. Call it an approximately zero-head intervention.
  Native constant-K/V trials bypass this approximation and provide an important
  independent replacement reference.

## Supported interpretation

Native constant-K/V replacement for H15 preserves 99.80–99.85% argmax agreement
across WikiText, Hume, and Plato, with mean clean-to-condition KL approximately
6e−6 to 1.5e−5. Layer-matched H0 replacements preserve similarly well. This supports
simple first-position constants in these tested conditions, not uniqueness of H15,
exact distributional equality, or measured compute savings.

The scalar gate depends materially on text distribution: average clean H15 key0
mass is approximately 0.502 on WikiText versus 0.247 on Hume and 0.245 on Plato.
Wiki-fitted scalar gate replacements transfer less well to the philosophical
corpora than Hume-fitted gates. Position-dependent gate fitting improves some
comparisons but does not remove the domain dependence. Deletion drives key0 mass
near 0.99 in all three, while value0 norm grows from about 0.92 to 4.15–4.21.

Large deleted-donor damage should not be equated with necessity of normal H15
output. Approximately zeroing that head in clean runs changes mean NLL by only
0.00548 (WikiText), 0.02441 (Hume), and 0.02204 (Plato), much less than poisoning it
with deleted-donor K/V. Removing the source gate/value and zeroing the entire head
are also distinct interventions; the former preserves the nonzero-key mixture.

No blocking implementation defect remains. The concrete labeling requirement is
to distinguish approximate zero-head and oracle decompositions from exact native
K/V operations. Paired intervals remain pointwise and conditional on these small
samples, chosen heads, and fixed calibration.
