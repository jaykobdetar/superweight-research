# Cross-model comparison: shared questions, different signatures

[Overview](../../../README.md) · [Documentation index](../../README.md) ·
[OLMo](../olmo/README.md) · [GPT-2](../gpt2/README.md) · [Llama](../llama/README.md)

The main comparison is [Report 8](../../TRANSFER.md), with a separate
[transfer reproduction guide](../../TRANSFER_REPRODUCING.md). OLMo provides the
original case study; the GPT-2 and Llama candidates were selected on development
contribution magnitude and tested with model-specific interventions. Similar
activation magnitudes do not establish the same mechanism.

| Question | OLMo-1B | GPT-2 | Llama-3.2-3B-Instruct |
| --- | --- | --- | --- |
| Scalar output versus whole selected neuron output | Scalar deletion can be much more damaging on the main panel, with whole-output counterexamples. | Both selected-candidate effects are small on the initial panel. | Whole-output deletion is consistently more damaging on the initial panel. |
| Earlier-newline signature at the chosen candidate | H14 suppression path supported under the stated eligibility and donor conditions. | Predeclared candidate screen fails. | Predeclared candidate screen fails. |
| Subsequent strongest scoped evidence | Conditional MLP relay and source K/V/geometry dependence. | Conditional coordinate regrowth and transmission of chosen gate/write perturbations. | Distinct source K/V roles and approximate fixed replacements. |

Read the model pages for exact evidence, controls, failed hypotheses and natural
counterexamples. Failed candidate transfer does not prove absence of all related
circuits in a model. GPT-2 follow-ups do not demonstrate a native mechanism tied
to the original OLMo super weight.

## What the comparison does not isolate

The models differ in architecture, size, training, instruction tuning and
precision. Each tokenizer encodes the fixed character passages independently;
cross-model inputs are not token-matched. Source 0 is the first position in a
freshly reset teacher-forced sequence, so first-token identities and calibration
overlap require explicit accounting. Natural/BOS variants share text.

Small fixed panels and selection-conditioned, unadjusted intervals do not
establish population-wide effects. An intervention's loss improvement does not
prove native computation is dispensable, a faithful output replacement or a
deployment benefit. Architectural closure tests validate implementation
boundaries; expected closures are not independent mechanism discoveries.

## Evidence and reproduction routes

- OLMo: [research report](../../REPORT.md),
  [coordinate erratum](../../COORDINATE_ERRATUM.md),
  [reproduction](../../REPRODUCING.md),
  [prior-literature comparison](../../NOVELTY.md).
- GPT-2/Llama comparison: [transfer report](../../TRANSFER.md),
  [frozen protocol](../../../experiments/pulse_transfer/PROTOCOL.md),
  [reproduction and replay limits](../../TRANSFER_REPRODUCING.md),
  [transfer numerical auditor](../../../scripts/audit_transfer.py).
- Later GPT-2 evidence and uncertainty: [model report index](../gpt2/README.md).
  Public numerical consistency, original private reviews and pretrained replay
  are separate checks with different coverage.

The [research log](../../../RESEARCH_LOG.md) preserves the earlier chronological
account. No artifacts are relocated by these indexes, and no new experiments
are part of the documentation reorganization.
