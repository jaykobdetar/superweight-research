# Llama-3.2-3B-Instruct: a distinct source-use pattern

[Overview](../../../README.md) · [Documentation index](../../README.md) ·
[Cross-model comparison](../cross-model/README.md)

Llama evidence is part of [Report 8](../../TRANSFER.md), using
`meta-llama/Llama-3.2-3B-Instruct` revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`, BF16/native SDPA. The
development-selected candidate is layer 1, output 588, intermediate 1419.
It was selected for a large typical contribution, not proved to be the model's
most causally important weight. All indices are zero-based.

## Supported findings

| Finding | Evidence and limitation |
| --- | --- |
| Whole-neuron-output deletion is more damaging than deleting its largest scalar output contribution in all 96 initial window/length/start comparisons. | [Initial Llama results](../../TRANSFER.md#llama). This differs from the earlier OLMo panel; squared weight energy alone is not a causal explanation. |
| Source keys, values and source access have distinct conditional roles. Restoring clean values helps more than restoring clean keys in the tested mixed states. | [Source K/V factorial](../../TRANSFER.md#2-llama-needs-source-values-as-well-as-source-access). Neither mixed state restores native outputs; effects are not additive mediation percentages. |
| Freshly calibrated fixed source K/V or signed activation means approach native loss on another panel while still changing predictions. | [Replacement tests](../../TRANSFER.md#4-fresh-calibration-of-simpler-llama-replacements). Attention routing remains endogenous; free-generation fidelity and speedup were not established. |

Both Llama and GPT-2 failed the predeclared newline-history screen at their
frozen candidates. That does not rule out other newline circuits. The Llama
replacement controls include a fixed matched-norm signed permutation that fails;
one fixed randomized control is not an ensemble of all matched directions.

## Reproduction and evidence

Use the [transfer reproduction guide](../../TRANSFER_REPRODUCING.md). Its ordered
workflow includes both models; it is separate from the OLMo core runner. The
Llama stages are `llama_run`, `llama_source_kv_run` and `llama_replacements_run`,
with dependencies listed in the guide. The public CPU-only
[transfer auditor](../../../scripts/audit_transfer.py) checks supplied numerical
exports without loading either model.

- [Frozen replacement design](../../../experiments/pulse_transfer/LLAMA_REPLACEMENTS.md)
  and [transfer protocol](../../../experiments/pulse_transfer/PROTOCOL.md).
- [Exported transfer evidence](../../../results/pulse_transfer/),
  [independent reviews](../../../reviews/pulse_transfer/),
  [model/tokenizer hashes](../../../manifests/transfer_models.json),
  [input sources](../../../manifests/input_sources.json).

Models tokenize the source passages independently. Size, training, instruction
tuning and arithmetic differ alongside architecture, so the comparisons do not
isolate which difference caused the changed signature. Four passages per
source/start and fixed calibration limit inference. Original-machine replay is
distinct from independent researcher replication; public arrays omit private
inputs, activations and full vocabulary logits.
