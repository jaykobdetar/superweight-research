# Documentation and evidence index

[Research overview](../README.md) · **[Complete research log](../RESEARCH_LOG.md)**

Research is paused through Report 19. These indexes organize completed work;
they do not approve or schedule the hypotheses and proposed experiments retained
in the archive. Existing reports, raw exports, scripts and handoffs keep their
established paths.

## Model-specific reading routes

| Model or scope | Start here | Main evidence |
| --- | --- | --- |
| OLMo-1B | [OLMo findings and evidence](models/olmo/README.md) | [Research report](REPORT.md), [coordinate correction](COORDINATE_ERRATUM.md), [claim boundaries and prior work](NOVELTY.md). |
| GPT-2 | [GPT-2 findings and report index](models/gpt2/README.md) | Transfer/follow-up through gate studies and Reports 16–19; chosen perturbations, without a demonstrated native link to the original super weight. |
| Llama-3.2-3B-Instruct | [Llama findings and evidence](models/llama/README.md) | Source K/V, scalar/column lesions and fixed-replacement tests within [Report 8](TRANSFER.md). |
| Cross-model | [Comparison and limits](models/cross-model/README.md) | Shared questions, different causal signatures, failed transfers and protocol differences. |

## CPU-only numerical audits

Use Python with NumPy 1.26.4 and run from the repository root. The README's
[quick start](../README.md#quick-start) sets up an audit environment. Each script
is a public numerical check, distinct from original private-tensor reviews and
from a full pretrained-model replay. Consult its guide for exact omissions,
dependencies and output options. These commands are documented entry points;
the research reorganization does not rerun scientific acquisition.

| Scope | Auditor | Guide |
| --- | --- | --- |
| OLMo exported historical arrays | [audit_results.py](../scripts/audit_results.py) | [OLMo reproduction](REPRODUCING.md) |
| GPT-2/Llama transfer and GPT-2 follow-ups | [audit_transfer.py](../scripts/audit_transfer.py) | [Transfer reproduction](TRANSFER_REPRODUCING.md) |
| GPT-2 gate subspaces | [audit_gate_results.py](../scripts/audit_gate_results.py) | [Gate reproduction](GPT2_GATE_REPRODUCING.md) |
| GPT-2 layer localization | [audit_localization_results.py](../scripts/audit_localization_results.py) | [Localization reproduction](GPT2_GATE_LOCALIZATION_REPRODUCING.md) |
| GPT-2 additive output/loss | [audit_additive_results.py](../scripts/audit_additive_results.py) | [Additive quick start](GPT2_GATE_ADDITIVE_QUICKSTART.md) |
| GPT-2 fresh Wiki calibration | [audit_wiki_calibration_results.py](../scripts/audit_wiki_calibration_results.py) | [Calibration quick start](GPT2_GATE_WIKI_CALIBRATION_QUICKSTART.md) |
| GPT-2 native source values | [audit_native_value_results.py](../scripts/audit_native_value_results.py) | [Native-value reproduction](GPT2_NATIVE_VALUE_REPRODUCING.md) |
| GPT-2 actual-write norm controls, Report 16 | [audit_post_write_norm_results.py](../scripts/audit_post_write_norm_results.py) | [Report 16 reproduction](GPT2_POST_WRITE_NORM_REPRODUCING.md) |
| GPT-2 write timing, Report 17 | [audit_query_write_timing_results.py](../scripts/audit_query_write_timing_results.py) | [Report 17 reproduction](GPT2_QUERY_WRITE_TIMING_REPRODUCING.md) |
| GPT-2 L11 K/V donors, Report 18 | [audit_l11_key_value_donor_results.py](../scripts/audit_l11_key_value_donor_results.py) | [Report 18 reproduction](GPT2_L11_KEY_VALUE_DONOR_REPRODUCING.md) |
| GPT-2 calibration worlds, Report 19 | [audit_calibration_world_comparison_results.py](../scripts/audit_calibration_world_comparison_results.py) | [Report 19 reproduction](REPRODUCE_GPT2_CALIBRATION_WORLD_COMPARISON.md) |

The separate [additive readout auditor](../scripts/audit_additive_readout.py)
has its own scope in the [additive guide](GPT2_GATE_ADDITIVE_REPRODUCING.md).
Numerical consistency is not independent model replication or evidence of a
universal mechanism. Later reports explicitly record unperformed second replays.

## Provenance, hypotheses and historical records

- [Checkpoint identity](../manifests/checkpoint.json),
  [transfer model identities](../manifests/transfer_models.json),
  [external input sources](../manifests/input_sources.json),
  [core same-machine validation](../manifests/core_validation.json).
- [Frozen protocols and experiment code](../experiments/),
  [exported results](../results/), [transfer outputs](../results/pulse_transfer/),
  [independent reviews](../reviews/), [all manifests](../manifests/).
- [OLMo hypotheses](HYPOTHESES.md),
  [GPT-2 native-value hypotheses](HYPOTHESES_NATIVE_VALUE.md),
  [Wiki-calibration hypotheses](HYPOTHESES_WIKI_CALIBRATION.md),
  [write-norm hypotheses](GPT2_POST_WRITE_NORM_HYPOTHESES.md),
  [write-timing hypotheses](GPT2_QUERY_WRITE_TIMING_HYPOTHESES.md).
  These are proposals or historical predictions; their presence does not imply
  execution or support.
- [Saved research handoff](HANDOFF.md) and
  [chronological README snapshot](../RESEARCH_LOG.md). Historical publication
  status in frozen reports belongs to its recorded date; current reproduction
  readiness is described in the linked guides.
