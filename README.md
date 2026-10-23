# Super-weight research

**What computation depends on a super-weight pulse, and how much of that
computation can a simpler intervention preserve?**

This repository studies that question through a causal case study of
`allenai/OLMo-1B-0724-hf`, followed by separate GPT-2 and Llama investigations.
Its strongest evidence concerns an OLMo newline-history pathway and the large
difference between removing one scalar output contribution and removing a
neuron's whole output. The later models reveal different conditional effects;
they do not establish a shared super-weight mechanism.

**New OLMo research separates signal production, native use, and training development.**
Read the [October 23 synthesis](docs/olmo/RESEARCH_UPDATE_2026_10_23.md) for the
checkpoint trajectory, alternate writers, native-source analysis and actual Adam
state, with explicit limits and [publication cutoff](docs/olmo/PUBLICATION_SCOPE.md).
GPT-2 and Llama remain separate investigations. The full previous
README, including the chronological updates and publication history, is retained
in the **[research log](RESEARCH_LOG.md)**. Start with the
[documentation index](docs/README.md) for the complete reading and audit routes.

## Strongest supported findings

| Finding | Evidence and scope |
| --- | --- |
| **OLMo: an earlier newline suppresses a later pulse through L0/H14.** Removing the selected attention edge restores the pulse; donor keys and values jointly transfer suppression. | Development-selected head, 16 separate constructed evaluation windows, then 12 eligible unedited book/code cases. [Pathway evidence](docs/REPORT.md#2-a-preceding-newline-suppresses-a-later-pulse). |
| **OLMo: the layer-0 MLP conditionally relays the history effect.** Endogenous layer-1 attention can amplify it. | Component swaps support a conditional carrier, without identifying a unique minimal circuit or an additive mediation percentage. [Component evidence](docs/REPORT.md#3-the-earlier-mlp-carries-much-of-the-history-effect). |
| **OLMo: deleting more of the neuron can damage less.** On the matched Wiki panel, scalar-output deletion gives a perplexity ratio of 1933.922, versus 1.445 for the whole neuron output. | Position-local interventions relative to the same native inputs. Whole-output deletion also has severe counterexamples; source K/V and normalized geometry matter. [Comparison and counterexamples](docs/REPORT.md#4-deleting-more-of-the-neuron-can-damage-less). |

### New developmental and optimizer evidence

- **Large coefficient does not mean mature circuit.** The primary weight becomes
  dominant long before the tested final-model H14 newline role appears. Twelve
  functional checkpoints distinguish coefficient growth from pathway development.
- **Adam comparisons depend on which coordinates are compared.** At native
  step 954000 the primary has the largest second moment across neurons in its
  output row. Yet within that same neuron, it has lower second moment than all
  128 preselected other residual axes, giving about twice their median adaptive
  gain. These are different comparisons, not evidence of historical cause.

See the [OLMo synthesis](docs/olmo/RESEARCH_UPDATE_2026_10_23.md) for source links,
negative findings, checkpoint identities and what remains unknown.

![OLMo causal pathways and their limits](figures/mechanism.png)

## Mechanism supported by the OLMo interventions

Earlier-token keys change L0/H14 routing, while values help determine the routed
content; the donor pair jointly transfers suppression. The layer-0 MLP
conditionally carries that history into production of the later
pulse. The neuron's output changes the source token's normalized geometry,
which later attention uses through coupled key/value states and dynamic routing.
A stable source representation can therefore support context-dependent use.
The [full OLMo report](docs/REPORT.md) separates these tested steps from open
explanations and failed replacements.

Here, “scalar deletion” removes one neuron's contribution to one output
coordinate at a specified token; “whole-neuron-output deletion” removes that
neuron's full down-projection contribution at that token. Neither means deleting
the entire MLP or permanently editing the checkpoint. All indices are zero-based.

## Read by model

| Entry point | What it establishes |
| --- | --- |
| **[OLMo-1B](docs/models/olmo/README.md)** | Newline history, conditional MLP relay, scalar versus whole-output interventions, source K/V and geometry. Begin here for the main super-weight case study. |
| **[GPT-2](docs/models/gpt2/README.md)** | Conditional coordinate regrowth, gate interventions, finite early-write transmission and calibration controls. These concern chosen perturbations; a native mechanism linked to the original super weight has not been demonstrated. |
| **[Llama-3.2-3B-Instruct](docs/models/llama/README.md)** | A different scalar/whole-output pattern, source K/V dependence and approximate fixed replacements on tested panels. |
| **[Cross-model comparison](docs/models/cross-model/README.md)** | Which signatures failed to transfer, what differs between protocols, and why no universal mechanism follows. |

## Limitations and claim boundaries

The causal studies use finite interventions; the new developmental study also
compares training checkpoints, without establishing historical training cause. Many later panels have
only four passages per source, fixed selection/calibration and correlated,
unadjusted bootstrap intervals. Fresh passages do not establish independence
from pretraining. A small loss difference does not establish identical
predictions, faithful generation, absence of an effect or a useful speedup.

The latest GPT-2 calibration comparison is particularly uncertain: **51/64
nominal primary intervals include zero, and 27/64 mean signs can reverse when
one passage is omitted**. The largest endpoint calibration shift is about
0.0265 millinats per later target in the removal-minus-control contrast, not
the raw removal-arm loss. Four passage clusters per source and one fixed
fit/control do not support an equivalence or population claim. See
[Report 19](docs/GPT2_CALIBRATION_WORLD_COMPARISON.md) and its
[complete per-passage comparisons](results/calibration_world_comparison_report/tables/all_64_primary.csv).

The [coordinate erratum](docs/COORDINATE_ERRATUM.md) is substantive: an older
secondary-coordinate experiment used L2 where the cited paper lists L1. Its
measurements remain attributed to the coordinate actually executed. The
[prior-literature comparison](docs/NOVELTY.md) and model reports credit closely
related work; this repository makes no broad novelty or universality claim.
Failed runs, negative tests and natural counterexamples remain in the record.

## Quick start

From the repository root, audit the supplied OLMo numerical results on CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4
python -B scripts/audit_results.py
```

This loads no model. It checks exported loss/KL arrays; it cannot independently
reconstruct omitted predictions or prove model-hook correctness. The
[audit index](docs/README.md#cpu-only-numerical-audits) lists all 11 public
numerical auditors and their model-specific guides.

For a future **OLMo nine-batch model reproduction**, first obtain the exact
checkpoint and input files and create the ignored `inputs.local.json` described
in the [reproduction guide](docs/REPRODUCING.md). In a separate environment with
a compatible CUDA PyTorch build, the recorded route is:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_inputs.py --config inputs.local.json --work .reproduce-1
python scripts/run_core.py --work .reproduce-1
python scripts/compare_core.py --work .reproduce-1
```

The core route uses 4,956 sequential forwards and covers nine later OLMo batches,
not every historical experiment. The original-machine allowance was 8 GiB VRAM,
16 GiB RAM and 10 GiB free disk; versions, hashes, measured resources and
same-machine validation are linked in the guide. Different hardware can change
rounding. GPT-2 and Llama use [separate reproduction routes](docs/TRANSFER_REPRODUCING.md);
later GPT-2 reports have their own guides and unperformed replay limits.
These commands document the earlier reproduction route. The new OLMo packages
have separate [offline audits and replay boundaries](docs/olmo/PUBLICATION_SCOPE.md).

## Contents

| Location | Contents |
| --- | --- |
| [research/olmo/](research/olmo/) | New model-specific compact evidence, scripts, protocols and stage-specific reports. |
| [docs/](docs/README.md) | Model-specific entry points, reports, limitations, hypotheses and reproduction guides. |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | Complete earlier README chronology, preserved as a historical snapshot. |
| [experiments/](experiments/) · [scripts/](scripts/) | Frozen designs and archived code; portable preparation, audit and reproduction tools. |
| [results/](results/) · [figures/](figures/) · [reports/](reports/) | Exported numerical evidence, counterexamples, figures and self-contained Reports 16–19. |
| [manifests/](manifests/) · [reviews/](reviews/) | Input/checkpoint hashes, provenance, validation and independent review records. |
| [results/pulse_transfer/](results/pulse_transfer/) | GPT-2/Llama transfer outputs at their established paths. |

Model weights, input corpora and private raw activation bundles are not
redistributed. Exact external sources and hashes are recorded in the guides and
manifests. The public numerical exports have narrower audit scopes than the
private original evidence. Existing artifact paths and saved handoffs are
preserved; the model indexes provide navigation without moving that evidence.

