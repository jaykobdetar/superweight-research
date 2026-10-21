# Super-weight causal circuits

A reproducible causal case study of the computation around an OLMo-1B super
weight: earlier newline history, attention K/V pathways, residual components,
source geometry, and limits of simpler replacements.

Start with the [research report](docs/REPORT.md), its
[primary-literature comparison](docs/NOVELTY.md), and the
[coordinate correction](docs/COORDINATE_ERRATUM.md). The correction is substantive:
an older secondary-coordinate experiment used layer2, whereas the cited paper
lists layer1. Measurements are preserved under their actual coordinates and a
separate corrected experiment is included.

The strongest scoped result is a development-selected L0/H14 path through which
an earlier single-newline token suppresses a later pulse. Separate key/value
interventions and natural-input tests support that path. A layer-0 MLP output
conditionally relays it. Downstream, removing one dominant output coordinate
can be far worse than removing the whole neuron output; mixed source K/V states
and source-normalized geometry help explain this intervention dependence.

This repository does not claim a universal mechanism, priority over all prior
work, or a faithful fixed low-rank replacement. Negative experiments and natural
counterexamples are retained. No UIT knowledge is needed.

The [model-transfer report](docs/TRANSFER.md) adds GPT-2 and Llama-3.2-3B-Instruct.
It finds conditional GPT-2 MLP coordinate regrowth and different Llama source
K/V dependence, including approximate fixed-source replacements. See the
[transfer reproduction guide](docs/TRANSFER_REPRODUCING.md) for eleven sequential
batches and their limits. The [GPT-2 follow-up](docs/GPT2_FOLLOWUP.md) localizes
the conditional L3 writes, separates source K/V transmission, and tests value
norm against direction. The [query-pathway report](docs/GPT2_QUERY_PATHWAYS.md)
separates allocation from the conditional non-source value mixture and finds
opposing loss effects on several corpora. These findings do not establish one
shared mechanism or faithful output replacement.

The [gate-subspace report](docs/GPT2_GATE_SUBSPACES.md) tests centered PCA
directions with causal retention/removal and matched controls. High calibration
variance does not yield a faithful scalar replacement: rank-one retention
raises perplexity about15–29% relative to the zero-value reference, while
rank-six retention still changes about5–10% of argmax predictions. Two failed
cross-length runs remain preserved; the completed study uses fixed512 execution.
See the [standalone reproduction guide](docs/GPT2_GATE_REPRODUCING.md): it
rebuilds old metadata without rerunning prior model experiments or loading
Llama weights. Audit exported gate results with
`python scripts/audit_gate_results.py`; this needs NumPy and no model.

The [layer-localization report](docs/GPT2_GATE_LOCALIZATION.md) then tests every
L4–L11 singleton and their joint intervention on fresh passages. L10 PC1 removal
worsens Wiki loss but improves both books; joint effects do not add in NLL,
and lower loss still does not preserve predictions. See the
[localization reproduction guide](docs/GPT2_GATE_LOCALIZATION_REPRODUCING.md).
`python scripts/audit_localization_results.py` checks all exported numeric
results without a model. The new primary batch is independently audited;
a complete second pretrained localization replay remains unperformed.

The [additive-output report](docs/GPT2_GATE_ADDITIVE.md) separates finite
output nonadditivity from the nonlinear loss calculation on another fresh
panel. The terms can oppose each other strongly; small net loss differences
do not imply faithful predictions or an internal compensating mechanism.
L10's scoped Wiki-versus-other-source pattern recurs, while a blanket book
benefit fails on Mill. Follow the [audit/replay quick start](docs/GPT2_GATE_ADDITIVE_QUICKSTART.md)
or run `python scripts/audit_additive_results.py` with NumPy only. This new
batch has independent saved-array and CPU readout audits; a complete second
pretrained replay remains unperformed.

Transfer numerical audit: `python scripts/audit_transfer.py`. Full rerun:
`python scripts/run_transfer.py --config transfer-inputs.local.json --work .reproduce-transfer`,
then `python scripts/compare_transfer.py --work .reproduce-transfer`.

## Quick start

CPU-only numerical audit of the supplied results:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4
python scripts/audit_results.py
```

GPU reproduction of nine core batches, using separately obtained inputs:

```bash
# Install requirements in an environment with a compatible CUDA PyTorch build.
python -m pip install -r requirements.txt
# Create inputs.local.json using docs/REPRODUCING.md and manifests/input_sources.json.
python scripts/prepare_inputs.py --config inputs.local.json --work .reproduce-1
python scripts/run_core.py --work .reproduce-1
python scripts/compare_core.py --work .reproduce-1
```

The runner uses one offline checkpoint load and4956 sequential forwards.
The original machine has an8GiB RTX3060Ti and16GiB system RAM; allow at least
8GiB GPU memory,16GiB system memory, and10GiB free disk for checkpoint, runtime,
and outputs. Budget roughly10–20minutes after inputs and dependencies are ready;
downloads and installation are additional. Exact measured validation time and
peak memory are recorded in the validation manifest. CPU-only audit needs no
model and is much smaller. Other GPUs/kernels can change FP16 rounding.

## Contents

- `docs/`: report, protocols, limitations, novelty and correction.
- `scripts/`: portable preparation, core reproduction, comparison, numeric audit,
  and figure generation.
- `experiments/`: archived intervention and analysis code, plus frozen designs.
  Local input prefixes are replaced with explicit `/EXTERNAL` placeholders.
- `results/`: all exported historical window metrics, summaries, failed
  conditions, and per-token loss/KL arrays. Input and prediction token IDs are
  excluded. Large private raw activation bundles remain local.
- `reviews/`: independent CPU audits and scientific review notes.
- `manifests/`: checkpoint, input hashes, seeds, dependencies, export provenance,
  and validation records.
- `figures/`: original explanatory diagrams and result plots.
- `transfer/`: subsequent model comparisons, when executed.

The intended GitHub destination is private; creation remains denied and no
remote has been configured. This local repository contains no credentials, checkpoint weights,
input corpora, or unrelated files. Retrieve external inputs under their own
terms; hashes and source links are supplied instead of redistribution.
