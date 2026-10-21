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

The repository is private. It contains no credentials, checkpoint weights,
input corpora, or unrelated files. Retrieve external inputs under their own
terms; hashes and source links are supplied instead of redistribution.
