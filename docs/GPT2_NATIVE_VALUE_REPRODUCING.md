# Audit or reproduce T19 native-value boundary effects

Read [Report15](GPT2_NATIVE_VALUE.md) first. This study restores native source
values and tests the L10 PC1 gate edit at all eight upstream/local/downstream
source-value boundaries. The primary evidence is a reviewed composite recovered
after interruption, not a successful uninterrupted original run.

## Numerical audit: no model or external data

From the repository root, with Python and NumPy1.26.4:

```bash
python -B scripts/audit_native_value_results.py
```

The script verifies all83 exported file hashes,41,632 arrays' declared formats,
1,088 available exact suffix metric/norm controls, current/native anchor metrics,
and independently recomputes all window rows, norm aggregates, summaries,
finite-difference coefficients, paired bootstrap intervals and paired-start
contrasts. Original statistical helper definitions are retained from the
independent full-raw audit; the public array loader is a separate adapter.
No Torch, tokenizer, weights, source text, GPU or network is used. Allow about
250MiB RAM and20seconds on the reference machine. The measured author rehearsal
used104.3MiB RSS and14.994seconds. It does not rederive agreements from omitted
prediction IDs or reconstruct omitted attention states/vocabulary rows/journals.

The immutable numerical evidence is under `results/gate_native_value/`, with
its external83-file inventory at `manifests/native_value_numeric_inventory.json`.
Creation records saying "pending independent comparison" are historical states;
the later independent acceptance is recorded under `reviews/gate_native_value/`.
Never overwrite the historical evidence to turn an old pending state into a pass.

## Lawful separately obtained inputs

The35-file immutable reproduction package is
`results/gate_native_value_reproduction/`. It includes aggregate T18 reference
maps/directions/manifest, safe historical commitments and the complete frozen
source closure. It contains no example tokens, source corpora or model weights.
It requires no earlier model experiment, no refit and no Llama model/tokenizer.

Use the pinned [GPT-2 checkpoint](https://huggingface.co/openai-community/gpt2)
revision `75e09b43581151bd1d9ef6700faa605df408979f`, including tokenizer/config
files and the checkpoint. Hashes are in the package's metadata dependencies and
`metadata/data_manifest.template.json`. Runtime: Python3.12.3,
Torch2.11.0+cu130, Transformers4.57.6, NumPy1.26.4, safetensors0.8.0,
Datasets5.0.1 and Tokenizers0.22.2. Use the repository requirements and a CUDA
build compatible with your hardware.

Supply these six source keys in a private `inputs.local.json`:

| Key | Source and processing |
|---|---|
| `wiki_validation` | [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1` validation Arrow file; join `text` records with two newlines |
| `utilitarianism` | [Project Gutenberg11224](https://www.gutenberg.org/ebooks/11224), plain-text Mill; frozen `strip_book` transformation |
| `ethics` | [Project Gutenberg3800](https://www.gutenberg.org/ebooks/3800), plain-text Spinoza; same frozen transformation |
| `adadelta` | [PyTorch source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/optim/adadelta.py) |
| `adamax` | [PyTorch source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/optim/adamax.py) |
| `asgd` | [PyTorch source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/optim/asgd.py) |

Optimizer files join in the listed order with two newlines. Every physical
file and processed text must match its recorded SHA256. Upstream web content
or Arrow serialization can change; these source links are retrieval starting
points, not a claim that a new download necessarily reproduces the exact
cached byte container. A byte mismatch must be reported, not silently bypassed.
A transport adapter for a different but text-equivalent Arrow container is
not implemented by this release. The current author rehearsal used the
existing pinned local sources, not independently retrieved downloads.

Configuration example (substitute your own absolute paths):

```json
{
  "models": {"gpt2": {
    "path": "/EXTERNAL/gpt2",
    "model_id": "openai-community/gpt2",
    "revision": "75e09b43581151bd1d9ef6700faa605df408979f"
  }},
  "sources": {
    "wiki_validation": "/EXTERNAL/wikitext-validation.arrow",
    "utilitarianism": "/EXTERNAL/pg11224-utilitarianism.txt",
    "ethics": "/EXTERNAL/pg3800-ethics.txt",
    "adadelta": "/EXTERNAL/adadelta.py",
    "adamax": "/EXTERNAL/adamax.py",
    "asgd": "/EXTERNAL/asgd.py"
  }
}
```

## Tokenizer-only reconstruction and model staging

Keep configuration, all new outputs and logs outside the immutable package.
Use fresh destinations and `python -B` to avoid bytecode artifacts:

```bash
python -B results/gate_native_value_reproduction/experiments/pulse_transfer/native_value_portable_draft.py rebuild \
  --config inputs.local.json --package results/gate_native_value_reproduction \
  --out .reproduce-native-inputs
python -B results/gate_native_value_reproduction/experiments/pulse_transfer/native_value_portable_draft.py stage-replay \
  --config inputs.local.json --package results/gate_native_value_reproduction \
  --inputs .reproduce-native-inputs --out .reproduce-native-model
```

Rebuild checks382 candidates,366 rejections,16 windows, the canonical tokens,
BOS identity and inherited577-hash exclusion history. Stage independently
rechecks the package and reconstructed data, then changes only the six-entry
`INPUT_HASHES` assignment. The remaining runner AST and all scientific source
bytes remain exact. Package inventory, symlink and output-containment guards
fail before source mutation. History is an inherited attestation; absent
historical streams are not newly audited. Sixty CPU fixtures and a separate
independent11-check probe cover the preparation path and rejected cases.

These commands do **not** load pretrained weights or run a model. The stage's
`replay_preparation.json` records the exact absolute command for a later explicit
2,184-forward run. Execute that command only after checking no other model job
is active, at least2GiB reserve remains beyond the checkpoint/environment/data
and output allowance, and your chosen environment meets the frozen protocol.
Use one GPU worker. The reference machine had an8GiB RTX3060Ti and16GiB host
RAM; observed model peak was about2.86GiB host RSS and1.074GiB GPU reservation.
Allow additional host/GPU space for runtime variability. The retained raw
shards are361.1MiB, with separate512MiB normal and256MiB failure bounds.

Run the frozen `analyze_gate_native_value.py --out RUN_DIRECTORY` on complete
raw replay output. It cannot read the public restricted NPZ schema. Compare
fresh raw metrics, snapshots, controls, model/source/input/precision guards and
all derived statistics. The archived primary full-raw auditors require the
historical local provenance layout; the public NumPy auditor is the standalone
release reanalysis route. There is no portable full-raw second-replay comparator
claimed complete here. A successful replay alone is not a verified match.

## Failures, resources and epistemic limits

The historical original returned1,110 calls, fully processed1,109 and committed
1,092 scientific forwards. RecoveryA ran616 calls including70 exact continuity
forwards; recoveryB ran546. Total actual returns2,272; retained scientific
forwards2,184. Eighteen uncommitted original returns have no recoverable arrays.
Original final parameter/version/precision guards remain unavailable. A/B end
guards passed, the70-forward raw continuity payload was exact, and all34
available original/recovery hidden/logit fingerprints matched. This is
same-machine continuity, not independent external replication.

Readout-key correction, interrupted process, sizing/overflow fixtures, rejected
portable packages and their incorrect early allowlist assertion are preserved
in the failure ledger. No threshold, exclusion rule or calibration was relaxed
to obtain a pass. A second full pretrained replay has not been performed.
Use four passage clusters per source; paired starts and boundaries reuse them.
Conditional bootstrap intervals omit calibration-fit uncertainty and have no
multiplicity correction. Lower loss is not faithful output preservation,
generation quality, a mechanism shared with OLMo/Llama, or a unique causal circuit.
