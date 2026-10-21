# Reproduce the model-transfer experiments

Read [the transfer report](TRANSFER.md) and [GPT-2 follow-up](GPT2_FOLLOWUP.md)
for claims and limits. The first six batches contain 4,568 evaluation records
from 4,592 forwards, including 24 development/calibration forwards. Three later
GPT-2 batches add 1,912 forwards/records: nine total batches, 6,504 forwards and
6,480 records. This is separate from the earlier OLMo core reproduction.
No UIT background is required.

## Audit supplied numbers without models

With NumPy1.26.4 installed, run `python scripts/audit_transfer.py`.
It reconstructs each record's NLL, KL, native NLL and argmax agreement from
numeric-only NPZ files. Agreement uses saved boolean equality masks; input and
predicted token IDs are not distributed. It cannot reconstruct KL from full
vocabulary logits or prove hook correctness. Independent reviews cover the
original private outputs/code; full reruns exercise the interventions.

## Obtain exact external inputs

Install `requirements.txt` in a compatible Python3.12/CUDA environment. Original
runs used Python3.12.3, PyTorch2.11.0+cu130, Transformers4.57.6 and native SDPA.
Use ordinary authorized model access; scripts are offline and alter no credentials.

- [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), revision
  `75e09b43581151bd1d9ef6700faa605df408979f`; safetensors SHA256
  `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct),
  revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`. Respect access/license terms;
  do not substitute the base model, another revision or quantized weights.
- Every model/tokenizer/config filename, size and SHA256 is recorded in
  `manifests/transfer_models.json`. Use the matching tokenizer; no fallback.
- The eight source filenames, hashes and lawful retrieval links are in
  `manifests/input_sources.json` and [the main guide](REPRODUCING.md).
  **Transfer preprocessing differs from the oldest OLMo panels:** Gutenberg
  markers are stripped for Hume, Du Bois and Hobbes. Wiki rows are joined by two
  newlines; optimizer files join sgd,adam,adamw in that order. Models tokenize
  the fixed character passages independently. Each transfer data_manifest
  records processed-text/passage hashes, offsets and tokenizer literal IDs.

Upstream files can change. A hash mismatch is an exact-input access limitation,
not permission to guess alignment. A separately labeled replication may use
lawful new inputs with an explicitly changed protocol and recorded identities.
Weights and corpus text are not distributed here.

Create ignored `transfer-inputs.local.json`:

```json
{
  "models": {
    "gpt2": {"path": "/absolute/path/gpt2", "model_id": "openai-community/gpt2", "revision": "75e09b43581151bd1d9ef6700faa605df408979f"},
    "llama": {"path": "/absolute/path/llama", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95"}
  },
  "sources": {
    "wiki_test": "/absolute/path/wikitext-test.arrow",
    "wiki_validation": "/absolute/path/wikitext-validation.arrow",
    "hume": "/absolute/path/hume.txt",
    "dubois": "/absolute/path/dubois.txt",
    "hobbes": "/absolute/path/hobbes.txt",
    "sgd": "/absolute/path/sgd.py",
    "adam": "/absolute/path/adam.py",
    "adamw": "/absolute/path/adamw.py"
  }
}
```

## Sequential private rerun

```bash
python scripts/run_transfer.py --config transfer-inputs.local.json --work .reproduce-transfer --prepare-only
# Confirm the GPU is free, then:
python scripts/run_transfer.py --config transfer-inputs.local.json --work .reproduce-transfer
python scripts/compare_transfer.py --work .reproduce-transfer
```

Preparation hashes inputs and copies immutable scripts/protocols into the ignored
workspace. Stages run sequentially in separate processes, releasing each model.
No existing run directory is overwritten and no partial run is silently resumed.
Use a fresh workspace for recovery. `--stage RUN_NAME` runs one not-yet-started
stage after its prerequisites have completed in that workspace:

| Stage | Needs | Purpose |
|---|---|---|
| gpt2_run | inputs | Development selection, starts/lengths, newline screen |
| llama_run | inputs | Initial protocol with model-specific tokenizer/GQA |
| gpt2_groups_run | gpt2_run | Frozen top-three column factorial, residual controls |
| llama_source_kv_run | llama_run | Independent source K/V donor factorial |
| gpt2_writes_run | gpt2_run,gpt2_groups_run | Third-panel E/A/M writer factorial |
| llama_replacements_run | llama_run,llama_source_kv_run | Fresh means, complement lesion |
| gpt2_layer_writes_run | gpt2_run,gpt2_groups_run,gpt2_writes_run | Fourth-panel layer-local writes |
| gpt2_regrowth_kv_run | all previous GPT-2 stages | Fifth-panel source K/V transmission |
| gpt2_value_geometry_run | all previous GPT-2 stages | Sixth-panel headwise value norm/direction |

Comparison defaults to exact scalar equality. Other hardware/kernels may differ.
`--atol` explicitly sets an absolute tolerance: justify it scientifically and
retain discrepancies rather than hiding input/intervention errors. Each run
also requires its own prescribed exact oracle/closure checks.
Use `--series six` or `--series followup` with the comparison script to inspect
one documented subset. The default requires all nine unique batches.

Allow8GiB VRAM,16GiB RAM and25GiB disk including checkpoints/environment/outputs.
The original six batches took518.85seconds recorded worker time, with6.36GiB
maximum allocated GPU and5.50GiB maximum worker RSS. Retrieval, installation,
review and packaging are additional; budget15–30minutes after inputs are ready.
The three follow-up batches added 120.54 recorded worker seconds on this machine.
CPU-only numeric audit needs no GPU or model.

## Audit boundaries

Frozen protocols live in `experiments/pulse_transfer/*.md`. Private runs retain
token tensors, selection/calibration tensors, per-token NLL/KL/predictions,
source vectors, checks and resource/model/code/input hashes. These are excluded
from Git. Generic source_contribution denotes the native potential activation
times its selected weight; mean-pulse conditions record the actual applied mean
separately. Source0 depends on first-token identity in this reset protocol, so
calibration/evaluation identity overlap is reported explicitly.

Export identities/private raw hashes are in `manifests/transfer_numeric_export.json`.
Repository arrays contain only losses, KL and equality masks. Independent reports
are in `reviews/pulse_transfer/`; private-data review scripts remain in the local
bundle. This repository provides a portable numerical audit and full regeneration
route rather than claiming those private-tensor audits work without their inputs.
Bootstrap intervals use n=4 per corpus/start, condition on frozen selection and
calibration, and do not establish population-wide or pretraining independence.
