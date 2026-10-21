# Audit or reproduce the additive-output experiment

This is the T17 experiment described in [the report](GPT2_GATE_ADDITIVE.md). It needs no UIT knowledge. The commands below assume the current directory is the repository root. No script downloads inputs, changes credentials or starts a prior model study automatically.

## Start with the supplied numerical results

```bash
python -m venv .venv-audit
source .venv-audit/bin/activate
python -m pip install numpy==1.26.4
python scripts/audit_additive_results.py
```

This reads the exact exported file universe, recomputes every primary and segment summary and paired bootstrap interval, verifies the additive loss identities and matches the FP64 loss components to the original per-token readouts. It uses CPU only, no checkpoint, tokenizer or corpus. It verifies numeric consistency of the supplied evidence; it cannot establish the hook implementation or recreate logits from scalar arrays.

## Prepare a model replay without executing it

Install the repository's pinned dependencies in an environment with a compatible CUDA PyTorch build. Obtain the pinned GPT-2 checkpoint and the six source files under their original terms. See `results/gate_additive_reproduction/metadata/data_manifest.template.json` for exact file and processed-text hashes. The model/config/tokenizer hashes are in `manifests/transfer_models.json`. Do not replace a source with a different current revision merely because its name matches.

Create a private `additive_inputs.local.json` outside the committed package:

```json
{
  "models": {"gpt2": {
    "model_id": "openai-community/gpt2",
    "revision": "75e09b43581151bd1d9ef6700faa605df408979f",
    "path": "/EXTERNAL/gpt2-checkpoint"
  }},
  "sources": {
    "wiki_validation": "/EXTERNAL/wikitext-validation.arrow",
    "mill": "/EXTERNAL/pg34901-on-liberty.txt",
    "wollstonecraft": "/EXTERNAL/pg3420-a-vindication-of-the-rights-of-woman-with-strictures-on-political-and-moral-subjects.txt",
    "adagrad": "/EXTERNAL/adagrad.py",
    "rmsprop": "/EXTERNAL/rmsprop.py",
    "lbfgs": "/EXTERNAL/lbfgs.py"
  }
}
```

Supply an existing verified eight-passage calibration directory, or rebuild it using the separate eight-forward route in [the detailed reproduction guide](GPT2_GATE_ADDITIVE_REPRODUCING.md). That inherited calibration-bootstrap route needs pinned Llama tokenizer/config files to reconstruct historical metadata, but never Llama model weights or prior model experiments. Once a verified reference exists, this T17 preparation needs only GPT-2 and its six source files.

```bash
python experiments/pulse_transfer/reproduce_additive.py \
  --config /EXTERNAL/additive_inputs.local.json \
  --metadata results/gate_additive_reproduction/metadata \
  --calibration-reference /EXTERNAL/verified-gate-calibration \
  --parent-metadata results/gate_additive_reproduction/parent_metadata \
  --source-dir experiments/pulse_transfer \
  --source-manifest results/gate_additive_reproduction/source_inventory.json \
  --work /EXTERNAL/new-t17-preparation \
  --prepare-only
```

Preparation performs zero model forwards. It reconstructs the selected token windows, frozen directions and all grid rejection decisions. It requires five canonical file identities and explicitly substitutes only three portable metadata hashes in a staged runner copy. Inputs, directions and scientific code are not relaxed. Keep the private workspace: it contains token payloads and paths deliberately excluded from this repository.

## Execute a new replay

Use the same command with `--execute` in place of `--prepare-only` and a **different empty `--work` directory**. The wrapper never overwrites an existing preparation or failure. It executes 1,992 forwards, then the saved-data analyzer, and records actual completed/attempted counts. This is a new model run, so schedule it when the GPU is available.

The measured original batch used one RTX3060Ti: 700.5 worker seconds, 0.948 GiB peak allocated GPU memory, 1.010 GiB reserved GPU memory and 2.594 GiB peak host RSS. Allow at least 4 GiB available GPU memory and 8 GiB system RAM for this isolated study, plus headroom for the desktop and unrelated jobs. The runner requires about 4 GiB free disk for its bounded outputs and reserve; allow at least 5 GiB free in the output filesystem, separately from checkpoint, environment and source-file storage. Initial retrieval and installation are additional. A CPU-only full hidden-state/readout audit is slower than the original GPU forward batch.

The preparation path and failure bookkeeping have independent checks. A complete second pretrained T17 replay has **not** been performed, and this wrapper has no automatic original-versus-replay raw comparator. The earlier fixed-shape batch has a separate exact replay; do not mistake that for validation of every T17 output. Preserve a mismatch on another platform and report its checkpoint, software/kernel versions, precision settings, input/map identities and first failing operation. Do not widen tolerances or substitute a new calibration to label it a match.

For a completed raw T17 run, independently reconstruct its readout on CPU:

```bash
python scripts/audit_additive_readout.py \
  --run /EXTERNAL/completed-t17-raw-run \
  --checkpoint /EXTERNAL/gpt2-checkpoint/model.safetensors \
  --output /EXTERNAL/new-readout-audit.json
```

This loads only the pinned tied embedding tensor, never a model. It streams all saved hidden states, compares the 744 saved GPU logit rows under explicit FP32 error bounds, and reports all-query metric, sign and argmax discrepancies. The portable adapter preserves the original auditor's scientific function ASTs exactly; only CLI paths, inline file/hash helpers and failure-status writing differ. It needs the private raw run or a new replay's raw output, which is deliberately absent from the repository. It is a numerical consistency check, not a second model execution.

## Evidence locations

- `results/gate_additive/`: all exported per-target numerical arrays and summaries, including negative results.
- `results/gate_additive_reproduction/`: safe input metadata, parent metadata and exact source inventory.
- `manifests/additive_numeric_export.json`: raw-file hashes and numerical export membership; private raw files are not redistributed.
- `reviews/gate_additive/`: independent implementation, numerical, reconstruction and report reviews; the export manifest lists private omissions.
- `experiments/pulse_transfer/`: frozen scientific code, original proposal, prospective storage addendum and reproduction helpers.

The private GitHub repository has not been created. This is a local package pending the specific publication confirmation; no credential or permission change is needed to run the local audits.
