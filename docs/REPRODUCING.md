# Reproduction without specialized background

## Three different checks

1. **Audit supplied numbers:** `scripts/audit_results.py` recomputes window NLL
   and KL from28 batches of per-token numeric arrays. It does not run a model.
   Prediction IDs are excluded, so this portable audit cannot independently
   reconstruct agreement; the original local reviews did check it.
2. **Repeat the model experiments:** the core runner regenerates nine batches
   from independently retrieved, hash-checked inputs with one model load.
   `compare_core.py` checks every window's identity, scoring span, NLL,
   agreement and KL against the recorded output. This has been tested using the
   original resident checkpoint on the original machine; consult the validation
   manifest for the actual outcome. It is not an independent researcher run.
3. **Test generalization:** use fresh data or another model and label it as a
   new experiment. Changing a tokenizer, file version, scoring span or precision
   is not an exact reproduction. Do not silently relax checks to obtain a pass.

## Obtain the external inputs

Use `manifests/checkpoint.json` for the exact Hugging Face revision and tokenizer
hashes. Download or locate `allenai/OLMo-1B-0724-hf` at that revision using your
normal authorized Hugging Face workflow. The experiment scripts operate offline
and do not download weights or modify credentials.

`manifests/input_sources.json` gives every source filename, SHA256, retrieval
URL, corpus token hash, exact starts and tokenization convention:

- Salesforce Wikitext, configuration `wikitext-2-raw-v1`, test and validation
  Arrow files. The historical dataset cache revision is recorded. Dataset
  packaging can change: the default reproduction requires the recorded file
  hash and additionally verifies tokenized contents.
- Gutenberg ebooks9662 (Hume),408 (Du Bois),3207 (Hobbes). Obtain the UTF-8
  editions lawfully for your jurisdiction. Hume wrappers are stripped using the
  exact marker rule in the script; the other two historical files are unstripped.
  Project Gutenberg may update files, so a current download can fail the hash.
  That is an exact-input access limitation, not permission to guess alignment.
- PyTorch `torch/optim/{sgd,adam,adamw}.py` from commit
  `70d99e998b4955e0049d13a98d77ae1b14db1f45` (tested2.11.0+cu130 source).
  Files are joined in that order with two newlines. These are input data;
  running their code is unnecessary.

Create a private, ignored `inputs.local.json`:

```json
{
  "model": "/absolute/path/to/pinned-model-snapshot",
  "sources": {
    "wiki_test": "/absolute/path/wikitext-test.arrow",
    "wiki_validation": "/absolute/path/wikitext-validation.arrow",
    "hume": "/absolute/path/pg9662-an-enquiry-concerning-human-understanding.txt",
    "dubois": "/absolute/path/pg408-the-souls-of-black-folk.txt",
    "hobbes": "/absolute/path/pg3207-leviathan.txt",
    "sgd": "/absolute/path/sgd.py",
    "adam": "/absolute/path/adam.py",
    "adamw": "/absolute/path/adamw.py"
  }
}
```

Preparation refuses to overwrite an existing work directory and verifies files
before creating it. It tokenizes with `add_special_tokens=False`, verifies full
corpus token hashes, and constructs frozen development/evaluation slices. Later
token edits never retokenize text. It copies experiment code into the work
directory so outputs do not alter the repository's recorded results.

## Run and inspect

Use the commands in the README. The tested runtime is Python3.12,
PyTorch2.11.0+cu130, Transformers4.57.6, FP16 and native SDPA. The base package
versions are pinned in `requirements.txt`; choose the matching CUDA wheel when
installing PyTorch. No training, paid service or model duplication is required.

Core order: `column_factorial`, `same_layer_pair`, `later_uses`,
`newline_history`, `newline_heads`, `newline_broadcast`, `newline_natural`,
`newline_components`, `source_geometry`. Dependencies are sequential: head
selection is rerun on development before its held-out tests; natural eligibility
is computed before component/geometry follow-ups. Earlier `upstream_run/tokens.pt`
in this work directory is an input container, not a claim that its earlier GPU
experiment was rerun. Preparation overrides only the two external-text readers
with verified frozen tensors. Intervention code is unchanged.

Each batch saves its manifest, per-window and per-token outputs, controls,
activation traces and model hashes. The runner checks the parameter hash and
PyTorch version counters after each batch. A failed assertion stops that run;
preserve the partial directory and diagnose before using a new directory.
`--jobs` exists for debugging and requires prerequisite outputs; the supported
complete reproduction is the default ordered run.

`compare_core.py` defaults to absolute tolerance1e-5 for scalar output metrics
and exact token/scoring identities. Original-machine equality is recorded
separately; different accelerators can need a scientifically justified numerical
comparison rather than bitwise equality. Never describe an assertion disabled
after seeing results as a successful original protocol.

## Historical coverage and limits

The portable core covers the main later circuit, row/column and geometry claims.
Earlier source-mean, prefix, centered-SVD, gate-PCA, and generation studies have
their scripts, protocols, hashes, numeric results and review notes archived in
`experiments/`, `results/`, and `reviews/`. They are **not all wired into the
nine-batch turnkey runner**. Their source code names the calibration and
preceding-run tensors it consumes; reproducing them requires regenerating those
earlier stages and resolving `/EXTERNAL` input placeholders. The original local
bundles preserve the full dependency tensors and immutable reports. This is a
documented packaging limit, not an independent replication claim.

No model weights or corpora are included. The numeric export retains loss and
KL but deliberately removes predicted token IDs. Figures are regenerated by
`scripts/build_figures.py`. Exact source-versus-export hashes and the input-path/private-service-identifier
transformations are recorded in `manifests/source_export.json`.

For the optional self-contained HTML report, install `requirements-report.txt`
(tested `markdown-it-py==4.2.0`), then run
`python scripts/build_report_html.py --output report.html`.

The old second-coordinate attribution must remain corrected in any replication:
`two_units.py` is the historical L1/L2 experiment; `same_layer_pair.py` is the
correctly cited L1/L1 pair. Do not relabel the old output.
