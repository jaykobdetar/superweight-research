# Reproducing the fixed512 gate study

Use the exact input files listed in `manifests/input_sources.json` and the
GPT-2 checkpoint in `manifests/transfer_models.json`. Follow the retrieval and
configuration instructions in `TRANSFER_REPRODUCING.md`; no UIT expertise is
needed. Keep the input config outside tracked files (for example
`transfer-inputs.local.json`). The same config schema is used here.

Only the pinned Llama tokenizer/config files are needed to reconstruct old
calibration exclusion metadata: `config.json`, `special_tokens_map.json`,
`tokenizer.json`, and `tokenizer_config.json`. The wrapper verifies their hashes
and never opens Llama weight files. Historical model experiments are not run.
GPT-2 uses its pinned complete local checkpoint in FP32. The runner is offline.

## Audit the supplied numerical results without a model

```bash
python -m pip install numpy==1.26.4
python scripts/audit_gate_results.py --output /tmp/gate-audit.json
```

This recomputes all available window loss/KL/agreement reductions and full,
early/later, unexposed and paired-bootstrap summaries. It also verifies export
hashes and keeps failed runs explicitly failed. It does not recompute KL from
full logits, validate private input IDs, or prove hook correctness. In each
failed run's last C512 record the P forward occurred but its record was not
appended; P-agreement is exported from the saved mask, while P baseline loss
cannot be independently reduced from the incomplete record set.

## Rebuild and rerun independently

Install `requirements.txt` in an environment with compatible CUDA PyTorch.
Use one unused workspace; the wrapper refuses existing calibration or study
outputs, and never resumes a partial run automatically.

```bash
python scripts/run_gate_study.py \
  --config transfer-inputs.local.json --work .reproduce-gate
python scripts/compare_gate_study.py \
  --candidate .reproduce-gate/pulse_transfer/gpt2_gate_fixed512_run
```

The default comparison checks the safe numerical repository export. It reads
and reduces actual candidate arrays, verifies all680 records and deterministic
summary/operation files, and requires completed/passed study state. It does not
compare private reference arrays absent from the repository. If the complete
private reference bundle is available locally, use:

```bash
python scripts/compare_gate_study.py \
  --reference /path/to/private/gpt2_gate_fixed512_run \
  --candidate .reproduce-gate/pulse_transfer/gpt2_gate_fixed512_run \
  --scope private --output /tmp/gate-private-comparison.json
```

Private comparison loads shard pairs sequentially and compares tensor contents,
not only saved hash claims. Copied reference metadata is distinguished from
new model inference. Production source/hash checks prevent silent edits.
Use `--prepare-only` on the wrapper to verify inputs and regenerate historical
metadata with zero model forwards. A later full invocation can use that same
prepared workspace if its staged files still match the frozen source export.

A full run performs eight GPT-2 calibration forwards and1352 study forwards.
The calibration raw/map hashes and tensor digest must match the reference before
evaluation starts. A different platform may fail exact matching; preserve that
attempt and investigate it as separate numerical evidence. Do not relax the
checks or overwrite the original reference to force a match.

The physical length is always512. Early/later statistics slice one execution;
they do not validate native128 execution. Both earlier cross-length runs remain
failed and are preserved. The runner retains all21 conditions, PCA/QR ranks1/3/6,
32 inputs,672 exact changed-suffix controls and all original1e-4 reconstruction
guards. Target/current gains can reach1.95e10, so precision assumptions matter.

## Resources and privacy

Original hardware: RTX3060Ti8GiB,16GiB RAM; Python3.12.3, torch2.11.0+cu130,
transformers4.57.6, NumPy1.26.4. Primary study:226.997seconds,0.901GiB peak
allocated GPU memory and2.533GiB process RSS. This excludes setup, input
retrieval, CPU analysis and review. Plan for an8GiB GPU,8–16GiB system RAM,
roughly5GiB free disk for the existing checkpoint plus one fresh raw output,
and about5–10minutes once dependencies/inputs are ready. Exact speed/memory
can differ. The NumPy-only exported-data audit uses about56MiB RSS; private
snapshot reviews use roughly1.2GiB and load no model.

The repository contains numerical outputs, code and source hashes, with
negative experiments included. It excludes weights, corpus text, source and
prediction token sequences, full logits, raw activation donors and credentials.
Private raw bundles remain local. Retrieve external inputs under their own
terms; identical hashes are required for exact reproduction.

## Rebuild figures and the readable report

The plot scripts accept the numerical exports; no model or private tensors
are needed. The default paths in the archived source match the original local
workspace, so use these explicit portable arguments from the repository root:

```bash
python experiments/pulse_transfer/build_gate_result_figures.py --run results/pulse_transfer/gpt2_gate_fixed512_run --out figures/gate_subspaces
python experiments/pulse_transfer/build_gate_spectrum_figure.py --run results/pulse_transfer/gpt2_gate_subspace_run --out figures/gate_subspaces
python experiments/pulse_transfer/build_prefix_diagnostic_figure.py --run results/pulse_transfer/native_prefix_diagnostic_run --out figures/gate_subspaces
python experiments/pulse_transfer/build_gate_diagram.py --out figures/gate_subspaces
python scripts/build_gate_report.py --output /tmp/gpt2-gate-report.html
```

All five regenerated PNGs matched the original report figures byte for byte
in the local portable-path check. SVGs are also retained. The report renderer
embeds the PNGs so its HTML can be read offline.
