# Reproduce the completed twelve-checkpoint investigation

Start with `combined/deliverables/training_development_report.pdf`. This is the completed eight-point initial study plus four adaptively selected later checkpoints, not a new twelve-point preregistration. The four later points are 256k, 512k, 800k and 1.2M; both protocols and every original prompt/token ID remain included.

The original eight-point report and original ZIP are retained unchanged as historical evidence. Their description of the then-paused extension is superseded by the new combined report. The extension is now complete. Do not substitute the nearby step1454000 branch for the exact release revision `d7cbab742d80589e714b1a2d7f838dcd21cbe143`.

## Offline analysis

Extract this archive into a fresh working directory and keep the ZIP unchanged. Python, NumPy, Matplotlib and psutil support the provided analyses; ReportLab and DejaVu fonts are additionally needed to render the report. PyTorch/Transformers are not required. From the extracted `olmo_training_origins` directory:

```
python combined/checks/validate_compact.py
OPENBLAS_NUM_THREADS=4 python combined/checks/analyze_combined.py
OPENBLAS_NUM_THREADS=4 python combined/checks/source_cross_time.py
OPENBLAS_NUM_THREADS=4 python combined/checks/query_cross_time.py
OPENBLAS_NUM_THREADS=4 python combined/checks/plot_diagnostics.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/audit_precision.py
OPENBLAS_NUM_THREADS=4 python phase3/checks/audit_precision.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/bound_selected_contributions.py
OPENBLAS_NUM_THREADS=4 python phase3/checks/bound_selected_contributions.py
```

These commands use stored outputs and small included tensors only; no network requests are made. Keep all saved NPZ state files in place so the cross-time scripts do not try to recreate them by loading a prefix. The first command verifies the original compact files before derived artifacts are regenerated. Subsequent analysis may replace derived artifacts in the extracted working copy, not the archived original evidence.

The bundle preserves both precisions, all per-example measurements, the initial endpoint validation, separate cohort audits, source-state and query-state arrays, immutable official URLs/range receipts, headers/configs/indexes, small down rows, selected gate/up rows, the pinned Transformers source snapshot, tokenizer, and scripts. Bulk prefix payload is omitted. The raw prefix tensors remain retained in the original cloud workspace; the compact bundle is not a full model.

## Full prefix-inference replication

This section requires fresh resource/network authorization; do not execute it automatically. A fresh replay requires approximately 4.1 GB of tensor payload on disk and a 4 GB working-memory allowance. Original normal inference used about 406 MB, the query diagnostic about 419 MB. No GPU or training is involved.

Use a separate extracted working copy. Rename each of `phase2/results` and `phase3/results` to `results_recorded` in its own phase directory, then create new empty `results` directories. Preserve the original bundle and receipts. Original acquisition logs are archived under `results_recorded/recorded_download_events.jsonl`, not under the active `sources/download_events.jsonl` paths, so previous transfers are not counted twice.

```
OPENBLAS_NUM_THREADS=4 python phase2/checks/acquire_prefix.py main
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python phase2/checks/run_all.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python phase3/checks/run_all.py
OPENBLAS_NUM_THREADS=4 python combined/checks/validate_replica.py
```

Explicitly acquire main first: the archive contains its manifest but deliberately omits most tensor files. Acquisition validates any existing cached slice before skipping it, uses only bounded official immutable Hugging Face byte ranges, requires HTTP 206 with matching ranges, and enforces each cohort's transfer cap. It does not bypass access restrictions. Inspect any `.partial` file or uncertain outcome before retrying. The inference runner refuses to overwrite raw result files. The numerical replica comparison is distinct from the original historical transfer-accounting and unchanged-evidence audits, whose recorded baseline refers to the original run rather than a newly executed replica.

Fresh transfers can be slightly smaller because selected gate/up rows are already in the compact bundle. Historical audited transfer was 2,715,887,454 bytes for the initial study and 1,356,912,941 bytes for the extension, including the extension's already-received interrupted portion. Total functional-study transfer was 4,072,800,395 bytes. The original bulk hash audits are supplied, but cannot be repeated from the compact bundle alone without reacquiring the omitted tensors.

## Scientific scope

The prefix includes all of layer 0, layer-1 attention and selected layer-1 MLP rows, stopping before the full layer-1 MLP output. It does not compute full-model loss, complete output-column geometry, gradients, optimizer moments, historical training batches or training counterfactuals. FP16 operation/storage rounding is an approximation, not bit-exact CUDA SDPA.

The four artificial prose panels and five single tokens are not a representative corpus. The final-identified heads are tracked; analogous earlier mechanisms in other heads are not excluded. Developmental brackets are between sampled checkpoints, not exact first-emergence times. Cross-time hybrids are native-coordinate diagnostic combinations, not natural model checkpoints or training interventions. The 8041 analytic bound is on one scalar's amplitude in a specified checkpoint, before floating-point roundoff, not on downstream importance.

No additional checkpoint downloads, training or paid compute are implied by these files. The causal tests in the report require a separate decision and an authenticated dense restart or a clearly labeled controlled replica.
