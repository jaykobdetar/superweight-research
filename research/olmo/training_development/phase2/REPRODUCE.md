# OLMo-1B training-development study: reproducibility

This bundle contains the completed initial eight-checkpoint study, not a completed later-checkpoint extension. Start with `phase2/deliverables/training_development_report.pdf`. All model coordinates are zero-indexed. The exact final revision is `allenai/OLMo-1B-0724-hf@d7cbab742d80589e714b1a2d7f838dcd21cbe143`; do not substitute the nearby `step1454000` branch.

## What is included

- Frozen eight-checkpoint protocol, fixed prompts/token IDs, immutable model revisions and numerical-sensitivity addendum.
- Original FP32 and FP16-approximation results, source-state snapshots, per-probe values, independently recorded acquisition receipts, audits and executable analysis/inference code.
- Official tokenizer and pinned Transformers source; original safetensors headers, index/config files and small down-projection rows. Selected gate/up row tensors are included for the analytic amplitude bounds.
- Post-hoc cross-time compatibility and residual-basis symmetry diagnostics, clearly distinguished from the frozen panel and from training interventions.
- Separate adaptive four-checkpoint protocol, code and interruption state. No extension inference result is included. Its acquisition was cancelled before an approval decision, not denied; do not infer user intent from this event.

The 2.714 GB initial prefix-tensor payload and 0.339 GB partially acquired extension payload are retained in the cloud workspace but excluded from this compact bundle. No full checkpoint or optimizer state was downloaded. Receipts are not independent whole-model hashes. The original transfer logs are archived as `phase2/results/recorded_download_events.jsonl` and `phase3/results/recorded_download_events.jsonl`, not under the active acquisition-log path. This prevents fresh downloads being counted twice when reproducing in a new workspace.

## Offline verification and analysis

Use a separate extracted working copy, preserving the bundle as the original evidence. Python, NumPy and Matplotlib suffice for these commands; psutil is additionally needed for inference. Report rendering needs ReportLab and DejaVu fonts. No PyTorch/Transformers package is required or executed.

From the extracted `olmo_training_origins` directory:

```
python phase2/checks/validate_bundle.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/analyze_prefix.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/audit_precision.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/bound_selected_contributions.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/source_cross_time.py
python phase2/checks/plot_cross_time.py
```

These steps use stored evidence and small included vectors, make no network requests, and regenerate derived artifacts only. `source_cross_time.py` replaces its derived addendum timestamp in the working copy, not the archived original; all eight source-state NPZs must remain present so it does not attempt prefix inference. The analytic bound is for the specified scalar in the exact checkpoint, in real arithmetic before floating-point roundoff, not for downstream importance.

## Re-running the bounded inference (requires fresh resource/network authorization)

Do not run this section automatically. In a fresh working copy with at least 4 GB free disk and 4 GB memory, preserve `phase2/results` as `phase2/results_recorded` and create a new empty `phase2/results`. Preserve the original bundle and receipts. Then explicitly acquire main before running the controller:

```
OPENBLAS_NUM_THREADS=4 python phase2/checks/acquire_prefix.py main
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python phase2/checks/run_all.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/analyze_prefix.py
OPENBLAS_NUM_THREADS=4 python phase2/checks/audit_precision.py
python phase2/checks/audit_initial_study.py
```

The first acquisition is necessary even though the compact bundle contains its manifest: the bulk tensor files are deliberately absent. The runner refuses to overwrite existing raw inference results. Acquisition uses only bounded byte ranges of immutable official Hugging Face tensors, checks HTTP 206/ranges and hashes, and enforces the initial 2.75 GB transfer cap. It does not bypass access restrictions. Inspect any `.partial` file or uncertain network outcome before retrying. The original source/metadata transfer counted 2,715,887,454 bytes; peak normal inference RSS was 405,008,384 bytes. The extra symmetry diagnostic peaked at 737,943,552 bytes.

The final model must pass the endpoint gate before earlier checkpoints are interpreted. The partial model includes all of layer 0, layer-1 attention and four selected layer-1 MLP input rows; it stops before the full layer-1 MLP output. It does not compute model loss, full output-column geometry, gradients, optimizer state or training counterfactuals. The FP16 mode is an approximation, not bit-exact CUDA SDPA. Four artificial prose probes and five single tokens are not a representative corpus.

## Paused adaptive extension

`phase3/protocol.json` fixes 256k, 512k, 800k and 1.2M after seeing the initial eight results, with the identical prompt panel. The total planned additional transfer is 1,356,912,941 bytes, including 338,614,573 already received. Of the first checkpoint, 190/262 slices are complete and verified; 72 small slices (655,360 bytes) remain. No acquisition process or partial file remains active. The four-point controller can resume cached files in the retained cloud workspace only after authorization; in this compact bundle, omitted payload would have to be acquired afresh. Do not merge adaptive data into the initial-eight audit or describe this as a completed twelve-checkpoint study.

## Interpretation

The primary coefficient becomes large before the mature H14/newline and H12/empty-history effects appear on these probes. Intermediate source-token contributions are split across tracked neurons, and early/final prefix/readout combinations are not interchangeable in the native basis. This narrows the developmental account; it does not identify Adam, weight decay, newline frequency or attention gradients as the causal training origin. See the report for discriminating controlled tests and the unresolved native optimizer-state/configuration provenance.
