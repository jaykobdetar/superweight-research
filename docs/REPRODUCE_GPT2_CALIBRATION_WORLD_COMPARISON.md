# Reproduce Report 19

These are separate routes. None of the first three loads model weights or requires a GPU or interactive UI. Use fresh output paths outside the immutable result packages; keep private configuration and rebuilt inputs out of Git. Each independent command block creates its parent directory first. The public numeric package has passed independent acceptance: all 30,704 permitted arrays and 32 ledgers matched the private originals, and public-only reanalysis reproduced all published statistics. The local tokenizer rebuild and source-only stage also passed their independent actual-binding acceptance. A repository convenience-wrapper run and second model replay are not claimed.

## 1. Rebuild the aggregate presentation

With Python 3.12, NumPy 1.26.4 and Matplotlib 3.6.3, run from the repository root:

```bash
mkdir -p .reproduce-t23

python3.12 -B results/calibration_world_comparison_report/build_report.py \
  --inputs results/calibration_world_comparison_report/data/report_presentation_inputs.json \
  --out .reproduce-t23/report_candidate

python3.12 -B results/calibration_world_comparison_report/finalize_report19.py \
  --input .reproduce-t23/report_candidate --out .reproduce-t23/report_final
```

The builder uses only the public JSON bundle and supplied templates. The finite finalizer applies the reviewed “128 paired comparisons” clarification and presentation labels; tables, figures and scientific values stay unchanged. Both output directories must be new. This copies saved statistics and renders figures; it performs no bootstrap, calibration fit or independent N/P/R audit. Historical build manifests retain their original checkpoint statuses and paths; those original private JSON paths are commitments, not runtime dependencies.

## 2. Reanalyze the public numerical observations

Python with NumPy 1.26.4 is sufficient:

```bash
mkdir -p .reproduce-t23

python -B scripts/audit_calibration_world_comparison_results.py \
  --report .reproduce-t23/public_numeric_audit.json
```

The pinned NumPy auditor checks the exact restricted package, 32 typed ledgers, 1,728 wide window rows, 432 summaries, 2,304 contrasts, 1,152 paired-start contrasts, all 64 co-primary comparisons and 256 matching-pair records, using the frozen shared 2,000 PCG64 resamples. It uses the same published statistical helpers; the original independent numerical audit is distinct. Five per-target observations are retained, including signed tiny KL and observed agreement. Prediction IDs are omitted, so agreement is not reconstructed from IDs. The signed gate projection `s` is a scalar trace, not a norm.

The exact public array inventory contains 30,704 permitted arrays and is authoritative. A private `--raw` audit can compare all allowed bytes and ledgers with the original shards, but requires separately held private raw files. Public auditing does not recreate QKV, contexts, hidden states, full distributions, learned projections or the independent raw N/P/R audits.

## 3. Validate metadata, reconstruct inputs, prepare commands

JSON-only metadata validation uses the standard library. Tokenizer reconstruction additionally requires the recorded Torch 2.11.0+cu130, Transformers 4.57.6 and Datasets environment, plus your lawful local tokenizer and six source files. These operations do not load GPT-2 weights.

```bash
mkdir -p .reproduce-t23

python -B results/calibration_world_comparison_reproduction/experiments/pulse_transfer/calibration_world_portable.py validate-metadata \
  --package results/calibration_world_comparison_reproduction \
  --report .reproduce-t23/metadata_validation.json

python -B results/calibration_world_comparison_reproduction/experiments/pulse_transfer/calibration_world_portable.py rebuild \
  --package results/calibration_world_comparison_reproduction \
  --config config.local.json --out .reproduce-t23/inputs

python -B results/calibration_world_comparison_reproduction/experiments/pulse_transfer/calibration_world_portable.py stage-replay \
  --package results/calibration_world_comparison_reproduction \
  --inputs .reproduce-t23/inputs --config config.local.json \
  --out .reproduce-t23/replay
```

Create the `.reproduce-t23` parent first. Match the [existing private configuration schema](GPT2_QUERY_WRITE_TIMING_REPRODUCING.md#lawful-external-inputs-and-private-configuration), [source retrieval guide](TRANSFER_REPRODUCING.md) and [model manifest](../manifests/transfer_models.json). Supply `models.gpt2.{path,model_id,revision}` and `sources.{wiki_validation,utilitarianism,ethics,adadelta,adamax,asgd}`. Exact physical and processed hashes, component boundaries, tokenizer files and literal commitment are asserted; fresh downloads are not presumed byte-identical.

The recorded reconstruction preserves the 841-hash selection guard (809 canonical histories plus 32 auxiliary suffix identities), all used intervals, four complete 8,192-character calibration buffers and all 1,090 candidate decisions: 16 accepted, 1,074 rejected. Calibration16 repeats four already excluded contexts; it adds no invented independent hashes. This is inherited recorded-workflow exposure, not a host-wide or pretraining-overlap proof. No prior GPT-2/Llama experiment or old filesystem rescan is rerun.

Staging records explicit metadata-only source/certificate bindings and prepares A/B/merge/analysis commands. It preserves scientific functions, guards, order and scoring; original, public and staged commitments remain distinct. It reuses original F plus the accepted CE aggregate fit. It does not rerun calibration or execute its commands. The tokenizer-only rebuild took 6.38 seconds with sampled peak RSS/HWM about 803,880,960 bytes; source-only staging took 2.91 seconds and peaked at 741,848 KiB. Their assertions and independent actual-binding review passed without weights or model calls. Rebuild memory was sampled every 0.05 seconds and is not an instantaneous all-process peak proof; stage timing/peak are coordinator-reported execution observations. The first rebuild attempt stopped during Torch import because a virtual-address-space limit was mistakenly used. Its preserved failure preceded tokenizer and output work; the retry correctly monitored process RSS/HWM. These are operational events, not experimental outcomes.

## Optional pretrained replay — unperformed

The prepared evaluation schedule is exactly 1,160 calls, split 580/580, using physical length 512 and native-000 evaluation. The original 16-call calibration and its original-F continuity gate are historical accepted prerequisites. A new calibration16 replication is a separate, unperformed operation; it is not silently included in the 1,160-call route. A second T22 full 776-call replay is also unperformed.

Use `openai-community/gpt2` revision `75e09b43581151bd1d9ef6700faa605df408979f`, checkpoint SHA256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`. The observed environment was Python 3.12.3, Torch 2.11.0+cu130, Transformers 4.57.6, NumPy 1.26.4 and CUDA 13.0. The original study used FP32 SDPA, no cache, seed 20261021 and the recorded precision settings.

The observed machine had 16 GiB host RAM and an 8 GiB RTX 3060 Ti. Keep one GPU process at a time. The reviewed primary process ceiling is 4 GiB with at least 6 GiB host MemAvailable before loading; review jobs remain below 3 GiB. A/B took about 353/359 seconds and peaked at 2,332,804/2,164,244 KiB RSS; these are observations, not universal minima or performance claims. Preserve the 2 GiB disk reserve, 512 MiB normal and 1 GiB failure allowances per execution tree. Allow at least roughly 7 GiB additional workspace for conservative fresh A/B/composite allowances and reserve, plus separately acquired model/tokenizer/source assets; actual admission checks remain authoritative. Do not delete earlier evidence to pass them.

Prepared commands and public numerical PASS do not validate a new provenance-aware raw N/P/R comparator or authorize a model launch. Any future model replay requires its own reviewed source/input/fit/precision/resource bindings, execution approval and raw audits. It is not already an independent replication.
