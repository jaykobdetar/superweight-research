# Audit or reproduce T21 query-write timing

Report 17 is the reviewed scientific snapshot in Library: `libfile_891ea43d5e308191b822cdfee35c0dca`. Its statement that portable checks were pending accurately describes the report's assembly time. Those local checks have now finished. The user subsequently authorized publication of completed Reports 16/17. This repository packaging preserves that scientific snapshot and adds the completed reproduction checks; it introduces no second model execution.

## What was actually reproduced

- The 45-file metadata package contains the 30 frozen scientific sources, two reviewed portable tools, sanitized selection/provenance metadata and the previously fitted calibration reference.
- The 75-file numerical package contains 17,904 allowed arrays and 32 scalar/control ledgers. Independent comparison found every array's dtype, shape and bytes, and every ledger, equal to the private raw evidence. Input and prediction IDs, corpus text, full activations, model weights, credentials and personal filesystem paths are excluded.
- NumPy-only reanalysis reproduced all 960 window rows, 24 anchor rows, 240 summaries, 408 contrasts, 204 paired-start contrasts and 192 support-pair summaries exactly. Torch was not imported.
- One tokenizer-only reconstruction rechecked all 747 deterministic candidate decisions, producing the original 32 input identities and byte-identical token artifact. This used existing lawful local sources; no model forward occurred.
- The private replay workspace is prepared. Only the seven-entry `INPUT_HASHES` assignment differs from the original runner; every other AST node and source text is unchanged. The 648 model forwards in its commands were **not executed**.

This distinguishes exact reanalysis and input reconstruction from an independent second model execution. Public scalar fingerprints and guard flags remain runtime provenance; they do not reconstruct omitted full timing states or full vocabulary distributions. Historical exposure is the inherited recorded-experiment chain, not a new host-wide or pretraining-overlap guarantee.

## Run the numerical audit without model weights or corpora

From the repository root, using Python with NumPy 1.26.4. The convenience wrapper derives both package paths from its own repository location:

```bash
python -B scripts/audit_query_write_timing_results.py --report t21_new_public_audit.json
```

The equivalent direct command is:

```bash
python -B results/query_write_timing_reproduction/experiments/pulse_transfer/audit_query_write_timing_public.py \
  --code results/query_write_timing_reproduction/experiments/pulse_transfer \
  --package results/query_write_timing \
  --report t21_new_public_audit.json
```

The report path must be new and outside the immutable packages. This is a command for another reader; it has not been rerun just to write this guide. The observed first public audit took 3.77 seconds and peaked at 214,184 KiB RSS. Its exact result is `reviews/query_write_timing/t21_public_audit_candidate_v1.json`.

## Rebuild inputs without rerunning previous experiments

The portable tool's `rebuild` action accepts a private configuration with `models.gpt2.{path,model_id,revision}` and `sources.{wiki_validation,utilitarianism,ethics,adadelta,adamax,asgd}`. Supply lawful local files with the hashes in the package's metadata; do not substitute a differently processed corpus. Checkpoint revision is `75e09b43581151bd1d9ef6700faa605df408979f`. Source processing, tokenizer hashes, component boundaries, exclusions and all candidate decisions are asserted by the tool. Existing acquisition/license guidance remains in the repository's reproduction documents and input-source manifest.

```bash
python -B results/query_write_timing_reproduction/experiments/pulse_transfer/query_write_timing_portable.py rebuild \
  --package results/query_write_timing_reproduction \
  --config config.local.json \
  --out .reproduce-t21/inputs

python -B results/query_write_timing_reproduction/experiments/pulse_transfer/query_write_timing_portable.py stage-replay \
  --package results/query_write_timing_reproduction \
  --inputs .reproduce-t21/inputs \
  --config config.local.json \
  --out .reproduce-t21/replay
```

These are separately optional reader commands, not instructions to repeat the completed local reconstruction. Staging writes commands and frozen sources; it does not run the model. Regenerated provenance metadata is explicitly adapted and inherited, while the token artifact remains byte-exact. No old T20, GPT-2 or Llama model experiment is required to rebuild that metadata. The local reconstruction peaked at 767,852 KiB; the stage step peaked at 675,516 KiB.

A full second replay would use the prepared A, B and merge commands in `replay_preparation.json`, run sequentially, with the reviewed resource and predecessor guards. It remains unperformed. Original study cost was 648 forwards, 299.57 seconds including loading/serialization/guards, peak 2,368,968 KiB process RSS and 1,161,822,208 bytes GPU reserved on an RTX 3060 Ti 8 GiB. Use one model/GPU process, a 3 GiB process ceiling, at least 4 GiB available host memory, and at least 4.5 GiB free disk at segment A admission, in addition to storage already needed for lawful inputs and the model. Do not delete evidence to meet these bounds.

## Evidence and limitations of the audit process

Author execution closure: `reviews/query_write_timing/t21_portable_candidates_v1_closure_1792627326869269227/report.json` (`30bf091fa0ca8175c6c1785f6a1d0a996e2b5d71935abbcc8caf4bece4a727a7`).

Independent numeric/privacy review: `reviews/query_write_timing/t21_numeric_candidate_independent_review_v2.json` (`3dfc6bb340cf7350db792cabceaed81f41ab06414ec01e4ac8291136e0fb3619`). The first reviewer harness retained T20's twelve-state assertion instead of T21's ten-state panel. Its failure was preserved; root reviewed the narrow correction before one retry. No candidate or scientific threshold changed.

Independent metadata/input-binding/stage review: `reviews/query_write_timing/t21_portable_metadata_independent_review.json` (`5a61ccde3c685bfd9363922e852fdb43702a84e7c7b223656ffb0226a9fa96bd`). Its first harness rejected generic personal-path strings inside privacy-checking source code. The reviewer executed the one-line regex correction before the requested root review, disclosed that procedural error and stopped. Root subsequently read the complete corrected source and exact diff, and accepted the observed result retrospectively in `reviews/query_write_timing/t21_metadata_review_root_post_execution_validation.json`; this is not recorded as prospective approval. Both attempts remain preserved.

The numeric reviewer authored the analyzer and P/R audits, but not the exporter, independent N audit or report. The metadata reviewer was separate from the exporter. The initial metadata export's standalone elapsed time and peak RSS were not instrumented; they are null, not inferred or measured by repeating the export.

Package identity: metadata `bc58c7b4ee0d77e4d96363ba67678d9b7f154d328006a238daa210343dff8cfd`; numeric export `799f91136775fb619e7bfada4b0e183be1da4e61471c9f45afcaa7100b76a298`. Original full model replay remains a separate replication task.

## Lawful external inputs and private configuration

Use the same six sources and frozen processing described in [the input acquisition guide](GPT2_NATIVE_VALUE_REPRODUCING.md#lawful-separately-obtained-inputs): WikiText validation Arrow (`text` records joined by two newlines), Gutenberg11224 Utilitarianism and Gutenberg3800 Ethics under the frozen `strip_book`, and PyTorch `adadelta.py`, `adamax.py`, `asgd.py` from commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`, joined in that order by two newlines. Supply exact physical/processed SHA256 bytes from `results/query_write_timing_reproduction/metadata/data_manifest.template.json`. Retrieval links do not guarantee a fresh download matches the cached byte container; no text-equivalent Arrow adapter is implemented. Report a mismatch rather than bypassing it. The completed reconstruction used existing lawful local files, not fresh independent downloads.

A private `config.local.json` has this schema (replace each placeholder with your own local path; keep it outside tracked/package directories):

```json
{
  "models": {"gpt2": {"path": "<local GPT-2 directory>", "model_id": "openai-community/gpt2", "revision": "75e09b43581151bd1d9ef6700faa605df408979f"}},
  "sources": {"wiki_validation": "<validation Arrow>", "utilitarianism": "<Mill text>", "ethics": "<Spinoza text>", "adadelta": "<adadelta.py>", "adamax": "<adamax.py>", "asgd": "<asgd.py>"}
}
```

Checkpoint SHA256 is `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`. Tested environment: Python3.12.3, Torch2.11.0+cu130, Transformers4.57.6, NumPy1.26.4, Tokenizers0.22.2, Datasets5.0.1, safetensors0.8.0. The numeric wrapper needs only Python/NumPy; reconstruction/staging use their declared additional dependencies. Model weights, tokenizer/corpus payloads, input/prediction IDs and full private activation/vocabulary rows are not redistributed. Published agreement values are supplied observations; omitted prediction IDs cannot be reconstructed from them. The standalone public audit uses frozen analyzer helpers; the original independent raw N audit separately implemented statistics.

Staging is not replay. Any future second run requires its own admission checks, original source/input/precision/journal/control review and comparison before a replication claim. Original raw N/P/R auditors retain original layout/provenance contracts; this public route does not promise a standalone second-replay full-raw comparator.
