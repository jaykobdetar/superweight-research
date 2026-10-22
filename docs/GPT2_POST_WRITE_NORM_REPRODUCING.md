# Audit or reproduce T20 actual-write norm matching

The scientific, actual-export and metadata/reproduction reviews passed; exact identities are in `manifests/post_write_norm_release.json`. Read [Report16](GPT2_POST_WRITE_NORM.md) for the estimand and limitations.

## Audit supplied numeric results without a model

From the repository root, with Python and NumPy1.26.4:

```bash
python -B scripts/audit_post_write_norm_results.py
```

The restricted package is `results/post_write_norm_matching/`. It supplies64 numeric shards and their control ledgers, the exact saved analysis, source/execution commitments and explicitly reduced summaries of the independent raw-data audits. It excludes token/prediction IDs, full attention states, hidden states, vocabulary rows and checkpoint tensors. The standalone audit checks exact declared membership, byte hashes, array shapes/dtypes, no-op and suffix controls, then recomputes all1,152window rows,288summaries,648contrasts,324paired-start interactions and128matching-quality records from19,824safe arrays. The public reanalysis uses the frozen analysis helper definitions; the independent original raw numerical audit separately implemented the statistical calculations. Published agreement observations cannot be rederived from omitted prediction IDs. This is not a model replay or a reconstruction of omitted attention states.

The final author public-only rehearsal took3.96seconds and128,284KiB RSS (125.3MiB), with no Torch, tokenizer, model or GPU. The separate author comparison against private originals checked all19,824arrays and64ledgers exactly in11.43seconds at225,092KiB RSS. The independent actual-export review passed in12.4seconds at278MB peakRSS, including privacy/membership checks and exact reanalysis. Allow300MiB and30seconds; these are numerical checks, not pretrained replays.

## External inputs and environment

Use the same six lawful physical sources, exact source bytes/processing and private config schema as the [T19 input instructions](GPT2_NATIVE_VALUE_REPRODUCING.md#lawful-separately-obtained-inputs): Wiki validation Arrow, Utilitarianism (Gutenberg11224), Ethics (Gutenberg3800), and PyTorch `adadelta.py`, `adamax.py`, `asgd.py` from commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`. Do not substitute the older optimizer-code panel. Exact hashes and processed text hashes are in the package's `metadata/data_manifest.template.json`; retrieval links do not guarantee current remote downloads match historical cached bytes. A text-equivalent Arrow transport adapter is not implemented; report an exact-container mismatch rather than bypass it.

Checkpoint: `openai-community/gpt2`, revision `75e09b43581151bd1d9ef6700faa605df408979f`. The source and parameter hashes, tokenizer/config hashes and environment are recorded in the package. Tested Python3.12.3, Torch2.11.0+cu130, Transformers4.57.6, NumPy1.26.4, Tokenizers0.22.2, Datasets5.0.1, safetensors0.8.0. Use a compatible CUDA PyTorch installation. This release changes no credentials, downloads no weights and does not load Llama.

## Tokenizer-only reconstruction and staging

The immutable metadata/source package is `results/post_write_norm_reproduction/`. Keep config, reconstructed private tokens and all new outputs outside that package, in fresh ignored destinations:

```bash
python -B results/post_write_norm_reproduction/experiments/pulse_transfer/post_write_norm_portable.py rebuild \
  --config inputs.local.json --package results/post_write_norm_reproduction \
  --out .reproduce-write-inputs
python -B results/post_write_norm_reproduction/experiments/pulse_transfer/post_write_norm_portable.py stage-replay \
  --config inputs.local.json --package results/post_write_norm_reproduction \
  --inputs .reproduce-write-inputs --out .reproduce-write-model
```

Rebuild checks the exact six sources/tokenizer,444grid candidates (16accepted,428rejected),32natural/BOS input hashes,713inherited historical hashes, four full8192-character F calibration buffers, and16T19spans/32start attestations. It reconstructs canonical `tokens.pt` bytes. Private1370-file scan/fixture trees and raw historical tokens are not required or redistributed: their independently reviewed history closure is inherited, explicitly labelled and hash-committed. No fresh audit of absent historical streams or pretraining-overlap claim is made.

Stage rechecks metadata and token bindings. Only the scientific runner's seven-entry `INPUT_HASHES` assignment changes to bind the explicitly regenerated portable provenance metadata; the intervention/schedule/scoring and all other AST remain unchanged. It does not pretend regenerated metadata is byte-identical to original private metadata. Package membership/symlink/containment controls fail before modifying source or running a model.

The author tested both commands on the existing pinned local sources. Reconstruction took4.47seconds at811,612KiB RSS; staging took2.89seconds at706,688KiB. The independent metadata review rehashed the source/tokenizer/input/metadata evidence and verified the actual staged AST without duplicating selection. This validates current-machine preparation, not independently retrieved source containers or a second model replay. A32-check hostile preparation/export fixture passed; its identity is in the release manifest.

These commands run no pretrained forward and fit no calibration. They write `replay_preparation.json` containing exact absolute commands for sequential segmentA(388), segmentB(388, bindingA), and merge. Inspect those commands and the current process/resource state before executing A thenB thenmerge. Execute the staged analyzer separately on the complete composite directory:

```bash
python -B .reproduce-write-model/pulse_transfer/analyze_post_write_norm.py \
  --out .reproduce-write-model/composite
```

A full second776-forward pretrained replay has **not** been performed. The archived primary full-raw auditors require original provenance/layout and checkpoint access as specified in their contracts; this release does not claim a standalone complete second-replay raw comparator. A completed new run is not automatically a verified match. Compare source/model/input/precision/journal guards, all matching/control evidence and raw saved values before making a replication claim. The public NumPy route checks released observations without a model.

## Resource budget and failures

One GPU job/model at a time. Reference machine:8GiB RTX3060Ti and16GiB RAM. T20's two segments took169.17s and164.06s, each near2.06GiB host RSS,1.064GB allocated/1.162GB reserved GPU memory. These are observed individual-process peaks, not hardware minimums or an inference speedup. Prospective RSS cap3GiB, minimum4GiB host MemAvailable before model load. Keep at least4GiB free beyond environment/checkpoint/inputs for outstanding normal/failure allowances and2GiB reserve;6GiB is a practical buffer. Three original raw shard inventories together total604,662,924B because the merged composite retains exact copies. Preserve source segments and failed outputs; no cleanup/storage replacement is part of reproduction.

Both preplanned segments completed on their first model execution:776actual/retained forwards, zero calibration and no recovery. All final guards and the independent N/projection/readout audits passed their first actual execution. Development hook/storage/history/ABI/export harness failures and prospective resource amendments remain documented in [the failure ledger](GPT2_POST_WRITE_NORM_FAILURES.md). No loss threshold was adjusted to obtain a pass.

The case study has one F fit, one orthogonal control, four passage clusters/source, paired starts and two endpoints. Conditional intervals are correlated and unadjusted, and omit calibration uncertainty. Approximate scalar norm equality does not prove every amplitude-only null impossible without a downstream sensitivity bound. Position/start effects and an individual Ethics reversal are retained. No unique circuit, mediation fraction, faithful replacement, generation benefit or priority follows.
