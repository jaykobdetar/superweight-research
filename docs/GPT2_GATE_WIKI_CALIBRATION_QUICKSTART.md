# Audit or reproduce the fresh-calibration experiment

The [T18 report](GPT2_GATE_WIKI_CALIBRATION.md) tests old/fresh Wiki calibration centers and directions separately. The main result is conditional on zero source values in L4–L11; it does not establish the effect of a gate-only intervention in intact GPT-2. No UIT knowledge is needed.

## Audit the supplied results

From the repository root, with Python and NumPy 1.26.4:

```bash
python scripts/audit_wiki_calibration_results.py
```

No model, tokenizer, corpus, GPU or network is needed. The auditor verifies 73 exported files, 26,024 arrays, identities, saved runtime checks, exact-copy/suffix controls, calibration fits, 4,032 summary groups, 912 paired contrasts and 162,432 norm reductions. Its measured run took 6.28 seconds and 117 MiB peak RSS. Thirty deliberately corrupted fixtures were rejected. This establishes consistency of supplied numerical evidence; omitted attention/readout arrays and source freshness cannot be independently reconstructed from the export alone.

## Prepare a new model replay

Follow the explicit commands and seven source-file requirements in the [reproduction guide](GPT2_GATE_WIKI_CALIBRATION_REPRODUCING.md). The pinned GPT-2 revision is `75e09b43581151bd1d9ef6700faa605df408979f`. Model/config/tokenizer hashes are in [transfer_models.json](../manifests/transfer_models.json), and exact source hashes and transformations are in [the safe metadata](../results/gate_wiki_calibration_reproduction/metadata/data_manifest.template.json).

Preparation uses only the GPT-2 tokenizer, an existing validated M8 calibration reference and local source files. It reconstructs four calibration and 16 evaluation windows, verifies every grid rejection and preserves eight canonical file hashes. Six metadata files intentionally change; the wrapper records their relation to the originals. The source inventory covers 37 dependencies. Original model-experiment sources are unchanged; only a staged input-hash assignment changes, with AST checks.

If an M8 reference is unavailable, the guide gives a separate eight-forward rebuild route. Its historical metadata bootstrap requires pinned Llama tokenizer/config files, but no Llama weights or earlier model experiments. A GPT-2-only M8 bootstrap is deferred. Do not omit that bootstrap's `--prepare-only` flag.

## Execute and interpret a replay

The wrapper defaults to preparation only. An explicit `--execute` in a new empty workspace runs 1,548 forwards: four fresh calibration, 776 main and 768 suffix controls. It then runs the frozen analyzer and separately reviewed layout adapter. The adapter restores QR tensor strides lost by NPZ storage, preserving values, original audit code and tolerances. The original failed audit and its diagnosis are retained.

The primary run took 262.6 seconds on one RTX3060Ti, with 2.163 GiB peak host RSS, 0.900 GiB GPU allocation and 0.982 GiB reserved GPU memory. Allow at least 4 GiB available GPU memory, 8 GiB system RAM and 3 GiB free output disk, separately from checkpoint, environment and source storage. The runner preserves a 2 GiB disk reserve. Initial setup and downloads are additional; other hardware may change rounding.

The preparation route, actual reconstructed inputs and wrapper accounting were independently reviewed. The execution path has not been used for a second complete pretrained T18 replay, and a standalone numerical replay comparator remains unimplemented. Successful execution alone is not a confirmed numerical match or independent external replication. Local raw outputs, immutable input hashes and the original frozen code provide the reference for further comparison.

## Preserved failures and limitations

The original mixed eight-passage recalibration design failed before any model run because no Hume window met the frozen exclusion rules. An independently reviewed Wiki-only revision followed; exclusions were not relaxed. The QR-layout audit failure, exporter schema failures and fixture setup failures remain recorded. See [the failure ledger](GPT2_GATE_WIKI_CALIBRATION_FAILURES.md).

The evaluation has four passages per corpus, paired natural/BOS starts, fixed 512-position execution and a single old/fresh calibration pair. Bootstrap intervals condition on those fits, use passage clusters and do not adjust for multiple comparisons. Similar PC1 directions are not proof of a single mechanism. The next prospective question is whether the causal loss pattern persists with native source values.
