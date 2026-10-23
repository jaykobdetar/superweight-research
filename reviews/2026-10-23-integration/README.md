# Integration verification, 23 October 2026

This review concerns publication packaging and exported numerical consistency, not a fresh model inference replication. It does not replace the stage-specific audits or retroactively change their original cutoffs.

The independently written `independent_downstream_reduction.py` uses Python scalar summation over the exported Q1 loss arrays rather than importing the experiment's reduction helpers. It recomputes 112 summary rows, 16 factorial rows, 472 numeric fields and eight reconstruction/add-back controls across 28 saved forwards. Maximum discrepancy is 1.78e-15. It does not audit head14-only attention arithmetic or establish that the actual forward implementation executed the intended intervention.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python reviews/2026-10-23-integration/independent_downstream_reduction.py research/olmo/signal_continuation
python reviews/2026-10-23-integration/verify_package_manifests.py .
```

The manifest checker validates the immutable study packages, not files outside those packages. Stage-specific commands also passed during integration: developmental compact integrity and seven stored-array diagnostics, signal-production compact decomposition, Q1 continuation compact checks, native-function numeric reductions and synthetic estimator algebra, authentic-optimizer scalar checks and the separately frozen axis-sample analysis/uncertainty audit. Reusing author-provided checks is not an additional independent scientific implementation.

Publication edits are limited to navigation, synthesis, explicit boundaries and source-text-safe exports. Original public evidence paths are preserved. The new package intentionally excludes full model weights, natural corpus texts/reconstructable excerpts, credentials and correspondence. Where input removal narrows reproduction, the corresponding guide states it instead of claiming a self-contained replay.
