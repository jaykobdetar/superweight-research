# Publication cutoff and reproduction scope

## Baseline and preservation

This integration builds on repository commit `0bf1747a8d1b4e6a0adfa3aab72005f7b13df096`. Reports 1–19, all established evidence paths, GPT-2 and Llama model indexes remain unchanged. The findings-first README and OLMo index gain navigation; the previous README remains preserved in the chronological log. The original independent review concerns `e59cfed2df13a1d3a6e6c607648a9d6844790654`, not the newer OLMo studies.

The evidence directories are self-contained packages so their internal paths and original manifests remain meaningful. New reports do not silently rewrite old reports. Their dates and scientific stages matter.

## Included completed studies

- Independent repository review of Reports 1–19: arithmetic, source-file integrity, syntax and statistical fragility checks; no new model inference.
- Coefficient trajectory: 38 training checkpoints plus the exact release.
- Functional development: eight initially selected checkpoints plus four adaptive follow-ups, separately frozen protocols, both recorded precision modes and all per-example results.
- Historical native provenance addendum: retained with its route-specific access failure explicitly superseded by the later row-identity and optimizer evidence.
- Signal-production study: full L1 decomposition and selected geometric interventions, with compact inputs and separate audit records.
- Signal-production continuation frozen at **2026-10-23 06:16 UTC**: reciprocal L0 donors, 28 completed downstream/control forwards and the original audit boundary. A separately dated integration check independently reduces all downstream summaries; it does not audit the model forwards or head14-only arithmetic.
- Native-function reanalysis: exact published output arrays, paired reductions, source-identity joins and explicitly exploratory interpretations. Publication export omits reconstructable natural-corpus excerpts.
- Optimizer-state addendum frozen at **2026-10-23 06:15 UTC**: actual native 954k first/second moments, settings, row-identity evidence, schedule-condition calculations and initializer/confound analysis.

- Separate optimizer-axis supplement frozen at **2026-10-23 06:30 UTC**: 128 preselected residual axes within neuron 1710, actual saved moments and finite-population uncertainty. This is distinct from the earlier exact all-neuron row comparison.

The accompanying manifest and per-package records identify files and their exact scopes. A later research result is not included merely because it shares a workspace or topic.

Completed implementation validation and 64 synthetic recipient baseline/access controls are frozen separately at **2026-10-23 06:22 UTC** in [validated_continuation](../../research/olmo/validated_continuation/README.md). They contain zero content-donor intervention outcomes and zero outcomes from the running fresh natural-text factorial.

## Not completed findings in this release

The fresh 24-window two-source factorial, content-donor intervention results, further post-cutoff analyses, and historical causal training experiments are excluded from completed findings. Completed protocol or implementation checks may be supplied as such; they are not positive scientific outcomes. No heavy duplicate inference, training, paid compute or automatic acquisition is implied by publication.

## CPU-only verification

Run in a disposable copy: some original analysis scripts regenerate their own derived outputs. Preserve the distributed evidence and manifests. Python with NumPy supports numerical checks; Matplotlib/ReportLab are optional for their respective figure/report generation. Use a writable cache directory if rendering. Commands below load no complete model and perform no network retrieval:

```bash
cd research/olmo/training_development
OPENBLAS_NUM_THREADS=1 python combined/checks/validate_compact.py
# Further stored-array diagnostics: combined/REPRODUCE.md

cd ../signal_production
OPENBLAS_NUM_THREADS=1 python audit_compact.py
OPENBLAS_NUM_THREADS=1 python audit_historical_paths.py

cd ../optimizer_state
OPENBLAS_NUM_THREADS=1 python checks/analyze_training_constraints.py
OPENBLAS_NUM_THREADS=1 python checks/derive_optimizer_slice_plan.py
OPENBLAS_NUM_THREADS=1 python checks/primitive_pickle.py
OPENBLAS_NUM_THREADS=1 python checks/analyze_optimizer_snapshot.py
OPENBLAS_NUM_THREADS=1 python checks/audit_bundle.py
```

For native-function numerical reduction and source-input reconstruction, use that package's [publication reproduction guide](../../research/olmo/native_function/README_REPRODUCTION.txt). Some token-conditioned checks require externally acquired inputs, intentionally absent from the public export. Do not mistake a missing input for a failed numerical result or silently replace it with a different corpus.

For the independent review, its [instructions](../../reviews/2026-10-22-independent/verification/README.md) require the original reviewed Git snapshot alongside the scripts. A newer checkout is not an equivalent substitute.

## Inference and acquisition are separate

Each study records exact checkpoint revisions, shapes/ranges, source hashes, intervention definitions and evaluation supports. The deterministic diagnostics do not acquire a new random seed by default; randomized historical procedures retain their original seeds/protocols. New small-panel selection was often exploratory or adaptive and is labeled accordingly.

Full inference requires the documented external model/input artifacts and appropriate resources. Follow the relevant study guide, preserve the recorded arrays, and compare fresh outputs separately. FP32 CPU, approximate FP16 arithmetic and original FP16/native SDPA are distinct numerical regimes. A compact arithmetic PASS does not establish bitwise model equivalence or an independent end-to-end replication.

## Data and interpretation safeguards

No complete model or optimizer shard, full natural corpus, credential, unsent correspondence, or private coordination record belongs in this publication. Small licensed tensor slices, constructed stimuli, public provenance metadata and non-reconstructive numerical outputs have separate roles. See [third-party notices](../../research/olmo/THIRD_PARTY_NOTICES.md).

A frozen epsilon sweep is not a changed optimizer training run. A native-model intervention is not historical training causality. A small teacher-forced loss difference is not output identity, faithful generation, compression or acceleration. Natural counterexamples and failed hypotheses remain part of the evidence.
