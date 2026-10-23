# OLMo-1B-0724: training-origin constraints and authentic Adam snapshot

This is a post-hoc research addendum, not a training experiment. The original
eight-point developmental probe and four-point adaptive extension are unchanged.

## Main findings

- The primary coefficient requires ongoing support under the released schedule;
  ordinary decay cannot account for its 512k-to-800k increase.
- At native step954000, the primary has the largest absolute Adam first moment
  and largest second moment within output row 1764. Its inverse-denominator gain
  is therefore the smallest in that row. A current tiny-denominator explosion
  account is contradicted by the actual state.
- Epsilon filters most comparison coordinates substantially more than the
  primary. The primary update's absolute rank is 101 with actual epsilon, versus
  2149 without epsilon at fixed moments. This is not a training intervention.
- Initial scalar size does not predict the final winner. However, coordinatewise
  truncated initialization and finite precision remain genuine confounds to
  attributing basis selection to Adam.
- Native and pinned HF step954000 output row 1764 are byte-identical. The claim
  is exactly one complete row, not whole-checkpoint identity.

## Offline reproduction

Python 3 with NumPy is sufficient for the numerical checks. The PDF renderer also
uses ReportLab and the DejaVu system fonts. No installation is performed here.

Run from this directory:

    python checks/analyze_training_constraints.py
    python checks/derive_optimizer_slice_plan.py
    python checks/primitive_pickle.py
    python checks/analyze_optimizer_snapshot.py
    python checks/audit_bundle.py

These commands make no network requests and do not load or run a model, compute
new gradients, apply an optimizer, or train. They read small FP32 byte arrays
using NumPy or struct. Re-render the report, if desired, with:

    python checks/render_report.py

Do not run the acquisition scripts as part of reproduction. They were separate,
bounded retrievals and contain receipt-based rerun guards. No retry of any denied
URL is authorized by this bundle. Existing receipts preserve both unsuccessful
metadata access and the later successful, distinct optimizer tensor range reads.

## New actual optimizer evidence

The selected tensors come from the official native release:

    https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/optim/

The acquisition used 12 exact HTTP 206 requests to rank9, rank8 and rank0. Body
bytes totaled 74,641:72,465 selected tensor bytes plus 2,176 header bytes. No full
optimizer shard was downloaded. All request ranges and SHA256s are preserved in
inputs/optimizer_slices/receipt.json. All settings are primitive data.

The uploaded optimizer metadata and an earlier archived official copy are
byte-identical:294,328 bytes, SHA256
743d81315a9aee7bb5fb31b2af9dc1dd24d2713e7dc670c04c1e9a8ee1059297.
The native configuration and the independently archived upcycle configuration
are also byte-identical. Neither match proves the complete historical schedule.

The saved optimizer group has beta=(0.9,0.95), epsilon=1e-5, decay=0.1, and
LR=0.00013046107895981594. The LR exactly matches the released cosine scenario
at checkpoint label 954000. The optimizer tensor's counter is 954001. This offset
is preserved rather than silently corrected. A May 2024 official serializer can
initialize optimizer state via a zero-gradient step, but that is only a possible
explanation, not an established provenance fact about this training run.

Settings are encoded as pickled U 8 byte arrays by the verified official old
serializer. checks/primitive_pickle.py is a static primitive-only interpreter:
it rejects GLOBAL, REDUCE, BUILD, persistent IDs, object construction and all
unsupported opcodes. It never calls pickle.loads or torch.load. Included
official code is reference text only and is never imported or executed.

## Important comparison axes

The measured output row compares 8,192 hidden neurons at fixed residual
coordinate 1764. It does not establish the optimizer's behavior across the 2,048
residual coordinates of neuron 1710's output column. Those are different
concentration questions. A coordinate permutation or arbitrary mixing of
SwiGLU neurons is not interchangeable with a residual-space rotation.

The frozen epsilon sweep changes a formula while holding m and v fixed. It
does not reproduce changed training, changed gradients, or a new next step.
The isotropic comparison is norm-matched within this one row only. It is not
an SGD training arm or a whole-model matched-progress experiment.

## Hypothesis status

1. Continued learned pressure and upstream/readout co-development: supported
   by the developmental trajectories and current adaptive direction. The source
   examples and causal learning pathway remain unidentified.
2. Coordinatewise optimizer dynamics, including epsilon: directly present and
   relevant to the current update, but not established as the historical cause
   of feature formation or residual-axis selection.
3. Initializer orientation and finite-precision symmetry breaking: real
   alternative/confounding ingredients, not proven causes of the super-weight.
4. Token-boundary statistics, packing and objective pressure: remain plausible;
   no new causal evidence here selects one corpus or packing explanation.

Disfavored or excluded within stated scope: largest-initial-scalar-wins; pure
passive decay; ancient momentum sustaining hundreds of thousands of steps;
and a tiny primary second moment at the observed 954k snapshot.

## Smallest causal training test, not executed

First obtain a complete baseline model and data batch, then verify a residual
rotation fixing the all-ones vector preserves logits and transforms gradients
as predicted. Compare a single update under native-coordinate Adam with a
function-equivalent rotated model using coordinatewise Adam. Reset moments in
both arms; add the conjugated-optimizer identity control. This isolates a local
optimizer-basis effect but is not authentic historical continuation.

Only if that validation passes should a bounded short paired continuation be
considered. Compare coordinate concentration and functional H12/H14 onset
separately, at matched progress as well as matched steps. A larger epsilon
needs update-norm/progress controls and both absolute/relative concentration
metrics. Initializer-distribution tests require a separate matched-variance,
multiple-seed comparison, not merely relabeling one rotated realization.

This is outside the present 4GB, no-training scope. Naive FP32 weights, gradients
and two Adam moments alone total about 20.5GB before activations, so it needs a
separate verified resource plan. No training, GPU allocation or paid compute
is requested or started by these artifacts.

Authentic historical causality additionally requires early model/optimizer/
trainer/RNG states near the 120k-to-256k and 512k-to-800k brackets, the actual code
revision and restart/config history, and exact data-order/sampler state. A new
optimizer started from HF weights is a useful controlled fork, not a replay
of the original run.

## Contents and preservation

- deliverables/training_origin_constraints.pdf: six-page readable report.
- results/optimizer_snapshot_summary.json: authentic late-state findings.
- results/optimizer_snapshot_selected.csv: primary, corrected secondary and
  comparison writers at the same checkpoint.
- results/frozen_epsilon_sweep.csv: explicit frozen-state counterfactuals.
- results/optimizer_budget.csv: conditional endpoint budgets over 39 checkpoints.
- inputs/: small archived tensor slices, configs, public source text, receipts
  and the matching user-supplied metadata. No complete model is included.
- BUNDLE_MANIFEST.json: byte sizes and SHA256 for all distributed files.

The package excludes rendered-review PNGs, Python caches and duplicate archive
containers. Their original evidence remains preserved in the work directory.
Public official code is included for provenance under its accompanying license.
No Ai2 outreach has been sent.
