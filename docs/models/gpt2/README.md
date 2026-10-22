# GPT-2: conditional perturbation and transmission studies

[Overview](../../../README.md) · [Documentation index](../../README.md) ·
[Research log](../../../RESEARCH_LOG.md)

These studies use `openai-community/gpt2`, revision
`75e09b43581151bd1d9ef6700faa605df408979f`, FP32/native SDPA. They begin with the
GPT-2 part of the transfer study, then examine coordinate erasure, attention
gates and chosen finite writes. They do **not** demonstrate a native mechanism
linking GPT-2's effects to the original OLMo super weight, intrinsic path
importance, or native-task necessity.

## What the completed work supports

- After selected source coordinates 447/138 are erased, later MLP writes regrow
  them and become conditionally important for loss. Blocking the same writes
  without erasure has much smaller effects. Regrowth is not restoration of the
  native trajectory. See [transfer findings](../../TRANSFER.md) and
  [coordinate follow-up](../../GPT2_FOLLOWUP.md).
- Gate directions and source-value choices can change loss and predictions;
  high explained calibration variance does not make a faithful simple
  replacement. Sign patterns depend on source, support, reference world and
  control. See the reports below.
- Early L10 write edits can affect later outputs through the remaining L11
  attention block. Either earlier keys or values can transmit the chosen
  perturbation; joint K/V donors close exactly in the tested recipient.
  Signed loss effects can oppose and cancel. This is finite transmission
  evidence, without establishing a unique native circuit.

## Report and reproduction index

| Study | Findings | Audit or reproduction |
| --- | --- | --- |
| Initial transfer screen and coordinate regrowth | [Report 8, GPT-2 sections](../../TRANSFER.md) | [Transfer guide](../../TRANSFER_REPRODUCING.md) |
| Layer-local writes and source K/V geometry | [Follow-up](../../GPT2_FOLLOWUP.md) | [Transfer guide](../../TRANSFER_REPRODUCING.md) |
| Query allocation versus conditional value mixture | [Query pathways](../../GPT2_QUERY_PATHWAYS.md) | [Transfer guide](../../TRANSFER_REPRODUCING.md) |
| Leading gate directions and failed cross-length studies | [Gate subspaces](../../GPT2_GATE_SUBSPACES.md) | [Gate guide](../../GPT2_GATE_REPRODUCING.md) |
| L4–L11 localization and joint effects | [Localization](../../GPT2_GATE_LOCALIZATION.md) | [Localization guide](../../GPT2_GATE_LOCALIZATION_REPRODUCING.md) |
| Finite output nonadditivity versus nonlinear loss | [Additive output](../../GPT2_GATE_ADDITIVE.md) | [Quick start](../../GPT2_GATE_ADDITIVE_QUICKSTART.md), [full guide](../../GPT2_GATE_ADDITIVE_REPRODUCING.md) |
| Old/fresh Wiki calibration with zero source values | [Wiki calibration](../../GPT2_GATE_WIKI_CALIBRATION.md) | [Quick start](../../GPT2_GATE_WIKI_CALIBRATION_QUICKSTART.md), [full guide](../../GPT2_GATE_WIKI_CALIBRATION_REPRODUCING.md) |
| Restored native source values | [Native-value report](../../GPT2_NATIVE_VALUE.md) | [Native-value guide](../../GPT2_NATIVE_VALUE_REPRODUCING.md) |
| Report 16: actual projected-write norm controls | [Report](../../GPT2_POST_WRITE_NORM.md), [HTML](../../../reports/report16.html) | [Report 16 guide](../../GPT2_POST_WRITE_NORM_REPRODUCING.md) |
| Report 17: early/later write support | [Report](../../GPT2_QUERY_WRITE_TIMING.md), [HTML](../../../reports/report17.html) | [Report 17 guide](../../GPT2_QUERY_WRITE_TIMING_REPRODUCING.md) |
| Report 18: earlier L11 keys and values | [Report](../../GPT2_L11_KEY_VALUE_DONOR.md), [HTML](../../../reports/report18.html) | [Report 18 guide](../../GPT2_L11_KEY_VALUE_DONOR_REPRODUCING.md) |
| Report 19: native versus value-lesioned calibration | [Current summary](../../GPT2_CALIBRATION_WORLD_COMPARISON.md), [complete report](../../../results/calibration_world_comparison_report/report19.md), [HTML](../../../reports/report19.html) | [Report 19 guide](../../REPRODUCE_GPT2_CALIBRATION_WORLD_COMPARISON.md) |

## Report 19: uncertainty remains central

Of 64 correlated nominal primary intervals, **51 include zero**. **27/64 mean
signs can reverse under at least one single-passage omission**: for each row
of the [complete primary CSV](../../../results/calibration_world_comparison_report/tables/all_64_primary.csv),
compare the full mean with each mean of the three retained `passage_*_nats`
values. This describes sensitivity of this fixed panel, not a new experiment.

The largest endpoint shift, D_CC − D_PP, is about 0.0265 millinats per later
target in the removal-minus-own-orthogonal-control contrast. It is not raw
removal-arm loss. Four passage clusters per source, one calibration fit/control,
unequal natural cross-cell doses, correlated starts and unadjusted intervals
limit inference. No equivalence, calibration irrelevance or population result
follows. Selected-row precision checks do not bound every unsaved tiny contrast.

The public observations reproduce the published statistics using the frozen
statistical helpers. A fresh 16-call calibration replication and full 1,160-call
evaluation replay remain unperformed; so does a second complete Report 18
776-call replay. Use current reproduction guides for readiness; historical
pending-publication statements in frozen report bodies retain their dated scope.

## Evidence, failures and open hypotheses

The [audit index](../../README.md#cpu-only-numerical-audits) separates public
numerical checks from private activation/readout reviews and model replay.
Transfer exports remain under [results/pulse_transfer/](../../../results/pulse_transfer/); later studies
retain their existing [results/](../../../results/),
[manifests/](../../../manifests/) and [reviews/](../../../reviews/) paths.

Preserved failure ledgers include [Wiki calibration](../../GPT2_GATE_WIKI_CALIBRATION_FAILURES.md),
[native values](../../GPT2_NATIVE_VALUE_FAILURES.md),
[write-norm controls](../../GPT2_POST_WRITE_NORM_FAILURES.md),
[write timing](../../GPT2_QUERY_WRITE_TIMING_FAILURES.md) and
[K/V donors](../../GPT2_L11_KEY_VALUE_DONOR_FAILURES.md).
The [saved handoff](../../HANDOFF.md) and
[hypothesis index](../../README.md#provenance-hypotheses-and-historical-records)
retain proposed continuations. Research is paused; proposed controls and native
mechanistic explanations are not completed findings.
