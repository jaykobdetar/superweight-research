# Layerwise KV0 mediation review

CPU-only review of `mediation.py`, saved paths, raw observations, and paired
summaries. No experiment edits or model/GPU loading. Peak audit RSS was 537MiB.

All 1,920 observations, 240 condition groups, and 1,936 forwards passed. Independently
replayed 240 summary rows and 480 paired contrast intervals; verified raw NLL/KL
means, baseline means, agreement, token hashes, code hash, and weight integrity.
Synthetic tests confirm layer/head/kind localization and all-position candidate
patches. Saved layer 0/1 KV0 is identical before and after the layer 1 source change.

## Exact cut and scoring scope

Both directions of the all-head KV0 cut at layers 2–15 reproduce donor losses,
predictions, and KL bitwise on every saved position across 24 windows. All records
uniformly score 510 predictions: predictor indices 1–510, predicting token indices
2–511. Predictor 0 is intentionally excluded because the source residual remains
unpatched. The clean/deleted arrays exactly match the preceding gate panel after
dropping its first prediction. Do not compare their aggregate metrics with prior
511-prediction metrics without acknowledging this scoring difference.

This cut is architecture-expected: a source-only position 0 perturbation reaches
other positions through attention K/V from that position. Replacing every such
edge leaves later-position computation equal to the donor while preserving a
different source residual. Exact equality validates intervention placement and
closes this specified causal boundary; it does not identify a unique minimal
circuit or establish that one head is necessary.

## Layer cuts and persistent source paths

Corrupting all layer 2 KV0 alone nearly reproduces full deletion NLL on all three
corpora. Repairing all layer 2 KV0 has asymmetric results: residual clean-relative
NLL is 2.41046 on WikiText, 0.13637 on Hume, and 0.12877 on Plato. Extending repair
through layers 3, 5, 8, 11 progressively reduces these residuals; the full 2–15 cut is
exact. L2/H15 alone leaves much larger residuals than all layer 2 heads.

Late-only repair scarcely helps a network whose later-token states were already
damaged by the first broadcast. Late-only damage into a clean recipient still
matters: layers 3–15 add 2.59655 nats on WikiText, 0.14625 on Hume, and 0.13820 on
Plato. These measurements support an early trigger plus continuing source-KV
effects, with major state and corpus dependence. They are overlapping nonlinear
cuts, not additive mediation fractions. Damage curves are not strictly monotone.

## Constants and downstream heads

Calibration uses the same development-only means as the preceding gate batch;
all layer 2 K0/V0 means match exactly. There is no evaluation fitting. All-layer
means beyond layer 2 have correct dimensions and finite values; individual raw
calibration activations were not retained for independent replay.

Replacing KV0 at layers 2–15 in clean runs preserves approximately 99.36–99.71%
argmax agreement across corpus/calibration-source combinations. Replacing all
layers 0–15 preserves approximately 97.65–97.94%, with clean-relative mean NLL
between 0.00199 and 0.00667 nats. Thus early cache entries contain some information
that the later entries do not need in these conditions. Small NLL differences do
not establish distributional equivalence. These hooks still compute the original
projections; actual prefix-cache bypass needs its own matched standard-cache
baseline and must explicitly handle any skipped first prediction.

Late-head interventions swap Q/K/V at all positions in one head, so they measure
downstream dependence rather than direct transmission from source 0. Queries now
have nonzero effects, unlike the first-broadcast layer's later queries. L12/H5
clean full-QKV donor rescue worsens all three corrupted recipients by roughly
1.11–1.29 nats; its clean-state damage is modest. L15/H11 QKV rescue helps, but
query-only rescue can worsen loss and matched H0 controls can show stronger
rescue than the nominated head. Neither head warrants a context-independent
“necessary transmitter” label from these interventions.

No blocking correctness issue found. Use pointwise paired intervals and preserve
the exploratory, reused-evaluation-window scope of this mechanism panel.
