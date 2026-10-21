# GPT-2 source-value geometry: independent numerical review

21 October 2026. **Audit passed.** CPU only, no pretrained model/GPU loaded. Reproducible audit: `gpt2_value_geometry_numerical_audit.py`; evidence and full ranges: `gpt2_value_geometry_numerical_audit.json`.

## Evidence checked

- 520 forwards/records, 32 input/start pairs, 16 main conditions each plus eight anchors. Every saved per-token loss/KL/argmax reduction matches its window record with zero error. All128 condition summaries, 352 passage-level paired contrasts and88 paired summaries/bootstrap intervals were independently replayed.
- All168 raw scored-output equality checks pass, including full-array repeat/identity anchors, plus32 E/B first-boundary equality checks. Runtime logs contain520 operation rows,4,160 projection-locality checks and5,888 exact K/V patch copies.
- Saved source vectors verify4,680 exact residual additions, selected-layer native intermediate/output invariance, source-only E erasure and B L3 MLP coordinate blocking. These are operation-level checks, not a claim that later endogenous source trajectories are unchanged.
- All2,048 geometry rows reconstruct from the raw native donors. Diagonal donor copies are exact; mixed directions/norms match the specified FP32 transformation and a separate FP64 reference. All6,144 randomized head vectors preserve absolute-value multisets exactly; the frozen seed/maps regenerate exactly. Maximum observed relative norm-reduction error is2.6183e-7, below2e-6.
- All source/protocol/parent identities match the manifest; parameters and version counters are unchanged. Actual local tokenizer/source preflight independently reproduces every token tensor and the complete data manifest:16 fresh512-token windows, disjoint from20 prior used spans per corpus and each other. Natural/BOS inputs share the intended suffix. Score index1 consistently means targets2..511.

Peak audit memory is recorded in the JSON (about1.16GiB), below3GiB. Applied recipient K/V arrays are not separately retained for every patched forward: assignment/locality evidence comes from runtime assertions and the prior synthetic hook audit. Native donors and their saved geometry diagnostics were independently reconstructed here. Full vocabulary logits are absent, so KL is reaggregated rather than recomputed from logits.

## Main result: both factors matter, nonadditively

The table gives ranges of corpus means relative to native C. All four factorial cells use **fixed E keys in a B recipient**. The magnitude factor substitutes the joint layer/head norm profile, not one scalar; direction also changes jointly across all tested heads/layers.

| Value condition | Natural PPL ratio | BOS PPL ratio |
|---|---:|---:|
| E direction, E norms |1.033–1.125|1.080–1.398|
| E direction, B norms |2.531–5.035|2.197–6.654|
| B direction, E norms |1.162–1.329|1.431–1.734|
| B direction, B norms |5.254–12.614|11.865–89.202|

Changing E→B norms increases NLL at either fixed direction on all32 inputs. Changing E→B direction increases NLL at either fixed norm profile on all32 inputs. All eight corpus/start bootstrap intervals for each simple effect are above zero. Thus neither direction alone nor magnitude alone explains the tested outcome.

The signed interaction `EE−EB−BE+BB` is positive in31/32 inputs and every corpus/start mean, with all eight descriptive intervals above zero. The exception is Wiki natural window0, −0.014844nats. It is therefore accurate to report a consistent group-level positive interaction, but not positive interaction in every passage. Do not assign additive mediation percentages or infer a globally monotone norm-dose law from this two-level factorial.

Reducing B values to the E norm profile is a large improvement relative to the B/B cell, but it is not full recovery: natural argmax agreement is79.71–88.77%, BOS70.00–80.29%; KL remains positive. The exact E/E restoration is an architectural full-donor control and is not a novel mechanistic finding.

## Magnitude-profile nuance

All384 observed head/input B:E norm ratios exceed one in each of L4..10. At L11, however, **only head8 has a smaller B norm**, in every input; the other11 heads have larger B norms. Head8's median ratio is0.4903. The whole L11 V vector has B:E norm ratio0.7795–0.8808 because that head dominates the aggregate. Consequently, “L11 values are smaller” is valid only for the whole-vector norm, not all per-head magnitudes. No head-specific causal attribution follows from this descriptive observation.

## Negative results and limitations

Native C with source V zeroed only in L4..11 has near-native mean loss: natural PPL1.0014–1.0095, BOS1.0021–1.0131. Agreement is95.15–97.30% natural and94.41–96.91% BOS. This is approximate loss preservation, not exact functional dispensability, and leaves earlier source transmission and source attention mass intact. Some group intervals include zero and others show a small positive NLL cost.

Zero V in B improves NLL on all32 inputs; with fixed E keys it gives natural PPL1.0327–1.1145 and BOS1.0415–1.2076. It does not prove values carry no content or that the whole source can be removed. Current-key zeroing and fixed-key zeroing are different counterfactuals. This batch has no new blocked-source arm, so source-block comparisons must explicitly identify the earlier batch/panel rather than silently pair different passages.

The frozen signed permutation of E values gives natural PPL1.102–1.226 and BOS1.191–1.353. It worsens NLL relative to E in27/32 inputs, but improves the BOS optimizer mean (all four windows; mean−0.0699nats, interval excludes zero), and the DuBois natural interval crosses zero. Thus native E direction is not universally better than this control.

The permutation of B values is still harmful relative to C (natural3.598–8.474, BOS4.920–11.991), yet improves NLL relative to the B/B diagonal in30/32 inputs and every group mean; seven of eight intervals exclude zero. This rejects a story that arbitrary directions at the B norm profile are benign, while also showing that the unpermuted B direction is not uniquely necessary for damage. These are two distinct matched controls, not one random direction tested at two norms, and only one seed was tested.

Report the result as a conditional projected-value geometry decomposition with dynamic recipient queries/gates. It does not identify one causal head/layer, prove an attention-sink or semantic mechanism, restore the native raw trajectory, or establish a universal GPT-2/self-repair theory. The small, adaptive research programme and n=4 descriptive intervals remain the relevant uncertainty limits. No correction to the root analyzer was needed.
