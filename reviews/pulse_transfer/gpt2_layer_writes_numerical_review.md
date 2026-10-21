# Independent numerical review: GPT-2 layer-specific coordinate writes

**Verdict: passed, with the interpretation limits below.** Reviewed the completed `gpt2_layer_writes_run`, its protocol, runner, inherited hooks and analyzer. This review used CPU only, loaded no model, and did not rerun inference. The final independent audit reached 1,021,992 KiB peak RSS (0.975 GiB). It does not assess literature novelty or repository publication, which are assigned separately.

The reproducible audit is `gpt2_layer_writes_numerical_audit.py` (`gpt2_layer_writes_numerical_audit.py`, private raw bundle); its machine-readable results, complete condition tables, paired confidence intervals and coordinate trajectories are in [`gpt2_layer_writes_numerical_audit.json`](gpt2_layer_writes_numerical_audit.json). Run from the repository/workspace root with a configured private `pulse_transfer/inputs.local.json` and the documented Python dependencies:

```bash
python pulse_transfer/review/gpt2_layer_writes_numerical_audit.py
```

The audit requires the matching local tokenizer and source files, but no checkpoint weights or GPU. The tokenizer's warning about encoding an extraction longer than the model context is harmless here: the audit tokenizes the 8,192-character extraction and then verifies the first 512 token IDs; it never feeds that extraction to a model.

## What was independently checked

- All 776 saved records/forwards: 24 unique interventions on each of 32 input/start pairs, plus eight exact native-repeat/identity anchors. Four documented aliases map to existing conditions; none is counted as an additional intervention.
- All per-token loss and prediction arrays and saved KL reductions were reaggregated independently. All 776 NLL, baseline NLL, agreement and KL scalar values matched exactly, with maximum scalar error zero. All 192 condition summaries, 128 paired contrasts and 32 paired summaries, including the seeded 2,000-resample bootstrap, matched.
- All 6,984 complete 768-coordinate residual additions at layers 3–11 matched exactly, with the model's actual addition order. Every source attention output was unchanged by its hook. The 4,352 blocked MLP vectors had exactly the two declared coordinates zeroed and every other coordinate unchanged. All 640 residual erasures affected exactly the declared source coordinates after layer 2. Early residuals and the captured pre-erasure layer-2 output were identical across conditions sharing an input.
- The 768 ordinary condition records contain 23,040 successful runtime locality assertions. Those runtime checks and the previously reviewed synthetic hook tests cover later-token locality; saved full-vector reconstruction independently covers the source position.
- All 40 closure checks passed: eight native/identity anchors and 32 last-layer negative controls. In every input/start pair, allowing only layer 11's selected source MLP writes was exactly identical to blocking all layers 3–11 on **every scored loss, KL and prediction element**. A source-only final-block MLP operation has no subsequent attention layer through which to transmit to scored later positions.
- All 16 fresh fourth-panel passages were independently retokenized with the matching tokenizer. Their actual 512-token character spans were disjoint from one another and from all 12 earlier evaluation spans per corpus. Source-file hashes, processed-text hashes, token hashes, extraction hashes, fixed offsets and used-span endpoints matched. Natural and BOS conditions keep tokens 1–511 fixed; BOS replaces token 0 with ID 50256. These are 16 passages with two matched start conditions, not 32 independent passages.
- Current protocol/runner/hook source hashes match both the completed manifest and the prereviewed implementation hashes. Parent-manifest, group-file and condition-file hashes match. The pinned model is `openai-community/gpt2`, revision `75e09b43581151bd1d9ef6700faa605df408979f`, float32 SDPA, seed 20261021. The recorded before/after parameter digest is identical (`2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527`), and recorded parameter versions are unchanged. This CPU audit verifies these recorded invariants and the code that produces them; it does not independently load and rehash the model parameters.

## Intervention scope and causal finding

All layer/coordinate indices are zero-based. `E` erases source-position residual coordinates `[447, 138]` after layer 2. A blocked layer has only those two coordinates of its source-position MLP output zeroed before residual addition. Other MLP coordinates, other positions, attention computations and subsequent responses remain endogenous. The random-coordinate control uses `[588, 104]`.

Within this intervention family, layer 3's selected MLP writes are a strong **conditional bottleneck after erasure**. Blocking them after E raises NLL in all 32 input/start pairs. The erasure-by-layer-3-block interaction is also positive in all 32 pairs. All eight four-passage group confidence intervals have positive lower bounds for both contrasts. The group mean added NLL from blocking layer 3 after E spans 1.182–2.351 nats/token for natural starts and 2.469–4.587 for BOS starts. Conversely, blocking layer 3 without erasure changes group perplexity ratios only to 0.999423–1.000147. The large effect therefore depends on the preceding intervention; it does not establish comparable layer-3 necessity during the native trajectory.

Allowing layer 3's selected writes while blocking the same coordinates in layers 4–11 leaves mean loss close to the E-only state on this panel. The paired NLL changes are −0.0113 to −0.00139 nats/token for natural starts and −0.00772 to +0.00945 for BOS starts. This is an observed small loss difference, not a prespecified equivalence test. The eight natural/BOS/corpus groups need not share an identical mechanism.

## Natural-start results

Entries are `exp(mean paired NLL difference)` relative to the native run for the same start/input, with four passages per corpus. The audit JSON contains confidence intervals, agreement and KL for every condition.

| Selected MLP writes allowed after E | Wiki | DuBois | Hobbes | Optimizer code |
|---|---:|---:|---:|---:|
| All layers 3–11 (E only) | 1.0592 | 1.0733 | 1.0418 | 1.0895 |
| Layer 3 only | 1.0577 | 1.0613 | 1.0316 | 1.0808 |
| Layers 4–11 only | 11.1139 | 4.8117 | 4.4547 | 3.5532 |
| None | 21.6422 | 11.9787 | 8.0456 | 4.2368 |
| Layer 4 only | 26.6940 | 8.8850 | 8.5205 | 9.9387 |
| Layer 5 only | 26.8239 | 13.7270 | 9.7772 | 5.3210 |
| Layer 6 only | 22.6103 | 12.3452 | 8.2885 | 4.4790 |
| Layer 7 only | 20.9974 | 11.6871 | 7.8825 | 4.0656 |
| Layer 8 only | 21.5205 | 11.9209 | 7.9720 | 4.1406 |
| Layer 9 only | 21.6841 | 11.9787 | 8.0239 | 4.1976 |
| Layer 10 only | 21.7267 | 11.9972 | 8.0667 | 4.2498 |
| Layer 11 only | 21.6422 | 11.9787 | 8.0456 | 4.2368 |

## BOS-start results

| Selected MLP writes allowed after E | Wiki | DuBois | Hobbes | Optimizer code |
|---|---:|---:|---:|---:|
| All layers 3–11 (E only) | 1.1909 | 1.0647 | 1.1167 | 1.3266 |
| Layer 3 only | 1.2022 | 1.0600 | 1.1081 | 1.3213 |
| Layers 4–11 only | 27.9945 | 12.5796 | 17.8149 | 130.2078 |
| None | 94.9890 | 29.8619 | 37.6871 | 576.3999 |
| Layer 4 only | 77.6496 | 29.7662 | 40.9131 | 393.6613 |
| Layer 5 only | 101.6995 | 31.9443 | 41.3986 | 792.4288 |
| Layer 6 only | 99.3438 | 31.0216 | 39.5058 | 674.1119 |
| Layer 7 only | 94.5466 | 29.6896 | 37.3824 | 569.2059 |
| Layer 8 only | 93.2823 | 29.4532 | 36.9002 | 528.3090 |
| Layer 9 only | 94.5909 | 29.7446 | 37.5379 | 575.6371 |
| Layer 10 only | 95.2601 | 29.9459 | 37.7988 | 579.7020 |
| Layer 11 only | 94.9890 | 29.8619 | 37.6871 | 576.3999 |

The layer-3 conditional effect appears in both start conditions, with much greater degradation under BOS for several comparisons, particularly code. Later selected MLP writes collectively provide a partial improvement when layer 3 is blocked; no tested individual later layer reproduces the small loss of the layer-3-only condition. Several isolated later writes increase loss compared with blocking them all. These observations reject a simple interchangeable-writer account on this panel; they do not identify all interactions among later layers. The full prefix-block sequence, including negative outcomes, is retained in the audit JSON.

## Why this is not native restoration or whole-network sufficiency

Native top-1 agreement in the layer-3-only condition is only 85.64–92.99% for natural starts and 77.50–88.68% for BOS starts. Source vectors also differ substantially. For example, the mean natural-start layer-3 residual coordinates `[447,138]` are approximately `[2653.894,770.345]` natively, versus `[529.196,66.066]` in the E/only-layer-3 condition. For BOS they are `[2664.038,783.146]` versus `[457.946,54.847]`. Similar mean NLL cannot be interpreted as restoration of the native trajectory, logits, or predictions.

Even blocking all selected MLP writes does not hold the residual coordinates at zero: unmodified attention can write to them. At layer 10 after E with all selected MLP writes blocked, the mean natural coordinates are `[24.790,-1.483]`; BOS coordinates are `[8.897,0.783]`. Large final-layer coordinate changes can coexist with exact later-token nontransmission. Neither large activation magnitude nor coordinate regrowth alone identifies a transmitting computation.

The no-erasure block controls span perplexity ratios 0.999423–1.005723, and the random-pair erase/block controls span 0.999871–1.000061 across the two start conditions. These controls support the specificity of the tested conditional coordinate pathway. One random pair is a limited control; no claim is justified about arbitrary coordinates or a unique minimal circuit. Entire blocks and the rest of the network were never removed in this experiment.

## Uncertainty, resources and next discrimination

This is one checkpoint, one 512-token length, two fixed residual coordinates and four passages per corpus/start. The fourth panel is fresh relative to the earlier three GPT-2 panels, but coordinate/layer hypotheses derive from earlier exploratory work. Bootstrap intervals describe passage variation on this small panel, have no multiple-comparison correction, and do not establish population generalization or equivalence. Saved KL reductions can be reaggregated, but full vocabulary logits were not retained for independently recomputing KL. Audit checks cannot replace a separately executed forward-pass replication.

The completed inference manifest records 43.221 seconds, 776 forwards, 0.706 GiB peak allocated / 0.734 GiB reserved GPU memory and 1.823 GiB peak process RSS. The audit stayed below the requested 3 GiB CPU budget and ran without model loading, GPU contention or external access. No access/resource blocker remained for this review.

The highest-value next causal discrimination is to separate the transmitting effect of layer 3's two-coordinate MLP write from its simultaneous changes to the other residual coordinates: use saved recipient-specific layer-3 writes for exact rescue in the E/block-layer-3 run, then compare equal-norm wrong-direction and cross-input writes while tracing the next attention K/V paths. A successful exact rescue would close the source-write intervention; transfer and directional controls would test whether magnitude, direction or context-specific structure drives the effect. Preserve a fresh evaluation panel and both natural/BOS starts. This suggestion is a hypothesis test, not a finding of the present run.
