# Independent numerical review: GPT-2 source K/V after layer-3 regrowth

**Verdict: passed.** All declared numerical, closure, input and source-vector checks passed. No access, resource or scientific-integrity blocker was found in this batch. This review establishes results for the declared intervention panel, not a unique mechanism or a complete replacement of the model.

The independent audit is `gpt2_regrowth_kv_numerical_audit.py` (`gpt2_regrowth_kv_numerical_audit.py`, private raw bundle). Its [`JSON evidence`](gpt2_regrowth_kv_numerical_audit.json) preserves file hashes, every native-relative table, additional E-relative paired summaries, all simple effects/interactions with confidence intervals, per-window sign counts and resource accounting. Run with the documented dependencies and local `inputs.local.json`:

```bash
python pulse_transfer/review/gpt2_regrowth_kv_numerical_audit.py
```

The audit loads saved tensors and the local tokenizer, with no pretrained model, GPU or external access. It uses one CPU thread and approximately 1.1 GiB peak RSS, below the 3 GiB limit.

## Independently verified evidence

- **616 records/forwards**, 19 conditions, 16 passages × two matched start conditions, plus eight anchors. No duplicate key, missing condition or extra independent sample was accepted.
- Exact independent reaggregation of all NLL, native baseline NLL, KL reductions and argmax agreement, with maximum scalar error **zero**. All 152 condition summaries, 576 signed paired contrasts and 144 paired summary groups match, including 2,000-draw seeded bootstrap intervals.
- All **18,480 full source vectors**, **5,544 complete residual additions**, **576 source residual erasures** and **288 layer-3 MLP coordinate zeroings** match the specified operation. Attention source vectors and all non-target coordinates are unchanged by the respective writer hooks. Pre-erasure captures and the source vectors through layer 3 match the appropriate recipient world.
- All **1,024 retained E/B source K/V donor vectors** have the expected layer, projection, shape and dtype. Every retained source boundary agrees with the saved writer trace. All 32 first-boundary records independently confirm matching later-position layer-3 residuals and layer-4 queries, matching source layer-3 attention, and source residual differences confined to coordinates `[447,138]`.
- All **128 diagonal**, **128 cross-recipient**, **32 source-block** and **eight native/identity** closures hold exactly on every scored per-token loss, KL and argmax element. Together with first-boundary and drift records, all **392** declared check records are accounted for. An additional audit check finds exact E/B recipient closure with E keys clamped and source values zero in all 32 pairs.
- Runtime records account for **18,480 writer locality assertions**, **4,928 projection locality checks**, **5,632 exact source K/V patch copies** and **512 first-query checks**. Expected patch counts are verified separately for factorial, zero-value, mask and anchor conditions.
- All 16 fifth-panel windows were independently retokenized and compared against actual source hashes, processed-text hashes and token IDs. Used token-character spans are disjoint from one another and all **16 earlier spans per corpus**. Natural/BOS inputs differ only at token 0; BOS uses token ID 50256. The edge fractions `[0,.05,.9,.95]` remain fixed and can include source front matter or code headers.
- Implementation hashes match the independently tested preflight and run manifest. Checkpoint revision, parent manifest and conditions hashes match. Recorded before/after parameter digests are equal and parameter version counters are unchanged. The completed run has no failed state; the failure handler was not triggered.

The matching checkpoint is `openai-community/gpt2`, revision `75e09b43581151bd1d9ef6700faa605df408979f`, float32 SDPA, seed 20261021. The model digest is `2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527`. The reviewed runner hash is `b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8`; analyzer hash is `ec2537cf37ebda5db4e4c666e70299af9db3ce78b05339a7582021c8266ed062`.

## What the interventions mean

All indices are zero-based. Native world C has no intervention. E erases residual coordinates `[447,138]` at source position 0 after layer 2. B performs E and additionally zeros those two coordinates of the layer-3 source MLP output. No later MLP coordinate write is blocked in this batch.

`KE`/`KB` and `VE`/`VB` denote source key/value projection vectors taken from the corresponding E/B runs, clamped at every layer 4–11. Queries and all later-position projections are unchanged by each patch. Their subsequent network responses remain endogenous. The E recipient supplies the unique factorial analysis; matching B-recipient arms validate closure and are not additional samples. Values below are perplexity ratios to the native C run with the **same** passage and start-token choice, computed as `exp(mean paired NLL difference)` over four passages.

## Natural-start results

| Condition | Wiki | DuBois | Hobbes | Optimizer code |
|---|---:|---:|---:|---:|
| E | 1.07060 | 1.05975 | 1.05611 | 1.04027 |
| B | 7.63598 | 5.62314 | 4.83020 | 4.72364 |
| KE + VB | 7.68204 | 6.08802 | 5.68542 | 8.21239 |
| KB + VE | 1.13137 | 1.05965 | 1.10693 | 1.01017 |
| E, zero V / endogenous K | 1.04523 | 1.06746 | 1.08829 | 1.03431 |
| E, zero V / clamped KE | 1.04494 | 1.06224 | 1.08233 | 1.03094 |
| B, zero V / endogenous K | 1.05220 | 1.06710 | 1.12184 | 0.99020 |
| B, zero V / clamped KE | 1.04494 | 1.06224 | 1.08233 | 1.03094 |
| Source blocked, either recipient | 3.90461 | 2.67980 | 2.60435 | 1.59106 |

## BOS-start results

| Condition | Wiki | DuBois | Hobbes | Optimizer code |
|---|---:|---:|---:|---:|
| E | 1.21946 | 1.10626 | 1.12452 | 1.29300 |
| B | 37.28172 | 16.64949 | 11.58826 | 130.72974 |
| KE + VB | 26.33069 | 12.90124 | 8.74931 | 83.12245 |
| KB + VE | 1.67213 | 1.33379 | 1.27098 | 1.43503 |
| E, zero V / endogenous K | 1.10455 | 1.08855 | 1.11909 | 1.15633 |
| E, zero V / clamped KE | 1.11908 | 1.09517 | 1.12538 | 1.16779 |
| B, zero V / endogenous K | 1.13561 | 1.09738 | 1.15094 | 1.11219 |
| B, zero V / clamped KE | 1.11908 | 1.09517 | 1.12538 | 1.16779 |
| Source blocked, either recipient | 4.08255 | 2.65458 | 2.43424 | 1.62581 |

## Supported causal comparisons

The conditional layer-3 effect replicates on the fresh panel: B-minus-E adds 1.513–1.965 nats/token for natural starts and 2.333–4.616 for BOS starts. Every group interval excludes zero. Clamping complete E/B source K/V states reproduces their donor outputs at scored later positions, with exact E/B recipient closure. This validates the tested source-state transmission pathway; architectural closure alone is not a new mechanism discovery.

**The value donor carries the large loss effect in the tested factorial.** Replacing VE by VB raises NLL under either key donor in every input/start pair: 64 positive simple-effect observations across the two key settings, using 32 matched pairs. All 16 four-passage group intervals exclude zero. Group mean value effects span 1.473–2.066 nats/token for natural starts and 2.052–4.512 for BOS starts. Changing keys with VE held fixed has much smaller loss effects on this panel, but they are not universally negligible.

Keys remain context-dependent. With VE, natural-start group key effects range from −0.02936 to +0.05521 nats/token; the Wiki interval is positive and the other three include zero. Under BOS, all four key-effect intervals are positive, with means +0.10422 to +0.31569. With VB fixed, natural Hobbes and code have **negative** key effects with intervals excluding zero, whereas all four BOS groups have positive effects. Thus “E keys are always beneficial” and “keys are irrelevant” both fail as descriptions of these results.

The signed interaction is `I = NLL(KE,VE) − NLL(KE,VB) − NLL(KB,VE) + NLL(KB,VB)`. Its mean and 95% paired bootstrap interval are:

| Corpus | Natural I [interval] | BOS I [interval] |
|---|---|---|
| Wiki | −0.06122 [−0.22620, +0.07256] | +0.03208 [+0.00301, +0.06115] |
| DuBois | −0.07935 [−0.23835, +0.02674] | +0.06802 [−0.00026, +0.13363] |
| Hobbes | −0.21002 [−0.32658, −0.13706] | +0.15858 [+0.07855, +0.22125] |
| Optimizer code | −0.52370 [−0.81690, −0.23050] | +0.34860 [+0.21122, +0.45525] |

These differing signs are incompatible with a single signed additive-mediation account across starts. The small-sample intervals do not prove absence of interaction in groups whose intervals include zero.

**Zero values and source blocking have different effects.** In all 32 input/start pairs, source blocking has greater loss than value zeroing for both recipient worlds and both key settings. All 32 group intervals for those four contrasts exclude zero. For E with KE clamped, the blocked-minus-zero mean difference is +0.434 to +1.318 nats/token for natural starts and +0.331 to +1.294 for BOS starts. The explicit causal-mask references have zero scored per-token loss drift and exact prediction agreement in all 64 comparisons; therefore raw and reference-adjusted contrasts agree in this run.

This supports preserving source attention allocation while avoiding its B-world value contribution as a useful intervention in this setting. It does **not** establish that source values encode useful linguistic content: literal zero values produce much smaller loss than VB. Nor does it establish that only attention normalization matters: later queries and non-source computations evolve after each intervention, and values/keys were changed jointly across eight layers.

## What remains unproven

Neither donor repair nor zero values restores the native outputs. With KB/VE, native argmax agreement is 77.79–84.75% for natural starts and 64.66–83.33% for BOS starts. Even E with zero V and clamped KE has only 90.29–93.58% natural and 85.20–91.27% BOS agreement. Aggregate loss close to the native level is not output equivalence, faithful generation or restoration of the original internal trajectory.

There are only four passages per corpus, with both starts derived from the same passages. The fixed edge-of-source panel is fresh but not representative sampling. Hypotheses were chosen using earlier panels. Bootstrap intervals use 2,000 draws with seed 20261021, have no multiple-comparison adjustment and do not establish population generality. The data cover one checkpoint, one context length and one all-layers source intervention family. No individual head, layer, value direction or minimum sufficient circuit has been localized by this batch.

The audit independently checks source writer vectors and retained E/B donors. Patched K/V tensors and later-position projections are not retained separately for every recipient: their exact copies/locality depend on recorded runtime assertions and independently exercised hooks. KL reductions can be reaggregated, but full logits were not retained. Parameter immutability is checked against recorded digests/version counters; no model was loaded to recalculate them. These are explicit audit limits, not failed checks.

## Highest-value next discrimination

Test whether VB is harmful because of its magnitude, its direction, or a small layer/head subset. Freeze source keys to KE, retain zero/VE/VB anchors, and compare VB normalized to VE's per-head norm with VE enlarged to VB's norm and matched random-direction controls. Use separate development/calibration and fresh held-out passages, both natural/BOS starts, fixed later token IDs, and preserve NLL, KL and agreement. First localize candidate layers/heads on development data so a held-out test targets implicated sets without selecting on evaluation outcomes.

This directly tests the strongest current competing explanations: suppressing an excessive source value vector versus preserving a particular harmless direction. A zero-value/retained-key result alone cannot choose between them. Subsequent query/attention-weight interventions should determine whether the retained source allocation is itself required, before proposing a simpler replacement for a larger computation.

The inference manifest records **40.362 seconds**, **616 forwards**, **0.708 GiB peak allocated / 0.736 GiB reserved GPU memory**, and **2.015 GiB process RSS**. The independent audit remained near 1.1 GiB CPU RSS with no GPU use. No failure handler was triggered and no blocker remains for reporting this bounded result.
