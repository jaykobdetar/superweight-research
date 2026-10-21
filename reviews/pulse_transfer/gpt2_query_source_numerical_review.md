# GPT-2 query bias × source access: independent numerical review

**Passed.** The completed batch matches its frozen design and saved numerical summaries. No correction to the results is needed. This review used saved tensors and eight small native bias vectors, never loaded a model, and used no GPU. Peak audit RSS was 1,399,780 KiB (about 1.34 GiB).

Evidence: audit script (`gpt2_query_source_numerical_audit.py`, private raw bundle), [audit results and allocation summaries](gpt2_query_source_numerical_audit.json), [prior implementation review](gpt2_query_source_implementation_review.md). Reviewed run: `gpt2_query_source_run`, runner SHA256 `c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5`.

## Numerical and provenance checks

- All 776 window records are present and unique: 24 conditions × 32 inputs plus eight repeat/identity anchors. All 192 four-window summary groups, 672 paired contrasts, and 168 paired bootstrap summaries independently replayed. NLL/KL/agreement reaggregation error is exactly zero. Targets are tokens2–511 in every arm, including the attention summaries' matching query positions1–510.
- All eight raw repeat/identity loss, KL, and prediction closures are exact. The 96 native-versus-explicit causal-mask references have zero token-loss drift. The 680 saved checks and 776 operation records pass. An additional independent check finds E and B source-blocked results exactly equal for all 32 inputs × three query modes, as expected: those worlds first differ in source-only L3 MLP writes, which cannot reach scored later positions after all L4–11 source access is blocked. This does not imply a corresponding C-versus-E closure, because their earlier L3 attention histories differ.
- Independently checked 4,608 source-Q/K/V layer equalities across query/mask arms, 6,984 exact raw source residual additions, all 6,208 layer diagnostics, and all 74,496 head-level summary rows. Source values and their weighted contributions are exactly zero in every zero-V arm. Blocked source probabilities are exactly zero at later queries; query0 retains unit source probability and zero non-source contribution.
- All twelve preregistered Wiki-natural-C snapshot states independently reconstruct in FP64 from their saved post-intervention Q/K/V and masks. Maximum discrepancy from actual native attention context is `1.0142341e-5`; maximum source-probability discrepancy is `6.9826692e-7`. Source-contribution norm error is zero; maximum non-source-contribution norm discrepancy is `9.6462798e-6`. Maximum runtime FP32 reconstruction error across all arms is `2.4795532e-5`, below the frozen `1e-4` threshold. These are numerical reconstruction controls, not exact bitwise SDPA identities.
- Native Q-bias arrays match the local pinned safetensors checkpoint. Seed20261024 independently recreates every permutation/sign array, and each randomized bias exactly retains its original per-head absolute-value multiset. No full matrix/model loading was needed.
- All source hashes, parent manifest, condition list, and bias-control file hashes match. Repeated tokenization/span preparation reproduces all sixteen fresh windows and the complete saved data manifest, with disjointness from twenty-four earlier actual used spans per corpus. Natural/BOS arms share scored suffixes. Parameters and version counters were unchanged according to the completed run. Recorded research runtime is 74.5175 seconds, peak GPU allocation 768,470,528 bytes, and process peak RSS 2,628,888 KiB.

## Findings and negative results

Ratios below are exponentiated within-window NLL differences from each input's original native C baseline, then averaged in log space within corpus/start (four windows). Ranges span the four corpus means; they are not confidence intervals.

| C-world condition, source V=0 in L4–11 | Natural PPL ratio | BOS PPL ratio |
|---|---:|---:|
| Learned bias present, source open | 1.0046–1.0165 | 1.0028–1.0182 |
| Learned bias removed, source open | 0.9792–1.1873 | 1.0056–1.1772 |
| Random matched subtraction, source open | 1.0092–1.0440 | 1.0041–1.0445 |
| Learned bias present, source blocked | 1.7929–3.7385 | 1.7893–4.1459 |
| Learned bias removed, source blocked | 1.9937–3.8135 | 1.9250–4.0081 |

Learned-bias subtraction clearly changes source allocation. Averaging equally over the four corpora, four windows, eight treated layers, twelve heads, and 510 scored queries in C, natural source mass falls from **0.486506 to 0.272559**; BOS falls from **0.488072 to 0.273953**. The randomized subtraction leaves those means at 0.478180 and 0.479557. These are descriptive uniform averages, not a weighting by causal importance. The effect is not source elimination: substantial source mass remains, and even the L4 natural mean is 0.0820 after removal versus 0.3590 before it. Later treated layers retain larger mean mass.

The loss effects are heterogeneous. In C with source access open, native-bias removal raises mean NLL on Wiki by 0.1239 nats (natural) / 0.1356 (BOS), and on optimizer code by 0.1654 / 0.1602. Those four paired bootstrap intervals exclude zero. It improves natural Du Bois by 0.0375 nats, with its interval excluding zero; Hobbes and BOS Du Bois intervals cross zero. In E, open removal effects are evenly split in sign across the 32 individual inputs; improvements in book-text groups are not merely one outlier. In B/BOS, Du Bois and Hobbes improve by 0.1189 and 0.1318 nats, respectively, with both intervals below zero.

The preregistered interaction is `(removed−present)_blocked − (removed−present)_open`. It is negative in C Wiki and code for both starts, but positive in C natural Du Bois. E/B and BOS do not yield a universal sign: E/BOS code has a positive interaction of 0.0565 nats, whereas B/BOS code's 0.0018 interval crosses zero. Thus the experiment supports context- and lesion-dependent effect modification by source access, not a uniform mechanism or an additive mediation fraction. Query-bias removal also has effects with source blocked, especially on code, confirming the intervention is not confined to source attention.

The randomized control generally has much smaller open-arm NLL effects than real bias removal in Wiki/code, but is not inert and is not uniformly dominated in every corpus/world. A single fixed randomized direction is not a significance test. Four-window bootstrap intervals are descriptive, sensitive to these small samples, and unadjusted for the many reported comparisons.

Loss preservation must not be confused with output preservation. In C, open zero-V with learned bias present retains mean argmax agreement 95.49–97.89% natural and 95.39–97.65% BOS. Removing learned bias lowers it to **73.24–90.20%** natural and **74.02–89.75%** BOS, despite modest or occasionally improved NLL. Corresponding mean KL ranges are 0.1499–0.2996 and 0.1558–0.3029 nats. Native B remains severely damaged on this new panel (PPL4.79–13.25 natural and 12.54–198.98 BOS), while literal source-V zeroing largely avoids that damage. This replicates the previous value contrast on new inputs; it does not establish a usable model edit or faithful generation.

## Highest-value next question

The remaining ambiguity is whether the bias-removal effect comes from changing the amount of non-source attention output, its conditional composition, or their interaction. With literal zero source V the exact algebra is `Y=(1−a0)u`, where `u` is the conditional non-source weighted value mixture. Query-bias subtraction changes both `a0` and `u`. Current allocation summaries alone cannot separate them.

A bounded next experiment could freeze this decomposition prospectively and cross **present-bias versus removed-bias donor gates** with the recipient's current conditional non-source mixture, in both directions, on fresh suffixes. Preserve zero source V and query0, include current-recipient reconstruction and same-world donor controls against native SDPA, and use conditional non-source softmax rather than unstable division by `1−a0` near zero. A full gate-plus-mixture donor control can test computational closure. Donor quantities taken from the same evaluation input would make this a diagnostic causal patch, not an input-independent approximation or generalization result. It would test the specific allocation-versus-mixture ambiguity directly; another head search or a claim that bias removal itself removes the sink would not.

## Scope and audit limits

This is a targeted extension of established GPT-2 query-bias/attention-sink mechanisms, not their discovery. All interventions concern positions1+ Q and source0 V/access in layers4–11, with earlier layers and non-source computations active. C/E/B conditions are distinct causal worlds; whole-layer or original trained-model necessity does not follow.

The saved per-token arrays support exact aggregation and closure audits, but full-vocabulary logits were not retained, so losses and KL were not independently regenerated from logits. Full raw Q/K/V/context snapshots cover only the twelve preregistered states; all-arm locality additionally relies on runtime assertions and the earlier synthetic implementation review. No pretrained inference or publication action occurred during this audit.
