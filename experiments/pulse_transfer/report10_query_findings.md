Removing learned query bias reduced source attention allocation, but did not reproduce the damage caused by blocking source access. Its loss effects depended on the corpus, starting token, and preceding lesion. These results support a conditional role for query bias; they do not establish that attention allocation alone mediates the loss changes.

The experiment used the pinned GPT-2 checkpoint in FP32 with SDPA, on sixteen fresh 512-token passages: four each from Wiki validation, Du Bois, Hobbes, and optimizer code. Each passage had natural and BOS starts with an identical scored suffix. Actual used spans were disjoint from all twenty-four earlier spans per corpus. **C** denotes the native upstream computation; **E** erases source0 residual coordinates447/138 after layer2; **B** additionally blocks their source0 MLP writes at layer3. All query/value/mask interventions occur in layers4–11. The factorial sets source0 V to literal zero and crosses learned-query-bias subtraction at positions1+ with source access open or blocked. Source0 Q is unchanged by the query-bias hook; zero-value changes in earlier layers can still alter its later trajectory. A fixed signed-permutation subtraction supplies a per-head norm-matched control.

The table gives PPL ratios relative to each input's original native C baseline. Each range spans four corpus means, calculated from paired mean NLL differences; it is not a confidence interval. “Bias present” and “bias removed” both retain source access and set source V to zero. The blocked condition also has zero source V and retains learned query bias.

| Upstream world and intervention | Natural start | BOS start |
|---|---:|---:|
| C, no intervention | 1.000 | 1.000 |
| E, no additional intervention | 1.013–1.088 | 1.059–1.397 |
| B, no additional intervention | 4.793–13.248 | 12.544–198.979 |
| C, zero V, bias present | 1.005–1.017 | 1.003–1.018 |
| C, zero V, bias removed | 0.979–1.187 | 1.006–1.177 |
| C, source blocked | 1.793–3.738 | 1.789–4.146 |
| E, zero V, bias present | 1.037–1.081 | 1.035–1.202 |
| E, zero V, bias removed | 0.980–1.198 | 0.945–1.219 |
| E, source blocked | 1.795–3.839 | 1.812–4.617 |
| B, zero V, bias present | 1.040–1.100 | 1.082–1.214 |
| B, zero V, bias removed | 1.051–1.269 | 0.960–1.300 |
| B, source blocked | 1.795–3.839 | 1.812–4.617 |

In C with zero V and source access open, removing learned bias increased mean NLL on Wiki by 0.124 nats naturally and 0.136 with BOS, and on code by 0.165 and 0.160. All four paired bootstrap intervals excluded zero. Conversely, it improved natural Du Bois by 0.0375 nats, with its interval below zero. In B/BOS, Du Bois and Hobbes improved by 0.119 and 0.132 nats, respectively. The preregistered interaction, `(removed−present)_blocked − (removed−present)_open`, was negative for C Wiki/code at both starts but positive for C natural Du Bois. E/B conditions also produced mixed signs. There is no universal beneficial bias-removal effect, uniform interaction, or additive mediation fraction.

Source allocation was measured at the scored query positions1–510, which predict targets2–511. Uniformly averaging across those queries, twelve heads, eight treated layers, and sixteen passages, C zero-V source mass fell from **0.4865 to 0.2726** after bias removal naturally, and from **0.4881 to 0.2740** with BOS. The randomized subtraction left those averages at 0.4782 and 0.4796. Substantial source allocation therefore survived removal. These averages describe allocation, not causal importance. With V0=0, the source's weighted vector contribution was exactly zero regardless of its probability mass; retained mass instead changes the weighting of non-source values. Query-bias subtraction also changes the conditional mixture of those values, so mass changes alone cannot explain the outcome.

Near-native loss did not imply faithful outputs. C zero-V with bias present retained 95.49–97.89% argmax agreement naturally and 95.39–97.65% with BOS. Removing bias reduced agreement to **73.24–90.20%** and **74.02–89.75%**, respectively; mean KL from native predictions was 0.150–0.300 and 0.156–0.303 nats. The fixed randomized control had generally smaller open-arm effects on Wiki/code, but was not inert or uniformly less harmful. It is one matched direction, not a randomization significance test. All confidence intervals use four paired windows per corpus/start, 2,000 bootstrap resamples, and seed20261021; they are descriptive and unadjusted for multiple comparisons.

Independent review replayed all **776 records, 192 summary groups, and 672 paired contrasts**. Eight repeat/identity controls were exact on saved per-token loss, KL and argmax; native-versus-explicit causal-mask token losses had zero drift. Source Q/K/V equality across zero-V query/mask arms passed, as did all source-blocking and literal-zero checks. E/B blocked-source saved per-token loss, KL and argmax were exactly equal in all 96 input/query-mode comparisons, an expected architectural closure rather than a new mechanism. All twelve frozen raw attention snapshots independently reconstructed in FP64: maximum actual-context error was `1.01e-5`; the maximum runtime reconstruction error over all arms was `2.48e-5`, below the frozen `1e-4` tolerance. Saved bias vectors matched the checkpoint, and the random control seed and multisets were verified. Full logits were not retained, so the audit reaggregated saved losses/KL/predictions rather than regenerating them.

The 776-forward run took **74.52 seconds**, with recorded peak GPU allocation **768,470,528 bytes** and process peak RSS **2,628,888 KiB**. Parameters and version counters remained unchanged. Independent review used no model or GPU and peaked at **1,399,780 KiB RSS**.

This is a scoped functional extension of prior work, not discovery of GPT-2 query-bias or attention-sink mechanisms. Ran-Milo et al.'s [*A Mechanistic Account of Attention Sinks in GPT-2*](https://arxiv.org/html/2604.14722v1), already credited in Report8, studies the query-bias route. The unresolved question here is how much the present intervention changes allocation versus the conditional non-source mixture; the current factorial establishes effect modification by source access, not mediation or a unique circuit.

Supporting evidence: independent numerical review (`review/gpt2_query_source_numerical_review.md`; see repository reviews), audit results (`review/gpt2_query_source_numerical_audit.json`; see repository reviews), and replay script (`review/gpt2_query_source_numerical_audit.py`, private raw bundle).
