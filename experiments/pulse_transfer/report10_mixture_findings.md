Separating non-source attention mass from its conditional value mixture exposed opposing effects of query-bias removal. In this fixed zero-source-value experiment, the bias-removed mass donor generally improved loss on books and code, while the bias-removed mixture donor worsened it. The result is a conditional factorial finding, not an additive mediation account of the native model.

This batch used native upstream GPT-2 computation, with **no E/B coordinate or writer lesions**. In zero-based layers4–11, source0 V was set to zero. Donor **P** retained learned query bias; donor **R** removed it from Q at positions1+. For each treated layer/head/query, non-source mass was `a=sum(j>0,pj)` and conditional non-source mixture was `u`. Hence attention context was approximately `a*u`, as checked against native SDPA. The intervention delivered the saved mixture-donor context `c_D*(a_G/a_D)`, copying diagonal donors exactly. Query0 remained untouched. Donors came from their own unpatched SDPA forward trajectories under the assigned Q/V conditions on the same input; crossed trajectories need not correspond to any native run.

Use **mass-first notation** throughout: `Y_PR` means P mass with R mixture; `Y_RP` means R mass with P mixture. All four pairs were delivered into both P and R recipients. The recipient duplicates produced exact matching outputs and final-hidden hashes, so they are controls, not additional observations.

The eighth fresh panel contained four 512-token passages per corpus, each with natural/BOS starts and identical scored suffixes. Actual used spans excluded all twenty-eight earlier passages per corpus. Every arm scored targets2–511. Table ranges span four corpus means, each expressed as the exponential of paired mean NLL difference from its own original native C baseline; they are not confidence intervals.

| Condition | Natural PPL ratio | BOS PPL ratio |
|---|---:|---:|
| `Y_PP`: P mass, P mixture | 1.003–1.020 | 0.995–1.022 |
| `Y_RR`: R mass, R mixture | 1.014–1.130 | 1.011–1.130 |
| `Y_PR`: P mass, R mixture | 1.087–1.259 | 1.081–1.236 |
| `Y_RP`: R mass, P mixture | 0.910–1.017 | 0.882–1.015 |
| P query bias, source blocked | 1.710–3.620 | 1.653–3.645 |
| R query bias, source blocked | 1.837–3.580 | 1.778–3.629 |

At either fixed mixture, replacing P mass with R mass improved mean NLL in Du Bois, Hobbes, and code at both starts; all twelve corresponding paired intervals were below zero. Wiki's mass effects were slightly harmful with intervals crossing zero. Conversely, replacing P mixture with R mixture worsened every corpus/start mean at either mass, with fifteen of sixteen intervals above zero; Hobbes/BOS at P mass was the exception. `Y_RP` beat the complete R donor in all eight corpus/start means. `Y_PR` was worse than the complete P donor in all eight, and worse than R in six, excluding Wiki.

The full interaction, `Y_RR−Y_RP−Y_PR+Y_PP`, had mixed signs: 17 positive and 15 negative individual-window values. Three group intervals excluded zero—positive Hobbes at both starts, negative natural Du Bois. These results do not justify a universal sign, an additive mediation percentage, or a unique repair pathway. Intervals use paired n=4 windows, 2,000 bootstrap resamples and seed 20261021, without multiple-comparison correction. Natural/BOS starts are paired variants; the experiment does not estimate broad population effects.

R donors had greater non-source mass on average: 0.5125→0.7279 naturally and 0.5111→0.7267 with BOS, uniformly averaged over scored queries/heads/layers/passages. R/P mass ratios exceeded one in 95.12% of scored entries. This is not universal at every head/query. Median gain was 1.442 and 95th percentile 4.916; maximum gain was **6.79 million**, with minimum positive mass **3.31e-12**. No mass floor or clipping was used. Direct conditional-softmax diagnostics remained within the fixed 1e-4 error bound, including these extreme cases.

Improved loss did not restore native predictions. `Y_RP` agreement was **80.93–90.10%** naturally and **80.05–90.00%** with BOS, with KL 0.0824–0.1560 and 0.0823–0.1686 nats. In comparison, `Y_PP` agreement was 94.85–97.35% and 94.56–96.96%. Same-input donor patches are diagnostic interventions, not usable input-independent replacements or evidence of faithful generation.

Independent CPU review reproduced all **520 records, 128 summaries, 256 contrasts and 64 paired summaries**, plus 840 saved checks and 3,840 mixture-layer checks. All 328 specified output/final-hidden-hash closures passed; native/explicit-mask token losses had zero drift. Thirty-two donor files were checked sequentially, including exact mixture arithmetic. All twenty frozen Q/K/V/context states reconstructed in FP64 with maximum native-context error **8.11e-6**; all sixteen patched snapshot contexts exactly matched the specified donor arithmetic. The largest runtime conditional-mixture discrepancy was 3.62e-5, below tolerance. Final-hidden closures were checked from recorded hashes; full hidden tensors and full logits were not retained for independent regeneration.

The run took **55.54 seconds**, with peak GPU allocation **768,470,528 bytes** and process peak RSS **2,312,288 KiB**. Parameters remained unchanged. Independent review used no model/GPU and peaked at **1,062,016 KiB RSS**.

Query-bias/sink mechanisms, attention-output decomposition, and zero-valued allocation have close primary precedents. The focused prior-art review (`review/gpt2_query_mixture_prior_art.md`; see repository reviews) distinguishes those established ideas from this specific donor factorial without claiming priority. Supporting evidence: numerical review (`review/gpt2_query_mixture_numerical_review.md`; see repository reviews), audit results (`review/gpt2_query_mixture_numerical_audit.json`; see repository reviews), and replay code (`review/gpt2_query_mixture_numerical_audit.py`, private raw bundle).
