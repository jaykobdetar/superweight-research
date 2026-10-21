# Independent Llama replacement numerical review

Status: passed. Evidence: `llama_replacements_numerical_audit.py/.json`. CPU-only, no model/GPU load; peak RSS809,360 KiB (about0.77GiB).

All528 forwards (eight calibration plus520 evaluation rows) match the frozen design. All expected conditions, eight full-array repeat/oracle anchors,32 clean-KV donor closures and32 pooled-mean-KV C/N scored closures replay. The320 projection-patch records and64 mean-pulse checks have correct counts, zero patch errors and declared applied activation. All128 summaries,128 individual contrasts and32 paired summaries reproduce from saved per-token arrays. Model hashes/version counters remain unchanged, and source hashes match the prelaunch independent review.

Calibration means replay exactly:156 pooled/Wiki/Hume per-layer K/V vectors were fitted in FP32 then cast once to BF16. All52 seeded permutation/sign controls replay exactly and preserve each mean vector's absolute-value multiset and BF16 dtype. FP32 norm reduction after permutation differs by at most1.91e−6; that is reduction-order rounding, not a change in the mathematical norm. The signed activation mean is exactly−426 in BF16.

Private-source tokenization, hashes and offsets were independently reconstructed. All eight calibration passages are disjoint from original development used spans; all16 evaluation passages are disjoint from both previous evaluation panels and each other. The eight calibration first tokens are distinct. Two natural evaluation windows share one calibration identity; the other14 natural windows each have a distinct unseen first token. BOS is one further unseen identity repeated across16 suffixes. The two shared-identity clean K/V donor sets exactly match their calibration donor states. This distinguishes source-state coverage from the number of downstream contexts.

Every D source output retains coordinate588 exactly, and every other source coordinate equals the corresponding full-column-deletion N output exactly. R and N give the same selected-coordinate output. Mean-pulse row/full modes also give the same selected-coordinate output, with all other coordinates unchanged in row mode. Generic saved `a0`/`source_contribution` remains the native potential contribution; use the frozen−426 activation and `pulse_checks` for the applied mean-pulse intervention. Later-position projection locality relies on runtime assertions and synthetic/static checks, as full later-token projections were not retained.

Across the four corpus means:

| Condition | Natural PPL ratio | BOS PPL ratio |
|---|---:|---:|
| Selected row R |1.0021–1.0224|1.0012–1.0244|
| Complement D |1.6849–2.0504|1.7484–2.2202|
| Whole column N |1.4085–1.5036|1.2061–1.3600|
| Pooled mean K/V |0.998479–1.001467|1.001892–1.008919|
| Wiki-only mean K/V |0.999375–1.001154|1.002622–1.009164|
| Hume-only mean K/V |0.998529–1.000862|1.001571–1.007326|
| Mean activation, selected row |0.997795–1.003499|0.993791–1.001756|
| Mean activation, full output |0.997793–1.001781|0.994713–0.998589|
| Signed-permuted mean K/V |5.4913–9.4905|5.4034–8.8843|

Pooled mean K/V is a useful approximation in this fixed panel, not exact fidelity. Natural argmax agreement is97.941–98.676%, KL0.001013–0.001484; BOS agreement96.814–98.137%, KL0.002286–0.002802. Across the14 unseen natural identities, descriptive aggregate PPL is0.999692, agreement98.2213%, KL0.001280. Across16 BOS suffixes it is1.006210,97.3652%,0.002556. These subgroup aggregates are descriptive, not newly selected confirmation tests or confidence intervals over independent source identities.

Both K-only and V-only means also retain near-native average loss; the pooled K-only natural range is0.998298–1.000604 and V-only0.998717–1.001858. BOS V-only is somewhat worse (1.000798–1.011776). Mixed replacement remains conditional because the unfixed complementary projection changes endogenously.

The prespecified complement interaction `N−R−D+C` is negative in every one of32 inputs and all eight group confidence intervals have negative upper bounds. N is less harmful than D in every input. The selected coordinate's contribution therefore does not simply add to the complement's loss effect: retaining it while deleting the rest can be more damaging than removing the whole column. This establishes conditional nonadditivity, not a semantic-content decomposition or an explanation based solely on squared weight energy.

One fixed random direction control strongly harms loss despite norm matching. This supports direction dependence for that comparator, not optimality of the arithmetic mean or a universal low-rank claim. Successful mean activation and K/V replacement remain related but distinct interventions. No trained replacement architecture, cached speedup, long-context equivalence or free-running-generation fidelity was tested here.

Scoring covers targets2..511; predictor0 is uniformly excluded for causal closures. Four passages per corpus, one checkpoint, natural/BOS starts, BF16/SDPA and fixed512-token teacher forcing limit generalization. Near-one NLL can coexist with altered outputs. Stored KL reductions were reaggregated, not independently rebuilt from absent full-vocabulary logits. All reporting should retain these limits and the explicit source-identity coverage.
