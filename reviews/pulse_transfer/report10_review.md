# Report10: independent scientific and rendering review

**Passed after three small clarifications.** The final report integrates both independently audited batches accurately, gives the important negative/heterogeneous results, and keeps its causal and novelty claims within the tested boundary. No remaining scientific-report blocker was found. Repository packaging/replay is a separate review.

Reviewed report SHA256 values:

- `REPORT10.md`: `691d07317d68bcecfed311821e3b44f287d071beee8f8c6fe8d8b5e7229a1fd2`
- `report10.html`: `2b03f250740b12eb03ebdce00e3558e44725b291a2d69dd02410a107125328f8`
- `build_report10.py`: `7686545e85e5fb5807153df772eaeae799a89641080e33e346eb1cc2c8b4c689`

Machine-readable evidence and all source/figure hashes are in [consistency audit](report10_consistency_audit.json), produced by the CPU-only audit script (`report10_consistency_audit.py`, private raw bundle). This review loaded no model and used no GPU. Raw-data scientific validation remains documented in the two individual numerical reviews; the integration audit checks report statements, summaries, plot inputs and rendering against those records.

## Scientific claims and numerical consistency

The report correctly separates the first experiment's C/E/B upstream worlds from the second experiment's native-upstream, literal-zero-source-V world. It identifies zero-based layers4–11, later-query Q-bias subtraction, unchanged query0 hook behavior,512-token teacher forcing, targets2–511, independent fresh panels, and paired natural/BOS suffixes. Same-input donor trajectories are distinguished from deployable or input-independent replacements.

The main finding is supported as a **conditional donor-factorial result**: at either fixed mixture, R rather than P mass improves the six book/code corpus/start means, with all twelve mass-effect intervals below zero. Wiki mass effects have slightly positive means with intervals crossing zero. R rather than P mixture raises every corpus/start mean at both masses; fifteen of sixteen intervals exclude zero. These are accurately described as opposing effects that can be obscured by the complete donor change. They are not stated as a universal improvement, additive natural mediation, a unique native mechanism or an exact preservation of predictions.

Mass-first indexing is consistent in prose, tables, diagram labels and analysis. The interaction is `Y_RR−Y_RP−Y_PR+Y_PP`; its17 positive and15 negative individual-window values and only three group intervals excluding zero are retained. The report also preserves corpus/start-dependent query-bias effects, residual source mass, non-inert random controls, near-native-loss versus argmax disagreement, and the Wiki exception to the favorable crossed arm. It does not convert a nonsignificant interval into proof of no effect.

The audit independently verified all **84 PPL table cells**, including range endpoints, across **five tables**, and both resource rows. The combined count is **1296 forwards**, with **130.05 recorded worker seconds**. Checkpoint ID/revision and FP32/SDPA specifications agree with both manifests. The numerical summary claims match the separately reviewed raw experiments:776/520 records;192/128 summary groups;672/256 contrasts;328 mixture-batch closure hashes and saved metric closures;840 saved mixture checks;3840 mixture-layer checks; and20 raw mixture snapshots with16 exact delivered-patch formulas.

The largest gain, about6.79million, and minimum mass, about3.31e-12, remain visible. The report says the frozen stability guards passed and gives their maximum error; it does not equate copied diagonal closure with proof that arbitrary rescaling is stable. It explains actual SDPA context rescaling, direct non-source probability summation, exact diagonal copying, independent conditional-softmax checks and the absence of clipping. Tiny errors are numerical diagnostics, not causal effect estimates.

The first batch's saved loss/KL/argmax closures are now described precisely. The second batch additionally hashes the full final normalized hidden tensor, and the report states that a later reviewer can verify recorded hash equality but cannot regenerate those hashes without the unsaved full tensors. Similarly, reaggregating saved KL is not presented as independent regeneration from full vocabulary logits.

## Figures, HTML and readability

All four scientific figures were visually inspected: labels and legends are readable, curves/points match the stated conditions, error bars are visible, and the donor-boundary diagram makes the two factors and fixed donor trajectories clear. The implementation's finite-precision formula is supplied in adjacent text, so the diagram's mathematical `a_G*u_D` notation does not conceal its actual SDPA rescaling.

The audit intercepted the builder's plotting functions without writing files and checked **56 error-bar points with their confidence bounds** against audited summaries. It also checked **48 source-allocation curve points** against the uniformly averaged head/passages data. The source-allocation plot is descriptive; the report explicitly states that these uniform means are not causal importance weights. The error bars use the declared paired bootstrap summaries rather than treating recipients or token rows as new independent observations.

The HTML body exactly matches the rendered reviewed Markdown. All **four PNGs are embedded byte-for-byte**, with alt text; there is no script. The viewport and horizontally scrollable tables are present for narrow displays. All **12 local report/evidence/figure links** resolve in the source report tree. Library rendering and repository-relocated paths are separate delivery checks.

Three changes requested in this review were applied before the final hashes:

1. Source0 Q is now described as unchanged by the query-bias hook, while acknowledging that earlier zero-V changes can alter its later trajectory relative to original C.
2. First-batch repeat/identity and E/B blocked-source closures explicitly identify saved per-token loss/KL/argmax, avoiding an unsupported full-output audit claim.
3. The copied prior-art note's Report8 link was relocated correctly, and future editorial wording was replaced with the actual comparison. The original historical prior-art note remains preserved separately.

The builder now reads its prior-art section from the root `report10_prior_findings.md`, allowing the portable staged builder to receive it without a private review-directory dependency. That source copy and builder hash are covered by the audit evidence.

## Novelty calibration checked against primary sources

I reopened the pinned primary versions on21October2026. Ran-Milo, Ofek and Mendel's paper explicitly studies learned query bias, transformed positional signals and key-projection structure. Its setup/interventions use40-token prompts, a latter-half allocation metric, an analysis layer range excluding the first three and final block, and query-bias nullification in every layer. Its limitations acknowledge residual sinks and unresolved functional benefit/cost. The report correctly treats these as established neighboring mechanisms and states why its estimand is not a matched replication. [Primary paper, sections3.2 and5](https://arxiv.org/html/2604.14722v1)

Sun et al. explicitly decompose source-associated and other weighted-value contributions in section4.2/Eq2. Section4.3 and AppendixB.4 distinguish learned extra K/V vectors from three alternatives: zero K/V allocation, appended Q/K features, and constant additive values. The latter three did not eliminate massive activations in their training experiments. The report credits these precedents and does not recast their training comparison as a test of the present fixed-checkpoint factorial. [Primary paper, sections4.2–4.3 and AppendixB.4](https://arxiv.org/html/2402.17762v2)

These targeted checks support the report's bounded novelty discussion. They are not an exhaustive literature/repository search, and failure to find this exact factorial in the inspected sections is not proof of priority. The final text explicitly avoids such a claim. Elementary output regrouping and zero-valued probability allocation are appropriately credited rather than introduced as discoveries.

## Remaining limits and next work

The small fixed panels, four-window bootstrap intervals, multiple comparisons, one random control direction, shared corpus sources and teacher forcing constrain generalization. Loss gains do not restore native prediction fidelity. The all-layer donor replacement uses trajectory information and does not establish efficient deployment, whole-network sufficiency or a universal circuit. Those limits are stated clearly.

A prospective calibrated gate-subspace experiment is documented separately; it is not blended into Report10's findings. Export hygiene, portable replay and a resulting local commit must be verified in their own completion step. No model inference or account-mutating external action occurred in this report review. The previously denied GitHub action remains unexecuted and was not retried.
