# Report 19 — Native versus value-lesioned calibration of the GPT-2 early-write intervention

**DRAFT — scientific interpretation and presentation pending final review.**

The calibration choice changed the estimated removal-minus-orthogonal-control later-loss contrast by small amounts in this finite experiment, while most confidence intervals remained broad relative to those effects. Calibration defines manual gate edits; it does not train or change GPT-2’s weights. This does not establish equivalence, absence of a calibration effect, or irrelevance of attention bias. The comparison asks a narrower question: when an intervention is fitted on the same four calibration texts, how does retaining versus removing source-token values during calibration alter its later-loss effect in an otherwise native GPT-2 evaluation?

Across the 64 prespecified primary comparisons, 51 nominal intervals included zero. The within-cell removal-minus-orthogonal-control contrast had both positive and negative means across source/start strata. The largest absolute mean change in D_CC − D_PP between the endpoint calibration cells was about 0.0265 millinats per later target. This is a change in the removal-minus-control contrast, not the raw removal-arm loss. These are descriptive results from four passage clusters per stratum, conditional on one fit; they are not an equivalence test or a population estimate.

## What was compared

Four fixed calibration texts were run in four worlds. **PE** removed source-token values and used an explicit causal mask; **PN** used the same value lesion with the native mask route. **CE** retained native values with the explicit mask, and **CN** retained native values and the native mask. The PE intervention uses the original frozen F maps and directions, not a convenient contemporary refit. Its 64 raw allocation arrays, maps and directions passed the historical continuity gate. Across all eight observed layers and four texts, the PE/PN and CE/CN mask-route comparisons were byte-identical for 128 stored allocation-logit/mass arrays in total.

At a non-source query q, b is a 12-coordinate vector of head-wise log-odds for non-source versus source attention mass. Subtracting log(q) removes the declared query-count reference, and α centers that residual. The leading component u and its orthogonal control v therefore live in the 12 head-wise gate-logit coordinates at layer 10. They are not principal components of whole hidden states or a count of causal mechanisms. The scalar projection onto u determines the removal and control proposals. For a given native incoming state, the source-value vector and conditional non-source value mixture stay fixed while this proposal changes their allocation gate.

Only PE and CE defined the primary evaluation parameters. Each supplies a center vector, α, and a direction pair: the first principal component, u, and its declared orthogonal control, v. The four cells separate their sources: PP uses the PE center and pair; PC uses the PE center and CE pair; CP uses the CE center and PE pair; CC uses both CE components. Changing the direction pair changes both u and v, so a cell difference cannot be attributed to the principal-component direction alone. PN and CN were mask-route diagnostics, not extra evaluation cells.

At layer 10, the PE–CE center distance was 0.11718032 in L2 norm; the corresponding u and v cosines were 0.99992906 and 0.99998459. These descriptive geometric similarities do not determine functional equivalence. Calibration retained the same texts, and the native-value world can already differ upstream at layers 4–9. A layer-10 fit difference therefore does not show that layer-10 values caused that layer’s own incoming queries or keys.

{{SCHEMATIC}}

## Intervention, panel and uncertainty

Evaluation kept source values native and used the native mask route throughout. Each cell first generated actual full removal and orthogonal-control proposals, including their actual attention output projections. The two proposals were then interpolated after c_proj to the smaller of their per-query full-change norms. Only earlier query rows 1–126 received the matched write; all other rows copied that variant’s baseline write. Later scored targets were 128–511, predicted from queries 127–510. Query 511 remained a control position without a next-token target.

Matching was within a cell and input variant, not across calibration cells. The largest actual paired relative RMS norm gap was 4.1807359×10⁻⁸. Nevertheless, the CC/PP active-target RMS ratio ranged from 1.00044852 to 1.01219647 across 64 main/suffix records. Different directions and different natural targets can yield different functional doses even when paired Euclidean write norms match closely. Equal vector norm is not equal computation or equal loss sensitivity.

The panel contained 16 passages from four sources—Wiki, utilitarianism, ethics and code—with natural and beginning-of-sequence starts, giving 32 identities. All calls used 512 positions with key/value caching disabled. The BOS variant changed token 0 only and preserved the other 511 tokens. Deterministic preparation tested 1,090 candidates, rejected 1,074 and accepted 16. It retained the complete calibration-buffer and interval exclusions and checked an 841-hash experimental-history guard. That recorded-exposure guard is not a pretraining-overlap proof, and these passages are not a random sample of a broad task population.

The study completed 16 calibration forwards and 1,160 evaluation forwards: 576 main, 576 rotated-suffix controls and eight native/identity anchors. Two planned 580-forward segments completed their source, parameter and precision guards. Segment A took 352.927 seconds with a peak of 2,332,804 KiB; B took 358.670 seconds with 2,164,244 KiB. Peak allocated GPU memory was 1,142,526,464 bytes. These are execution resources for this study, not a speed comparison or a generation benchmark.

## All primary results

For a cell, **D = removal − orthogonal control** in signed later negative log-likelihood. Positive values mean the removal had higher loss than its control; negative values mean lower loss. Four D contrasts plus four fixed factorial contrasts produce eight estimands in each of eight source/start strata. The center main effect averages the center change across direction pairs; the direction-pair main effect averages the pair change across centers. The joint endpoint change equals the sum of those two main effects, and the interaction is the remaining fixed difference of differences. These algebraically related comparisons are deliberately all reported, not independent opportunities for a “win.”

Intervals use 2,000 shared bootstrap resamples of the four passage clusters, with seed 20261021. Starts, arms, views, metrics and contrasts share the index array. These paired nominal intervals are conditional on the fixed calibration and canonical FP32 scorer; numerical error is not included in the bootstrap. The nominal 95% intervals are unadjusted and correlated, and omit calibration-fit uncertainty. No interval crossing rule was a scientific success criterion. Neither a zero-containing interval nor a small point estimate establishes absence or practical equivalence.

The four D families each had six positive and two negative stratum means, with 28 of their 32 intervals including zero. D_PP means ranged from −0.293120 to +0.941390 millinats per target; D_CC ranged from −0.296286 to +0.961002. The endpoint calibration shift, D_CC − D_PP, had a maximum absolute mean of 0.0264775 millinats. The center, direction-pair and interaction rows expose the full decomposition, but their small differences should not be ranked as resolved mechanisms.

{{FOREST}}

The appendix table retains all 64 comparisons in a fixed order, with mean and interval in millinats per later target. The accompanying CSV and JSON retain all four passage values in original nats as well as display units. The complete frozen analysis is also embedded as a download in the self-contained HTML, including every secondary contrast, norm summary and paired-start row. No new bootstrap, refit, significance test or selective secondary analysis is performed by this report builder.

## What the independent checks establish

The numerical audit checked 41,871 stored arrays and independently rebuilt the saved-statistic summaries and fixed interval calculations. The projection audit checked 19 correlated layer-10 learned projection relations, totaling 7,471,104 coordinates. The readout audit checked all 72 predetermined full-vocabulary rows from the 18 selected Wiki natural-start states. These are complementary checks of saved artifacts, not a second model replay, ordinary whole-panel reconstruction or a universal bound on GPU arithmetic.

For those 72 selected rows, the maximum saved-FP32 versus saved-logit-FP64 KL discrepancy was 4.76524×10⁻⁷ nats; the largest defined error-to-signal ratio was 0.166774. Eight exact-zero FP64 references had undefined ratios, recorded as null. The maximum NLL discrepancy was 5.48602×10⁻⁷ nats. A 10⁻⁴ consistency threshold is not a sensitivity proof for tiny interactions, and selected-row diagnostics cannot bound every reported passage or contrast. Signed tiny KL values remain signed rather than clipped.

Infrastructure corrections were preserved separately from scientific hypotheses. They included an input-audit count-key error, a CPU-witness acceptance count-sum error, missing execution-review provenance fields detected before array access, and later receipt/policy-filename mistakes after literal activation. The fixes did not repeat model forwards or change scientific criteria. An earlier portable-workflow containment fix is likewise documented infrastructure history. Such events are neither negative experimental outcomes nor evidence that every historical transient failure operand was saved.

## Established context, limits and proposed next step

Extreme activations, their persistent effects, attention allocation as a gate, and GPT-2 attention-bias circuits already have primary literature. The role of this study is a calibration control of the GPT-2 transfer branch, not a new discovery of a GPT-2 super weight. The candidate increment here is this controlled, finite calibration-center/direction-pair comparison with actual early-write matching, explicit uncertainty and saved-artifact checks. It does not establish a unique or universal mechanism, intrinsic principal-component importance, task necessity, causal mediation fractions or priority over that earlier work.

A proposed next step is a sign-balanced ensemble of multiple orthogonal controls, evaluated at a common per-query target based on actual post-c_proj proposals, with a fixed native C fit and fresh held-out texts. Independent calibration texts would follow. No call count is an approved schedule: the components, evidence layout and resources still need prospective review. No implementation or execution of that next experiment is authorized by this report.

Aggregate presentation can be rebuilt from the supplied public JSON bundle. Portable model reproduction and the public numeric export remain pending separate acceptance. A second full 776-forward T22 replay has not been performed. This report is a draft account of the completed T23 study, not a claim that those additional reproduction steps are complete.

{{LITERATURE}}

## Complete primary appendix

{{PRIMARY_TABLE}}

{{ARTIFACTS}}

{{PROVENANCE}}
