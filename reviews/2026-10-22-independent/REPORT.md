# Independent review of superweight-research

## Executive assessment

**Verdict: a credible, unusually auditable set of finite causal case studies, with a strong OLMo core and a progressively narrower GPT-2 perturbation program. It is not yet a general explanation of super weights, a uniquely identified native GPT-2 mechanism, or a demonstrated compression method.**

22 October 2026. Reports 1-19 at commit **e59cfed2df13**. Repository-only review; no user-computer access, new model runs or repository changes. Unpublished local T24 work is excluded.

The most defensible findings are:

1. **OLMo's earlier-newline suppression has a specific causal route through L0/H14 and a conditional layer-0 MLP relay.** Development-only selection, reciprocal K/V interventions and natural-input cases make this more than an activation correlation, but not a complete or unique semantic circuit.
2. **A dominant scalar contribution is not an additive measure of neuron importance.** In OLMo, removing it can be catastrophically worse than removing the entire source-token neuron output. The selected Llama case reverses this ordering.
3. **GPT-2 conditionally depends on later MLP writes after source-coordinate erasure.** Regrowth is not restoration of the native trajectory or evidence of the same necessity in the intact model.
4. **The gate edits have scoped effects, including in native GPT-2.** Loss depends on direction, corpus, query support and intervention boundary. Equal write norm is not equal computation; better loss is not faithful output preservation.
5. **Both L11 K and V transmit chosen earlier perturbations into later outputs.** This does not establish intrinsic path importance, native-pulse necessity or a new attention/path-patching method.
6. **Report 19 is a limited calibration control.** Its small changes and zero-containing intervals do not establish equivalence. The report mostly says this correctly.

I found **no demonstrated arithmetic failure invalidating the principal published summaries**. All 2,364 files match the immutable Git tree; all 11 supplied CPU-only auditors pass; all 412 Python files parse. My separate implementation reproduces Report 19's 5,256 exported metric values and all 64 primary means and intervals exactly. This is independent summary reanalysis, **not independent model replication**.

The highest-value next move is a compact independent implementation/replay, stronger direction controls and genuinely new evaluation/calibration data. Choose whether the next scientific claim concerns the native OLMo mechanism or the distinctiveness of the GPT-2 gate direction, rather than extending another closely related factorial by default.

## 1. Scope, method and evidence strength

### Immutable acquisition

Full reviewed commit: **e59cfed2df13a1d3a6e6c607648a9d6844790654**. Tree: **ca708f97aca9f051a2a798b22036acd683c43a7a**. Remote main matched at acquisition and again at 20:17 UTC, with no intervening difference found.

The complete repository snapshot contains 2,364 files totaling 1,353,201,281 bytes. Every extracted file was checked against its Git blob hash and size; no mismatch occurred. Repository references below are commit-pinned, not moving main links. The review read the scientific reports, current handoff and reproduction guides, inspected intervention/analysis implementations, and executed read-only static and archived-numerical checks. It did not manually prove every function in all 412 Python files correct.

The consolidated OLMo report covers the first seven reports; the separate transfer and GPT-2 reports cover Reports 8-19. Historical failed attempts, the coordinate erratum, current public exports and proposed next work were considered. A proposal is not a result. In particular, the repository's T24 status is source preparation with no acquired parameters, selected inputs or model calls, and no inaccessible local continuation is included. [Current scope and next-step status](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/HANDOFF.md#L1-L58).

### Evidence ladder

The following distinctions should remain visible in any paper or presentation:

- **Integrity:** files match their committed bytes. This establishes what was published, not whether an experiment was correct.
- **Recorded-observation reanalysis:** retained loss, KL, agreement and norm arrays reproduce tables. This checks reductions, indexing and contrasts but does not regenerate omitted logits, predictions, attention primitives or model executions.
- **Independent primitive checks:** projection/readout checks can corroborate selected saved states. Public receipts for private checks are provenance commitments; they are not the underlying operands, and this reviewer did not independently inspect those private operands.
- **Same-environment replay:** the OLMo and early transfer replay records are valuable repeatability evidence. They are not independent implementation/hardware replication.
- **Independent experimental replication:** a new acquisition from verified inputs/model with an independent implementation and evaluation is not supplied by this review. Later full model replays remain explicitly unperformed.

The latest public guide makes these distinctions unusually well. Its public auditor explicitly reuses the frozen statistical helpers, rather than pretending to be a wholly independent analysis implementation. [Report 19 reproduction routes](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPRODUCE_GPT2_CALIBRATION_WORLD_COMPARISON.md#L1-L72), [public auditor scope](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_reproduction/experiments/pulse_transfer/audit_calibration_world_public.py#L1-L15).

### What I actually recomputed

Audit environment: Python 3.12.14, NumPy 2.3.5, CPU only, one numerical thread. This differs from the original pinned Python 3.12.3 / NumPy 1.26.4 environment; the listed checks nevertheless passed. No Torch/model execution was used for these checks.

All supplied entry points passed: audit_results.py, audit_transfer.py, audit_gate_results.py, audit_localization_results.py, audit_additive_results.py, audit_wiki_calibration_results.py, audit_native_value_results.py, audit_post_write_norm_results.py, audit_query_write_timing_results.py, audit_l11_key_value_donor_results.py and audit_calibration_world_comparison_results.py. Report 19's JSON-only metadata validation also passed. The earlier auditors cover 32,068 OLMo and 7,776 transfer records; records are repeated intervention observations, not independent passages.

My separate checker did not import the repository's analysis helpers. It read the latest NPZ scalar arrays directly, reproduced 1,728 window rows plus 24 anchor rows across three metrics (5,256 numbers), reconstructed all four passage values for each primary contrast, and recomputed their means and frozen PCG64 intervals. Every value matched exactly. It also independently reaggregated selected OLMo row/column and geometry comparisons from their window records. The script, machine-readable results and audit receipts accompany this review in the verification bundle.

What this **does not** independently verify: that each stored scalar loss or KL was originally computed correctly from the entire vocabulary, that an omitted activation really had the recorded fingerprint, or that the full model intervention would reproduce on a new machine. For the H14 pulse claims, the public pulse summaries, development-screen fractions and intervention code were inspectable, but the original per-input activation traces were not exported. I independently recovered the H14 selection ranking from the saved development fractions, not from a new activation reconstruction. Those limits must not disappear behind a large count of passing arrays.

## 2. Priority findings and corrections

Priority labels rank scientific risk and information value, not confirmed software defects. No critical numerical defect was demonstrated in the checks performed.

### P1: the native-mechanism bridge is still missing

The GPT-2 sequence is scientifically coherent as a series of controlled perturbations, but its connection to the original super-weight question becomes increasingly indirect. The selected GPT-2 scalar contribution was weakly consequential; the program moved to coordinate erasure, source-value lesions, then fitted gate directions, matched early writes and donor transplants. Each transition changes the estimand.

A path capable of transmitting an imposed perturbation is not thereby the path responsible for an ordinary native behavior. A result about a leading covariance direction is not automatically a result about the original massive coordinate. The repository usually acknowledges this; external summaries should not erase that qualification. The latest K/V paper is explicit about “chosen finite perturbations,” which is the correct level of claim. [Transfer selection and initial negative](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/TRANSFER.md#L11-L52), [Report 18 conclusion](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_L11_KEY_VALUE_DONOR.md#L3-L15).

**Correction:** organize the program as two linked but distinct case studies, not an increasingly complete single mechanism. Add an explicit causal diagram identifying the untested links from native source/pulse variation to the chosen L10 gate direction and onward to behavior. If the central goal remains super weights, testing one of those links should outrank further descriptive subdivision of the engineered edit.

### P1: more independent data, not more nominal comparisons

Most later strata have four passage clusters. Natural and BOS starts share the same text; arms, scoring views and contrasts reuse the same clusters; many reported contrasts are algebraically related. These are not hundreds of independent replications.

For Report 19, enumerating all 4^4 = 256 ordered bootstrap resamples produced exactly the same percentile endpoints as the saved 2,000-draw bootstrap. Thus I found **no bootstrap Monte Carlo error**. But deleting one passage flips the sign of **27 of the 64** primary means in at least one leave-one-out calculation. Sixteen of those are the four within-cell D contrasts in the same four strata: Utilitarianism/BOS, Ethics/natural, and code/natural and BOS. This is sensitivity, not proof that a mean is false.

The report correctly retains 51 zero-containing intervals, all four passage values and fixed-fit/multiplicity caveats. Stronger wording such as “the calibration effect is absent” or “the sign generalizes by domain” is unsupported. With a deterministic finite panel, bootstrap intervals are best described as **conditional resampling summaries**; nominal 95% population coverage is not established by this design. A four-pair sign-flip reference has minimum two-sided probability 0.125 under its symmetry assumption, which is another illustration of limited independent information, not a valid replacement test automatically licensed by this nonrandomized experiment. [Estimator implementation](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_reproduction/experiments/pulse_transfer/analyze_calibration_world_evaluation.py#L75-L109), [declared uncertainty](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_report/report19.md#L27-L37).

**Correction:** retain the full descriptive results but stop using interval counts as persuasive sample-size substitutes. For a confirmatory claim, predeclare one or a few estimands, a population/sampling scheme and practically meaningful effect or equivalence margin; acquire substantially more independent documents and independent calibration sets. More bootstrap draws will not fix four clusters.

### P1: tiny latest effects need estimand-level numerical resolution

The reported arithmetic checks are careful, but a consistency threshold is not a sensitivity bound. In Report 18, selected-row FP32-versus-FP64 KL error reaches 3.05994e-7 nats, with a maximum defined error/reference ratio of 0.441326. In Report 19, the corresponding maximum KL discrepancy is 4.76524e-7; selected-row NLL discrepancy is 5.48602e-7. These are sampled-row diagnostics, not upper bounds for every unsaved contrast.

Ten of Report 19's 32 calibration-factorial primary mean contrasts have absolute magnitude below 2e-6 nats. The smallest is about 1.89e-7. This does **not** show the contrasts are artifacts: averaging, cancellation and correlated errors matter. It shows that the available diagnostics do not justify a whole-panel precision guarantee at their scale. Conversely, these small arithmetic issues do not plausibly explain away OLMo's hundreds-to-thousands-fold PPL lesions or GPT-2's large conditional lesion effects. [Report 18 numerical scope](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_L11_KEY_VALUE_DONOR.md#L29-L39), [Report 19 numerical scope](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_report/report19.md#L43-L47).

**Correction:** in the next acquisition, compute stable FP64 scalar NLL/KL from the actual FP32 logits for every scored row, in bounded chunks, and retain paired scorer differences for the actual primary contrasts. Separately test upstream hook/backend sensitivity on a small prespecified subset. FP64 readout arithmetic alone cannot certify the upstream FP32 trajectory. Do not resolve tiny interactions merely by pointing to a passing 1e-4 tolerance.

### P1: one signed orthogonal control does not identify a special PC axis

The orthogonal control uses the coefficient of the principal component, not its own projection: s = (b - log(q) - alpha) dot u; PC removal subtracts s u; the control subtracts s v. This is a legitimate equal-logit-norm control, but it is **not removal of the orthogonal component**. The code implements the stated construction. [Gate map](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_reproduction/experiments/pulse_transfer/gate_native_value.py#L39-L50), [direction construction](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/experiments/pulse_transfer/gate_localization_inputs.py#L47-L68).

There is a useful algebraic warning: replacing u by -u leaves PC removal s u unchanged, but flips s v if v is held fixed. Canonical signs make the experiment reproducible; they do not make the sign of a single orthogonal control scientifically privileged. Loss differences from that control can reflect its orientation, ordinary directional sensitivity or curvature, not a distinctive PC mechanism.

The proposed sign-balanced control ensemble is therefore well motivated. However, matching all controls to the smallest proposal norm can shrink the common dose as controls are added. Report zero/tiny target frequency, actual dose distributions and dose-response behavior; a near-null result after shrinking everything is not evidence that directions are interchangeable. If practical, include both signs of the PC-axis perturbation as a secondary local geometry diagnostic, clearly distinguished from the directional operation called PC removal.

### P2: calibration-cell changes are compound interventions

Report 19 changes both u and its control v when changing the direction pair. It also matches write norms only within each cell, not across cells. CC/PP target RMS ratios range from 1.00045 to 1.01220. Therefore a “direction-pair effect” is not an isolated PC-axis effect, and the center/direction factorial does not hold every downstream dose fixed.

The PE/CE axes are extremely close: u cosine 0.99992906 and v cosine 0.99998459. This is a local robustness test for these same four texts, not a broad stress test over calibration populations. The report's own language correctly states these limits. [Calibration geometry and matching](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_report/report19.md#L11-L25).

**Correction:** preserve the result as a valid compound-intervention comparison. If the goal becomes isolating calibration orientation, hold control construction and a common actual-write dose fixed across cells, or explicitly estimate dose separately. Independent calibration texts are more informative than repeatedly changing the computational world on the same four texts.

### P2: public reproducibility is strong for summaries, incomplete for primitives

The public export intentionally omits model weights, token/prediction IDs, Q/K/V, contexts and most hidden/logit states. Lawful-input retrieval and rebuild routes are supplied; that is reasonable. But a public-only reviewer cannot independently regenerate the original KL/argmax metrics or all causal guards from exported scalar observations alone.

The amount of provenance is impressive, yet a hash, a PASS field and a prepared command remain different from executable replication. Historical source copies and certificate-binding layers also increase the burden of identifying the current scientific implementation. [Public export limits and optional replay](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPRODUCE_GPT2_CALIBRATION_WORLD_COMPARISON.md#L25-L72).

**Correction:** add a small canonical claim-to-evidence index and an independently written minimal reference implementation for a few decisive interventions. Preserve historical versions and failures; do not delete them to simplify navigation. The first independent model check need not replay every historical batch to be valuable, but a small sentinel replay must not be labeled a full-panel replication.

### P3: clean up current-versus-historical status without rewriting history

Some older prose still says no remote repository/push exists; Report 17 says publication is paused; the embedded Report 19 says public numeric export and portable reproduction remain pending. Current README, handoff and reproduction guides show later public packaging. These are understandable historical snapshots but contradictory if read as current status. Report 14 already uses a useful current-status banner.

**Correction:** add dated banners linking the authoritative current status. Preserve original historical assertions below them. Also rename “row deletion” in headline prose to “single-coordinate contribution deletion”: the experiment does not delete an entire matrix row or permanently remove a model weight everywhere. [Transfer historical footer](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/TRANSFER.md#L180-L182), [Report 17 footer](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_QUERY_WRITE_TIMING.md#L142-L154), [Report 19 historical status](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_report/report19.md#L51-L57), [precise lesion definition](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPORT.md#L24-L33).

## 3. Findings by research stage

### Reports 1-7: the OLMo core

**Stable source, dynamic use.** The fixed-source pulse/K/V replacements and failed fixed-gate simplifications support a useful distinction: a comparatively stable source state can participate in query-dependent downstream computation. Source-value zeroing and source blocking are not equivalent because blocking renormalizes attention over other tokens. The failed fixed low-rank/gate replacements and generation mismatches are important negative results, not loose ends to hide. The reported bypass did not establish a speed benefit. [Source and gate evidence](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPORT.md#L52-L80).

**Newline-history route.** L0/H14 was selected on development data before a separate 16-window constructed panel. Earlier newline donor K and V jointly suppress a later pulse; neither alone suffices in that recipient. In reverse, either can break suppression. Twelve eligible natural book/code cases corroborate induction under edge removal; there were no eligible Wiki pairs in that panel. The implementation freezes selection before held-out evaluation and includes H0, sham-edge and oracle controls. This is the strongest focused causal sequence in the repository. [Report](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPORT.md#L82-L107), [selection and interventions](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/experiments/pulse_continuation/newline_heads.py#L86-L112).

It does not establish a general “newline detector,” a complete memory mechanism or compensation for deleting the earlier pulse. The constructed prefixes deliberately alter tokenizer entries and prior newline history. Natural eligibility itself selects a subset. Those boundaries are scientifically acceptable when explicit.

**MLP relay and empty history.** The three-component donor factorial supports layer-0 MLP output M0 as a conditional carrier, with endogenous A1 amplifying the response. It does not assign a unique mediation fraction to that entire 2,048-dimensional output. L1/H12's empty-history involvement does not transfer the H14 newline-history state and must remain a separate observation. [Component evidence](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPORT.md#L109-L132), [component swap implementation](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/experiments/pulse_continuation/newline_components.py#L9-L45).

**Single contribution versus entire neuron output.** My independent window reaggregation recovered natural-start PPL ratios of 1933.92, 537.43, 648.00 and 596.21 for the scalar lesion, versus 1.445, 1.731, 1.033 and 1.011 for whole-column deletion across Wiki, Du Bois, Hobbes and code. These are enormous intervention differences, not a small-significance phenomenon. But the Du Bois whole-column mean conceals one individual ratio of 8.55; other natural-newline counterexamples are retained. “Deleting more can hurt less” is supported. “Whole-neuron deletion is harmless” is not. [Definitions and main comparison](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPORT.md#L134-L165), [window observations](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/pulse_continuation/column_factorial_run/windows.jsonl).

**Geometry.** The mixed normalized-source counterfactuals strongly support relative state geometry rather than scalar weight energy as part of the explanation. On Wiki, a clean alignment/radius/mean bundle with the row-lesion orthogonal direction gives PPL 1.791; the column-lesion bundle with that same row direction gives 324.729. A negative-axis-only replacement gives 1.307, not faithful reconstruction. Crucially, the so-called angle donor also carries mean and radius. These are multi-layer interventions at L2-15, not a proof that one angle at one layer alone is the cause. [Geometry interpretation](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/REPORT.md#L167-L196), [actual bundled construction](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/experiments/pulse_continuation/source_geometry.py#L38-L54).

**Coordinate correction.** The erratum is substantive and correctly handled. Yu et al.'s Table 2 lists both OLMo-1B entries in layer 1, including [1764,8041]; an earlier experiment actually tested the secondary entry in layer 2. Its measurements remain exploratory evidence about the executed layer-2 coordinate, not a replication of the paper's pair. The corrected same-layer experiment does not retroactively validate the old attribution. Removing one down-projection column cannot change another column's already-computed intermediate activation in that same MLP forward. This correction does not invalidate the primary [1,1764,1710] pathway. [Erratum](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/COORDINATE_ERRATUM.md), [original Table 2](https://arxiv.org/html/2411.07191v2#S3.T2).

### Report 8: transfer to GPT-2 and Llama

The transfer result is a valuable negative against assuming interchangeability, not an architecture-controlled experiment. GPT-2's candidate selection maximized typical contribution magnitude rather than causal importance, so its weak lesion effect cannot establish that GPT-2 lacks other super-weight-like dependencies. The selected Llama full column is consistently more harmful to remove than its largest scalar contribution; different energy concentration, precision, training, scale and tokenization confound architectural attribution.

GPT-2's E-by-MLP-write lesion interaction is strong and correctly called conditional. Source coordinate regrowth does not restore native values; later signs can differ. Llama's K/V factorial and approximate mean-source substitutions are useful, but K and V interventions do not have matched functional doses. A larger rescue from V is not an intrinsic V-importance ranking. [Transfer definitions and results](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/TRANSFER.md#L11-L140).

A conceptual simplification would help: in a reset, deterministic causal transformer, source-position-0 states depend only on token 0 and position 0, not the future passage. Repeated BOS passages therefore do not provide independent source-state variation. The report already checks source identities, including unseen identities for Llama mean replacement; those checks are essential. This deterministic fact does not explain away variation across different first tokens or the dynamic later query route.

### Report 9: conditional L3 dependence and value geometry

Localizing the post-erasure dependence to L3 MLP writes is a useful refinement. The claim remains conditional on E and on the chosen output coordinates; it is not a unique native repair circuit. The later source-value donor factorial distinguishes direction from a joint headwise norm profile. All four finite simple effects raising NLL on the tested pairs supports rejection of a purely amplitude-only or purely direction-only account within that intervention family.

Do not collapse the headwise profile into one layer norm: L11's total value norm can shrink while most heads grow. Fixed signed permutations sometimes improve loss and are not uniformly destructive. That negative is informative evidence against a uniquely privileged “correct” value direction under the lesioned recipient. [Report 9 geometry and counterexamples](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_FOLLOWUP.md#L149-L220).

### Report 10: query bias, allocation and non-source mixture

Subtracting learned query bias changes more than source allocation; it also changes the conditional mixture among non-source values. The all-layer donor factorial separates specified allocation and mixture donors and reveals cancellation. That is a valid intervention result, but it is not a unique natural mediation decomposition. The report correctly distinguishes zero source values from blocking source access. [Report 10 definitions and interpretation](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_QUERY_PATHWAYS.md#L17-L143).

### Report 11: gate PCA and failed execution-length checks

The high-variance directions fail as faithful scalar replacements: rank-one retention raises PPL roughly 15-29% relative to the value-lesioned reference, and rank six still changes a material fraction of argmax predictions. This is a meaningful negative against equating covariance compression with functional compression. It does not rule out every nonlinear or differently parameterized low-dimensional mechanism.

The two cross-length attempts failed declared numerical/causality checks and were not relabeled successful. Moving to fixed physical length 512 is a legitimate narrowed protocol, not evidence of length generalization. Very small non-source mass makes the old context-scaling formula ill-conditioned; finite outputs and passing local guards alone do not prove whole-model fidelity. [Failed attempts](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_GATE_SUBSPACES.md#L66-L158), [causal results and conditioning](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_GATE_SUBSPACES.md#L160-L264).

### Report 12: layer localization

The L10 Wiki-versus-book loss pattern is the substantive finite finding; it is not shared identically by all layers or all directions. L4's leading direction is overwhelmingly concentrated on one head yet has weak loss effects, an especially useful counterexample to treating a PCA loading as importance. Joint-minus-sum NLL is nonzero, but nonlinear downstream computation and the loss function already permit that. It does not identify feedback. [Findings, covariance and numerical scope](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_GATE_LOCALIZATION.md#L13-L107).

### Report 13: additive logits versus nonlinear loss

The N = J + R decomposition is mathematically sound. It distinguishes the loss of the actual joint intervention from the loss of a synthetic additive logit vector, and then from the sum of singleton loss changes. It is not a decomposition into biological-style independent causal mechanisms. R need not be nonnegative: the additive logits are an affine extrapolation, not a convex mixture. Opposing large J and R terms can cancel. This is valuable clarification of the earlier nonadditivity result, not discovery of a new feedback pathway. [Definitions](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_GATE_ADDITIVE.md#L31-L55), [implementation](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/experiments/pulse_transfer/gate_additive_readout.py#L41-L75).

The independent CPU readout differences and a few argmax/sign changes are disclosed rather than hidden. Those checks increase trust in the numerical account but do not upgrade teacher-forced loss gains to generation quality or speedup.

### Report 14: fresh Wiki calibration

Crossing old/fresh centers and directions usefully challenges dependence on one original fit. The persistent L10 sign pattern on that panel and the constant common-state center-offset identity are valid finite findings. However, the reference still has source values zeroed, and improvements relative to it can partly return loss toward the intact model. The report explicitly identifies this issue and proposes the native-value test rather than overclaiming. One pair of four-text fits does not estimate calibration-population stability. [Report 14 main findings and crossed fit](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_GATE_WIKI_CALIBRATION.md#L11-L108).

### Report 15: native-value factorial

This is an important substantive upgrade: the tested L10 gate operation harms Wiki and improves the selected books/code even with native source values. Thus the gains cannot all be repair of the source-value-zeroing lesion. Upstream, local and downstream value boundaries change effect size without creating the entire sign pattern. This is still a frozen edit, not native necessity or a generally beneficial transformation. Roughly 1.7-4.1% changed predictions and nonzero KL rule out faithful replacement. The interrupted acquisition and limited continuity replay are correctly separated from a complete independent replay. [Report 15](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_NATIVE_VALUE.md#L5-L111).

### Report 16: actual post-projection norm matching

Matching the actual residual attention write rather than just gate-logit or preprojection context norm is a meaningful control improvement. The PC-minus-control domain pattern survives that finite matching, so a naive “PC always has the larger vector perturbation” explanation is inadequate for these runs.

It is not exact amplitude isolation or equal functional dose. Residual mismatch is small but nonzero, pointwise relative mismatch can exceed the aggregate threshold, and no downstream sensitivity bound translates it into NLL error. More fundamentally, equal vector norms can align very differently with the downstream loss gradient. The report explicitly says this. [Matching and primary result](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_POST_WRITE_NORM.md#L11-L90), [finite-precision limits](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_POST_WRITE_NORM.md#L120-L132).

### Report 17: query-support timing

Unlike merely scoring early/later targets from the same intervention, this actually changes the intervention's query support. The protected later L10 state and L11 QKV make the cross-position attribution well designed: earlier write changes can affect later outputs through the final attention layer.

Early-only effects are small and can oppose later-only effects. The later support has 385 query rows versus 126 earlier rows, so larger later-only responses cannot establish intrinsically greater timing importance. These are different doses and supports. Positive mean KL establishes recorded distributional transmission, not a uniform signed loss benefit. [Support and results](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_QUERY_WRITE_TIMING.md#L11-L82).

### Report 18: earlier L11 key/value donors

The implementation preserves every recipient query, source 0, later K/V and the baseline upstream trajectory while splicing earlier K, V or both. The combined donor closure is exactly the right implementation test. It is also an expected architectural consequence once all changed earlier K/V are copied, not an independent discovery.

Both single paths produce positive recorded later KL on every passage; signed NLL changes differ and cancel. V's larger KL cannot identify intrinsic V dominance because its natural donor perturbation has 1.28-1.60 times the K RMS dose. The optional context interaction is an exact bilinear attention identity with finite-precision residuals, not a new mediation law. [Results and dose](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/docs/GPT2_L11_KEY_VALUE_DONOR.md#L9-L49), [actual splice and protections](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/l11_key_value_donor_reproduction/experiments/pulse_transfer/gate_l11_key_value_donor.py#L369-L419).

### Report 19: native versus value-lesioned calibration

This is a properly scoped calibration comparison, not a breakthrough mechanism result. Explicit/native mask routes agree on the four tested calibration texts. PE/CE centers and direction pairs are close; 64 paired primary results are transparently reported. My independent recomputation confirms the largest absolute mean endpoint change is **2.6477493937e-5 nats**, or **0.0264774939 millinats**, and 51/64 intervals include zero.

D is removal minus orthogonal-control loss, not removal minus native loss. A change in D can come from either arm. The center/direction-pair main effects sum to the endpoint change by construction. The study did not isolate a universal calibration-invariant PC, prove equivalence or resolve every tiny interaction. Its current README and handoff mostly make exactly the right claims. [Complete primary definitions and table](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_report/report19.md#L31-L135).

## 4. Mathematical interpretation: what the controls identify

### Attention gating is exact algebra, not by itself a discovered circuit

For one head and a fixed incoming state, let a be total attention mass on non-source positions, m their normalized conditional value mixture, and v0 the source value. Then:

c = a m + (1 - a) v0.

A chosen allocation change gives c' - c = (a' - a)(m - v0). When v0 is zero this reduces to c = a m, so scaling context by a'/a implements the real-arithmetic allocation change, but divides by a potentially tiny number. The later native-value delta implementation avoids that division. Because the runtime uses diagnostic softmax and actual SDPA outputs, reconstruction/rounding checks remain necessary; the symbolic identity is not a proof of byte-identical native arithmetic.

The local postprojection sensitivity to gate logits has the form of the learned output projection acting on headwise (m - v0) vectors weighted by a(1 - a). Consequently, PCA variance of b - log(q) is not a proxy for downstream effect: gate saturation, mixture/value direction, output projection and loss-gradient alignment all matter. This explains why equal logit norm is insufficient, why actual-write matching helps, and why equal write norm still does not imply equal loss sensitivity. These are organizing deductions, not newly measured mechanisms. [Implemented map and delivery](https://github.com/jaykobdetar/superweight-research/blob/e59cfed2df13a1d3a6e6c607648a9d6844790654/results/calibration_world_comparison_reproduction/experiments/pulse_transfer/gate_native_value.py#L39-L74).

### Factorials identify finite contrasts, not universal importance fractions

A donor factorial identifies responses to the specified substituted states. K/V hybrids and multi-layer clamps may lie outside the native state distribution. They can show sufficiency or dependence relative to that recipient without uniquely identifying the natural causal mediation of an ordinary input difference. Reciprocal swaps, exact diagonals and native-state controls strengthen the interpretation; they do not remove this distinction.

Likewise, loss nonadditivity is not automatically internal feedback. If z_add = z_base + sum(z_single - z_base), the additive output may not correspond to any model trajectory. The Report 13 decomposition is an accounting identity over finite outputs and nonlinear cross-entropy. At fixed queries, the Report 18 context interaction (C_KV - C_K) - (C_V - C_base) = (A_donor - A_base)(V_donor - V_base) is ordinary bilinearity; residual analysis checks how closely captured arithmetic follows it.

### Low rank is representation-dependent

A PCA rank is a property of a chosen coordinate space, centering, sampling distribution and metric. Twelve gate coordinates at each layer, a 96-dimensional cross-layer covariance and hundreds of residual/logit perturbation directions answer different questions. None alone counts causal mechanisms. Failed linear low-rank substitutions rule out the tested substitutions, not every compact nonlinear description.

## 5. Novelty relative to primary literature

The repository's restrained novelty language is largely warranted. I checked the following close primary sources; this is a targeted prior-art check, not proof of global priority.

- **The Super Weight in Large Language Models:** single-weight sensitivity and its relationship with extreme activations precede this work. The original OLMo coordinate table confirms the repository's erratum. The extension here is the particular causal pathway and intervention comparison, not discovering super weights. [Yu et al., v2](https://arxiv.org/html/2411.07191v2).
- **Massive Activations in Large Language Models:** stable extreme activations, mean replacement, first-token behavior, implicit attention bias and explicit K/V alternatives are prior art. The present source replacement and gating interpretation must be positioned as detailed case studies. [Sun et al., v2](https://arxiv.org/html/2402.17762v2).
- **Attention Sink Forges Native MoE in Attention Layers:** source attention mass as an implicit gate is already explicit. Reexpressing attention as allocation times a conditional mixture is not a novelty claim available here. [Sink-aware training paper, v3](https://arxiv.org/html/2602.01203v3).
- **A Mechanistic Account of Attention Sinks in GPT-2:** the primary paper names coordinates 138, 378 and 447 and studies a query-bias/positional-MLP/key-projection route. The current raw-coordinate and gate interventions are different estimands, not discovery of those coordinates or a replication of that entire route. [Ran-Milo et al., v1](https://arxiv.org/html/2604.14722v1).
- **GPT-2 weight-dictionary work:** the pinned public exp_s8_loop.py already studies channel 447 at the block-2/3 boundary, regulators and downstream compensation. I inspected that code rather than relying only on the repository's summary of it. This constrains broad claims of discovering coordinate regulation. [Pinned source, commit 33504a4](https://github.com/sunmoonron/weight-dictionary-decomposition/blob/33504a4d90840bc01d0ef75609265e48a34403bb/scripts/exp_s8_loop.py).
- **Transient Reserves, Sink Dampers** and **The Life Cycle of a Massive Activation:** architecture dependence, collective magnitude, redundant carriers and perturbation controls precede broad nonuniversality or magnitude-versus-identity claims. They do not subsume every specific donor experiment here. [Li, 2026](https://arxiv.org/html/2607.09279v1), [McClendon et al., 2026](https://arxiv.org/html/2610.00423v2).
- **Activation-patching methodology:** metric choice, corruption choice and path interventions are established and can change conclusions. Exact closure is excellent engineering, but not a new causal-inference method. [Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042).

**Best candidate contribution:** a transparent OLMo case study tying specific token history to a selected head/MLP pathway, together with unusually explicit scalar-versus-neuron-output counterfactuals and cross-model negatives. The later GPT-2 studies add a careful finite perturbation dataset and successive controls. Their novelty is narrower and should be sold as such. I would not frame the contribution as discovery of attention gates, self-repair, GPT-2 outlier coordinates, general direction sensitivity or a universal sink mechanism.

## 6. Highest-value next experiments

These are recommendations, not execution authorization. None was run in this review. The repository's T24 proposal is considered only as a published proposal; no unpublished local implementation was assessed.

### Priority A: an independent, compact replication of decisive claims

**Question:** do the central causal results survive a separately implemented intervention and checked scoring path?

Choose a small prespecified sentinel set: an OLMo constructed/newline natural pair with H14 K/V and sham controls; an OLMo scalar-versus-column case; a native GPT-2 PC/control matched-write case; and one L11 K/V donor closure. Use verified original inputs or a clearly labeled fresh replication panel, pinned checkpoint and correct tensor conventions. Recompute metrics independently, retain enough raw primitives to check the actual operation, and compare the intended estimand rather than requiring every unrelated byte to match across hardware.

**Success condition:** reproduce the large qualitative effects and the specified intervention boundary within prospectively justified numerical tolerances. A small sentinel replay is not a full 1,160-forward replay. If it fails, diagnose precision, indexing and intervention semantics before expanding to more conditions.

### Priority B: settle control orientation and meaningful dose

**Question:** is the PC removal response distinctive relative to multiple prespecified orthogonal directions and signs, at informative matched actual-write doses?

The published sign-balanced ensemble is a sensible next GPT-2 experiment. Add a compact dose sweep and report the distribution of actual per-query target norms, including how much the common minimum shrinks relative to the original PC edit. Predeclare a small number of primary contrasts, use fresh documents and retain the native baseline. Do not interpret a finite set of seeded directions as a random-direction population unless its sampling/inference scheme supports that claim.

**Discriminating outcomes:** persistence across signs/directions and nontrivial doses supports a direction-specific finite effect; disappearance under sign balance suggests the original one-control contrast was orientation-dependent; disappearance only at vanishing dose is inconclusive. Neither outcome alone identifies a unique native circuit.

### Priority C: reconnect to native behavior with a predictive bridge

**Question:** what naturally occurring computation does the proposed mechanism perform, and does its measured state predict a held-out intervention response?

For OLMo, vary newline identity, position, distance and preceding history on separately selected natural documents. Freeze the head/path hypothesis before evaluation. Test necessity and rescue for the same native behavioral/pulse contrast, with matched alternative-token controls, and distinguish the empty-history H12 phenomenon from H14 history suppression. A model of when the pulse turns on or off, tested on new histories, would add more than another retrospective donor decomposition.

For GPT-2, first state a native contrast linking source-coordinate/pulse variation to the L10 gate direction. Test whether the natural contrast actually changes that direction and whether an appropriately localized intervention transfers or suppresses the relevant behavior. Without that bridge, label continued work as gate-perturbation sensitivity, not explanation of the original super weight.

**Success condition:** predict and causally manipulate a native held-out contrast with less flexibility than an all-layer donor clamp. Failure is informative and should narrow the mechanism claim.

### Priority D: independent calibration and a real inferential target

After the operation is fixed, use multiple independent calibration text sets and a much larger document-level evaluation sample. Distinguish within-fit evaluation variation from between-fit variation. If robustness/equivalence is the goal, choose a meaningful equivalence margin before acquisition, based on behavioral or practical relevance rather than observed tiny differences. Account for the small family of confirmatory claims; retain exploratory detail separately.

### Lower priorities until these are resolved

- More fine-grained K/V or head rankings without dose control and a native-behavior target.
- Another long sequence of tiny center/direction interactions on four passages.
- More covariance-rank measurements presented as mechanism counts.
- Compression/speed claims before faithful output and actual wall-time tests.
- A universal cross-architecture theory inferred from one magnitude-selected candidate per checkpoint.

## 7. Recommended presentation and immediate non-experimental fixes

1. Lead with the OLMo causal finding and the scalar/column intervention paradox; separate the GPT-2 gate branch visually and conceptually.
2. Put a claim ledger next to each headline: exact estimand, model/revision, input unit, intervention boundary, evidence file, verification level and strongest failed alternative.
3. Preserve the coordinate erratum prominently. Do not silently rewrite layer-2 history into layer 1.
4. Use “single-coordinate contribution deletion,” “whole neuron-output deletion at this token,” “finite gate edit,” “recorded-observation audit” and “same-environment replay” consistently.
5. Add current-status banners to historical reports. Keep scientific negatives distinct from infrastructure failures and missing guards.
6. Promote passage-level plots and leave-one-out sensitivity alongside small-n interval tables. Show baseline fidelity metrics even when loss improves.
7. Publish a minimal independently understandable recipe for a handful of central claims, with lawful input instructions and a canonical current source map. Keep the historical evidence intact.

## 8. Final judgment

This work is strongest when it refuses simplistic conclusions: large weights are not interchangeable with important neurons; more deletion can be less damaging; source values and source access are different interventions; regrowth is not restoration; covariance rank is not circuit count; improved loss is not faithful replacement; and tiny calibration differences are not equivalence.

The repository generally already respects those distinctions. My review therefore does **not** recommend discarding the research or alleging a hidden fatal bug. It recommends making the scientific center of gravity clearer, independently replicating a few decisive causal operations, strengthening the control/data design, and declining to turn increasingly precise bookkeeping about engineered perturbations into a broader native-mechanism claim than the experiments identify.

The strongest next paper would make one narrow mechanism more predictive and independently reproducible, rather than trying to make every completed report part of one universal explanation.
