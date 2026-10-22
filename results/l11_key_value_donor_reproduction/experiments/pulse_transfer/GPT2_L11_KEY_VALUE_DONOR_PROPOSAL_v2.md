# T22 prospective design v2 — earlier L11 keys versus values after an early L10 write edit

**Design v2 only. Not frozen or approved for implementation, selection, checkpoint access, model execution, or publication.** Existing T20/T21 sources and evidence remain unchanged. Publication is explicitly paused; the user controls visibility. This proposal requests no historical model rerun and introduces no general experiment framework.

## Question and evidence motivating it

T21 completed648 planned forwards in two planned first-execution segments; independent numerical, projection and readout audits passed. Its early-only L10 matched writes at queries1–126 produced positive mean later baseline-to-trial KL in all64 passage/start/direction cases, while later L10 post-attention, MLP and block outputs and incoming L11 Q/K/V at queries127–511 remained baseline bytes. Source/start/direction mean later KL ranged from1.14585e−5 to5.81764e−5 nats. Signed later NLL remains uncertain:15of24 primary nominal intervals include zero. These are conditional results about one frozen fit and one orthogonal control, not evidence that PC is unique or that an exact amplitude-only null has been excluded.

The next question is which finite changes to **earlier L11 keys and/or values** reproduce that cross-position consequence on fresh passages. T22 holds the recipient's upstream trajectory and every query vector at baseline, then copies earlier keys and/or values from a genuine early-write donor. It separates a specified donor's K and V interventions; it does not estimate a mediation fraction or a population circuit decomposition.

Evidence commitments:

- T21 completed manifest: `c4de3d3473eb93556bcdeb1161d1fc7d88f8c4e20c9fb38ad10b7096058299a9`.
- T21 analysis: `94d8473b35e449b494b5c44c7d29651213f99399f99a40c83cb8fac8a9d56748`.
- T21 N/P/R audit files: `9cd0778f3d418b2d7f408b08146c52684276161ef1ddc0d6db2d9b8d00edc2c2`, `09e586fcff0ffcd38b4ba6306f37af21d182f908e405cf9cb63bb70b9033c6df`, `e92400d073855d8c2b3905cf748de09f9fe44d45db88d3073b84d5ce76a55da3` respectively.
- Frozen T21 protocol v4: `a17e3e7e8150935950d58df1d2b266dac2bf91a6c65c4a8539d7e28fcf3e0819`; frozen runner `539565fb028eb9fd217d49ed43b03e5ba6869c8ecc0d3ed2a98c0eca66064c92`; support hook `f9f2eb64795b6b40ce19a2b6187e4dbb7263d838ff152bf2d081b06260a0f0ab`.

The motivating summary reads already completed aggregates only. No T22 outputs exist.

## Fixed elements and fresh inputs

Retain the checkpoint/revision, FP32/SDPA precision settings, batch1/no-cache fixed512 execution, original model masks, frozen F/PC direction/one orthogonal direction, source processing, natural/BOS replacement, split and scorer from T21. Native000 only: no source-position value zeroing. No fit, direction ensemble/search, scaling sweep, query-cutoff search or outcome-dependent arm/row selection. Layer indices are zero-based; L11 is the last of GPT-2's12 attention blocks.

Select four new512-token passages from each of the same four sources, paired natural/BOS starts:16 passages/32 identities. BOS replaces only token0; it is not inserted. Selection occurs only after a separately reviewed exact selector/source/history closure. Inherit the audited745-hash T21 known-history chain, append all32 actual T21 natural/BOS identities and16 actually used source-linked512-token spans, and independently verify all32 are new relative to745: expected union **777**. Retain all prior source-linked used spans and all four full8192-character F calibration buffers. Do not substitute the old optimizer panel or enlarge used512 spans to unrelated extraction buffers.

The canonical T21 data-manifest commitment is `f0ff4671117fb5b1f540dded7329a21fc640e524d9f08648c721b37156011dd8`; its inherited known-history chain is `67106b29dc1991dc86c5f7af4e03e4accc7d09c18fcb50f1415f0f9dd71d103e`. Reconstruct append records from canonical accepted input records/component offsets and actual run/input bindings, not report text. Recheck current source/tokenizer bytes, offsets, natural/BOS suffix equality, grid traversal, interval and token-identity exclusions. Preserve every candidate and rejection. Exhaustion fails closed; no stride/source/exclusion relaxation.

Require a new coordinator attestation about recorded authorized experiments since T21, with independent known-chain review. It is not a claim of host-wide absence of unrelated jobs, an exhaustive repeat of the earlier private1370-file scan, or knowledge of GPT-2 pretraining overlap. No historical model/Llama run is needed to rebuild metadata. Proposed preparation cap remains32MiB pending an actual bounded metadata plan.

## Donors: exactly the existing early matched operation

For each input and main/suffix variant, obtain native baseline `B`, the two actual full L10 projected-write proposals, and the unchanged T20 CPU-FP64 querywise min-norm interpolation recipe. The full proposal arms are necessary to form the real rounded matched writes; they are not new standalone mechanistic evidence. Compute the recipe once per variant, retaining the existing byte commitments and source/zero/one/interior branches. Copy the already-rounded matched write only at `E={1,…,126}`; inactive queries/q0 copy baseline bytes.

Call the resulting genuine early-write donors `D_PC` and `D_orth`. Preserve T21 donor guards: every later L10 post-attention residual, MLP and final block output and original L11 later Q/K/V at `L={127,…,511}` equals that variant's baseline bytes. Preserve all-layer source-q0 checks, the frozen analytic and masked RMS norm guards, actual device-copy/attention-write equality and separate full-proposal diagnostics. Matching concerns the two L10 donor write changes, approximately in FP32; it does not match K/V changes at L11 or make amplitude irrelevant.

Cache each donor's **actual original L11 c_attn output**, before any transplant, plus baseline references. Each cache key binds corpus/window/start, physical512, actual main/suffix token SHA, model/revision/parameter and precision/freeze commitment, F digest, direction, donor operation, support and slice/layout version. Full tensor dtype/shape/byte fingerprints are immutable. No token-equivalent or causal-prefix shortcut permits cross-variant borrowing.

## Recipient K/V factorial

Every recipient starts from the **native baseline trajectory through the whole L10 block and L11 pre-attention state**. Its L10 delivered write, post-attention residual, MLP/block output and original L11 c_attn input/output must equal baseline bytes at all512 queries. It does not run the early L10 edit again.

Let original L11 projected Q/K/V be `Q_B,K_B,V_B`, each `[1,12,512,64]` logically, with the natural c_attn output `[1,512,2304]` laid out as contiguous Q/K/V width768 blocks. For direction `d`, construct four recipient conditions:

| Condition | Earlier keys q1–126 | Earlier values q1–126 | All queries Q | Source q0 and later K/V q127–511 |
|---|---|---|---|---|
| Neither/shared baseline | Baseline | Baseline | Baseline | Baseline |
| K only, `R_dK` | Actual donor d | Baseline | Baseline | Baseline |
| V only, `R_dV` | Baseline | Actual donor d | Baseline | Baseline |
| K+V, `R_dKV` | Actual donor d | Actual donor d | Baseline | Baseline |

Copy **all12 heads**, exactly the selected FP32 bytes, at the L11 c_attn output before attention consumes it. No interpolation, renormalization, softmax replacement, head selection, source-position edit or mask/kernel-shape change. Preserve every query vector, including earlier Q. Source q0 is never copied from another trajectory. Preserve original baseline later K/V, including unscored q511.

A current/sham arm explicitly exercises identity L10 delivery and the new L11 copy path using baseline earlier K/V bytes. It must equal the installed baseline at all512 logits and relevant states. Neither is shared between PC/orth; do not add duplicate inferential baseline arms.

### Natural versus delivered records and hook order

The transplant must execute **before** the immutable `GateNativeValue.combined` observer at L11, for example as an explicitly owned prepend hook. Otherwise the observer would reconstruct SDPA from baseline pending Q/K/V while SDPA used the edited tensor. Do not patch frozen pending states after the fact. Inspect/verify the owned hook roster/order, including unhooked-anchor removal/reinstallation and absence of duplicate handles.

Record separately:

1. Original natural c_attn input and output, observed before transplant.
2. Expected donor-spliced output and actual delivered output with byte equality.
3. Actual post-transplant attention-input Q/K/V observed by the frozen native diagnostic.

Delegated L11 native projection records for recipients now describe **post-transplant actual attention inputs**, not the unedited Conv1D projection. Projection auditing applies the weight/bias relation only to the **original** natural output, then independently verifies byte splicing. It must never test the transplanted output as if the unchanged linear projection had generated it.

## Hard combined-KV closure before separate effects

For each direction and variant, require `R_dKV` to reproduce the actual `D_d` donor at **all later queries127–511**, byte-for-byte, at:

- L11 attention context and actual projected attention write;
- L11 final block hidden output;
- final `ln_f` hidden output;
- the complete vocabulary logits at every later query, including511.

Every later recipient query, later residual input and later K/V equals its donor/baseline, while its earlier K/V now equals the donor. Earlier Q and residual streams can differ between recipient and donor, but earlier attention outputs cannot feed later outputs inside the same parallel attention operation; the remaining L11 MLP, final normalization and readout are positionwise. This predicts later closure under the fixed architecture, **not whole512-output equality**. Earlier recipient outputs may differ from the donor and are still scored/reported.

This is an empirical hard bitwise closure test, not a tolerance-based scientific effect. If it fails at any checked coordinate/row, preserve complete compared operands/caches and abort; do not interpret separate K/V effects, relax tolerance, omit a row or silently switch SDPA/precision. CPU fake-model fixtures will not by themselves guarantee GPU bitwise compatibility. The first checks occur within the predeclared study arms, with no extra pretrained pilot or old-input rerun.

Within an input, both variants and both directions must pass their KV closures **before that input's four K-only/V-only arms run**. Any later input failure makes the planned panel incomplete; earlier raw results remain preserved, not selectively released as a complete study. Zero later donor-versus-KV contrasts are closure controls, not evidence that keys and values individually contribute additively or that a mediated fraction is100%.

## Exact proposed arms, order, counts and segmentation

Twelve scientific arms, each MAIN then SUFFIX before the next arm:

1. `baseline` (`B`)
2. `current`
3. `remove_full` (auxiliary full proposal)
4. `orth_full` (auxiliary full proposal)
5. `remove_write_matched_early` (`D_PC`)
6. `remove_l11_kv` (`R_PCKV`, hard closure)
7. `orth_write_matched_early` (`D_orth`)
8. `orth_l11_kv` (`R_orthKV`, hard closure)
9. `remove_l11_k` (`R_PCK`)
10. `remove_l11_v` (`R_PCV`)
11. `orth_l11_k` (`R_orthK`)
12. `orth_l11_v` (`R_orthV`)

Pairing each donor with its combined closure allows only one direction's main/suffix full later-logit reference pair to be live at once. Keep immutable full donor QKV caches for both directions until the input completes. The other direction's recipe is still formed from both full proposals; order never changes the matched target.

At natural-start passage0 of each source, insert truly unhooked native-repeat and hooked identity-writer anchors immediately after baseline MAIN, before baseline SUFFIX. Neither advances scientific order or replaces a reference cache. Preserve the current T21 native-anchor semantics; the new identity/sham path is additionally tested.

Across4sources×4passages×2starts=32 inputs: **384main +384suffix +8anchors =776 planned forwards**, zero calibration. Segment A Wiki+Utilitarianism has388; segment B Ethics+Code has388 and requires exact complete marker-free A metadata/predecessor review.32shards total,16/segment, one per passage/start/native000. Reuse explicit attempted/returned/processed/committed counters, failure records, manifest-last completion, source/input/parameter/runtime/precision guards and safe predecessor/output containment. No source edits during execution and no model at merge.

Controls to roster explicitly:64current comparisons,384suffix-prefix comparisons,128donor later-state checks,384recipient baseline-upstream checks,128KV closure events and8anchors. Source-q0 checks cover704nonbaseline scientific calls; baseline/anchor source checks remain separately recorded. A guard failing in a model-end hook means output was observed internally but the model call did not return; journal those facts separately instead of incrementing returned/processed counters.

## Outcomes, finite contrast universe and uncertainty

Retain full targets2–511 (510), early2–127 (126), later128–511 (384), all from physical512. Primary causal readout region is **later**. Save per-query NLL, baseline-to-arm `KL(p_B || p_arm)` and baseline argmax agreement for all12arms, every source/start/passage, including failed-sign or tiny effects. Preserve T21 scorer arithmetic and no clipping/tolerance tuning; independent numerical recomputation must examine very small KL values explicitly. Agreement is coarse: unchanged argmax is not unchanged distribution. Report exact equality and effect size, not only interval exclusion.

Proposed co-primary numerical outcomes are signed later NLL and later baseline KL for the K-only/V-only effects, their paired K−V difference and per-direction K/V interaction. Agreement and other views/direction comparisons are planned secondary. This prioritizes distributional transmission prospectively after T21's small KL/nonuniform signed-NLL result; it is not a retrospective revision of T21. Final metric-priority choice needs independent prospective review before freeze.

For any scalar outcome `Y` (NLL, baseline KL, agreement), use the following **19 fixed coefficient definitions**, separately for all3views and3metric families:

| Definition | Coefficients |
|---|---|
| Donor PC effect | `D_PC − B` |
| Donor orth effect | `D_orth − B` |
| PC K effect | `R_PCK − B` |
| PC V effect | `R_PCV − B` |
| PC KV effect | `R_PCKV − B` |
| Orth K effect | `R_orthK − B` |
| Orth V effect | `R_orthV − B` |
| Orth KV effect | `R_orthKV − B` |
| PC K−V | `R_PCK − R_PCV` |
| Orth K−V | `R_orthK − R_orthV` |
| PC K/V interaction | `R_PCKV − R_PCK − R_PCV + B` |
| Orth K/V interaction | `R_orthKV − R_orthK − R_orthV + B` |
| K PC−orth | `R_PCK − R_orthK` |
| V PC−orth | `R_PCV − R_orthV` |
| KV PC−orth | `R_PCKV − R_orthKV` |
| Donor PC−orth | `D_PC − D_orth` |
| Interaction PC−orth | `R_PCKV − R_PCK − R_PCV − R_orthKV + R_orthK + R_orthV` |
| PC recipient-KV minus donor | `R_PCKV − D_PC` |
| Orth recipient-KV minus donor | `R_orthKV − D_orth` |

The final two are exact-zero hard controls in the later view; early/full views compare different query/residual conditions and need not be zero. Do not treat a nonzero early/full control contrast as a failure of later closure. Auxiliary full-proposal metrics remain disclosed in window/arm summaries without standalone mechanistic inference.

Expected scalar-metric window rows:1152, anchor-view rows24, arm summaries288. For **each** contrast metric family:456contrast rows (19×4sources×2starts×3views) and228paired-start rows; across NLL/KL/agreement:1368contrast and684paired-start rows. State metric dimension explicitly to avoid confusing coefficient definitions with expanded row counts. KL interactions/differences can be negative and are not themselves KL divergences; agreement effects use probability points; NLL/KL units are nats/target.

The co-primary later family contains **128 nominal comparisons:8estimands ×2metrics ×8source/start strata**. There is no “any-one-CI” study-success criterion, adjusted population claim or significance-only routing. The full primary family, passage values, uncertainty and numerically unresolved cases must be reported.

Use2000shared PCG64 passage-cluster bootstrap draws, seed20261021, the same passage-index matrix across starts, arms, views, metric families and contrasts. Four passage clusters/source; one fixed F/PC/orth. Report passage-level values and nominal conditional95% intervals without multiplicity adjustment, fit uncertainty, independent-replication interpretation or equivalence claims from zero-crossing intervals. Do not select strata/metrics because NLL cancels. KL near zero and agreement ties need explicit numerical scope, not optimistic rounding.

### Small KL arithmetic and selected readout audit

Preserve original FP32 KL metrics exactly, including tiny values, signed negative rounding residues and zeros; do not clip, round to positive, discard, or replace them after outcomes. For each of the48predeclared saved full-vocabulary rows, independently compute stable FP64 `KL(p_B || p_arm)` **from the saved actual GPU FP32 logits**, separately from the existing FP64 hidden×embedding reconstruction. The saved-logit FP64 calculation audits scoring arithmetic; the embedding-dot calculation audits readout drift. Do not conflate the two references.

Report each row's savedFP32 KL, saved-logit FP64 KL, signed error `(FP32−FP64)`, absolute error and absolute error-to-signal ratio when the FP64 reference is nonzero. At exact zero reference the ratio is null with an explicit zero-case flag, not a fabricated zero/infinite success score. Preserve signed FP64 roundoff if present. Display absolute scales alongside any ratio and include baseline/current exact-identity witnesses.

Selected-row discrepancies are not a whole-panel error bound. Do not infer a very-small-KL path ranking, absence or intrinsic importance when numerical resolution is inadequate; report the unresolved conclusion and all original finite outcomes. NLL, KL and agreement can disagree without a failed experiment. No dense all-panel FP64 evaluation, additional pretrained rerun or general numerical framework is requested; exact runtime closure remains independent of these scalar metrics.

## Interpretation and dose limits

A successful combined closure and nonzero K-only or V-only later response identify finite controlled consequences of transplanting that donor's earlier keys or values into a baseline recipient. A K−V difference compares these natural donor perturbations, not equal-sized or equally informative interventions. The L10 donors are approximately norm matched, but their L11 K/V change norms, logits and downstream sensitivities need not match across paths or directions. Do not normalize them after observing outcomes or interpret unequal effects as intrinsic K/V importance.

A nonzero interaction can reflect attention-weight/value multiplication and nonlinear residual/readout/loss; it is not a mediation fraction. Zero interaction does not imply absence of transmission or a linear model. The design does not isolate heads, semantic content, a unique PC circuit, a generic sink mechanism or population benefit; does not measure Jacobians; does not establish direction independence from amplitude; and does not test generation, fidelity, long contexts or speedup. One orthogonal control is not a directional ensemble. Findings are conditional on the fresh finite panel and the exact baseline/donor interventions.

K/V and query/key/value path patching are established methods, including [Wang et al.](https://arxiv.org/html/2211.00593v1) and [Goldowsky-Dill et al.](https://arxiv.org/html/2304.05969v1); metric sensitivity, including KL, also has prior work. T22's candidate contribution is this specific fixed-fit held-out donor experiment and exact closure control, not a new patching method. See [primary-source method notes](T22_METHOD_PRIOR_NOTES.md).

The [prospective readout/small-KL note](review/t22_prospective_readout_small_KL_scope_v1.md), SHA256 `515a10ffb68574fb267f9bfb63a60345973dbe103abceee26da701c21382935f`, supplies the minimal selected-row error diagnostics and null/denominator semantics. It adds no whole-panel sensitivity bound or new acceptance threshold. Bind its exact N/R fields prospectively before outcomes.

## Minimal implementation/API delta — not implemented here

Proposed new thin wrapper `GateL11KeyValueDonor` reuses immutable T20 arithmetic/native observer and the reviewed T21 support-copy/byte-commitment/causal-guard ideas. It needs its own explicit12arm/24call state machine; do not spoof a frozen T21 ten-arm order or execute unused T21 later/both arms merely to advance it. No existing scientific source is edited.

Suggested interface:

```
gate.start_input(metadata, boundary='000')
gate.reset('000', arm, variant='main'|'suffix', token_sha256=actualSHA,
           snapshot=l10_panel_selected, observe_only=False)
output = model(...)                   # owned hooks capture real L11/end-state operands
# Before finish: runner fingerprints/copies current tensors for declared panels.
gate.finish()                         # rejects missing donor/closure/guard/observations
```

Model/L11-block/final-normalization observers bind actual donor and recipient hidden/logit operands; closure verification must not trust caller-supplied unbound references. `.finish()` refuses a KV arm until its actual model-output closure completed, and refuses a single-path reset before both directional/variant closures. Capture model-output-observed versus call-returned separately.

Expose existing `write_cache`, `write_norms`, proposal gate diagnostics, source states, failure capture, close/unhooked, plus `donor_cache`, `l11_current`, `l11_records`, `l11_norms`, `closure_checks`. All identities include variant and frozen provenance. `l11_current` exposes complete original/expected/delivered operands until finish; ordinary normal records include full512 fingerprints and protected-region exact flags before arrays clear. Normal selected raw rows are fixed below. Whole failure state retains complete defined comparison operands/caches with no post-hoc truncation or deduplication.

L11 norm contract:9FP64 arrays[512] per arm—actual Q/K/V norms; Q/K/V deltas from baseline; Q/K/V deltas from the original natural projection. These distinguish a donor's changed original state from a recipient's transplant. Recipient Q delta is0 everywhere; K/V transplant deltas vanish outsideE and sourceq0; donor transplant delta is0 because no c_attn transplant occurred. Save full per-query arrays and active-only E summaries; do not label a donor's natural change as an inserted projection delta. Retain inherited L10 common/matched norms only where semantically applicable; recipient L10 is baseline, not a hypothetical full gate proposal.

## Predeclared raw panels by purpose

All panels use **Wiki passage0, natural, MAIN only**; runtime all-query guards/fingerprints/norms still cover every arm/variant/input.

- **L10 full recipe/interpolation panel,5states:** baseline, remove_full, orth_full and the two genuine early donors. Retain the existing complete L10 QKV/mask/context/stable-mixture/gate/proposed-write/actual-write evidence needed to reconstruct the unchanged recipe. Current and recipient L10 equality use complete runtime byte guards/fingerprints, rather than duplicate full L10 raw snapshots.
- **L11 original/projection/transplant/context panel,9states:** baseline, both donors and all6recipients. For each retain full c_attn input, original natural c_attn output, actual delivered QKV, actual SDPA context, actual attention projected write and L11 final block hidden output. Compare expected splice from separately saved baseline/donor references; expected full tensors can be reconstructed from those exact saved operands, while any failure retains the complete runtime expected operand. No post-transplant tensor is misrepresented as natural projection output.
- **End-state/readout panel,12states:** all12arms; full final hidden[1,512,768] and vocabulary rows q1,126,127,510 (48rows total), targets2,127,128,511. q511 still receives runtime full-vocabulary closure/source/control checks even though it has no scored target.
- **Five-row timing panel,12states:** q1,126,127,510,511 at L10postattention/L10MLP/L10block/L11delivered-QKV (240row vectors with the declared widths). Preserve source-q0 evidence and full protected-domain checks separately. These samples do not independently reconstruct unsaved full states.

These counts describe separate evidence purposes, not12duplicated full mechanistic snapshots. Freeze exact key/dtype/shape inventories and analyzer denominators before execution. Full contexts/QKV/hidden/vocabulary rows remain private; any later public numerical package requires separate explicit authorization and privacy/export review.

## Conditional resource and complete-failure assessment

Keep one model/GPU worker, process RSS≤3GiB, host MemAvailable≥4GiB, reserve2GiB plus outstanding allowances. Provisional normal cap512MiB and failure cap1GiB are the **reviewed T21 caps**, not a demonstrated T22 bound. No assumption that the T21 witness covers T22.

Known uncompressed FP32 component sizes at512×768 and50257vocabulary:

| Component | Bytes |
|---|---:|
| Full hidden array | 1,572,864 |
| Full combined QKV array | 4,718,592 |
| One donor's earlier KV rows | 774,144 |
| One donor's later hidden array,385rows | 1,182,720 |
| One donor's later full-vocabulary logits,385rows | 77,395,780 |
| One direction's two-variant later-logit reference pair | 154,791,560 |
| Baseline plus two donor fullQKV caches, both variants | 28,311,552 |
| Nine selected L11 raw panels, six specified arrays/10width units | 141,557,760 |
| Twelve final hidden arrays | 18,874,368 |
| Forty-eight vocabulary readout rows | 9,649,344 |
| Twelve five-row timing samples, six width units/row | 1,105,920 |

The four fixed normal components above total171,187,392bytes **before** five complete L10 snapshots, per-query metrics/norms, all JSON/inventories, checkpoint/journal/finalization and codec overhead. This is a component count, not a complete upper bound or a claim that512MiB fits. References retained for closure are actual observed full later-vocabulary arrays, not guessed from metrics or merely a hash; both compared operands must remain available if closure fails. Their bounded lifetime ends after the corresponding main/suffix combined closures, leaving small donor QKV caches for singles; this changes no stored file.

The T21 synthetic conservative failure encoded831,398,069bytes. Its logical array universe cannot simply be reused or incremented by one known donor tensor: T22 failure includes main/suffix donor caches, original/expected/actual transplant operands, full later hidden/logit closure references, current native/write/timing state, completed metrics/counters and the uncommitted new-panel data. Multiple wrapper/runner failure branches and copies must be included exactly as the new codec serializes them. No omitted operands or outcome-driven evidence reduction is allowed.

**Before implementation/run authorization**, review a symbolic complete normal/failure array-tree plan and bounded lifetimes/RSS estimate against the fixed3GiB cap. Implementation-only CPU rehearsal then needs separate authorization: one smallest changed-ABI causal fixture plus an exact high-entropy whole-layout/failure witness, with production shapes and scalar metadata preserved. No excess synthetic forwards are needed for repeated field/cap-only changes. If measurement exceeds cap, preserve it and seek a prospective resource-only amendment; do not relax numerical thresholds, drop failed fields or delete previous artifacts. A cap increase is not preapproved. Do not launch if complete failure evidence cannot be preserved within reviewed RSS/storage/disk budgets.

The [separate symbolic resource plan](GPT2_L11_KEY_VALUE_DONOR_RESOURCE_PLAN_draft.md) refines this section into branch-specific owned array trees, cache lifetimes and conditional NPZ/RSS envelopes. It must be independently reviewed and its producer/metadata/ownership assumptions tested before any run; the existing1GiB cap is not automatically inherited as evidence.

## Review and readiness gates

Still undecided/unapproved:

1. Independent scientific approval of finite K/V estimands, the19coefficient/metric universe, co-primary NLL/KL priorities and conditional uncertainty language.
2. Exact5/9/12purpose-specific raw panel and full normal/failure array-tree/peak-RSS plan, including retained donor logit references. The new1GiB failure bound is unproven and could fail.
3. Thin hook/API design, prepend/delivery-observer ordering and honest natural-versus-delivered record schema. Required CPU hostiles include wrong-head/axis/slice/source/Q/later copy, swapped direction/variant/F/token cache, changed donor bytes, wrong closure reference, skipped closure, late hook mutation, inactive signed-zero/subnormal bytes and unchanged-mask/current/suffix/anchor controls.
4. Fresh777known-history/input closure and no-unaccounted-recorded-experiment attestation, only after selector/source review. No token/corpus selection has been performed for this proposal.
5. Runner/776roster/A→B predecessor/disk/failure/journal/manifests, analyzer19contrast×3metric scope and future independent N/P/R policy. No source edits while running; no historical-model or duplicate-worker workaround.

The future numerical audit must reconstruct donor recipes and exact transplanted QKV from selected saved operands, independently check all saved norm/guard/closure identities and all statistics/shared bootstrap indices. Projection audit may read only required L10/L11 projection tensors on CPU under separate authorization and must distinguish original linear outputs from inserted tensors. Readout audit may read only the tied embedding and the48selected rows/12hidden states; it cannot replace whole-panel runtime closure checks. All actual audits precede interpretation. No implementation, selection, checkpoint access, model/GPU test, publication or visibility/security change is authorized by this design.
