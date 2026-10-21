# Prospective additive-logit study of finite gate effects

Status: draft for independent scientific and resource review. No new model outputs, tokenizer eligibility run or frozen input selection exists for this proposal. It does not authorize inference by itself. Report12 and the localization panel are development evidence for this question; their files and both older failed gate studies remain unchanged. Implementation and CPU tests require separate approval before a model run.

## Question, reference and algebra

Report12 found positive joint-minus-summed-singleton mean NLL changes in all24 corpus/start/family groups, with22 intervals excluding zero. That observation does not determine how much of the residual comes from applying a nonlinear loss to finite logit changes, or from the joint model output differing from an additive output counterfactual. This study distinguishes those quantities without naming either a feedback or mediation term.

Keep the exact GPT-2 checkpoint, FP32/default SDPA, physical512 execution, original eight-passage calibration, directions, seeds and gate interventions. Native C is the second reference; P zeros source V at q0 in L4–L11. Every scientific arm uses that same boundary at all eight layers. Test all three families—remove_pc1, orthogonal_control, position—at each singleton L4…L11 and jointly. No chosen-layer confirmation, recalibration, new direction or rank is included. Singleton/orthogonal matching remains at each arm's own current state. The all-layer trajectories do not have matched actual perturbation norms across arms.

For family f, query q and full-vocabulary logits, define

```
delta_l = z_single(f,l) - z_P
z_add(f) = z_P + sum_l=4..11 delta_l
S = sum_l=4..11 [L(z_single(f,l), y) - L(z_P, y)]
N = L(z_joint(f), y) - L(z_P, y) - S
J = L(z_joint(f), y) - L(z_add(f), y)
R = L(z_add(f), y) - L(z_P, y) - S
N = J + R
```

All logits supplied to this calculation are the actual model FP32 outputs. Promote them to CPU FP64, add singleton differences in fixed L4→L11 order, and compute full-vocabulary log-sum-exp/gather in chunks of32 query rows. Maintain one family accumulator at a time. J/R/N use independently recomputed FP64 losses for P, all eight singletons, joint and additive outputs; do not mix the inherited FP32 CE ledger into their identity. Require finite tensors and max absolute per-query identity error≤1e−12. Keep the original FP32 metrics as a companion ledger and report their drift from FP64 CE, requiring maximum absolute drift≤1e−4 across all retained target rows. No clipping, replacement of failed values, change of ordering or automatic rerun follows failure.

R is the loss functional evaluated on the explicit additive logit construction. Convex cross-entropy does not imply that R is positive for arbitrary multiple directions. J compares the joint model output to that construction; it is not a pure internal-interaction, feedback or mechanism contribution. Their signs may oppose one another. No percentages of mediation and no division by a near-zero N are reported. The additive output is an oracle diagnostic using eight singleton executions, not a proposed efficient network replacement.

## Readout disagreement and fidelity

Per query retain KL(joint||additive), KL(additive||joint), KL(P||additive), KL(C||additive), and additive top-1 agreement with joint/P/C. Report actual KL roundoff without clipping, with the prospective inherited helper guard that no FP64 KL is below−1e−10.

Center vectors across vocabulary at each query, not across passages: center(x)=x−mean_vocab(x). Record L2, RMS and maximum absolute value of center(z_joint−z_add), and L2 of center(z_joint−z_P) and center(z_add−z_P). Store actual full-vocabulary selected rows so those centered vectors can be independently rebuilt. Do not infer agreement of output distributions from a small signed mean loss difference. Uniform vocabulary shifts must leave all loss, KL, agreement and centered-discrepancy results invariant. No covariance rank or common-mechanism claim follows from these vectors.

Targets2…511 (queries1…510) are primary. Query0/target1 is retained as unscored diagnostic evidence. Early targets2…127 and later128…511 are correlated views of the same512 execution. Additive calculations cover the existing511 output rows q0…510. The saved full hidden state includes q511, but there is no target512, primary loss, or saved actual-logit row at that query.

## Inputs and pre-output eligibility

Proposed corpus IDs are wiki_validation, mill, wollstonecraft, optimizers_new. Use existing local Wiki validation plus Mill's On Liberty (pg34901), Wollstonecraft's A Vindication of the Rights of Woman (pg3420), and the ordered joined adagrad.py, rmsprop.py, lbfgs.py. The availability-only inventory is review/future_calibration_resource_inventory.json. It is not eligibility evidence. Its exact source SHA256 values are pinned in the proposed helper; the code source inventory records PyTorch revision70d99e998b4955e0049d13a98d77ae1b14db1f45.

Matching pinned GPT-2 tokenizer files only; no checkpoint/model load. Preserve the existing book-wrapper stripping and two-newline code joining rules. Record original-file and processed-text hashes and code-component character ranges. New source paths are explicit helper arguments, not inferred from similar filenames. The private legacy config supplies the existing Wiki source and matching tokenizer. Natural/BOS uses one tokenization and changes token0 only; retain starting-token/calibration-overlap evidence without assuming new identities.

The prospective selection algorithm is frozen in this draft before any model response: grid j/1000 for j0…1000, preferred numerators125,375,625,875 in that order. For each preferred position, consider candidates by (absolute numerator distance, numerator), deduplicating identical character starts. Start=floor(j*(processed_text_length−8192)/1000); tokenize that exact8192-character buffer without special tokens, retaining first512 tokens and exact used-character extent. Select the first eligible span. Exclude overlap with historical exposed spans, full calibration/development buffers on the same verified source, and previously selected new spans. Require each optimizer used512-token span to be inside one component file; the unconsumed8192-character buffer may extend across a join. Reject exact full-window token hashes previously present in the scanned research manifests, including across source names. Four windows per corpus are required. If capacity is insufficient, stop with complete candidate/rejection evidence; do not silently reduce n, change grid, weaken exclusions or substitute a source.

Reuse the already independently verified localization_inputs_v2 exclusion inventory and verify its hash plus every referenced provenance hash. This inherits the GPT-2/Llama/shared-source OLMo span validation and full calibration/development buffers; it is not a new independent OLMo-tokenizer reconstruction. Add all15 localization passages from its hashed data manifest/tokens, including the four new Wiki windows. Verify current Wiki processed-text identity before reusing coordinates. Distinct-source coordinates must never be applied numerically to a different book or code file.

Before selection, scan every historical manifest/data-manifest/calibration-input JSON under the three known canonical super-weight research roots pulse_transfer/, pulse_continuation/ and pulse_causal_tranche/. Record the sorted inventory of paths and SHA256 values. Search source filenames, file hashes and processed-text hashes for the new books/code; any match is an unresolved historical exposure and must stop the helper pending exact source/span mapping, rather than being waved away. Record matching JSON pointers. Availability-only inventory and the new output directory are not historical runs. This scope does not examine or restart UIT or search unrelated personal work; absence of a match is only a statement about the inventoried research metadata, not all activity on the machine or model training text. Derivative metadata repetitions are recorded rather than treated as extra independent exposures.

The helper must preserve successful partial selections, rejection reasons, source inventory, hashes and failure status. It is written now but not executed or frozen until independent design review approves it. A successful prospective preflight would save16 passages/32 model inputs, tokens, directions copied byte-for-byte, data/exclusion manifests, start identities, and model_forwards=0/model_loaded=false/gpu_used=false. No new calibration fit or model forward is permitted there.

## Conditions, ordering and numerical controls

31 conditions: C/P/current/fullrank plus3 families×(8 singletons+all).32 inputs yield992 primary main forwards. Two native repeat/coordinate-identity anchors on natural window0 of each corpus add8:1000 main records. One same-shape changed-suffix control for every primary condition adds992:1992 forwards total, zero calibration forwards. Process one input and one family accumulator at a time. Current/fullrank remain exact P copies, repeat/identity exact C copies. They are controls, not independent samples.

Use the frozen GateLocalization implementation unchanged. Retain its1e−4 context/manual reconstruction guards, positive finite current mass, finite target sigmoid endpoints without clipping,1e−6 within-state directional-norm guard, exact source q0/Q/K/V relative to P, exact unaffected-layer contexts and exact singleton incoming b/mass/context and directional s relative to P. Record source/hash/mask/parameter versions and precision/backend settings as before. Gates outside the saved subset still get full runtime checks and independently computed fingerprints; storage reduction does not remove execution checks.

Rotate tokens128…511 by one for each suffix control; compare unchanged-prefix hidden, actual logits, gates, source states and current/delivered contexts exactly. Do not gather or score changed target128 onward. Gates for q1…127 may be retained, including unscored q127; scored control targets2…127 use queries1…126. Keep separately computed original/control fingerprints with dtype/shape. Additive readouts are computed only for the main original input; no extra additive donor, suffix efficacy score or model run is introduced.

## Raw evidence and independent readout reconstruction

Capture the actual input of lm_head, after final LayerNorm, not a nearby residual tensor. Save FP32 [1,512,768] arrays for all32 inputs and C/P plus27 scientific conditions:928 distinct main hidden states. Current/fullrank and anchors use exact hidden/logit equality to saved P/C plus independent hashes. Validate any post-LN-hook comparison at runtime; the lm_head input is the authoritative capture. Record the exact unembedding tensor name, aliasing, dtype, shape, content hash, optional bias and before/after parameter versions. Do not copy or export the model weights.

Save raw gates b/mass/target_b/target_mass for P at all8 layers, each of24 singleton arms at its affected layer, and each of3 joint arms at all8 layers:56 states per input. All32 inputs give1792 main states of [12,511] and1792 corresponding suffix-prefix states of [12,127]. C has no zero-value gate. Other layers and copy arms retain separately computed runtime fingerprints. Save raw source states for P for32 inputs; other-source equality remains runtime/fingerprint evidence unless present in an attention snapshot. No redundant full trace file, all-run prefix-hidden file, or full norm-tensor file. Save scalar operation/guard maxima and score-aligned norm summaries; do not describe unsaved context arrays as available for offline recomputation.

Actual raw full-vocabulary logits are saved for corpus window0/natural in all4 corpora, all31 primary modes, queries{0,1,31,63,126,127,254,510}:124 FP32 [8,50257] arrays. Also save each of3 additive FP64 [8,50257] arrays on those4 inputs. This selection precedes outcomes, covers q0 and both score-view boundaries, and is not a worst-case-selected sample. Four attention states are fixed: natural Wiki window0, main P and remove_pc1_all, at L4 and L11. No suffix or other attention snapshot is added after seeing results.

Offline review loads only the matching unembedding tensor and saved hidden states, in bounded CPU chunks. It does not load a whole model or rerun inference. Compute FP64 hW(+bias) and compare against the frozen actual-logit rows; reconstruct all other query readouts as an independent numerical approximation. Actual per-token FP64 losses and runtime additive metrics remain primary; reconstruction does not replace them silently.

Prospective sanity rule: with d=768, u=2^−24 and gamma_n=n*u/(1−n*u), n=2d+2, require each sampled actual-FP32 versus FP64-reconstructed logit error≤gamma_n*(sum_k abs(h_k*W_kv)+abs(bias_v)) + n*2^−126 + explicit FP64 slack1e−12. TF32 is off and the head operands/accumulation use FP32; record relevant runtime settings rather than inferring backend from an enable flag. Require all inputs/errors/bounds finite, exact h/W/bias identities/shapes and explicit zero-bias evidence if absent. This conservative dimension/dtype bound is a sanity guard, not a proof of scientific fidelity or an outcome-tuned constant.

Report max absolute error, RMS, relative L2 (zero denominator separately defined), and observed/bound ratios on sampled rows. Also report NLL/KL differences. For the actual sampled row max logit error delta, test NLL drift≤2*delta+1e−10 and both softmax KLs≤delta²/2+1e−10. These bounds apply to the two distributions on that row, not directly to J/R or an entire inference pipeline. For all-query reconstruction, report discrepancies of independently rebuilt J/R/N/KL/centered metrics versus saved runtime scalars; the full-vocabulary sampled rows directly validate actual GPU output only at their predeclared positions. Any material mismatch remains visible and stops claims until resolved without changing production outputs.

## Analysis, uncertainty and limits

Keep every corpus/start/family, singleton and passage. Main summaries:248 full and496 correlated early/later. Primary singleton/joint contrasts remain27 per input:864 records,216 groups. Additive decomposition produces96 per-input family records and24 corpus/start groups, with J/R/N means and paired intervals;48 segment group records repeat the calculation on early/later views. Use all four passages,2000 paired bootstrap draws, seed20261021, with the same resampling indices for J/R/N so the per-draw identity can be checked. Do not add interval endpoints or call natural/BOS independent samples. No multiple-comparison or population guarantee is claimed; the selected panel and frozen calibration condition the intervals.

Report both signed terms and their passage/token distributions, all disagreement/fidelity metrics, numeric sanity results, negative outcomes and failed hypotheses. New books/code change domains, so a different result cannot isolate calibration mismatch from domain/content differences. This study does not test the separate center×basis factorial, establish one shared scalar mechanism, identify compensation/feedback, improve generation, or deliver a faster replacement. Any selected-layer pattern remains a future fresh-confirmation question.

## Measured storage, prospective memory and stop rules

The previous localization directory was measured directly; its largest avoidable components are suffix arrays1,159,452,934 bytes, full snapshots534,252,975 bytes, prefix hidden duplicates369,166,376 bytes and traces195,449,621 bytes. Its measured RSS peak was3,008,792KiB and GPU allocation966,884,864 bytes. Those are historical measurements, not predictions of this runner.

review/gate_additive_storage_budget.json measures synthetic per-unit PyTorch serialization using the exact planned dtypes/shapes. It projects the following bytes:

| Component | Count | Serialized estimate |
|---|---:|---:|
| Post-LN hidden records |928|1,461,166,624|
| Main selected gate states |1792|180,140,800|
| Suffix-prefix selected gate states |1792|48,056,064|
| Actual full-vocabulary sampled rows |124|199,665,420|
| Additive sampled rows |12|38,622,012|
| Main metrics including FP64 CE |1000|17,681,000|
| Additive metric records |96|10,535,136|
| P source-state records |32|2,609,504|
| Four attention states allowance |4|50,331,648|
| Metadata/scalar JSON allowance ||50,331,648|
| Dependencies/input/code allowance ||8,388,608|
| Bounded failure capture reserve ||67,108,864|
| **Total** ||**2,134,637,328**|

The2GiB cap is2,147,483,648 bytes, leaving only12,846,320 bytes margin in this conservative unit model. The final complete synthetic writer layout must be measured before freeze; no inference if it exceeds the cap. Before model load require≥4GiB free (2GiB maximum new outputs+2GiB disk reserve). Before each next input, check remaining free≥2GiB and remaining projected bytes. Enforce the output cap on cumulative serialized outputs; preserve at least64MiB inside it for a bounded first-failure capsule. If metadata/snapshots exceed their allotted size, stop before the next model group rather than deleting old work or omitting declared raw evidence. No compression assumption or hardlinking of restricted/reference outputs is needed.

Failure preservation first checkpoints completed current-input hidden/gates/suffix/readout into their normal partial shards, which count toward the already declared payload rather than duplicate pending bundles. The separate≤64MiB capsule retains the current gate diagnostic, one full failing-layer QKV/current-delivered context pair, and both actual/reference logit rows at the first maximal-error query (or first nonfinite query), plus the frozen sampled rows when applicable. Keep full per-query scalar error arrays, independent full-array fingerprints, exact query/layer/vocabulary indices, input/map hashes and last-forward/record/check identities. It does not preserve every full pending logit matrix or accumulator; that audit limit is explicit. No unrelated tensors are duplicated. Implementation must measure its worst capsule size before freeze; if it exceeds the allocation, revise storage before outputs rather than truncate failure evidence afterward.

CPU payload estimates: FP64 [511,50257] accumulator205,450,616 bytes; each baseline/current FP32 matrix102,725,308 bytes; one32-row FP64 matrix12,865,792 bytes. Stream hidden/gate/snapshot outputs promptly and release completed family accumulators. Do not retain eight singleton full logits or all inputs/snapshots in RAM. CPU audit jobs stay≤3GiB RSS, one at a time if memory requires; require the production worker≤4GiB host RSS and one GPU model. Actual peaks/time and exact file inventory must be measured. These are prospective bounds subject to implementation testing, not claimed measurements of the future run.

Preflight/setup and runtime failures save distinct zero-forward or partial failed manifests, actual last-forward identity, pending record identity, active comparison, elapsed wall time, forward attempts/completed returns, memory and disk. All earlier runs, reports and archives remain untouched. No paid compute, credentials change, GitHub retry, external communications or unrelated UIT work.

## Required prelaunch tests and review

1. Tokenizer helper: pinned source/hash rejection, historical new-source collision stop, inherited-coordinate provenance, exact localization exclusion addition, actual-span/token-hash rejection, grid ties/boundaries, exhausted corpus, code join rejection, natural/BOS identity and zero-model failure evidence. Run only after independent design approval.
2. Readout: independent dense FP64 algebra versus chunked fixed-order accumulation; all3 families; null and opposing singleton changes; positive/negative R cases; row-wise vocabulary shifts; wrong order, duplicate/missing layer, shape/nonfinite failure; per-query J+R and actual score slicing.
3. Capture/reconstruction: random CPU model, actual lm_head input, nonzero biases where supported, exact aliases and hidden-to-logit readout; gamma-bound and scalar inequalities; perturbed hidden/weight/hash/row mismatch and zero-denominator cases. Predeclare comparisons before pretrained outputs.
4. Storage/orchestration: full32-input/1992-forward synthetic sequence, exact raw groups/counters, no changed-suffix scoring, exact singleton/source/copy controls, all4 snapshot states, fixed logit-row indices and all928 hidden states; actual serialized-size cap and bounded failure capsules; stale pass cleared and disk pre-load/partial failures preserved.
5. Independent scientific design review, then independent frozen implementation/CPU review. Root alone launches one pretrained model after both pass. Independent numerical review precedes any finding or report update. This draft itself is not frozen or approved for launch.
