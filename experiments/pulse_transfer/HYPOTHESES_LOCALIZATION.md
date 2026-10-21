# Living transfer hypothesis ledger

Updated21 October2026 after six sequential transfer batches. Research remains open. OLMo findings are scoped to its pinned checkpoint; shared magnitude is not assumed to imply shared mechanism. T1–T5 were declared before the initial results; later entries are explicit follow-ups, not retroactive predictions.

| ID | Current status | Evidence | Remaining discriminating test / limitation |
|---|---|---|---|
| T1 | Mixed; largest contribution is not necessarily strongly loss-critical | GPT-2 selected row/column loss effects near1×; Llama whole-column effects substantially larger | One activation-weighted candidate per model does not exhaust all super weights |
| T2 | Universal row-worse-than-column rule rejected for these candidates | Llama column-minus-row NLL positive all96 initial comparisons; GPT-2 both weak | Precision, training, size and inputs are uncontrolled across models; no unique architecture attribution |
| T3 | Predeclared history signature absent at both candidates | Neither satisfies tenfold median suppression plus relative-magnitude screen | No claim that other newline circuits are absent |
| T4 | Supported conditional source K/V roles in Llama; qualitative OLMo pattern differs | Mixed donor factorial; clean V helps more than clean K; zeroV about3× versus source blocking19–54× natural PPL; exact boundary closures | Need query/gate interventions to separate fixed value bias from dynamic gate control; no additive mediation percentages |
| T5 | Approximate fixed source replacement supported in Llama; independent numerical review passed | Fresh natural calibration,14/16 unseen natural evaluation starts and unseen BOS; pooledKV natural PPL.9985–1.0015,agreement97.94–98.68%; BOS1.0019–1.0089,agreement96.81–98.14% | Teacher forcing only; no faithful generation/computational speedup claim; dynamic gates remain endogenous |
| T6 | Simple within-L2 three-column hidden-loss redundancy explanation unsupported in GPT-2 | Joint top3 deletion greatly lowers MLP447 but leaves loss near native; whole MLP447 zero similarly weak | Other redundant computations not excluded |
| T7 | Conditional later-MLP coordinate regrowth supported in GPT-2 | E×M positive in all32 inputs; blocking M without E weak; full-vector reconstructions; both coordinates reappear afterL3 | Native raw trajectory not restored; current lesion spansL3..11, so minimal layer/neuron circuit remains unknown |
| T8 | Additive E/A/M interpretation rejected | After E+M, additionally blocking A improves all32 inputs | Need recipient-dependent geometry and layer-local interventions |
| T9 | Early regrown coordinate values alone insufficient for fidelity | Rescaling remaining residual after E yields similar early writes but much worse loss | Does not uniquely isolate LayerNorm or semantic content |
| T10 | Llama complement lesion is more harmful than whole-column lesion | D removes everything but output588; D>N in all32 inputs, negative complement interaction intervals | Conditional geometry needs directional interventions; squared weight-energy share alone is not explanatory |

## Seventh batch update, 21 October 2026 13:01 UTC

Independent numerical review passed the layer-local GPT-2 batch (776 forwards,
fourth fresh panel). T7 is now localized: after E erases source coordinates
447/138 after L2, blocking their L3 MLP writes increases loss in all32 inputs,
all eight group intervals exclude zero. Natural-start PPL ratios are1.042–1.089
for E alone,1.032–1.081 when only L3 can write these coordinates, and3.553–11.114
when L3 is blocked but later writers remain. BOS equivalents are1.065–1.327,
1.060–1.321 and12.580–130.208. Blocking L3 without E is near neutral
(.999423–1.000147). Final-layer-only writes cannot transmit to scored tokens,
confirmed exactly in32 cases. The only-L3 condition still depends on native
attention, other coordinates, later MLP computations and a changed trajectory;
it is not whole-network sufficiency or faithful native restoration.

T11 (new prospective test): Is the L3 conditional benefit transmitted via
downstream source keys, values or a joint state? The frozen GPT2_REGROWTH_KV
protocol compares E with B=E+blocked L3 writes, patching all source K/V in
L4..11 separately on a fifth fresh panel. The first-boundary query equality,
diagonal/cross-recipient closure and zero-value/source-block controls delimit
the claim. Numerical results are pending.

Closest prior work directly addresses GPT-2 massive coordinates, layer/neuronal
compensation, conditional reserves and coordinate/complement decomposition.
These experiments are targeted reproduction/extension, not discovery of the
coordinates or compensation. See revised Report8 and its pinned primary-code
citations; the inspected WDD author code is not a reproduced prior result.

Next priority is the declared GPT-2 K/V test, then evidence-dependent query
or directional interventions. Llama gate freezing or a calibrated gate subspace
remains an informative follow-up. Freeze any selection on development inputs
before new evaluation. Centered low rank alone will not be called one mechanism.

Private repository creation remains denied and unexecuted. No retry or alternate route is authorized by the parent while confirmation is pending; local research and review continue.

## Eighth/ninth batch update, 21 October 2026 13:24 UTC

T11: source-value transmission is supported in the E/B lesion family. The
616-forward fresh-panel K/V factorial and independent audit show B values
raise NLL under either donor-key state in all32 input/start pairs. The full
source donor boundary closes exactly; first later-token queries are unchanged.
Keys have smaller but nonzero, start-dependent effects, and their interactions
are not additive mediation fractions. Zero V preserves much better loss than
source blocking in all32 pairs; this does not imply exact output restoration.

T12: neither value amplitude alone nor direction alone suffices. The520-forward
headwise norm/direction factorial with E keys passed independent review. Both
simple factors increase loss in all32 pairs. The interaction is positive31/32
(one Wiki natural exception), with all8 group intervals positive. E-direction/
B-norm natural PPL2.531–5.035; B-direction/E-norm1.162–1.329. This is a joint
layer/head profile, not a globally larger scalar norm: L11 head8 is smaller
in B, whereas its other11 heads are larger. Randomized-direction controls have
meaningful exceptions, including improved loss in the BOS optimizer group.

Native C with zero source values L4..11 gives PPL1.0014–1.0131 across groups,
but only94.4–97.3% argmax agreement. Low loss is not faithful replacement.
Thus an indispensable linguistic-content explanation for these source values
is unsupported by these interventions; a harmful-value suppression role after
the imposed lesion is supported, with scope limited to the tested boundary.

T13 (prospective): learned query bias may interact with retained source access
when V0=0. The GPT2_QUERY_SOURCE protocol crosses bias removal with source
blocking, includes a fixed norm-matched randomized-bias subtraction and saves
actual attention diagnostics/snapshots. It tests effect modification by source
access; a main effect or attention-mass correlation alone cannot prove a
source-only mechanism. Seventh-panel preflight passed. No pretrained result yet.

All three new batches reproduced through the portable driver:1,912 records
and every retained raw tensor file byte-identical. Six earlier independently
verified rerun outputs were reused as immutable dependencies. This is an
additional same-machine validation, not an independent researcher replication.
Report9 has passed independent scientific/figure/HTML review and is saved in
Library. Useful query/gating and calibrated causal-subspace tests remain.

## Tenth batch update, 21 October 2026

T13: reviewed776-forward query/source factorial supports a conditional query
bias role, not a uniform source-only mediation account. Native C with zero V
has mean scored source mass .4865 natural/.4881 BOS; removing learned bias
reduces it to .2726/.2740, while matched random subtraction leaves .4782/.4796.
Source allocation remains substantial. Bias removal PPL ranges .979–1.187
natural/1.006–1.177 BOS, compared with source blocking1.793–3.738/1.789–4.146.
Loss and interaction signs vary across corpus/world/start; natural Du Bois
improves under bias removal. Argmax fidelity drops to73.2–90.2% naturally.
The twelve raw snapshots independently reconstruct in FP64 within1.02e-5;
all source-state and mask controls passed. Known query-bias routes are prior
art, not novel discovery. Full details in report10_query_findings.md.

T14 (prospective): query-bias removal changes both non-source allocation mass
and its conditional value mixture. GPT2_QUERY_MIXTURE separates their donor
states at the c_proj input in native zero-V GPT-2, with both recipient closures
and a fresh eighth panel. Stable conditional-softmax diagnostics explicitly
guard small-mass division; no clipping or posthoc threshold changes. This is a
defined boundary intervention, not native additive mediation. Results pending.

Reviewed Report9 local release commit80b9c2d74e29d6bf5db0e3703bc837b3446b32fa
and its private raw archive are preserved. GitHub creation remains denied and
unexecuted; no retry or alternate route. Research continues locally.

## Eleventh batch update, 21 October 2026

T14: the520-forward fresh eighth-panel donor factorial passed independent
numerical audit. At fixed P mass, R mixture gives PPL1.081–1.259 across groups;
at fixed P mixture, R mass gives .882–1.017. Mass effects have negative grouped
intervals in books/code at both starts and both mixture states (12/16groups),
while mixture effects have positive intervals15/16groups. Wiki mass effects
are small with intervals spanning zero. Interactions have mixed signs. Thus
allocation and mixture changes oppose one another on these book/code panels;
they are not interchangeable and no additive mediation fraction follows.
Mixed-cell agreement with native is80.0–92.0%, so improved loss is not faithful
restoration. All328 pertoken/final-hidden-hash output closures passed. Small
donor masses (minimum3.31e-12) and gains up to6.79million were retained without
clipping; direct stable-mixture and FP64 snapshot checks stayed below1e-4.

Next prospective line: fit centered gate subspaces only on fresh WikiTest/Hume
calibration, then causally retain/remove directions on fresh held-out corpora.
Use a position/key-count log(q) control because pooled covariance can mistake
a shared length trend for a common learned mechanism. Compare bounded gate
replacements, fixed means and matched random subspaces, with identical-token
128/512 prefixes. This line is still in design review; no calibration fit or
held-out result exists yet. Known attention algebra and zero-value allocation
are credited to primary prior work, including Massive Activations AppendixB.4.

Report10 passed scientific/figure/HTML and numeric-export reviews, and is saved
in Library. Portable query replay reproduced1,296 records and all44 recursive
raw tensor files byte-identically, with independently regenerated files and
nine explicitly reused verified prerequisite outputs. The final independent replay
audit passed; local release packaging is in progress. This is same-machine
validation; no external researcher replication or GitHub creation occurred.

Report10 release is complete locally: commitdbf7f2dbfa25d51219272bccb8b89b307cdc5c21,
899-file private archive verified by per-file SHA256/ZIP CRC, while only the
readable HTML was uploaded to Library. The next calibrated gate implementation
is in independent review; no new calibration/model inference has begun.

## Gate-subspace execution failure, 21 October 2026 14:22 UTC

T15 remains unresolved. The reviewed, frozen gate protocol stopped after8
calibration and25 evaluation forwards (24 records appended), at the first
native-C128/512 prefix comparison. Maximum final-normalized hidden drift was
.00227355957, exceeding the preset absolute.001 threshold; scored loss drift
was.00016212463 and argmax agreement1. The original failed manifest, raw
calibration, first-input partial interventions and failure tensors are retained
in gpt2_gate_subspace_run. No threshold was changed and no completed-study
claim is made. A separate native-only numerical diagnostic is being designed
to distinguish execution-shape sensitivity from suffix dependence. Synthetic
CPU checks did not predict this pretrained numerical failure. Calibration and
any partial outcomes must not be used to select favorable ranks or controls.

The separately reviewed18-forward native-only diagnostic completed. Default
SDPA reproduces the failed comparison exactly; hook-free, observer and frozen
study-hook outputs agree exactly. Within each backend, fixed512-shape rotated
and constant suffixes and the explicit padding-mask control leave the prefix
exact. Both128/512 pairs still exceed the old hidden absolute threshold:
.00227356 default versus.00106049 forced math. Their mean FP64 prefix KL is
about6.55e-10; predictions agree. Observed operators confirm memory-efficient
attention versus math. The first recorded length divergence occurs at L0's
QKV linear output, before attention, with identical input and ln1 states.
These are diagnostic observations awaiting independent raw-data audit; they
do not pass the failed gate study, uniquely blame SDPA, or establish universal
prefix invariance. A separately versioned numerical policy and same-shape
suffix controls are under review before further gate inference.

The native diagnostic passed independent audit (all18 shards,25 pairs,
276 FP64 score comparisons,192 context reconstructions,8 failed alignments;
maximum saved-summary discrepancy1.42e-14). The conclusion is bounded
numerical shape sensitivity on this input, not a universal theorem.

T15 V2 policy is independently approved for implementation, not yet inference.
It keeps all original maps/ranks/controls and the default backend; uses
explicit post-failure local relative-L2<=1e-4/global<=1e-5 hidden budgets plus
the unchanged loss/context guards; reports old absolute failures; and adds
all672 fixed-shape suffix controls. The first Wiki passage is marked exposed
in both starts/lengths; a fixed unexposed n3/n4 view accompanies original n4
summaries. Primary V2 will reuse frozen calibration and consume2032 new
forwards. An explicit separate8-forward rebuild is being independently
reviewed for portable reproduction. Neither policy approval nor diagnostic
success supplies a completed gate-dimensionality result.

The standalone calibration-only rebuild completed eight forwards in3.61379s,
with peak GPU allocation645,195,776bytes and RSS1,828,584KiB. Original raw gates
and fitted maps reproduced byte for byte (d41d5a91… and ed62f089…); independent
actual-output review is pending. No evaluation was forwarded or used to choose
a map. This is same-machine reproducibility, not independent scientific
replication or evidence for a low-dimensional causal mechanism. Primary V2
continues to use the original reference and remains pending implementation review.

Independent rebuild audit passed:176 tensors,64 operation checks, original
reference hashes unchanged; no pretrained/GPU reload by reviewer. V2 sources
are frozen for independent implementation review. No V2 inference yet.
The figure/report wording scopes source-query identity to the zero-V baseline
P; zeroing V0 can itself change source-query computation relative to native C.

## Second gate failure, 21 October 2026 15:05 UTC

The independently reviewed V2 run failed its declared local relative-hidden
criterion on firstWiki0/BOS native C128/512:1.0108874094e-4>1e-4 at query110.
Globalrelative2.44156e-6, maximum lossdrift.0001912117, predictions agree.
The failure remains failed, with90 completed/attempted forwards,68 appended
main records,21 suffix records,10.59108s,966942208GPUbytes,2235664KiBRSS.
Original and V2 source files, calibration and failure tensors are frozen;
there is no automatic tolerance increase or restart. T15 remains unresolved.

A separately named fixed512 study is proposed for independent review. It
retains all21 arms/maps/ranks and32 inputs, exactsuffixcontrols and original
context guards, but explicitly abandons standalone128 execution validation.
Early/full/later scoring would be separate views of512-token executions,
not a context-length causal claim. Both failed studies remain in evidence;
firstWiki exposure and fixedn3/n4 views remain explicit. No new inference is
authorized by this proposal alone.

V2 failure independently audited:the failed C pair is distinct from the
most-recent P512 forward,23 completed natural prefix pairs and21 exactsuffix
controls verified,60 FP64snapshotstates maximum componenterror2.9832e-5.
No final parameterclosure was reached. The separately declared fixed512
proposal passed independent design review; implementation/finalreview pending.
It costs1352forwards ifcompleted and keepsall21scientificarms; it narrows the
execution-shape estimand, with no newhidden tolerance or standalone128claim.

### 2026-10-21 fixed512 gate milestone (reviewed)

The narrower frozen fixed512 study completed1352forwards (680main+672suffix),227.0worker seconds,966876672BpeakGPU and2655644KiBRSS. Independent saved-output audit passed: all reductions/views/contrasts,672saved suffix closures,640source identities,10240gatechecks,40FP64snapshots. Full logits/contexts remain partly hash-only. Both earlier cross-length runs remain failed, and independent cross-length numerical validation is unresolved.

T15: leading gate covariance does not establish a faithful scalar replacement. Detrended PC1 explains40.47–61.12%calvariance; retaining it perlayer yields1.148–1.292PPL/P and80.07–87.40%agreement. Six retained directions yield1.0105–1.0954 and90.39–95.20%. PCAretention improves agreement versus oneQRbasis all24mean rank/corpus/start comparisons, but loss reverses on books. RemovingPC1 worsensWiki withpositivepairedintervals and improvesDuBois withnegativeintervals; Hobbes/code mean improvements remainuncertain. These are conditional boundary interventions, not oneglobalmechanism, nativefaithfulreplacement or priority.

Numerics: minimumcurrentmass5.11798e-11, maximumgain1.9538970624e10, noexplicitclipping, some sigmoidtargetssaturateto1. Everyunchangedreconstructionguardpassed, maximumFP64snapshoterror2.9832e-5; inference remains precision/inputscoped.

Competing hypotheses remain: accumulated/feedbackeffects of applying maps at8layers; mean/positionbaseline misfit; calibration-domain mismatch; highvariancedirectionsencodingnativepreferences withnonuniform empiricalutility. Independent reviewer recommends frozen single-layer vsjoint panel before refitting to isolate layer accumulation without newcalibrationdegreesoffreedom. Prospective design andfreshheldout inputs required; donotselectranksusingnewtestoutcomes. Same-machine standalone1360forwardreplay isbeingreviewed andhasnotyetexecuted. GitHubdenialremainsunexecuted; noretry.

### 2026-10-21 standalone gate replay and next localization design

The standalone1360-forward route completed (8newcalibration+1352study) in258.179driver seconds. After preserving an initial metadata-schema comparison failure and independently reviewing a narrow compatibility fix, all680 numerical records and123828 saved tensors matched exactly. All105 rawfiles werebyteidentical withdistinctinodes, independently provenance-audited; this remains same-machine reproduction, notexternalreplication. Original calibration bookkeeping counters missing from rebuildrecords were not invented; allsharedfields/64gatecheckscompared. Report11reviewed/delivered andlocalpackagecommitted; GitHubremainsdenied/unexecuted.

T16 prospective: test whether PC1removal effects concentrate inindividualL4..11 layers orappearprimarilywhenjointlyintervening. Compare eachsingleton/allset PC1removal, a fixedorthogonal equal-local-logitnormdirection, andposition-onlymap; includeC/P/exactcopycontrols, unchangednumericalguards andsuffixcontrols. Incoming singletonstateisP; jointconditionsdeveloptheir ownstates, so normmatchingislocal/hypothetical, notequalperturbationsacrossdifferentjointtrajectories. Joint-minus-summed-singletons isdescriptivenonadditivity, notmediation. Thisdoesnotaloneexplainallrank1retentionloss.

Inputpolicy failure preserved: original16-passagefrozengrid foundonly3optimizerwindows afterverifiedshared-source OLMo/Llama/GPT2 exclusions. Intervalpacking provesmax3onthegridonly. Reviseddesign retainsall12proseand3coderecords exactly, withactualn3code/n4prose. No exclusionsweakened, nofinerstartsearch, nohiddenfile substitution. Tokenizer-onlypreflight0forwards;1192prior-spannonoverlapchecks and48fullcalibration/developmentbuffers independentlyaudited. Revised31condition×30inputscope=938mainrecords(including8anchors)+930suffixcontrols=1868forwards,0newcalibration. Independentdesignapprovalcomplete; implementation/CPUreview requiredbeforeGPU.

Competing predictions: one strongsingleton versusmatchedcontrolsupports layer-specificdirectiondependence;weak singletonswithstrongjointdamage motivate a laterpair/blocktest butcannotseparatefeedbackfromnonlinearcomposition alone. Ifcross-corpussigns persistatlocalizedlayers, usefreshsame-domainandbalancedcalibration withseparatecenter×basis crosses. Do notchoose a layerfromnewheldoutresultsandclaimconfirmationonthesamepassages.

### 2026-10-21 reviewed layer-localization milestone

T16 is now measured on the frozen15-passage panel (n4 prose/n3 code):1,868
forwards,938 main records and930 suffix controls,334.222worker seconds,
966884864B peak GPU and3008792KiB RSS. Independent raw audit passed all
summaries/contrasts,720 singleton b/mass closures,480 directional s closures,
14,400 gate checks and52 FP64 snapshot states. The original maps remain frozen.

Joint PC1 removal increases Wiki perplexity/P to1.0514/1.0533, both intervals
above1. Du Bois means0.9488/0.9476 recur in the prior direction but their new
intervals cross1; Hobbes0.9668/0.9734 has intervals below1; code mean benefits
remain uncertain. Native output fidelity is not restored: joint-removal
agreement withP87.60–91.11%. L9/L10 removal lowers book loss, while L10 raises
Wiki loss and has opposite direction-control contrasts across Wiki/books.
L5's book benefit is not uniquely PC1-specific. L4's nearly single-head
calibration loading does not make its frozen-direction removal loss-critical;
equal logit norms produce unequal context changes. These are scoped causal
patch effects, not head dispensability or a shared mechanism count.

All24 joint-minus-sum NLL means are positive,22 intervals exclude0 (the two
DuBois removal groups do not). This rejects a simple additive finite-loss
description but does not isolate network composition from loss/readout
nonlinearity. Neither feedback nor additive mediation follows.

Numerical limits: current mass down to9.2813e-12 and gain up to1.0774e11,
without clipping. The worst-gain DuBois0/BOS position_L4 case lies outside
fixed raw snapshots; its runtime check passed unchanged1e-4. Successful full
output logits are absent; attention contexts are raw only in fixed snapshots,
with runtime/hash evidence elsewhere. Initial independent audit failure was
only a5.33e-15 CPU/GPU FP64 norm reduction equality; preserved source/log,
audit-only2e-12 comparison bound, no model rerun or production change.

T17 prospective: separate finite NLL composition using a fresh-panel
additive-logit counterfactual z_add=z_P+sum(z_single-z_P), retaining all8
singletons and all3 families. DefineJ=L(z_joint)-L(z_add) and
R=L(z_add)-L(z_P)-sum(L(z_single)-L(z_P)); nonadditivity equalsJ+R exactly.
Report centered-logit discrepancy, KL(joint||additive) and per-token terms.
NeitherJ norR is a pure mechanism/mediation estimate; convex cross-entropy
does not guaranteeR's sign for arbitrary directions. New source eligibility,
frozen protocol, bounded storage and independent review precede inference.

T18 deferred: fresh-calibration center×basis factorial, especially the
now-development L10 candidate and all-layer maps, on independently selected
confirmation inputs. It addresses calibration sensitivity/corpus sign rather
than the NLL nonadditivity question. No new calibration or selected-layer
confirmation has yet been run. GitHub creation remains denied and unexecuted;
local research, reviewed Library reports and the local repository continue.
