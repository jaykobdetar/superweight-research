# Calibrated causal gate subspaces in GPT-2

Prospective protocol, 21 October 2026. This follows the reviewed query-bias mass/
conditional-mixture donor factorial. That experiment uses same-input donor
trajectories and is not a deployable replacement. This experiment asks whether
freshly calibrated, simple gate maps preserve behavior while keeping the
current non-source conditional value mixture. No dimensionality result is
assumed in advance. Calibration, directions and ranks must be frozen before
any held-out inference. No checkpoint or weight-value sweep is involved.

Use pinned GPT-2 FP32 SDPA. Indices zero-based. Native C has no coordinate
lesion. P sets source V0=0 in all heads of L4..11, leaves Q biases intact, and
uses the explicit causal-mask reference. Every active gate intervention
preserves source query0 and acts at each layer's current attention context,
not a cached donor trajectory or future-layer state.

At query q>=1 define b_h(q)=logsumexp(scores_h,q,j for1<=j<=q)
minus scores_h,q,0. Then a_h=sigmoid(b_h) is non-source probability mass.
Actual model context c is measured from native SDPA; current mass is also
summed directly from probabilities. Preserve the current conditional mixture
using c*(a_target/a_current). Check this against a_target*u_stable where
u_stable uses non-source-only softmax. Record finite positive current mass,
minima, gain and absolute reconstruction errors; threshold1e-4, no clipping.
Finite projected logits can produce exact FP32 sigmoid endpoints0 or1;
these are allowed and counted. Only the current denominator mass must be
strictly positive. Tiny roundoff above1 in a native probability sum is
recorded, not clipped; fitted constant mean mass must be in(0,1].
Keep exact current/fullrank copies for identity closures. Validate full-rank
projection reconstruction separately, rather than relying only on bypasses:
full-rank logit error, mass error and rescaled-context error each must be
at most1e-4. Its delivered context and target mass still copy the actual
current values exactly. These numerical diagnostics are not alternate runs.

Position/key count is a competing explanation: even homogeneous non-source
keys introduce log(q) in b. Therefore fit causal directions primarily to
r(q)=b(q)-log(q), centered by its per-head calibration mean alpha. Fit one
12x12 sample covariance (normalization1/(N-1)) and orthonormal eigensystem
per layer, FP64 on CPU. Eigenvalues descend; compare covariance eigenvalues
to squared centered-data SVD singular values/(N-1). Canonicalize each right
singular vector's sign to make its largest-absolute coordinate positive. Also
retain centered spectra/covariances/SVD singular values for raw mass, raw b,
detrended r, and b after subtracting its calibration mean at each query
position. Document covariance normalization and total sample count. Do not
equate low rank with a single mechanism. Query-position centering is a
descriptive nuisance check, not an evaluation-fitted replacement.

Calibration: eight natural-start512-token passages, four each Wiki test and
Hume at .25,.45,.65,.85 of the legal8192-character-buffer start range. Exclude
complete buffers of earlier GPT-2 development and Llama fresh calibration;
verify hashes/offsets and freeze calibration tensors before evaluation.
Fit on scored queries1..510 (4080 vectors/layer); save all raw calibration
gates. No rank, mean, basis or control selected by held-out performance.
Use a fixed CPU QR random basis per layer, seed20261025, with deterministic
sign convention (nonnegative QR diagonal); one random control is not a
significance test. Save eight leave-one-passage-out spectra per space and
principal-angle cosines at ranks1/3/6 against the full fit, descriptively;
do not use them for selection. Natural starting-token IDs and overlap with
held-out natural/BOS identities are recorded; unseen identities are not
assumed. Calibration metrics have no external C reference: their saved KL
zeros mean Runner self-reference, not fidelity to native C.

Held-out ninth panel: four passages each from Wiki validation, Du Bois,
Hobbes and optimizer code at .075,.275,.475,.675. Verify actual used512-token
spans exclude all32 earlier spans/corpus. Each passage has natural/BOS starts
(token0 replacement only) and lengths128/512 from the identical token prefix.
Score targets2..127 and2..511. Length groups are correlated, not independent
new samples. Report shared-prefix metric/hidden-state drift at identical
positions, plus later-target outcomes separately; different endpoint averages
do not by themselves show an effect of unseen future context. Require maximum
shared-prefix per-token loss drift<=1e-3 and full final-normalized hidden
drift<=1e-3. Report argmax agreement and gate drift descriptively; a near-tie
argmax change need not be exact. Shared scores use126 targets2..127; later
512 scores use384 targets128..511. All calibration statistics are fixed;
projection acts independently at each current query and is suffix-independent.

Predeclared21 conditions per input:
* C, P_native-mask, P_explicit-mask, P_current-copy, P_fullrank-copy;
* fixed calibration mean b, fixed mean mass, position-only alpha+log(q),
  and globally centered raw-b rank1 projection;
* detrended-r PCA retain ranks1/3/6, and remove those ranks;
* matching fixed QR retain/remove ranks1/3/6.

For retain: b_hat=alpha+log(q)+U_k U_k^T(r-alpha).
For remove: b_hat=alpha+log(q)+(I-U_k U_k^T)(r-alpha).
Apply sigmoid to obtain bounded mass; no posthoc clipping. Fixedmeanmass uses
the direct calibration mean of actual a. Fixedmeanlogit uses sigmoid(mean b).
Global-b-rank1 uses its own calibration mean and leading raw-b eigenvector,
an explicit comparison to the positional nuisance model. The position-only
map uses the same alpha as detrended PCA. All maps work per current query/
layer across12 heads and require no later-layer lookahead.

64 evaluation inputs×21conditions=1344forwards; native repeat/coordinate
identity controls at natural window0 per corpus and both lengths add16;
eight calibration forwards give1368total and1360evaluation records. Do not
quietly discard a condition to meet an earlier rough estimate. Preserve all
negative/failed experiments. Require exact current/fullrank/native repeats
on per-token metrics and full final-hidden hashes; report mask drift. Save
actual before/after context and Q/K/V/mask for a frozen subset sufficient
for independent FP64 reconstruction and causal projection checks. The frozen
subset is Wiki validation window0, natural start, both lengths, L4/L11, ten
modes: P/fullrank/meanlogit/meanmass/position/global1/PCAkeep1/PCAremove1/
QRkeep1/QRremove1 (40 layer snapshots). Record
source-state identity, immutable parameters, finite shapes and hook locality.
There are1488 runtime control records (1280 source identities,128 copy
closures,16 native anchors,64 mask comparisons),680 paired prefix checks,
10240 evaluation gate-context checks plus64 calibration checks. Every
forward has eight projection/query/context checks. Save the raw calibration
tensors and maps once; freeze their byte hashes and a tensor digest before
held-out inference. Save raw evaluation gates per input and complete paired
prefix hidden states in separate files to keep memory bounded. Failure
checkpoints retain current Q/K/V/mask, b/current/target mass and map hash.

Evaluate paired NLL, KL and argmax agreement relative to both native C and
zero-V P, because P already changes native predictions. KL directions are
KL(C||trial) and KL(P||trial); argmax agreements use the corresponding
reference predictions. Both baseline logits are retained only for the
current input; no extra forwards are required. Report paired n=4
bootstrap2000 seed20261021 per corpus/start/length; intervals conditional on
fixed calibration and inputs, no multiple-comparison/population claim.
Predeclare32 paired NLL contrasts/input: each of16 active maps minus P;
six PCA retain/remove arms minus matched QR and the same six minus position;
global1 minus PCAkeep1; meanmass minus position; meanlogit minus position;
and P minus C. This gives2048 paired contrast records and512 grouped
summaries. Report all336 condition summaries and504 separate segment
summaries; anchors are checks, not extra samples.
Compare PCA retain/remove with rank-matched QR and position-only controls;
do not select the winning rank as confirmatory. Report calibration spectra
separately from causal outcome effects and natural/BOS/length transfer.

One local GPU model; CPU fit/review under3GiB where feasible. Implementation
and final protocol still require independent review before any calibration
or evaluation run. No faithful generation, speedup, universal architecture
or priority claim follows from this design alone.
