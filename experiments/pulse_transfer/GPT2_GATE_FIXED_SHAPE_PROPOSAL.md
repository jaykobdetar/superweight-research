# Proposal: causal gate subspaces at a fixed execution shape

Status: design proposal for independent review. No inference or implementation
is authorized by this file. It supersedes neither failed run.

## Reason for a different study

V1 failed its native 128/512 absolute-hidden guard. V2 failed its revised
relative-hidden guard at the first Wiki passage's BOS native baseline:
max query relative L2 0.00010108874094104775 versus limit0.0001, at query110.
Global relative L2 was2.441561531347229e-6; maximum scored loss difference
was0.00019121170043945312 and predictions agreed. This is another failure,
not an almost-pass. V2 consumed90 forwards, saved68 main and21 suffix-control
records, and took10.5910792351 seconds. Failure resources and frozen code are
preserved. Independent saved-output audit is pending.

The native diagnostic already showed that cross-length execution can change
arithmetic before attention. Another tolerance adjustment based on these
cases would not resolve whether calibrated gate directions matter. Instead,
separate that unresolved execution-length numerical question from a narrower
causal intervention study at one fixed physical execution shape.

## Scope and preserved scientific choices

Every model forward has exactly512 input tokens. Reuse all sixteen original
passages, both natural/BOS starts, the exact original calibration maps and
all21 scientific conditions with ranks1/3/6, retain/remove, matched fixed QR,
fixed means, position-only, raw-logit PC1, C/P and exact-copy controls. No
rank, map, input or condition is selected using outcomes. The failed V1/V2
runs remain visible and their partial records are not spliced into this run.

The first Wiki passage remains exposed for both starts. Retain all n4 tables
and the same fixed unexposed view (Wiki windows1..3 n3, other corpora n4),
with actual-n paired bootstrap2000 seed20261021. No independent-dataset claim.
The primary analysis compares intervention outcomes at the same512 shape
and identical input. Calibration remains eight separate natural-start
passages from Wiki test/Hume; no new fitting.

Report whole scored targets2..511, early targets2..127 and later targets
128..511 separately, relative to both C and zero-V P. The early segment is
an early-position score view of a512-token execution, NOT a fresh128-token
execution, an independent sample, or an experiment changing available past
history for matched targets. No short-execution fidelity claim is made.
The study therefore gives up the original cross-length numerical validation;
it does not report that check as passed or silently enlarge its tolerance.

## Controls and unchanged numerical guards

Keep all32×21=672 same-shape suffix-rotation controls, each rotating positions
128..511 by one while preserving0..127, masks, mode and frozen maps. Require
exact first128 hidden/logit/current-and-delivered-context/source-QKV/gate
states and exact shared scoring, preserving chunk32 arithmetic. Never gather
changed targets128 onward for these numerical controls. Save successful
prefix hidden/gate/source/metric tensors, separate logit/context hashes and
equality flags, fixed raw snapshots, and both full sides on a failure.

All original1e-4 attention/conditional-mixture/gate/context reconstruction
guards remain in force, with finite positive current mass and no clipping.
Current/fullrank copies, repeated native, coordinate identity, source-state
identity relative to zero-V P, map immutability and model integrity remain
required. Native/explicit-mask differences are reported. No cross-length
comparison is run or tested against a new tolerance. Every failure is preserved
with attempted/completed counts, timing and peak memory; no automatic remedy.

No padded128 control is proposed for the primary run: it would add another
mask variation but would not reproduce physical128 execution. The exact
changed-suffix controls and static query-locality review directly address the
causal locality relevant to this fixed-shape study. A separately declared
future numerical study could revisit length/backend sensitivity.

## Expected accounting

32 inputs×21 conditions=672 main forwards; natural window0 per corpus has
two native repeat/coordinate-identity anchors, adding8. Main records680,
suffix controls672, total1352 new forwards, zero new calibration forwards.
Main runtime checks744 (640 source identities,64 copy closures,8 native
anchors,32 mask comparisons); gate-context checks5120 main+5120 suffix=10240.
Every forward has eight projection/query/context checks and30 writer-locality
checks. Main snapshots20 plus suffix20: Wiki window0 natural, L4/L11, same
ten modes as the original protocol. No snapshots are chosen after results.

Expected whole-condition summaries168; early/later segment summaries336;
32 predeclared contrasts per32 inputs=1024 raw contrasts,256 grouped
summaries. Both full and unexposed versions remain available. Store raw
per-token metrics, gate tensors, prefix hidden states, source states,
calibration dependencies, frozen scripts and exact manifests.

Total gate-line production expenditure would be33(V1)+18(native diagnostic)
+8(rebuild)+90(V2)+1352(fixed-shape)=1501 forwards if this study completes.
Same-machine replays and CPU synthetic tests are separately accounted.

## Interpretation limits

Covariance describes variation; causal retention/removal tests whether the
specified directions matter at this boundary. A separate map per layer is
not one global scalar mechanism. Altered earlier layers change later current
states; this is not a donor-trajectory mediation decomposition. P itself
zeros source V and can change query0 relative to native C; gate projection
preserves query0 relative to P. Better loss is not exact output preservation.
The experiment concerns GPT-2's analogous allocation circuit, not proof of
transfer of OLMo's original single-weight pulse. Known gate algebra and prior
gated-attention work must be credited. No priority, speedup, generation or
universal architectural conclusion follows.
