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
