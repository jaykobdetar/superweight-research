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

Next priority after packaging: localize GPT-2's conditional writes across layers and test first-writer sufficiency; then test Llama gate freezing or a calibrated gate subspace with native attention recomputation closures. Any component selection from perturbed development inputs must be frozen before a new evaluation panel. Centered low rank alone will not be called one mechanism. Preserve failed controls and actual raw traces.

Private repository creation remains denied and unexecuted. No retry or alternate route is authorized by the parent while confirmation is pending; local research and review continue.
