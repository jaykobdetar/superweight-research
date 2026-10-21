# Independent novelty and attribution review

Reviewed 2026-10-21 against the six requested primary papers, using the versions linked below. This is a targeted literature comparison, not an exhaustive priority search. “Not located” means absent from my inspection of these papers, not proof that no prior work contains the result. The experiments are in one pinned OLMo-1B-0724-hf checkpoint; independent CPU replay is an audit, not an independent model-level replication.

## What is already established

| Primary source | Relevant overlap and boundary |
|---|---|
| [Yu et al., The Super Weight in Large Language Models, v2, 7 July 2025](https://arxiv.org/html/2411.07191v2) | Early MLP down-projection scalar deletion can catastrophically impair output, and scalar outliers generate persistent activation outliers. Section3.2 already reports incomplete recovery when restoring the large activation after weight deletion. Thus “the large activation does not explain the entire deletion effect” is not new. Table2 supplies the OLMo target coordinates. Our position-local row/whole-column factorial and incompatible K/V donor combinations are more specific interventions than that paper’s scalar deletion/restoration comparison. |
| [Sun et al., Massive Activations in Large Language Models, v2, 14 August 2024](https://arxiv.org/html/2402.17762v2) | Section2.2 explicitly identifies the starting token and first period/newline in LLaMA2-7B. Sections3–4 test fixed activation replacement and describe implicit attention biases, including nearly constant value updates. First-newline localization, low input variability, and bias-like effects must therefore be credited as established. Our candidate addition is a named OLMo head’s causal dependence on a preceding newline and its separable K/V requirements, not discovery of first-newline spikes. |
| [Sun et al., The Spike, the Sparse and the Sink, v1, 5 March 2026](https://arxiv.org/html/2603.05498v1) | Section3.1 develops directional quadratic FFN amplification and rank-one dominance. First-token self-only attention acts as a static VO map; delimiter self-sinking can follow a related trajectory. Section3.2 links normalization to bounded, sparse, nearly constant states and attention-sink geometry. Architectural ablations distinguish spikes from sinks. This substantially overlaps broad early-attention, isolation, amplification, and normalization explanations. Our history-sensitive OLMo head counterfactual is narrower. OLMo’s parameter-free LayerNorm also prevents directly importing that paper’s learned RMSNorm-scaling explanation. |
| [Mao et al., Massive Activation Gating Channel in Large Language Models, v1, 6 October 2026](https://arxiv.org/html/2610.07661v1) | Identifies a single FFN-input channel whose value controls massive activation emergence, derives large-input asymptotics, and connects dominant down-projection column geometry to the output pattern. It tests six models across four families and relates gating interventions to sink strength. An input gating coordinate is not the same object as the internal gate_proj factor of neuron1710. General gating and dominant-column explanations are established; our conditional residual-component and output-coordinate interventions remain distinct. I found no OLMo H14 experiment here. |
| [Owen et al., A Refined Analysis of Massive Activations in LLMs, v1, 28 March 2025](https://arxiv.org/pdf/2503.22329v1) | Tests zero/mean activation interventions across architectures, including OLMo-7B variants, with and without BOS. Section3.1 separately calibrates starting and non-starting activation means on RedPajama and evaluates other corpora. It finds architecture-dependent removal effects and limits to KV-bias mitigation; later experiments combine mitigation methods. Cross-corpus constant replacement, start-versus-later controls, and rejection of universal necessity or universal KV-bias remedies are not new. These are activation-value interventions, not the specific neuron output-coordinate/complement factorial tested here. |
| [Shi et al., A Single Layer to Explain Them All, v2, 12 May 2026](https://arxiv.org/html/2605.08504v2) | Studies an emergence layer, residual propagation, and joint RMSNorm/FFN effects. It emphasizes directional rigidity and introduces RMSNorm-weight-guided masking before later attention layers. AppendixE distinguishes successful selective masking from harmful random/magnitude masking. Therefore representation direction, structured removal, and normalization-mediated sink behavior are established. Our all-layer source-vector angle/direction swaps can test a different conditional causal decomposition, but cannot be advertised as discovering that direction or normalization matters. |

## Narrow findings with the best case for an extension

1. **A prior-newline-dependent suppression path in this OLMo checkpoint.** A development-only screen selected L0H14. On held-out constructed inputs, suppressing its single q→p attention edge restores the later newline pulse. Controlled token IDs distinguish187 from535 and other alternatives. In local head-context recomputation, donor K and V jointly suppress the pulse in one direction; either alone fails there, while either reverse donor breaks suppression. Oracle recomputation and H0 controls support localization. Twelve eligible unedited book/code windows reproduce edge-induced pulses. I did not locate this named head, prior-token counterfactual, and bidirectional K/V factorial in the six papers. This supports a specific circuit case study, not a universal newline detector or first report of delimiter spikes.

2. **Coordinate deletion and neuron-column deletion are different causal questions.** Removing the selected scalar’s output contribution can be far worse than removing the full neuron output at the same position. The row/complement factorial isolates this difference while keeping the intermediate activation fixed; identical raw target-coordinate outcomes can accompany different normalized vectors and downstream behavior. The model-specific outcome is the empirical extension; the intervention decomposition itself has close prior art in Weight-Dictionary Decomposition, credited below. It does not overturn prior scalar-deletion results or introduce the general scalar/complement distinction.

3. **Downstream K/V compatibility is a specific candidate mechanism for that fragility.** Clean, row-deleted, and column-deleted donor source states yield asymmetric separate-K/separate-V transfers. In the controlled prior-newline panel, column-world K into the row world is near baseline, while column-world V with row-world keys is catastrophically harmful. Blocking the source column often removes most additional row-deletion damage. These experiments expose harmful mixed states; they do not prove a universal “correct K/V pair” law, that keys encode no content, or that the values carry semantics. Whole-column deletion has material natural-input exceptions, especially the two DuBois p0 cases and one code window.

4. **Conditional upstream component decomposition.** Across28 constructed/natural contexts, dynamic M0 donor swaps induce or suppress the pulse, frozen M0 supplies a partial pulse, and joint frozen M0/A1 restores a strong pulse. Frozen A0 alone with downstream components held clean does not. This identifies a conditional carrier/amplifier arrangement in this circuit. It does not establish an additive mediation fraction, a unique minimal pathway, or unification with the separate empty-history H12 result. Reciprocal H12 context swaps failed to transfer the newline pulse state.

## Attribution correction that must accompany publication

Table2 of the super-weight paper lists both OLMo-1B coordinates in zero-based layer1: `[1764,1710]` and `[1764,8041]`; the caption says these indices directly apply to Hugging Face layers. The historical experiment instead used the explicitly recorded pair L1/1710 and L2/8041. This is an attribution error, not an ambiguity between zero-based and one-based indexing. The old L2 measurements remain measurements of L2 and must not be relabeled as the paper’s second target.

The saved original `main_run_metadata.json` already contains the L2 command-line override; `pulse_run_metadata.json` inherits it, while bundled core defaults correctly specify both in L1. The reason for that override cannot be recovered from these records. `COORDINATE_ERRATUM.md` should be prominent in the repository and linked beside historical report5. Preserve immutable reports and raw data. Present the corrected same-layer experiment separately; it does not erase the old limitation. The ongoing first-target H14/column studies use the correct L1/1710 target.

## Required limits and wording

- Use “a mechanistic extension in one OLMo checkpoint” or “a candidate new causal case study,” not “a new explanation of massive activations” or “fully understood.” No reviewed paper provides a license to generalize from this checkpoint to all models.
- Specify the source position, score exclusion, exact token IDs, natural versus constructed contexts, FP16 execution, and donor world. Prior-newline experiments condition on a narrow location/range and do not prove arbitrary-history rules or natural resets.
- Separate source-row deletion from permanent weight deletion across all positions. Later uses matter greatly in some corpora; the original global effect cannot automatically be assigned to source0.
- Near-baseline perplexity is not exact recovery. Report agreement/KL and material per-window exceptions. The natural whole-column findings rule out “whole-column removal is harmless” as a universal summary.
- Static mean/template fits, centered gate PCA, and fixed-gate failures assess their named approximation families. They do not refute all bias models or establish a minimum universal circuit rank.
- Full all-layer K/V or normalized-source donor restoration, and complete downstream masking, are expected architectural controls. Their exactness supports implementation, not novelty.
- The corpus panels are small, several later follow-ups reuse them, and adaptive research generated many comparisons. Paired bootstrap intervals describe these windows; they do not correct selection or establish population-wide significance.
- The CPU auditor did not run an independent checkpoint replication. A publishable package should distinguish audited replay of stored measurements, executable reproduction, and genuinely new model-level replication.

## Most informative remaining checks

A narrow transfer test should preserve definitions rather than hunt until it finds the same head number: fresh contexts, tokenizer-specific single versus double newline IDs, predeclared edge/patch controls, and per-model normalization architecture. A claim about a shared mechanism would require matching causal signatures, not merely observing a large first-newline activation. Source-geometry results should be interpreted against reconstruction drift and random directions before adding them to the explanation. Neither extension is needed to report the present results honestly, but both constrain broader claims.

## Additional primary context

[Xiao et al., Efficient Streaming Language Models with Attention Sinks, v4](https://arxiv.org/abs/2309.17453v4) established the practical importance of retaining initial-token K/V caches for streaming attention. Our source cache experiments build on this idea rather than discovering initial-token dependence.

[Subramanian et al., Super Weights in LLMs and the Failure of Selective Training, v1](https://arxiv.org/html/2607.08733v1) studies pruning and selective training, including coordinate versus broader structured updates. It cautions against equating parameter importance with isolated trainability. We do not train weights here; our conditional activation and K/V interventions address a different question.

Search date:2026-10-21. The initial targeted search covered super-weight column removal, massive activation newline history, OLMo H14 and the cited papers. A later exact-coordinate search found close precedents in both target coordinates and intervention families, credited below. The remaining contribution is the scoped comparative dataset and causal controls, without a priority claim.

## Transfer addendum, 21 October 2026

The subsequent [transfer report](TRANSFER.md) tests two additional pinned models.
It does not extend the original OLMo conclusions by assumption: the selected
Llama candidate reverses the initial row-versus-column comparison, neither
candidate passes the prescribed newline-history screen, and GPT-2 shows
conditional coordinate regrowth after a larger lesion. The fuller
column/complement factorial in Llama is also nonadditive.

GPT-2 massive source activations and a starting-token-only category already appear
in [Massive Activations](https://arxiv.org/html/2402.17762v2), along with explicit
K/V-bias experiments. Fresh mean-source replacement here is a controlled
checkpoint-specific extension, not discovery of a constant-source idea.
[The Hydra Effect](https://arxiv.org/abs/2307.15771) and
[Explorations of Self-Repair](https://arxiv.org/html/2402.15390v1) establish
compensation after ablation; the latter includes GPT-2, normalization and sparse
MLP effects. Our raw-coordinate E/A/M factorial has a narrower measurement and
does not claim general self-repair novelty or recovery of the native trajectory.

The strongest candidate additions are these particular source-coordinate,
writer-family and K/V donor counterfactuals, the cross-model failures of transfer,
and the paired quantitative controls. Fixed K/V can retain near-native loss
while changing argmax outputs and leaving query-dependent gates dynamic. Neither
this nor the earlier low-rank fits establishes a universal single mechanism,
faithful generation or a computational speedup. Independent audits and a rerun
on the original machine are distinct from replication by another researcher.

## Exact-coordinate attribution addendum

[Weight-Dictionary Decomposition](https://github.com/sunmoonron/weight-dictionary-decomposition)
is a particularly close precedent. At pinned commit
`33504a4d90840bc01d0ef75609265e48a34403bb`, its author code scales GPT-2 channel447
at block2→3 (`exp_s8_loop.py`), traces attention/MLP and neuron compensation
(`exp_t2_compensation.py`), crosses scaling with regulator ablation
(`exp_u1_reserves.py`), and separately removes a neuron's channel447 contribution
or non447 complement (`exp_u2_leash.py`). The complete OpenReview PDF was
access-blocked; these claims rely on the pinned primary code, inspected without
execution. See [the independent close-prior review](../reviews/pulse_transfer/late_prior_art_review.md)
and its immutable links. No claim of independent reproduction of that work is made.

[The Communication Map of a Transformer](https://arxiv.org/pdf/2608.22007)
already uses exact dimensions447/138 for causal direction deletion at repeated
layer boundaries. [A Mechanistic Account of Attention Sinks in GPT-2](https://arxiv.org/html/2604.14722v1)
identifies138/378/447 and a positional-MLP/key/query-bias pathway. Our source-only
joint erasure and selective later output-write blocks are a narrower estimand,
not discovery of those coordinates or coordinate regulation. Loss dependence
alone does not establish attention-sink mediation or the full positional circuit.

[Transient Reserves, Sink Dampers](https://arxiv.org/html/2607.09279v1) already
compares architecture-dependent consequences of repeated massive-coordinate
clamping. [The Life Cycle of a Massive Activation](https://arxiv.org/html/2610.00423v2)
tests collective magnitude, redundant carriers and training-time restoration
with magnitude controls. Our inference-only regrowth must not be equated with
its continued-training recovery. General redundancy/nonuniversality claims are
prior art; random-coordinate controls here remain unmatched for magnitude.

The GPT-2 experiments are best described as targeted replication/extension with
different intervention scope, fresh paired evaluation and explicit source K/V
controls. The specific OLMo H14 history/K/V outcome remains a scoped case-study
claim. None of these differences establishes priority or a universal mechanism.
# Additional gate-subspace evidence

The later [gate report](GPT2_GATE_SUBSPACES.md) uses an established attention
identity: with source values zeroed, the context is a conditional non-source
mixture multiplied by its probability mass. Fu et al. explicitly derive the
sigmoid/log-sum-exp gate and test source-value zeroing; this project does not
claim those algebraic or intervention ideas as new.
[Primary derivation](https://arxiv.org/html/2602.01203v1).

Explicit head-specific post-attention sigmoid gates, learned low-rank/cross-head
gates, and synthetic sinks also precede this experiment.
[Gated Attention](https://arxiv.org/html/2505.06708v1),
[Hybrid Gated Attention](https://arxiv.org/html/2608.11805v1).

The additional evidence is checkpoint- and protocol-specific: fixed centered
gate subspaces, directional retention/removal with matched controls, differing
loss versus prediction-fidelity rankings, and preservation of failed numerical
checks. This does not establish priority, one underlying mechanism, or an
efficient replacement. A potentially relevant SSRN preprint's full text remains
inaccessible, so its method comparison is explicitly unresolved.
[Unresolved comparison](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6886378).
