# Independent report8 review — five completed batches

Reviewed `REPORT8.md`, `build_report8.py`, and all four generated PNG figures against the independently audited GPT-2/Llama initial panels, GPT-2 groups, Llama source K/V and GPT-2 writes. This review covers the five-batch draft, not the pending replacement results or future sanitized package.

The main claims, numerical ranges, condition baselines, input/start scopes and resource totals are consistent with saved audits. Totals are4,064 forwards,4,048 evaluation records and416.32 recorded batch-wall seconds. Writer BOS sensitivity is explicitly included; trajectory plots correctly show natural-start means/ranges, not confidence bands. The raw-coordinate text properly distinguishes regrowth from restoration of the native trajectory. No claim of complete understanding, unique minimal circuit, clean-model MLP necessity or additive mediation was found.

Three corrections sent to parent before freezing:

1. The initial GPT-2 table rounds confidence bounds to three significant digits, producing `[1,1]` for nonzero intervals and obscuring bounds above one. Use at least four decimal places for near-one effects. For example, natural512 DuBois row has[1.00074266,1.00348335], and optimizer row[1.00030728,1.00360477]. A narrow interval is not an exact identity.
2. Initial figure legend “One output coordinate” can be confused with the later full-coordinate-zeroing control. Relabel as “Selected weight contribution” or equally precise wording.
3. “Explicit causal-mask reference forwards matched native output exactly” exceeds what was saved. Say “matched scored per-token losses and argmax IDs exactly.” This applies to96 cases; full logits were not retained. Existing KL limitations are otherwise accurate.

All four plots were visually inspected. Initial panels use differing scales but explicitly disclose that. K/V and writer figures include both natural/BOS conditions and log PPL axes, with the appropriate saved bootstrap bars. Regrowth plots use the proper raw residual coordinates/layers and observed min/max envelopes. No numerical plotting mismatch was found.

Novelty wording is appropriately constrained. Primary sources independently checked:

- [Sun et al., Massive Activations in Large Language Models, v2](https://arxiv.org/html/2402.17762v2): GPT-2 is explicitly studied; attention-bias analysis and training with learned explicit K/V biases precede this work. Its AppendixB.4 also distinguishes zero-K/V and other bias variants. Offline fixed-source replacements should not be advertised as the invention of explicit attention biases.
- [McGrath et al., The Hydra Effect](https://arxiv.org/abs/2307.15771): downstream compensation after ablation and late-MLP counterbalancing are established prior art.
- [Rushing and Nanda, Explorations of Self-Repair, v1](https://arxiv.org/html/2402.15390v1): imperfect/noisy self-repair, LayerNorm contributions, sparse MLP anti-erasure and GPT-2 experiments are established. The report correctly separates raw-coordinate reappearance from direct-effect restoration.

The candidate contribution is the specific frozen coordinate/writer factorial and scoped cross-model source-K/V comparison with numerical controls. The targeted search's failure to find an exact duplicate is correctly not presented as a priority proof. Model size, training, instruction tuning, precision and tokenization confounds are disclosed; the model comparison does not isolate architecture as the cause.

Uncertainty is correctly described as n=4 paired-window bootstrap conditional on frozen selection/calibration, without multiple-comparison or model-population coverage. Near-one loss is distinguished from exact fidelity, though the table precision correction is needed to preserve this distinction visually. Random controls are explicitly not magnitude-matched/exhaustive.

Publication status accurately records the denied private-GitHub creation, no verified remote, and no retry. This review authorizes no publication action and does not revisit that denial. Sanitized repository export, complete new-batch reproduction and final immutable bundle remain separate checks. Replacement results need their own numerical review and final report update before the six-batch report can be marked ready.

Follow-up verification: the report builder now includes all three requested corrections (four-decimal CI bounds, “Selected weight contribution” legend, and exact agreement scoped to scored losses/argmax IDs). A regenerated final six-batch report still needs a final consistency check after the newly audited replacement interpretation is inserted.
