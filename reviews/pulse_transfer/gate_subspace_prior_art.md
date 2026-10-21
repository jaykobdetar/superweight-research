# Additional primary comparisons for the prospective gate study

Inspected21October2026. This note concerns scope and comparison, not a claim
of novelty or a completed result from the currently failed gate study.

[Qiu et al., Gated Attention for Large Language Models, v1](https://arxiv.org/html/2505.06708v1)
§§2.2,4.1–4.3 already study sigmoid gating after attention, headwise and
elementwise choices, head-specific versus shared gates, and the relation to
attention sinks. Their low-rank discussion concerns the value/output linear
mapping and introducing nonlinearity. Those contributions rule out treating
attention-output gating, head-specific scalar modulation, or its connection
to sinks as new here. Our proposed intervention instead projects existing
source-allocation logits in a fixed checkpoint, with no training of new gates.
This is a different experimental estimand, not evidence of priority.

[Zhou et al., Hybrid Gated Attention, v1](https://arxiv.org/html/2608.11805v1)
§3.2 uses low-rank factorization for learned gates; §3.3 jointly uses all heads'
attention outputs to produce headwise gates, with a distinct learned
cross-head projection; §3.4 combines gating with learnable sinks. Thus
cross-head coordination and low-rank gating are also existing ideas. The
prospective study's centered covariance of native allocation logits, log(q)
nuisance control, complementary retain/remove maps and matched random
subspaces ask a narrower causal replacement question. We have not reproduced
HyGA, and its architecture-training results do not establish that frozen GPT-2
has one low-dimensional gate mechanism.

A relevant additional title was located: Ahmed Ghazouani, *Variance Geometry
is Not Functional Geometry in Transformer Residual Streams*, SSRN6886378,
posted26June2026. Both the primary abstract page and PDF returned403 at this
reading. Record it as an unresolved comparison, not a fully inspected paper;
do not use an unavailable method as evidence of either novelty or absence.
Primary target: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6886378

The already reviewed Massive Activations attention decomposition and zero-value
sink controls, and GPT-2 query-bias mechanistic account, remain direct prior
comparisons. Covariance rank is descriptive; directional interventions and
fidelity metrics are required to assess functional consequences.

## Additional close primary comparison, 21 October 2026

Fu et al., *Attention Sink Forges Native MoE in Attention Layers*,
https://arxiv.org/html/2602.01203v1, §3.1 and Appendix B.2 explicitly derive
non-source conditional mixing times a head gate, including the exact sigmoid
of non-source log-sum-exp minus source logit (Eq.16). Thus neither the gate
identity nor that logit is new here. Their Appendix A tests source-value
zeroing; Table3 reports Qwen3-8B average70.36→64.10 for all heads versus70.38
for selective zeroing, so broad negligible-impact language needs qualification.
Their PCA illustration in §3.2 concerns Q/K geometry, not our fitted gate
retain/remove intervention. No priority inference follows from this difference.

The SSRN delivery URL returned403 again; full-paper comparison remains
unresolved. Teo/Nguyen's https://arxiv.org/html/2406.13762v1 (§2) concerns
kernel PCA over keys, a different object from centered cross-head allocation
logits; it does not validate our causal conclusions.
