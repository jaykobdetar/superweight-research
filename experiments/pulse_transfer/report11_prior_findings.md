## Prior work and the limits of novelty

The gate algebra is established. Fu and colleagues explicitly derive the
non-source conditional mixture multiplied by a head gate under zero or
negligible sink-value contribution, including the gate’s sigmoid/log-sum-exp
expression, in §3.1 and Appendix B.2. Their source-value
zeroing experiments also precede this study; outcomes depend on which heads
are zeroed. We test fixed calibrated subspaces of these gates in a pretrained
checkpoint, rather than claim discovery of implicit gating.
[Attention Sink Forges Native MoE](https://arxiv.org/html/2602.01203v1).

Qiu and colleagues study explicit head-specific sigmoid gates after attention,
including query dependence and sparse gating. Zhou and colleagues study
learned low-rank and cross-head gates with synthetic sinks. These are relevant
precedents for simplifying or coupling gates; we do not reproduce their
training experiments or claim those architectural ideas as new.
[Gated Attention](https://arxiv.org/html/2505.06708v1),
[Hybrid Gated Attention](https://arxiv.org/html/2608.11805v1).

Reports8–10 distinguish established super-weight, massive-activation and
GPT-2 query-bias research from this project’s conditional-regrowth and source
K/V experiments. Massive Activations already separates source and non-source
attention contributions and tests synthetic zero-value alternatives. The
present intervention preserves the current conditional mixture locally while
retaining or removing calibrated gate directions; its outcome cannot by itself
identify one shared mechanism throughout the residual stream.
[Massive Activations](https://arxiv.org/html/2402.17762v2),
[GPT-2 attention-sink mechanism](https://arxiv.org/html/2604.14722v1).

A closely titled preprint, *Variance Geometry is Not Functional Geometry in
Transformer Residual Streams* (SSRN6886378), was located, but its primary
full text returned403. That comparison remains unresolved. Kernel-PCA
analyses of keys concern a different representation from the cross-head gate
vectors tested here. Our searched comparison supports careful attribution,
not a claim of priority or an exhaustive absence of similar work.
[Unresolved SSRN comparison](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6886378),
[Kernel-PCA attention analysis](https://arxiv.org/html/2406.13762v1).
