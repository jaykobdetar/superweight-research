## Relation to established work and limits of novelty

The original [Super Weight paper](https://arxiv.org/html/2411.07191v2)
establishes extreme sensitivity to individual weights and their relation to
super activations. This study's GPT-2 gate experiment is a follow-up to a
larger causal investigation; it is not a new discovery of super weights and
does not reclassify every large GPT-2 activation as the same mechanism.

[Massive Activations](https://arxiv.org/html/2402.17762v2), particularly
sections 4.2–4.3 and Appendix B.4, already studies implicit attention biases
and explicit alternatives. The allocation/value decomposition and the role
of source values are established context. [Fu et al.](https://arxiv.org/html/2602.01203v1)
derive the sigmoid/log-sum-exp form of the implicit gate and analyze related
attention variants. That algebra is not a novelty claim here.

The [GPT-2 attention-sink study](https://arxiv.org/html/2604.14722v1)
analyzes learned query bias, positional transformations and key structure
through causal interventions. It explicitly distinguishes mechanism from
function and limits architectural generalization. The present calibration
factorial instead tests finite changes to empirically fitted centers and
directions under an explicit zero-source-value boundary. Its contribution is
the measured, scoped pattern and its falsification controls—not discovery of
attention sinks, a new PCA method, or proof of a universal circuit.

The four primary texts were rechecked for this report. The repository's
broader [novelty review](https://github.com/jaykobdetar/superweight-research/blob/main/docs/NOVELTY.md)
and preceding reports retain further related work and unresolved access
limitations. We have not established that no prior study contains this exact
factorial; no first-result or priority claim is made. Low rank, stable axes,
loss gains and cross-model activation similarities remain distinct from a
validated common causal mechanism.
