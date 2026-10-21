## Relation to prior work and limits of the account

The source-value decomposition and its gate are established. *Massive
Activations in Large Language Models* analyzes attention bias behavior and
zero-value/synthetic alternatives ([paper, sections 4.2–4.3 and Appendix B.4](https://arxiv.org/html/2402.17762v2)).
Later work writes the non-source allocation as a sigmoid/log-sum-exp gate and
tests zero-value variants ([Fu et al., section 3.1 and Appendix B](https://arxiv.org/html/2602.01203v1)).
Explicit query/head gating also predates these experiments
([Qiu et al.](https://arxiv.org/html/2505.06708v1)).
GPT-2 query bias, positional contributions and massive key coordinates are
likewise prior research ([GPT-2 attention-sink study](https://arxiv.org/html/2604.14722v1)).
The earlier report contains the broader super-weight, compensation and
massive-activation comparison. A potentially close residual-stream variance
paper could not be fully inspected because its primary text returned 403;
that comparison remains unresolved.

This continuation contributes a precisely defined causal screen: original
calibration directions, singleton versus joint layer patches, a local
logit-norm control, verified fresh research passages, and preserved negative
results. It does not establish priority for causal direction tests, layer
localization, low-rank gating, or attention-sink algebra. Results on one GPT-2
checkpoint are not evidence that OLMo's single super weight, GPT-2's conditional
coordinate regrowth and Llama's source-value pathway share one mechanism.

The current account is conditional: source state and queries control an
allocation gate and a value mixture; changing either can have different effects
on loss and native-output fidelity. Centered covariance identifies variation,
not a causal mechanism count. We have not shown that one scalar can replace
the downstream computation, that the selected interventions improve generation,
or that a replacement is faster. The experiment does not validate independent
128-token execution; both earlier cross-length failures remain failures.

Any layer chosen from this screen must be tested on new data. A fresh
calibration-center by calibration-basis factorial remains useful for separating
mean misfit from direction misfit. Joint-minus-sum results can motivate selected
pair/block interventions, but they alone cannot distinguish downstream
compensation from other nonlinear composition. Such follow-ups need separately
frozen designs and resource checks.

## Next discriminating experiment

The next priority is a fresh-panel additive-logit diagnostic, retaining every
singleton and all three intervention families. Form the explicit output
counterfactual `z_add = z_P + Σ(z_single − z_P)`. Then split the measured
nonadditivity exactly into:

- `J = loss(z_joint) − loss(z_add)`.
- `R = loss(z_add) − loss(z_P) − Σ(loss(z_single) − loss(z_P))`.

The observed residual equals J + R. R measures the loss functional applied to
additive singleton logit changes; convex cross-entropy does not guarantee its
sign for arbitrary multiple directions. J compares the joint output with that
counterfactual, but is not a pure feedback or mechanism term. Centered-logit
differences, KL(joint‖additive) and per-token terms will accompany loss so that
cancellation cannot hide output disagreement. This needs new forwards because
full successful logits were not saved for every original condition. A new
frozen input panel and independent design/implementation review precede it.
