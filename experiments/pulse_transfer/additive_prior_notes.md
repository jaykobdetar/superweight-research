# T17 primary-source comparison before results

Checked 2026-10-21. The additive output counterfactual and N=J+R identity are elementary finite-difference accounting, not a claimed new attribution method or discovery of feedback. The empirical question is their measured behavior for the frozen GPT-2 source-value/gate interventions on this particular held-out panel.

[Anthropic's Circuit Tracing methods paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) constructs linear feature-attribution graphs after freezing attention patterns and normalization denominators, and validates proposed mechanisms through interventions. T17 instead combines actual output differences from separately rerun singleton gate interventions, without freezing the remaining network computation. Its J term therefore cannot identify any one internal cause. This is a comparison of estimands, not a claim of priority over circuit tracing.

The existing Report12 novelty comparison and original massive-activation, super-weight, gating and GPT-2 sink citations remain applicable. Search did not establish whether another paper has already reported precisely this gate-panel decomposition. No exhaustive novelty or first-result claim is justified.

Access limits: the original 2021 Transformer Circuits framework page exceeded the web reader's content-size limit; the attempted Hydra HTML revision2 endpoint returned404. These failures are not evidence of absent prior work. Existing repository references remain available; no new detailed claim is grounded in those failed fetches.
