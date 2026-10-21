# Independent Report12 review

Scientific/numerical review passed. All five corrections below were verified in the rebuilt report and figure; no new experimental defect was found. Portable-wrapper/package readiness is recorded separately. No complete second pretrained localization replay is claimed.

Reviewed final report hashes: `REPORT12.md` SHA256 `5ac54190635a922d008b68ad87ae4170c0fefc4c3c2ca1be03cbe0728dcabdf3`; `report12.html` SHA256 `2e9d50bdd53d363949c9c226c3946abfe29a9b1f4d21d6bef05415e24df8c932`. The 216-cell claim audit was rerun on these final Markdown bytes.

## Verified

`report12_claim_audit.py/.json` independently checks all 216 table cells against the audited result arrays, the eight L9/L10 book intervals versus P, six signed L10 direction-control intervals, the two DuBois L5 control comparisons, all 24 positive nonadditivity means/22 positive intervals, L4 direction concentration, reported L4/L10 context and logit norms, fidelity ranges and all start-token/calibration comparisons. The raw result audit covers all 938 records, every summary/contrast, exact controls and fixed snapshots. DuBois's new joint-removal intervals cross zero in NLL; the report correctly does not promote its mean improvement to resolved group evidence.

All four figures were visually inspected and their plotting source checked. They show every prespecified family, layer and start, use the P reference correctly, include paired intervals for loss/nonadditivity, and label agreement as means. There is no selected-winner omission. The mechanism figure's equations and within-current-state control scope are correct.

The report preserves P versus native C, the distinction between lower loss and fidelity, n3 code/n4 prose, correlated starts/score views, conditional bootstrap, no multiplicity correction, inherited calibration, grid-specific eligibility limits, and selected-layer development status. It acknowledges the large unsnapshotted gain, the audit-only FP64 norm correction, unchanged production guards, failed V1/V2, absent new full replay, and lack of external replication. The proposed additive-logit follow-up correctly separates two exact loss terms without calling either a mediation or feedback estimate.

## Corrections sent to the author — now resolved

1. Full attention contexts exist in the fixed raw snapshot subset; full successful model-output logits do not. Replace the sentence implying both full logits and contexts are saved there. All full-logit equality relies on runtime comparisons/fingerprints.
2. Narrow the opening claim from “no single layer explains” to the observable “no tested singleton reproduces the joint loss pattern.” The intervention screen does not identify a complete underlying explanatory circuit.
3. Say no layer was selected *for this run* using its outputs. L10 is now a selected candidate for fresh confirmation.
4. Align two prose roundings with the accurate tables: DuBois natural joint ratio 0.9488, code natural joint ratio 0.9859.
5. Shorten the nonadditivity figure's repeated rotated y-axis label or use a shared label; the current long labels overlap adjacent rows.

## Focused primary-source recheck

The report's limited prior-art credit is supported. Sun et al. §4.2 decomposes attention contributions, §4.3 trains explicit key/value bias alternatives, and Appendix B.4 includes a zero-key/zero-value variant. [Massive Activations](https://arxiv.org/html/2402.17762v2).

Fu et al. §3.1 and Appendix B.2 explicitly give non-source conditional-mixture scaling and its sigmoid/log-sum-exp form. This is prior algebra, not a new result of this study. The report does not repeat a universal negligible-effect claim for zeroing source values. [Fu et al.](https://arxiv.org/html/2602.01203v1).

Qiu et al. §2.2 and §4.2 discuss head-specific, query-dependent output gating. [Gated Attention](https://arxiv.org/html/2505.06708v1). Ran-Milo et al. §3.2.5–3.2.6 and Appendix A.2 cover query-bias/key-coordinate mechanisms and coordinates 138/378/447. [GPT-2 attention-sink study](https://arxiv.org/html/2604.14722v1).

These checks support the report's credits and explicit refusal to claim priority for gate algebra, localization or direction tests. They are not an exhaustive novelty search. The inaccessible residual-stream paper remains an unresolved comparison, not evidence of novelty. No cross-model unification, generation improvement, speed claim or complete causal understanding is established here.
