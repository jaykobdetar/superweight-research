# Independent gate-path review

Static and completed numerical review of `gate_feedback.py`: no blocker found. No model/GPU loaded. Replay script and evidence are `gate_feedback_audit.py/.json`; peak RSS approximately 558 MiB.

Both worlds clamp all source keys L2..15 to the same frozen mean, and all source values to the mean or zero. Native gate donors are captured separately per window/world with standard causal attention, before gate replacement. Each reset creates a new cache dictionary, so stored donors survive subsequent resets without accidental alias mutation.

The cross-world intervention substitutes donor gates only in the named layer set. Other layers reconstruct using current-recipient gates. Conditional non-source mixtures and source values remain those of the recipient world. Query 0 is left unchanged, and common scoring excludes predictor 0. The all-layer zero-world cross patch includes a clean/deleted invariance assertion.

Current-recipient and same-world donor reconstruction are separate numerical controls. Even same-world donor replay can accumulate differences after the FP16 conditional-mixture recombination changes an earlier recipient state; quantify both controls before interpreting cross-world changes. Frozen Wiki/Hume global gates are calibration controls, while per-window donor gates are diagnostic oracles rather than fitted or deployable replacements.

Cross-world comparisons can test whether altered source gates exacerbate or mitigate missing-value effects within these controlled worlds. They do not supply a unique causal mediation percentage, independent direct/indirect decomposition, or a literal recurrent feedback mechanism. Conditional attention mixtures and downstream recipient states can respond to every earlier intervention.

All 768 raw records, 96 summary groups, 96 paired contrasts and 128 reconstruction checks replay. Prior mean-world native/current/global anchors match exactly, zero-native matches the quiet-anchor fixed-mean-key zero-value control exactly, and saved native mean-world donor gates match the prior held-out masses exactly. Cross-all in the zero-value world has exact clean/deleted invariance. Synthetic CPU execution of the extracted context hook confirms layer selection, conditional mixture formula, and untouched predictor 0.

Mean/zero donor gates at L2 are exactly identical because no earlier fixed-value difference has propagated to that layer's queries. Correspondingly both cross-L2 output conditions exactly match their current-recipient controls. This is an informative causal/localization check.

| Corpus | Zero-V native | Zero-V with mean-world gates | Mean-V with zero-world gates |
|---|---:|---:|---:|
| Wiki | 218.92157 | 10.11420 | 7.99026 |
| Hume | 1.28660 | 1.16261 | 1.02143 |
| Plato | 1.26865 | 1.14440 | 1.01770 |
| Python | 3.80015 | 1.63891 | 1.40632 |

Values are PPL ratios against the same native-clean reference. Mean-V native is near 1 in all corpora. In Wiki, transplanting mean-world gates into zero-V reduces loss by 3.07477 nats [2.79531, 3.31992]; the opposite gate transplant adds 2.07798 nats [1.83920, 2.31150]. Every corpus's full cross-world contrast has an interval excluding zero in the corresponding direction. Python intervals are wide. The Wiki early-only repair instead increases loss by 0.48635 nats [0.40824, 0.56826] to 356.049×, while middle-only repair reaches 35.0499×. These partial interventions cannot be added to estimate independent mediated shares.

Current and same-world reconstruction are close but not exact: corpus-level absolute mean drift is at most 0.0001141 nats; largest individual token-loss difference is 0.17866 nats in Python/zero/current, and maximum condition/corpus argmax mismatch is 29/4,080. The major gate-swap changes exceed these numerical controls by a wide margin.

Frozen Hume global gates with zero values achieve 3.05096× Wiki PPL, better than the mean-world donor-gate intervention's 10.11420×. Therefore donor gates from the intact-value world are not optimal counterfactual gates when values are missing; avoid calling their residual error the irreducible direct value effect. These results extend the tested family of zero-value routing controls but still do not show high-fidelity recovery on Wiki.
