# Independent Llama replacement implementation review

Status: static/synthetic pass; no launch blocker. Evidence: `llama_replacements_cpu_test.py/.json`, including reviewed source hashes. CPU-only; no pretrained model or GPU loaded. Peak RSS818,200 KiB.

Calibration runs before evaluation forwards. Eight clean source0 activation/K/V observations are fitted with signed FP32 arithmetic means and cast once to BF16, separately pooled/Wiki/Hume. The fit is saved and hashed before evaluation. Seeded CPU sign/permutation controls retain each pooled BF16 vector's exact absolute-value multiset and BF16 dtype. The one fixed control is appropriately scoped.

The complement D uses a copied native down matrix, zeros column1419 then restores its selected588 entry, recomputes full-sequence F.linear, and replaces only the source0 output. The unchanged selected-coordinate assertion is correct. The MeanPulse hook runs after Probe's down hook, requires no simultaneous weight patch, replaces only the selected source intermediate in a cloned full input tensor and recomputes F.linear. Row mode changes only output588; full mode changes the source output vector; later positions remain unchanged. Probe's saved post-output is updated to the actual result. Five synthetic BF16 checks validate row/full replacements, identity activation, complement preservation and random-control magnitude preservation.

The inherited K/V hooks operate on native8-head projections at layers2..27 before RoPE/group repetition, modifying source0 only. Pooled K-only/V-only conditions retain endogenous complementary projections. Pooled fixed K/V C/N equality and clean full-donor rescue are correctly checked after predictor0. Repeat/oracle controls are also checked on scored positions at runtime; their full saved loss/prediction arrays remain available for an independent all-position check.

Independent real-source/tokenizer preflight reproduces eight calibration windows with eight distinct first tokens, all disjoint from earlier development used spans. Sixteen third-panel evaluation windows are disjoint from both previous evaluation panels and each other. Two of16 natural evaluation starts occur among the calibration first-token IDs; BOS does not. Coverage is saved explicitly and should accompany generalization claims.

Count: eight calibration forwards +32 inputs ×16 conditions +eight anchors =528 forwards,520 recorded evaluation rows. Expected72 scored closures (32clean-KV donor,32pooled-mean-KV C/N, eight anchors),320 K/V patch records,64 pulse replacement checks. All selection/recipient definitions agree with the revised protocol.

Reporting caveat sent to parent: generic `a0` and `source_contribution` remain the native pre-replacement potential activation/contribution in mean-pulse conditions. The actual applied activation is the frozen `activation_mean`, also recorded in `pulse_checks.mean_activation`. Do not present the generic value as the applied contribution.

Runtime completion, immutable parameter checks, mean/control replay, per-token summaries, D interaction and interpretation await numerical review. No experiment scripts were changed by this review.
