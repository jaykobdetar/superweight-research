# Query allocation × conditional mixture: independent numerical review

**Passed; no result correction required.** Reviewed runner SHA256 `0ee05a000051e34636211f10b04b290b78f0c475d650f558e503ba35e788b8ed`, protocol `0e3cd228104e1473deeb4c52e52c1b379d9fde586c5214713e3234b60ab47028`. Evidence: audit code (`gpt2_query_mixture_numerical_audit.py`, private raw bundle), [audit JSON](gpt2_query_mixture_numerical_audit.json), and [report findings](../../docs/GPT2_QUERY_PATHWAYS.md). Review loaded donor files sequentially, used no model/GPU, and peaked at 1,062,016 KiB RSS.

## Coverage and replay

All 520 records are unique and complete: sixteen conditions on 32 passage/start inputs plus eight native repeat/identity anchors. All 128 summary groups, 256 contrasts, and 64 paired bootstrap groups independently reproduce. Raw per-token NLL/KL/agreement aggregation error is zero. Target2–511 scoring, natural/BOS suffix matching, token hashes, all source hashes, original checkpoint/model hash, parameter-version flags, and file hashes pass. Actual fresh-token preparation reproduces the saved manifest and sixteen 512-token windows, excluding twenty-eight earlier used spans per corpus.

The 840 saved checks comprise 480 zero-V source-state checks, 320 factorial/current output closures, 32 mask references, and eight native anchors. Independently verified 328 raw loss/KL/prediction closures and matching complete-final-hidden hashes. All 3,840 source-Q/K/V layer comparisons pass; every saved zero-V source vector is exactly zero. All 4,680 raw source residual additions and the absence of coordinate/writer lesions pass. Current-context and cross-recipient closures are computational controls, not independent samples. Explicit/native causal-mask token loss drift is zero.

Thirty-two donor files, totaling 512 donor layer states, reproduce their saved geometry and all 3,072 head rows. Independently recomputed 2,048 donor-dependent layer formulas against operation checks. All 3,840 mixture-layer checks are finite and satisfy the fixed `1e-4` guards. For the twenty frozen Q/K/V/mask/native/delivered-context states, FP64 native reconstruction has maximum error `8.1127e-6`; donor mass error `9.1164e-7`; stable conditional-mixture error `1.0703e-5`. All sixteen patched snapshot contexts match the specified FP32 donor arithmetic exactly. Their maximum discrepancy from the donor-mass × stable-mixture reconstruction is `1.2570e-5`. Query0 context is unchanged.

## Arithmetic stability must remain visible

Non-source mass is summed over non-source probabilities, not obtained as `1−p0`. Minimum saved/runtime mass is `3.3077e-12` (natural Hobbes window3, L4H11, query1). Maximum R/P gain is **6,785,536.5** (natural Du Bois window0, L4H11, query1). These are scored queries, not discarded positions. The maximum runtime conditional error is `3.6240e-5`; mixed reconstruction error is `1.9073e-5`, and placement/formula error is zero. No clipping or denominator floor was applied. The large gains require disclosure even though all declared guards pass.

Across scored donor layer/head/query entries, median R/P gain is 1.4423, 95th percentile4.9159, and 99th12.0819; 95.12% exceed one, and 271 exceed100. Small FP32 summation excursions allow maximum mass1.0000002384; this is floating-point error, not a substantive probability violation. Exact diagonal copies deliberately avoid needless divide/multiply drift and do not validate arbitrary rescaling on their own; mixed-state snapshots provide that independent check.

## Interpretation

Use **mass-first indices**: write `Y_PP`, `Y_PR`, `Y_RP`, `Y_RR`, with first index the non-source mass donor and second the conditional mixture donor. P/R mean learned Q bias present/removed, both with V0=0 in zero-based layers4–11. Native C supplies the loss/fidelity reference. There are no E/B coordinate lesions in this batch.

At either fixed mixture, R rather than P mass lowers NLL on Du Bois, Hobbes, and code for both starts: twelve of sixteen mass-effect group intervals are below zero. Wiki mass effects are slightly positive with intervals crossing zero. In contrast, R rather than P mixture raises every corpus/start mean at either mass; fifteen of sixteen intervals are above zero (Hobbes/BOS at P mass crosses zero). At R mass every individual window's mixture effect is positive; at P mass30/32 are positive.

`Y_RP` has PPL0.9097–1.0166 naturally and0.8818–1.0151 with BOS, versus native C. It is below the complete R donor's NLL in all eight corpus/start means, and below P in six (all except Wiki). `Y_PR` has PPL1.0869–1.2588 and1.0809–1.2360; it is worse than P in every group and worse than R in six (again excluding Wiki). Thus the modest complete Q-bias-removal effect can combine opposing conditional effects in this finite factorial. That is not an additive mediation percentage or a decomposition of the native trained circuit's total causal effect.

The interaction `Y_RR−Y_RP−Y_PR+Y_PP` is positive in17/32 windows and negative in15/32. Only three of eight group intervals exclude zero: positive Hobbes at both starts and negative natural Du Bois. No universal interaction sign follows. All inference remains conditional on the donor trajectories, eight patched layers, chosen passages, and zero-source-value world. Four-window bootstrap intervals are unadjusted for multiple comparisons; natural/BOS are paired variants, and duplicate recipients are closures.

Low loss is not fidelity: `Y_RP` mean agreement is80.93–90.10% natural and80.05–90.00% BOS; mean KL is0.0824–0.1560 and0.0823–0.1686 nats. These same-input donors use held-out native trajectories, not learned input-independent constants. Mixed trajectories need not match any native forward. The prior-art note [here](gpt2_query_mixture_prior_art.md) credits established query-bias mechanisms, source/other output decomposition, and zero-valued allocation; no novelty claim is warranted by this audit.

## Resource and verification limits

Research runtime:55.5363 seconds; peak GPU allocation768,470,528 bytes; process RSS2,312,288 KiB. No parameter changes or failures are recorded. Final normalized hidden tensors are represented by hashes, not saved full arrays: the audit verifies recorded hash equality but cannot regenerate those hashes. Full vocabulary logits are absent, so loss/KL are reaggregated rather than regenerated. Full Q/K/V/context verification covers the twenty frozen states; other arms rely on saved donors, runtime locality/formula assertions, and the separate implementation review. No publication action occurred.
