# Independent fixed512 numerical review

**Saved-result audit passed.** This is a completed fixed512 study, not a successful rerun of either failed cross-length protocol. Evidence: `gpt2_gate_fixed512_numerical_audit.py/.json`; concise numerical ranges are in `gpt2_gate_fixed512_findings_ranges.json`. No model/GPU was loaded. Peak reviewer RSS was1,211,524 KiB.

## What was checked

The independent audit verifies680 main records,672 suffix controls,744 main runtime checks,10,240 gate checks,640 source-state identities,72 raw metric/hidden identity closures,680 saved prefix references and40 fixed snapshot states. Source hashes match reviewed code; calibration raw/maps remain byte-identical to the earlier independently audited calibration. FP64 means/covariances/eigenvalues were independently checked again (maximum eigenvalue error1.21e-13). Final checkpoint hash/version, map and precision closure are recorded by the completed runner; the reviewer did not reload the checkpoint to independently remeasure them.

All per-token scalar reductions and dual C/P aggregates match, including full/early/later score slices, seeded2000-resample n4 tables, actual-n3 Wiki unexposed tables,1,024 full and960 unexposed raw contrasts and512 grouped full/subset paired summaries. All672 saved control hidden/gate/source/metric comparisons are exact. Full successful logits and contexts outside the snapshot subset remain independently recorded hashes/runtime flags; they cannot be reconstructed from those hashes. The audit verifies saved KL arrays and reductions, not unavailable logits for every record.

The40 QKV snapshots independently reconstruct in FP64. Maximum errors: gate logits2.9832e-5, mass8.3767e-7, conditional mixture1.0593e-5, actual context7.4842e-6 and delivered context2.6375e-5. The saved FP32 delivery formula is exact;80 context-prefix hashes match raw snapshots. Independent CPU map reconstruction differs by at most7.63e-6. All are below the frozen1e-4 reconstruction guard.

Numerical conditioning deserves disclosure: maximum target/current mass gain is1.9538970624e10, with minimum mass5.1179769e-11 in Wiki window3/BOS, L4, meanlogit. This is finite, and declared per-forward reconstruction checks pass. It is outside the fixed raw snapshot input, so that worst case has runtime-check evidence rather than a new independent QKV reconstruction. Sigmoid output1 occurs without explicit clipping; zero targets do not occur in the audited active records. These facts do not demonstrate infinite precision or universally stable scaling.

## Findings and limits

The following ranges are over four corpora and both starts in the **unexposed view** (Wiki n3, other corpora n4), full targets2..511. They are separate from the retained full n4 tables.

- Zero-V P has0.99564–1.01349× perplexity relative to C, but94.84–98.19% argmax agreement. P is already a changed model state, including source query0.
- Keeping one calibrated residual PC per layer gives1.14798–1.29222× P perplexity,80.07–87.40% agreement with P, and KL(P||trial)0.12471–0.23275. It is not a faithful one-direction replacement.
- Keeping three gives1.03092–1.18609× P perplexity and86.18–93.53% P agreement. Keeping six gives1.01046–1.09540× and90.39–95.20%, respectively. Improved loss and agreement with retained rank do not imply exact preservation.
- Calibration PC1 explains40.47–61.12% of centered residual-logit variance per layer; PCs1–6 explain88.14–95.44%. Thus this calibration is not nearly rank1. In the descriptive full-panel held-out P covariance, projection onto those development PCs retains21.51–77.58% for rank1 and83.58–96.20% for rank6 across layer/corpus/start groups. These are variance summaries, not causal importance or unexposed-only estimates.
- Removing PC1 worsens Wiki relative to P (1.06064× natural;1.05704× BOS). It improves DuBois mean loss (0.96761×/0.97284×), with both conditional bootstrap intervals below1. Hobbes/code means also improve slightly, but their intervals include no change. All remain imperfect P predictions: overall removal agreement87.21–93.63%. Do not state that removal reliably improves every non-Wiki corpus.
- The fixed QR rank3 control has lower mean loss than PCA rank3 on DuBois and Hobbes, while PCA rank3 has higher argmax agreement. DuBois paired PCA-minus-QR NLL is+0.13886 natural (CI+0.10025..+0.18627) and+0.11928 BOS (+0.04785..+0.18229). Hobbes natural's interval includes zero; BOS barely excludes it. Wiki and code favor PCA rank3 loss. Thus PCA variance, fidelity and target loss are distinct; one fixed QR basis does not support a universal random-subspace ranking.

The full n4 view is preserved and separately audited: rank1/P ranges1.13906–1.29222 and rank6/P1.01446–1.09540. Do not silently substitute unexposed ranges into n4 captions. Early/full/later views share one512 execution and passages; they are neither short-execution replications nor independent samples. Small n, fixed calibration, one QR seed and many planned contrasts limit uncertainty claims. The native/exact-copy controls establish implementation fidelity; they do not validate a one-dimensional mechanism, additive mediation, priority or generalization to generation/other models.

## Highest-value next discrimination

First separate **simultaneous layer accumulation/feedback** from **calibration-domain mismatch**, without refitting maps. A prespecified single-layer-at-a-time projection panel across all eight layers, with the same frozen rank/QR controls and joint all-layer condition, is the cleaner next intervention. Compare KL/agreement to P as primary fidelity outcomes alongside target NLL. This asks whether small isolated distortions accumulate through later endogenous states or whether individual layers already show the corpus-dependent mismatch. The joint-minus-individual contrast need not be additive and is not a mediation percentage. Do not choose the layer/rank from evaluation performance.

If corpus differences persist at isolated layers, a separate fresh-development calibration comparison can distinguish centering/position profile from basis orientation, using equal calibration counts and fixed held-out confirmation. Merely refitting a corpus PCA after seeing these outcomes would confound domain adaptation with outcome-guided fitting. Current results alone establish neither explanation.

The completed production run reports1,352 forwards,226.9971876 seconds,966,876,672 GPU allocated bytes,1,061,158,912 reserved bytes and2,655,644 KiB process RSS. Cumulative production count through this study is1,501 including failed V1, native diagnostic, separate calibration rebuild and failed V2. Both failures remain preserved; no thresholds were retroactively relaxed.
