# Independent T18 numerical review

**Passed**, with the original saved-state audit failure retained and resolved by a separately reviewed, exact storage-layout adapter. No model/GPU inference was performed by this review. The experiment remains the single completed 1,548-forward run.

## Evidence checked

`gate_wiki_numerical_audit.py/.json` independently replayed 32 input shards, 776 main records, 768 suffix records, 1,128 main checks, 11,808 gate checks, 3,136 norm states and 11 fixed attention snapshots. It reproduced 4,032 summary metric groups, all 912 paired contrasts and their seeded n=4 passage bootstrap intervals, and 162,432 norm-summary reductions with zero discrepancy. All main/suffix saved arrays and exact copy/source/singleton identities checked as applicable. Other successful full-context/logit identities remain runtime comparisons plus hashes, not independently reconstructible full arrays.

Fresh calibration reproduced 32 full fits, 128 leave-one-out fits and 384 principal-cosine comparisons exactly. Frozen O and M maps match their earlier artifacts; fresh fitting precedes evaluation. The immutable map/source/input/inventory hashes and model/parameter/precision closures passed.

The first frozen saved-state audit stopped because NPZ serialization discarded QR tensor strides. All map values were exact. Restoring only the 24 `basis_qr` matrices from stride(12,1) to their original stride(1,12) recovered every saved direction/hash/scalar and geometry value exactly. No tolerance changed. Original auditor/source/log remain preserved. See `gate_wiki_direction_replay_diagnosis.*` and `gate_wiki_layout_adapter_fixture.*`.

FP64 snapshot reconstruction maxima: b3.064e-6, mass5.538e-7, conditional mixture3.249e-6, native context6.135e-6, target b9.279e-7, target mass2.357e-7, delivered context6.537e-7. These are descriptive higher-precision comparisons; the unchanged runtime guards were separately checked.

`gate_wiki_readout_audit.py/.json` loaded only the pinned GPT-2 unembedding tensor and reconstructed the 44 fixed GPU vocabulary rows: 11 singleton/P states × queries1,126,127,510 on natural Wiki window0. The prospective conservative dot-product error bound passed (largest error/bound0.022365). Maximum logit error0.000377515; RMS7.872e-5; relative L2 9.245e-7. Maximum CPU-versus-actual-logit NLL difference0.000245419, bidirectional KL3.583e-9, and zero argmax differences. No reference norm was zero. This is not full-query, all-input, or native-C readout validation; saved runtime metrics remain primary.

## Findings and limits

The experiment supports robustness of the **conditional zero-source-value intervention** to this particular within-Wiki calibration replacement. It does not establish universal calibration invariance, optimality, native-model improvement, or a unique mechanism.

For all four Wiki4 center/direction cells, L10 PC1 removal lowered full-view loss relative to P on Mill, Ethics and new optimizer source for both start identities (24 intervals below zero), while slightly raising Wiki loss (8 intervals above zero). The FF cell gives:

| Corpus | Natural PPL/P | BOS PPL/P | Natural agreement/P | BOS agreement/P |
|---|---:|---:|---:|---:|
| Wiki validation |1.003240|1.002658|95.39%|96.03%|
| Mill, Utilitarianism |0.982692|0.983074|97.79%|97.65%|
| Spinoza, Ethics |0.980979|0.980339|97.79%|97.25%|
| Optimizer code |0.986016|0.983824|98.58%|98.09%|

Every non-Wiki PC1-minus-orthogonal L10 contrast for those four cells has a negative interval. The matched control is one frozen orthogonal direction per basis, with equal logit-change norm at the same incoming state. It does not cover all orthogonal directions, and jointly intervened trajectories need not have equal changes. Wiki's relative PC1/control difference is mostly unresolved even though both can harm loss.

The full eight-layer intervention is a different result: FF PPL/P is1.049624/1.046619 on Wiki,0.989786/0.995658 on Mill,0.955246/0.958827 on Ethics, and1.002882/0.983806 on code (natural/BOS). Mill and code intervals cross zero; Wiki harm and Ethics benefit remain clear in this finite panel. FF all-layer agreement/P is only84.36–91.37%, so small or improved loss must not be described as exact fidelity.

Calibration-center and direction effects are not interchangeable. At L10, replacing O center with F center slightly raises NLL at either fixed basis in all16 contrasts (means+0.000233 to+0.000729nats; all intervals positive). Across all layers the same center replacement lowers NLL in all16 contrasts (means−0.005942 to−0.001780; all intervals negative). Replacing the basis across all layers raises mean loss in all16 contrasts, with12 positive intervals and four code intervals crossing zero. All16 removal-family center×direction interactions have positive intervals. These are contrasts in the loss of conditional interventions, not additive causal contributions or a decomposition of model computation.

At L10 O/F PC1 cosine is0.996647; fresh PC1 explains60.47% of detrended calibration variance and its leave-one-out cosines are0.998596–0.999849. Across layers O/F cosine ranges0.932976–0.998835, mean displacement0.367814–0.853018 and fresh leading relative eigenvalue gap0.321631–0.720378. These diagnostics describe these four calibration passages; the n=4 evaluation bootstrap does not integrate calibration-sampling uncertainty. M8 anchors differ in both composition and sample count and do not isolate a mixture effect.

**Native C remains a distinct baseline.** For example, Ethics FF L10 PPL/C is1.000043 natural and1.000783 BOS with intervals crossing zero: improvement over P largely returns to native loss here. Wiki FF L10 is worse than C on average, and other corpus/start comparisons vary. Do not infer that removing PC1 from native-source-value attention would help.

The worst conditioning lies outside the raw snapshot subset: maximum gain316,352,692,224 occurs at L4, Ethics window3/BOS, remove_FF_all; minimum mass3.161028947e-12 is at the same input/layer in P. Largest conditional reconstruction error3.695488e-5 occurs at L7, optimizer window3/natural, orthogonal_M8_all. All runtime guards passed, but the 11 raw L10 snapshots cannot independently bound the worst-gain context.

Production resources:262.573109s, peak GPU allocated966,477,312B and process RSS2,267,564KiB. Independent core12.09s/RSS1,054,716KiB; sampled one-tensor readout1.66s/RSS1,482,036KiB. Zero reviewer model forwards/GPU use.

## Most informative next scope test

A prospective native-source-value test is well motivated. Keep the selected layer/arms/maps fixed and cross value-world with intervention; do not use these evaluation outcomes to choose another best layer or calibration. For non-source mass a, native context is c=a*u+(1−a)*v0. Holding the current conditional mixture and source value fixed while changing allocation gives c'=c+(a'−a)*(u−v0). This should preserve query0 relative to each world's own baseline and include exact current/full-rank/native reconstruction controls, paired P/native baselines and fresh passages. It tests transfer of the same map-defined intervention to native values; maps fitted in P are still a conditioning choice. It would not by itself identify mediation or show that the same gate coordinates have identical distributions across worlds.
