# T19 implementation and evidence layout addendum

Prospective addition before pretrained execution. The original proposal and tokenizer-only input preflight remain pinned under their original hashes. This addendum makes numerical, observation and storage choices explicit; no T19 model outcome informed them. The 2×2×2 intervention, fixed F calibration, 34 conditions, 2,184 forwards and scientific bridge thresholds are unchanged.

## Native execution and arithmetic bridge

All conditions preserve the model's incoming attention-mask arguments, including implicit causal SDPA. The hook builds an effective causal mask only for diagnostics. Thus000 baseline is an observer of the native model. The111 scaling arms reproduce the T18 **local scaling arithmetic on the same T19 incoming state and mask**. They do not claim a full reproduction of T18's older explicit-mask execution protocol. Both arithmetic paths use identical live states and masks within each bridge comparison.

The reconstructed FP32 non-source mass is retained without clipping. It may exceed1 by rounding, with a prospective bound of1+1e−6. Finite0 or1 endpoints remain valid for the delta operation; legacy scaling separately requires positive mass, finite gain and its unchanged conditional/reconstruction guards. The mapped target sigmoid remains in[0,1], except the exact unchanged/current bypass retains the original diagnostic mass. FP64 decomposition and delta diagnostics are reported; the existing1e−4 manual/native/explicit-source reconstruction guards remain hard requirements.

Every forward uses `use_cache=False` and returns hidden states with `output_hidden_states=True`; this returns already computed states without modifying the mathematical intervention. Eight counted anchors occur at natural window0 of each corpus. The native-repeat anchor removes all intervention hooks and compares its actual logits/final hidden/metrics exactly with the000 observer baseline. The native-identity anchor transiently installs the historical writer identity operation and compares it with that same reference. The loaded model is reused throughout.

## Fresh input audit and inference scope

The new panel contains16 accepted512-token natural windows and paired token0-only BOS replacements. Its independent audit reconstructs all382 grid candidates, including366 rejections, against the unchanged selection rules. All561 inherited duplicate hashes remain in the577-hash preselection union; four fresh T18 Wiki-test calibration buffers remain full8192-character exclusions. The current evaluation uses six source files: Wiki validation, two books and three optimizer files; the Wiki-test fit and its source exposure are inherited from pinned T18 evidence, with no new calibration or tokenizer fitting.

Primary scores are targets2–511. Early/later views use the same512-position forward and are secondary. Four-passage paired bootstrap draws use seed20261021 and2,000 repetitions, reused across arms and paired starts. The analysis prespecifies64 finite-effect/factorial contrasts per corpus/start/view, plus paired-start interactions. They are correlated comparisons, with no multiplicity adjustment or population calibration uncertainty. Neither a loss interaction nor a similar PC1 direction identifies a unique circuit or mediation fraction.

## Evidence layout and resource accounting

The normal evidence cap is512MiB; a separately measured single-failure allowance is256MiB; preserve2GiB free space. These replace the earlier unmeasured128/32MiB design target after exact storage recovery created headroom. This is a resource choice, not a changed scientific estimand. A full high-entropy writer fixture must account for all32 shards,1,088 main conditions,1,088 suffix conditions,eight anchors,eight full snapshots/readouts,actual check dictionaries,frozen source closure and32MiB metadata/input/analysis allowance.

Every successful condition retains per-target losses, C-relative and own-boundary KL/agreement, local gate arrays, coefficient and query-level norms, source identities and runtime checks. The14 norm fields distinguish511 query values from the12 source-head norms. The unhooked native-repeat anchor has no intervention norm capture; this empty record is explicit. The native-identity anchor has the ordinary observer capture.

Optional ordinary-condition copies of per-head source probabilities and source/non-source vector norms are represented by fingerprints and scalar min/max/mean/RMS summaries. Complete query-level context/gate/mass magnitudes and perturbation norms remain explicit. Full vectors can be independently reconstructed in the eight predetermined saved L10 snapshots; ordinary conditions have runtime evidence and fingerprints rather than full attention replay. This limit must accompany later claims.

Failure capsules preserve the actual compared sides and provenance for causal-order, source/current, suffix and bridge failures. Fixed and worst-discrepancy full-vocabulary rows plus all per-query maxima are retained; full vocabulary tensors are not silently claimed to be supplied. A failed hard bridge halts the accepted batch, preserving both sides, without relaxed thresholds or dropping exposed inputs.

The first interrupted sizing attempt and the second attempt that exceeded its complete-output allowance remain preserved. A revised layout removes only the optional repeated vectors above, before any pretrained output. Do not delete the failed fixtures. Any later change to layout or failure membership requires renewed sizing; narrow guard/freeze/metadata-only changes may reuse exact serializer measurements only with explicit source-difference and independent compatibility review.

## Immutability and user storage restriction

The user now requires specific approval for any deletion, cleanup, deduplication or storage replacement. No further such operation is part of T19. Preserve all fixture directories and failures. Experiment checkpoints use unique immutable files and an append-only hash-linked event journal; final canonical manifest/inventory files are created once. Existing raw evidence remains protected. The completed105-pair storage transaction predates this restriction; both logical paths retain their verified bytes but now share storage, as recorded in the local storage annotation. Do not write those shared files in place.

Independent source/input/implementation/resource review and a final freeze are required before loading pretrained weights. There is no new user-approval request implied by these scientific review gates; the ongoing local research is already authorized.
