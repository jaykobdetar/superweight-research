# Independent bounded L1 signal-production audit

Result: no substantive algebraic or numerical error found. Final direct-exclusion revision rechecked at 2026-10-23 05:07 UTC. All computations are conditional on the recovered FP32 normalized prefix states, not an independent full-model forward. No network request, checkpoint download, or modification to parent analysis files was performed. Independent scripts used two BLAS threads, sequential blockwise matrix products, and peak RSS about 232 MiB.

## Evidence recomputed

- Nine relevant recovered input files, including both normalized-state caches, query metadata, reference FP32 results, source-generation code, safetensor header/index, and selected down row, match the recovered bundle manifest's sizes and SHA256 values.
- All three acquired tensors match their receipt SHA256, expected byte sizes, shapes, safetensor header ranges, and pinned revision URL. This establishes local cache/receipt consistency, not independent verification against a publisher-signed model hash.
- Cached state labels and selected gate/up rows match the new replay. FP64 recomputation from the exact FP32 cache differs from saved FP32 gate/up values by at most 5.98e-6/4.75e-6, activation by 2.47e-4, and centered full MLP output by 1.26e-4.
- The sum of all per-neuron centered-axis writes agrees with the independently reconstructed full MLP axis to 4.55e-13 in FP64.
- All 189 masked-neuron output interventions were recomputed by zeroing/retaining activation coefficients before a fresh down-projection product. Max discrepancy from saved interventions: axis 4.86e-6, alignment 5.35e-7.
- All 147 final source geometry conditions agree within 2.36e-7 alignment; all 16 scalar gate/up factorials within 1.50e-4 scalar contribution; all 16 final component swaps within 1.09e-7 alignment. The independently recomputed direct non1710 MLP differs from saved other_mlp by at most 3.15e-6; the final scalar-edge-deleted vector differs by at most 1.56e-5.

## Algebra and intervention meaning

Let C = I - 11^T / D and a = C e_1764 / sqrt(1 - 1/D). For centered output x, the reported axis is x dot a = x_1764 / sqrt(1 - 1/D). Neuron j contributes activation_j times (down_1764,j - mean(down_:,j)) / sqrt(1 - 1/D). Carry + all MLP writes decomposes linearly as implemented.

The centered pre-MLP carry can be reconstructed from a non-affine LayerNorm direction and the recorded centered norm. The saved FP32 state is only approximately mean zero, so this is a near-exact reconstructed centered vector, not recovery of the uncentered residual stream. The mean is unnecessary for these centered geometry diagnostics.

“row_deleted” is a misleading shorthand unless explicitly defined: it deletes only the scalar edge down[1764,1710], not the entire down-projection row. “column_deleted” removes all of down[:,1710] times activation_1710. Their geometry need not agree because the scalar deletion leaves the rest of 1710's column intact. The raw column squared-energy fraction at coordinate 1764 is 0.97904937; the centered axis squared-energy fraction is 0.97927737. Neither means 97.9% of all model signal or a causal fraction of behavior.

## Findings independently supported

- Source tokens 510, 15, 535, and 209 have substantial negative surviving axis after the 1710 column is removed. Their alternate-MLP signed shares are about 91.0–95.7%; carry supplies the rest. This is a signed residual-axis decomposition, not a universal fraction of signal or loss.
- Source 187 is different: negative alternate writes sum to −7.9191305, positive alternate writes to +7.7953667, giving −0.1237638. Adding carry +0.0191772 leaves −0.1045866, with alignment −0.0097299. Its small net result reflects cancellation, not absence of alternate writers.
- Source0 510's largest alternate writers include 7579 (−3.92935), 7845 (−1.70964), 6947 (−1.56847), and 2637 (−1.12123). The H14-edge-deleted query states instead have leading negative writers 3950 and 4786 in all four fixed panels. This supports a state-dependent writer composition, not one universal backup neuron set.
- The output-level gate/up factorial agrees numerically. The R gate with E up stays near zero; E gate with R up yields a moderate negative write; E/E yields the much larger negative write. These are post-hoc scalar combinations, not natural independently manipulated prefix states.

## Precision and inference cautions

- Resolved precision issue: the final main script computes non1710 output directly with activation1710 set to zero, and computes scalar-edge deletion by replacing only raw output coordinate 1764 with its directly recomputed counterpart before centering. This is algebraically equivalent to deleting only down[1764,1710], while avoiding dominant subtraction in the affected coordinate. H14 carry/other swaps now use saved direct other_mlp. The separately recorded diagnostic shows direct-versus-subtraction discrepancy up to 1.87704e-4. Source187 final other-axis is −0.123763121 and column alignment is −0.009729807. Direct masked FP64 replay confirms both. Qualitative findings are unchanged.
- The signed share is unstable or exceeds 100% when carry and MLP cancel. In R controls it can be negative or several times the net axis. Do not interpret it as a nonnegative attribution percentage.
- Removing a mathematically defined axis component necessarily removes its projection by construction. It is a geometry diagnostic, not independent empirical evidence of behavioral necessity.
- Alternate sets were selected after inspecting these same states. No held-out identification or generalization claim is warranted.
- The calculations do not test later layers, loss, logits, induction, copying, a trained counterfactual, or the network's ability to function without these writes. Source token samples are isolated one-token prefixes; query samples are the four cached fixed panels.
- Acquisition reuse validates an existing tensor against its receipt hash/size but does not itself assert the receipt's key/URL/range equals the currently requested job. The audit verified that equality for these actual files, so this is a reuse-hardening opportunity, not a current data defect.

## 8041 amplitude bound

For centered gate/up rows g and u and ideal non-affine LayerNorm h, mean(h)=0 and ||h||^2 <= D. Since |SiLU(g dot h)| <= |g dot h|,

|SiLU(g dot h) (u dot h)| <= D/2 (||g|| ||u|| + |g dot u|).

Multiplying by |down[1764,8041]| reproduces 0.16032984931200406. This is a real-arithmetic prompt-independent upper bound on the raw coordinate-1764 write, not a measured maximum or a roundoff-certified floating-point theorem. The centered-axis analogue is 0.16069531235012433. The analogous whole-column L2 upper bound is 25.72159013539151, so the raw-coordinate bound does not establish globally negligible whole-vector or downstream impact. Max observed raw 8041 write over these 21 states is approximately 4.994e-5.

## Artifacts

- audit.py: independent FP64 replay, local integrity/provenance checks, selected values, and 8041 bounds.
- audit.json: computed results for all 21 states.
- audit_interventions.py: independent masked-coefficient intervention checks.
- interventions.json: aggregate discrepancy results for all saved intervention conditions.

## Final version snapshot

FINAL_ARTIFACT_MANIFEST.json captures file sizes and SHA256 values for the final parent artifact snapshot. Hash inclusion is not an assertion of substantive audit for unrelated newly added files. The audited final analysis script SHA256 after portable-input packaging is 243d038c1abec971ccc914425e61a4881cc5e3e2a4250471142d166783452989; intervention script SHA256 is 386621e46def2b75f142fa58f13624392eb6239022d0fbab5b5c89040d29fa92; replay NPZ SHA256 is 9c626edd6738af1b2dfbe352cfd06c2e73f3bc3a6232409e4e1fb1c27f4abdc4. Independent audit scripts and numeric JSON outputs have been rerun against this revision.

Packaging refresh, 2026-10-23 05:15 UTC: acquire/analyze now resolve original inputs through OLMO_SOURCE_ROOT or packaged inputs/olmo_training_origins. All 12 packaged original files match the original bundle manifest, and the packaged manifest itself is byte-identical to the original. Numerical branches are unchanged, and replay NPZ plus all previously audited numerical output files retain identical hashes; only check-file resource metadata changed on rerun. FINAL_ARTIFACT_MANIFEST.json has been refreshed to include final script, input, report, and result hashes. REPORT.txt was checked for consistency with the bounded new replay; no substantive local-result overclaim was found. Its additional historical full-model claims were outside this audit scope.
