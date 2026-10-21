# Independent V2 failure review

The failed run is correctly marked failed; it does not establish a completed gate-subspace result. No tolerance was changed and no retry was performed by this reviewer. The audit script and full numeric evidence are `gpt2_gate_v2_failure_audit.py/.json`.

The actual failing pair is `failure_state.pt['prefix_pairs']['C']` for Wiki window0, BOS start, native C, physical lengths128 and512. `active_input` and `current_hidden` instead refer to the subsequently computed P512 state; using them would misidentify the failure. The saved C pair independently reproduces the reported failure:

- Maximum symmetric per-query relative L2 is **1.0108874094104775e-4** at query110, exceeding the frozen1e-4 bound by1.0887%. Only query110 exceeds that bound.
- Whole-prefix relative L2 is2.441561531347229e-6 and RMS absolute hidden drift2.0466428326788e-5. Both the global relative and original loss criteria pass; that does not override the failed local criterion.
- Maximum absolute hidden difference is0.004055023193359375 at query110, coordinate496:41.18998718261719 to41.19404220581055. Three entries exceed the old absolute0.001 bound. Query110's short/long vector norms are62.47053653/62.47048660.
- Source query0 maximum absolute difference is9.918212890625e-5, relative L2 1.7704645075435122e-6. Query127 maximum difference is4.57763671875e-5, relative3.3957632886455594e-7.
- Across126 scored shared targets2..127, maximum loss drift is0.00019121170043945312, mean signed delta7.47595e-7, MAE2.63769e-5 and RMS4.22710e-5. Predictions agree exactly. Minimum top-two margins are0.01119995 and0.01120758.

All23 completed natural-start cross-length pairs independently satisfy V2's recorded relative/loss guards. All21 completed natural-start fixed-shape suffix controls have exact saved prefix hidden, gates, source states and metrics; recorded independent logit/context fingerprints and flags also agree. The available60 Q/K/V snapshot states were independently reconstructed in FP64. Maximum errors were2.9832e-5 for gate logits,8.3767e-7 for mass,1.0593e-5 for conditional mixture,7.4842e-6 for native context and2.6375e-5 for delivered context, all below1e-4. Delivered context exactly matches the saved FP32 scaling formula. Query0 preservation is relative to zero-V P, not native C.

These controls establish numerical properties on the completed natural-start input. No BOS fixed-shape control completed before the failing C pair, so they do not prove BOS suffix independence. The failure occurs in a native baseline without an active gate projection, and the earlier separately audited native diagnostic establishes shape sensitivity before attention for its natural input. This is consistent with numerical execution-shape sensitivity; the saved BOS final states alone do not establish a unique cause, rule out all other defects, or attribute the difference solely to SDPA. The threshold failure remains a real protocol failure, not an almost-pass.

Accounting independently matches90 returned/attempted forwards:46 natural main +21 natural suffix +21 BOS short main +one BOS long C +one completed but unrecorded BOS long P. There are68 main records and21 suffix records, zero new calibration forwards. Raw/map hashes and frozen source hashes remain unchanged. This failed run has no completed final model hash/version closure; do not claim that final check ran. Recorded resources are10.5910792351 seconds,966,942,208 GPU allocated bytes,1,090,519,040 GPU reserved bytes and2,235,664 KiB process RSS. The reviewer used CPU only, no model, and peaked at1,185,756 KiB RSS.

No rank efficacy, variance-to-fidelity relationship or full scientific summary was inferred from this truncated run. A completion-gated full-result audit was not executed.
