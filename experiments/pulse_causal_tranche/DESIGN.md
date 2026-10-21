# Frozen first-tranche design

Written before new model measurements, 2026-10-21. Scope: the first target
(zero-based L1, row1764, column1710) in pinned OLMo-1B-0724-hf. No weight sweep.

1. Replicate baseline, global first-target deletion and the original fixed pulse
on all 32 saved evaluation windows. Preserve model hash and compare old per-window
NLL. Use FP16, native SDPA, no cache, no special tokens.
2. Fresh WikiText test split: 16 calibration windows, 8 development windows and
12 held-out evaluation windows, disjoint with guards. Nominees L2/H15, L12/H5,
L15/H11; matched same-layer H0 controls. Use position-zero-only source deletion
to isolate transmission. Patch donor key/value projection outputs before native
clipping and RoPE; donors and recipients use identical tokens and positions.
Keys alone change routing, values alone change content. Patch position0 and all
positions; patch complete head contexts as a comparison. Clean-to-deleted rescue
and deleted-to-clean damage are distinct experiments. Also sabotage pulse-restored
runs with deleted keys/values. Do not interpret a head's large response as causal
necessity or sum head effects into mediation fractions.
3. Fit centered token-by-feature delta covariance on development positions9+
at outputs of blocks2,12,15. Keep the mean separate. Save full covariance,
eigenspectra (equivalently squared singular values), top vectors and raw deltas.
Evaluate causal projections on fresh held-out windows, top1/8/32, random8 with
per-token matched norm, mean-only, mean-plus-top8, and full donor patch controls.
Use both rescue and damage for leading directions. Oracle donor-dependent patches
are diagnostics, not implementable replacements.
4. Freeze an implicated candidate set from development head-context interventions
only: mean NLL damage >0.02 nats or rescue improvement >0.02 nats. After causal
tests, try development-fitted constants for key/value0 and complete head contexts
(separate first/later means), in clean and pulse-restored runs. Compare equal-size
same-layer controls. No claims of measured speedup.
5. Generalization panel: WikiText test and local Gutenberg Hume, 16 fresh
calibration starts and 8 held-out starts per corpus; lengths128/512/1024 and
natural/eos/'The'/'.' start-token IDs. Substitute only ID0; suffix IDs and reset
procedure remain fixed. Fresh pulse is mean natural contribution on calibration
starts under the same token substitution, no sweep. Compare original pulse and
fresh pulse against matched clean and deletion baselines. Score common first127
and later ranges separately. Add continuous-prefix versus reset comparisons and
uniform RoPE-position offset controls when budget permits.

Uncertainty: paired bootstrap over windows, 2000 draws, fixed seed; report NLL
differences and exp(mean difference), with selection confined to development.
Intervals are conditional, not simultaneous, and do not cover corpus or model
selection. Report calibration dispersion; do not imply 16 starts are independent
when every first token is the same. Preserve all condition specs, IDs, per-token
losses, predictions and hashes. One model process, four CPU threads, no network
downloads, no other jobs touched. Independent CPU-only review at milestone,
roughly <=3 GiB, without loading model weights.
