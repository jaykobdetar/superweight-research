# Independent localization implementation review

**Ready for one prespecified pretrained run at the hashes below. No remaining launch blocker.** This is implementation approval, not a result or efficacy finding. The reviewer loaded no pretrained model and used no GPU. Production scripts were not edited.

## Frozen files

- `gpt2_gate_localization.py`: `eafab996ce06c697d081d8e0d06583cf9d3c0866ab2f27213cfcb4387503e238`
- `analyze_gate_localization.py`: `3998aed2c976c85999795f39d547db90270c2881f03725a720eb7e70269f2d28`
- `GPT2_GATE_LOCALIZATION.md`: `e97057f0f28d05f2d1d71baed68c012fdae10a286347700236f54e50c8fef415`

## Evidence and corrections verified

The earlier two findings are resolved: the analyzer now independently compares saved singleton incoming `b`/mass and directional `s` against P; setup input/hash/model-load exceptions preserve zero-forward failure status and resources. Frozen source copies are verified before model loading. Initial 7 GiB and subsequent 3 GiB free-disk guards stop execution without deleting files. The reviewed author failure fixture verifies both guard paths, bad input hash, model-load error, and a pending-C-record failure after a completed P forward, preserving the actual tensor identity separately.

The independent physical-512 integration used a random-weight 12-layer/12-head GPT-2 with nonzero QKV and output biases, small vocabulary and inner width. Its 27 CPU forwards covered C/P/current/fullrank and all/L4/L11 for every scientific family, each with a changed-suffix pair. Native C hooks exactly matched no-hook logits; current/fullrank exactly matched P; independent FP64 loss/KL matched within 2e-6 (maximum KL discrepancy 3.908e-7); singleton incoming states, source/query0 identity relative to P, zero source V, unaffected delivered contexts, prefix equality and cleanup passed. Maximum hypothetical norm-match error was 1.120e-7, below the unchanged 1e-6 guard. Peak reviewer RSS was 1,080,560 KiB.

The independent analyzer test explicitly credits reuse of the author's full fabricated-panel generator. It exercised 938 main records, 930 suffix controls, 1,688 checks, 14,400 layer checks, the 52-state snapshot inventory, 248 full and 496 segment summaries, 810 contrasts/216 groups, and 90 nonadditivity records/24 groups. It verified actual n3 code/n4 prose bootstrap and signed joint-minus-summed-singleton arithmetic. These are fabricated fixture counts, not experimental outcomes. Added one-ULP saved b, mass and s corruptions were rejected by their exact consistency checks even after refreshing file inventories. Failed status with a completion timestamp, changed early-score endpoints, scalar corruption and raw identity corruption were also rejected; stale pass status was cleared. Maximum analyzer child RSS was 891,740 KiB. The final-hash author 62-forward CPU fixture additionally covers every one of the 31 modes and all eight singleton layers, with independent NumPy map calculations and query-row locality.

## Scientific and accounting scope

The implementation matches the approved prospective families: remove `s*u`, subtract `s*v` using the same own-state coefficient, or replace with frozen alpha+log(q). No test fitting, calibration recomputation or outcome-based direction/layer choice occurs. Thirty inputs derive from 15 frozen passages, with n3 code explicitly retained after the preserved input-capacity failure. Counts are 930 primary + 8 anchors + 930 suffix = 1,868 new forwards; new calibration is zero. The eight inherited calibration forwards are dependencies. All physical inputs are 512 tokens. Main full/early/later target ranges are 2–511/2–127/128–511; suffix scoring uses only unchanged targets 2–127. Norm summaries use the corresponding predicting queries and exclude unscored q511, while all-query numerical guards include it.

Singleton prepatch equality makes the directional coefficient shared at that layer. All-layer arms have different endogenous trajectories: no across-arm norm match is asserted. Logit norm matching does not equalize mass, context or output norms. Joint-minus-singleton sums are descriptive nonadditivity, not mediation or proof of feedback. Source invariance is to P with V0 zeroed, not native C. One fixed orthogonal direction is one control. The small selected corpus panel, calibration-conditional intervals and correlated start/score views remain limitations. Any layer selected from its results needs fresh confirmation.

Successful full contexts/logits are not retained throughout: exact runtime comparisons and independent fingerprints support those closures outside 52 fixed raw states. The analyzer cannot independently reconstruct every successful full-context norm from raw tensors; that limitation is disclosed. A numerical audit of actual raw per-token arrays, gates, singleton states, suffix shards and FP64 snapshots remains required after completion. Preserve original V1/V2 failures and all new failures without automatic retries, threshold changes or input substitution.

## Review artifacts

- `gpt2_gate_localization_independent_integration.py/.json`
- `gpt2_gate_localization_independent_analyzer.py/.json`
- `gpt2_gate_localization_implementation_readiness.json` records frozen hashes and evidence hashes.
- Earlier `gate_localization_design_review.md` / `gate_localization_design_audit.json` cover frozen inputs, exclusions and direction construction.
