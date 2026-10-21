# Independent report 4 review

Reviewed `build_report4.py` and generated `report4.html` against the five completed CPU audits. No model/GPU loaded. Main numerical claims and tables agree with saved results, including the Gutenberg-footer defect and seven-window sensitivity.

Verified resource totals: 4,048 forwards, 295.4388721 seconds, maximum allocated GPU memory 2.6410565 GiB, lifetime worker RSS 5.1447296 GiB. The five appendices contain 520 primary condition groups (188/80/68/96/88). All five manifest weight hashes and parameter-version checks pass. Report correctly distinguishes lifetime RSS, bootstrap scope, rank-16 gate passthrough, nonlinear clamping, static source templates versus dynamic gate covariance, and non-optimal counterfactual donor gates.

Requested corrections before freezing:

- The full-appendix paragraph says non-baseline paired comparisons are in each `contrasts.csv`. Those generic tables mainly compare clean, and confirmation compares clean/deleted. Non-baseline pair effects cited here reside in the relevant independent audit JSONs and `value_layers_run/interaction_contrasts.csv`. Name those actual files or export a consolidated paired table.
- Qualify L2 swaps having “exactly no effect” as equality to the current-recipient reconstruction control. They are not necessarily bitwise equal to native execution because reconstruction changes FP16 arithmetic.
- Define the deletion intervention explicitly: remove only the contribution of L1 `down_proj[1764,1710]` at source position 0; this is not global weight deletion.
- Narrow the lead's categorical “neither a constant bias nor a single shared scalar transformation” to the tested bias/constant/bin/PCA approximation families. These experiments do not exclude every nonlinear scalar representation.

Reproducibility packaging review is pending files. In addition to the five completed runs and their scripts, replay/audit dependencies include `cache_run/calibration.pt`, `residual_paths_run/tokens.pt` and `data_manifest.json`, `quiet_anchor_run/tokens.pt` and `per_token.pt`, sibling `pulse_causal_tranche/run/tokens.pt`, `audit_common.py`, the imported implementation modules, and the exact confirmation source material or an explicit frozen-token replay path. Original confirmation preparation resolves installed torch optimizer source files, so a different installed torch version can change replay text unless pinned/validated or saved tokens are used. Audit scripts currently verify absolute original source paths; document or support remapping to archived provenance copies for portability.

An archive/Library identity is described in the report before packaging exists at initial inspection. Verify those claims against the final bundle and upload before delivery. Archive inspection and stub replay tests must not load the model or run GPU experiments.

Follow-up: all four requested report wording corrections are applied in the rebuilt report. `replay_report4.py` now checks source hashes before loading the worker; stub tests pass for success, late/split worker-failure markers, bad model hash/parameter versions, incomplete/missing manifests, existing output paths, and changed/missing source files. Evidence is `report4_packaging_audit.py/.json`. All exact archived text copies match their recorded hashes, and frozen dependencies exist.

The bundle plan includes the required sibling first-tranche files, quiet-anchor audit references, all five raw runs, shared helpers, sensitivity CSV, and report4 audit artifacts; it excludes the active upstream run. Final immutable-archive verification remains pending creation. Two final portability clarifications were requested: distinguish source-free metric replay from full confirmation provenance auditing, which still needs original paths, and preflight the optimizer files resolved from the current Python environment as well as the recorded original paths.
