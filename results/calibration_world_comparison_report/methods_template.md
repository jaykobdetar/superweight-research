# Report19 aggregate-presentation rebuild — DRAFT

The public bundle alone supplies every data input to this presentation builder. No private R report, raw manifest, tokens.pt, NPZ, learned weights, corpus, source activation or local study directory is needed. Original private JSON hashes remain provenance commitments; the builder never opens those original paths. The report is an aggregate presentation, not an independent audit or model replay.

The bundle contains the exact original full analysis JSON bytes encoded in base64, with SHA256 8dc01b8911cd95c08c9ba9c46c95a00e0cf5c54177d5bc70f7eda635bd1797b5. It retains the complete safe primary readout and literature record. Geometry is reduced to the one declared layer-10 PE/CE row of scalar center distance and u/v cosines. N/P/R, calibration/scientific acceptance and the model manifest are restricted to the typed status, count, hash-binding and scalar error-aggregate fields used by the report. No R per-state predictions, targets or mismatch IDs are included. The finite pointer projection proof records original and public commitments separately; it is not a claim that the private audit was rerun.

All 64 primary comparisons retain four passage values. The builder copies saved means and interval endpoints and converts nats to millinats; it does not recompute bootstrap samples, intervals, calibration fits or scientific tests. The table, primary extraction, claim sheet and plot functions are unchanged from the preserved v1 source. Both source/start labels appear in every forest panel in fixed declared order, with panel-specific horizontal scales explicitly labeled. The 12-dimensional head-wise gate-logit PC is not a hidden-state PCA or a count of mechanisms.

The self-contained HTML includes inline figures and the full 64-row table. Its complete analysis JSON, primary CSV, primary JSON and methods downloads are embedded data URIs, so the single HTML file works without companion Library assets. Separate copies remain available in the report directory. No scripts or remote requests are required to view it; the primary literature links open only when the reader chooses them.

Use Python 3.12 with NumPy 1.26.4 and Matplotlib 3.6.3. These dependency versions come from the existing reviewed build environment; an unspecified system Python 3.13 is not the documented environment. With that environment available, run from the task/repository root into a fresh output directory:

```bash
python3.12 -B pulse_transfer/calibration_world_report_candidate_v2/build_report.py --inputs pulse_transfer/calibration_world_report_candidate_v2/data/report_presentation_inputs.json --out pulse_transfer/report19_rebuild_v2
```

The bundle hash is pinned by the builder, while the bundle does not pin the builder, avoiding a self-hash cycle. The builder verifies templates and the public bundle before and after, writes exclusively to a fresh directory outside its source and input trees, and records versions, timing, process peak and file hashes. Matplotlib/NumPy run on one thread with a 3 GiB process bound. The persistent Matplotlib cache is outside the report artifact; no cleanup, deletion, deduplication, publication or Library action is performed.

The v1 source remains preserved: it incorrectly required the full private R JSON for a public presentation rebuild. This v2 projection/input change repairs that presentation-reproducibility gap, not a model or scientific failure. Independent N/P/R and the completed study are not repeated. Portable model reproduction and the public numeric export remain pending their own acceptance; a second full 776-forward T22 replay is unperformed.
