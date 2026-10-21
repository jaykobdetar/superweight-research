# Periodic continuation report claim audit

Reviewed `build_report.py` against saved continuation summaries and independent
audits. No model/GPU use and no experiment/report-builder edits.

## Concrete corrections sent to the report author

1. Add an explicit source definition near the opening: source-deletion experiments
   remove only the L1/output 1764/input 1710 contribution at position 0; later uses
   of that scalar and the second target remain intact. The draft does not name
   input coordinate 1710. Replace “globally damaged context” in the later-head
   paragraph with “source-deleted recipient” to avoid suggesting global-weight
   deletion. Cache-only execution uses clean weights and no deletion.
2. Fix the literal claim that every `*_run` contains frozen token IDs and calibration.
   `mediation_run` and `normalization_run` reference the shared `gate_run/tokens.pt`
   by hash; they do not each contain a token file. `factorial_run` has no calibration
   file because it does not fit calibration constants. Say artifacts are stored
   or explicitly referenced across the shared reproducibility bundle.
3. The author plans to add complement-rescue argmax agreement 89.83%/94.31%/93.85%.
   Include it alongside the small NLL differences so “nearly recovers” is not read
   as exact output recovery.

Optional precision improvement: replace “reject a denominator-only account” with
“donor-denominator restoration alone is insufficient in these tested interventions.”
The latter directly states the measured result without excluding other state-
dependent scalar or normalization mechanisms.

## Verified claims and interpretation

- Factorial damage values and intervals agree with saved data. The report clearly
  distinguishes the later Plato gate batch from the initial WikiText/Hume factorial.
- Gate/value native comparisons, exact identities, diagnostic reconstruction error,
  and approximate zero-head wording match the independent audit. V0-zero rescue
  ratios are 20.277/8.104/7.752, supporting the rounded “roughly 8–20×” statement
  when understood as H15 in the source-deleted recipient.
- H15 is explicitly not established as necessary. The contrast between modest
  near-zero-head cost and destructive donor poisoning is scientifically important
  and preserved. Matched H0 constant controls are acknowledged.
- Exact full-KV and full-normalized-vector closure is properly described as
  architecture-expected validation rather than discovery of a unique minimal
  circuit. The 510-prediction score scope and retained source residual are explicit.
- Complement and denominator-only normalization ratios match saved results.
  Full restoration is exact only for the stated later-position scoring range.
- Cache packing/execution, dictionary versus full-context oracle distinction,
  hybrid layers 0/1, exact first-logit reuse, and approximate mean-cache behavior
  are accurately described. The ordinary single-token cache is shown as the
  changed-execution numerical control.
- Timing medians and quantiles are derived from the saved 24 nonwarmup rounds.
  The approximately 5% slowdown versus full cached prefill is correct. Copying
  overhead is included; dictionary/calibration work is excluded. No practical
  speedup is claimed, and full-no-cache/full-cache paths are distinguished.
- Intervals are explicitly paired, pointwise, conditional, and based on eight
  windows per corpus. Reused mechanism-test evaluation sets, limited corpus/model
  breadth, unavailable generation/task evidence, and ongoing research are stated.

No numerical headline correction or full-understanding overclaim was found.
Remaining changes are the source-scope, artifact-location, and output-agreement
clarifications above; they do not invalidate measured experiments.
