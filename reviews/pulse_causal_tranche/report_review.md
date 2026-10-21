# Final report claim audit

Reviewed `build_report.py` against the audited CSVs and saved CPU loss/prediction
tensors. No report-builder edits, model loading, or GPU use occurred.

**No blocking numerical correction was found.** Specifically verified:

- Continued suffix deletion ratios: WikiText 5,248.306827 and Hume 1,615.715241.
  Pulse ratios: 1.004969435 and 1.004418471. The reported reset/continuation
  interpretation correctly acknowledges changed source identity and distance;
  scoring aligns the final 511 target IDs.
- Uniform-offset clean ratio: 0.999895522. Ratios computed against that offset's
  own clean baseline are deletion 2,590.471199 and pulse 1.005847926, matching
  reported values. These use eight windows, whereas the main causal table uses 12.
- Selected full-context replacement in clean: perplexity increases 4.870400%,
  changed argmax predictions 11.415525%, matching rounded 4.87% and 11.42%.
  Clean/pulse K0/V0 and full-context replacement ratios, agreements, and quoted
  intervals also match. Percent perplexity changes use exp(mean NLL difference),
  not the mean NLL difference multiplied by 100.
- L2/H15 K-only, V-only, and K/V all-position patches exactly match their
  position-zero versions in saved loss and prediction, in both directions.
  Complete-context and K0/V0 outputs match exactly as claimed.
- Fresh EOS/`The`/period loss and prediction arrays exactly match clean through
  length 512. `The` continues to match at 1024; EOS/period do not. Natural starts
  do not have exact restoration. The report distinguishes these cases correctly.

## Suggested precision clarification

The report acknowledges small fresh-pulse differences at 1024. It could make
common-prefix comparability more explicit: across every corpus/identity/state
combination, first 127 loss and prediction arrays are identical at lengths 128
and 512. At1024, the *clean* mean first 127 loss drift across cells ranges from
−0.001085 to +0.000396 nats, with maximum individual token difference 0.192091.
Across all four states, maximum absolute cell-mean drift is 0.002499 nats and
maximum single-token difference is 0.435324 nats. Thus aggregate drift is small,
but it is not uniformly tiny at every token. The numerical cause was not isolated.

Suggested optional sentence: “The shared first 127 loss and prediction arrays
match exactly between 128 and 512 tokens; at 1024, small aggregate changes also
occur in clean runs, with some larger token-level differences. Their numerical
source was not isolated.”

The report's main scope, centered/raw-delta distinction, nonadditive rescue versus
damage interpretation, pointwise uncertainty, replacement limitations, and
non-uniqueness qualifications are supported by the reviewed outputs.
