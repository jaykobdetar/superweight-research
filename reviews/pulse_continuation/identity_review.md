# Fixed-source identity audit

CPU-only audit passed for all 1,536 observations, 192 summaries, and 384 paired
contrast intervals. Raw NLL/KL means, baseline means, agreement, token hashes,
code/template hashes, score alignment, and model integrity passed. No experiment
edits or model/GPU use; peak RSS approximately 515MiB.

Only ID0 is replaced. Reconstructed natural/`The`/EOS/period inputs match every
saved input hash and preserve identical suffix IDs. All records score predictors
1–510, targeting token indices 2–511. Natural clean/deleted and template conditions
reproduce the preceding template experiment bitwise in loss, prediction, and KL.
Full source-KV cuts remain exact in both directions. Every fixed template produces
identical later outputs from clean and deleted states. The independently installed
normalization hook is reset after each template call, preventing carryover into
subsequent baselines or source cuts.

## Source-cache equality evidence

The code checks every layer's K0/V0 against a shared reference for each fixed ID
and source state, across all corpora and suffixes. All 144 runtime assertion records
are present and true, covering three fixed IDs × two states × 24 windows. Saved
reference caches have valid shapes and finite values; layers 0/1 are identical
between source states as expected.

The bundle retains one reference cache per identity/state rather than every compared
cache. Thus the CPU audit verifies the code, coverage, references, and recorded
assertion results; it cannot independently repeat all 144 tensor comparisons from
persisted per-window caches. This is a reproducibility limit, not contradictory
evidence or a failure of the executed assertions.

## Supported interpretation

With `The` fixed at position 0, layer 2 all-head rescue leaves perplexity ratios
8.11836/1.13066/1.09216 for WikiText/Hume/Plato. The pure analytic-axis replacement
gives 1.38057/1.01139/1.00547. The same qualitative gap persists for EOS and period.
Because fixed-identity source caches match, first-token identity alone cannot
explain these measured cross-corpus differences; suffix-dependent computation
remains important.

This does not mean identity is irrelevant. WikiText layer 2 rescue ranges from
6.60511 with EOS to 12.66463 with period. The controlled result rules out a sole
identity explanation within the tested three IDs, fixed length, and sampled
windows; it does not establish a universal causal effect of corpus labels.

Rank2-residual-plus-axis templates retain approximate preservation across source
identities: WikiText agreement remains about 97.38% for fixed IDs, with nonzero KL.
This supports the previously fitted static source template's transfer within this
panel. It does not turn the static 14×2048 across-layer residual SVD into evidence
for low-rank downstream dynamic mechanisms.

The study explicitly reuses the template confirmation windows for exploratory
interaction testing and performs no fitting. Preserve this scope, the paired
same-suffix design, and pointwise conditional intervals in reporting. No blocking
correctness issue found.
