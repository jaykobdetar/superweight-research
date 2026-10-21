# Independent CPU review: greedy and matched-history generation

The implementation and raw results pass review. The audit reconstructs 96 condition records for 16 prompts from saved per-step metrics and checks 13,216 forwards, all source/calibration/script hashes, unchanged model hashes/version counters, EOS counts and cache lengths. No model/GPU was loaded; peak audit memory was about 489 MiB. Experiment artifacts were not modified.

## Implementation

The clean baseline performs normal cached full-prefix prefill and sequential decoding. The full-donor first-token cache is copied before clean decoding mutates the original cache. Each replacement constructs a fresh suffix-prefill cache, then clones it for independent free-running and forced-clean-history decoding. Generated tokens are computed normally after the initial first-token cache substitution; no source or normalization hook remains installed by this experiment.

The cache positions and off-by-one accounting are correct: prefill holds 128 tokens, emitting 64 predictions requires 63 further token forwards, and the final cache length is 191. Free and matched-history paths have identical predictions and margins through their first divergence from clean, inclusive; all 96 cases pass this independent saved-tensor check.

KL has the intended clean-to-condition direction and is measured under the clean generated history. Reference NLL targets the clean greedy next token. It is not ground-truth corpus NLL or a quality score. The stored KL/reference-NLL aggregates replay exactly, but underlying logits are discarded; distribution-level recomputation of those per-step quantities is unavailable. Their formulas were reviewed statically.

## Results

Across 16 prompts (four per corpus), each with exactly 64 emitted tokens:

| Initial cache | Identical full sequences | Free token agreement | Matched-history agreement | Mean matched-history KL |
|---|---:|---:|---:|---:|
| Exact full donor | 15/16 | 94.3359% | 99.9023% | 0.00000532 |
| Exact independent single token | 16/16 | 100% | 100% | 0.00000502 |
| Wiki mean, all layers | 7/16 | 64.4531% | 98.4375% | 0.00139505 |
| Exact layers 0–1, Wiki mean layers 2–15 | 13/16 | 91.5039% | 99.5117% | 0.00015476 |
| Static rank-two residual plus axis, layers 2–15 | 8/16 | 76.5625% | 98.9258% | 0.00145362 |
| Zero cache | 1/16 | 34.3750% | 91.3086% | 0.16998821 |

The exact full-donor control diverges on Plato window zero at generated step six (zero-based). The clean FP16 top-two logit margin is exactly zero at that step; the condition's margin is 0.0078125. This one changed decision leads to a different free-running continuation, while only one matched-history decision differs across all 1,024 predictions. The result is consistent with amplification of tiny numerical differences and precludes claiming exact free-running equivalence of that control. The independent single-token cache's 16/16 match here does not establish greater general fidelity than the full donor.

Frozen means and static rank-two templates produce small matched-history distribution changes but can produce substantially different free trajectories. Thus previous near-baseline teacher-forced loss does not imply preservation of generated text. Conversely, different generated text does not itself establish worse quality. The mean-late condition preserves 13 sequences, but this small fixed panel cannot establish a general exact-generation guarantee or a universal ordering of approximations.

The generator intentionally emits 64 tokens even after EOS and reports EOS occurrences. No sampling, repetition penalties, alternative seeds or quality evaluation are present. Prompts reuse the extended-context panel; no new fitting or selection occurs. Prompt-level bootstrap intervals are descriptive for these fixed four corpora, with only four prompts each; pooled intervals resample the mixture of 16 prompts without enforcing fixed corpus counts.

All 30 summary rows and their prompt-bootstrap intervals have now been independently replayed against the saved per-step metrics. The final status is recorded in `generation_audit.json`. No implementation or aggregate-numerical discrepancy was found; the unavailable logits remain the distribution-level audit limit described above.
