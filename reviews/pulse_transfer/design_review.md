# Independent review of transfer tranche1

Reviewed21October2026 before transfer evaluation. CPU-only protocol inspection; no model or GPU load. The denied repository-publication action was not retried or bypassed.

**Assessment:** the proposed tranche is a sound bounded test of selected causal signatures in two locally available models. It deliberately selects by development contribution rather than held-out damage, preserves negative results, and distinguishes model size, instruction tuning, normalization, positional encoding and precision differences. No architectural transfer conclusion follows merely from observing a similarly large activation.

## Material clarifications before evaluation

1. **Freeze all selection details.** Define the intermediate as the post-activation input to the down projection: GELU output for GPT-2 and SiLU(gate)×up for Llama. Specify zero-based indices, deterministic argmax ties, rejection of nonfinite scores, and an on-disk selection/control checksum written before evaluation. The stated score is a legitimate large-contribution heuristic, not a general search for the maximally causal weight. The exclusion of the final decoder layer is an explicit restriction of that search.

2. **Specify random controls exactly.** State whether each seed samples a new column with its largest-weight row or an independent row/column pair, and whether sampling excludes the selected candidate and duplicate controls. Prefer distinct control columns, with the selected-row or per-column-max-row convention fixed in advance. Do not call these energy-matched controls unless their amplitude/norm is actually matched. This removes avoidable ambiguity from later selected-versus-control interpretation.

3. **Preserve native projection numerics.** The copied-weight oracle must use the architecture's actual output operation, including its bias. GPT-2 Conv1D storage transposition is for analysis; its native addmm/reshape output path should remain the intervention path. A generic mathematically equivalent F.linear call can round differently. Fail the numerical control before interpreting small causal effects, and retain repeat-native results to distinguish kernel drift from patch error. The existing full-sequence operation and per-token closure requirement are appropriate.

4. **Separate a triggered follow-up from confirmation.** If these evaluation outcomes motivate a later head screen/KV factorial, the present history panel is discovery evidence for that decision. Reserve fresh evaluation passages for confirming the frozen later head choice; reusing this panel remains a useful exploratory mechanistic follow-up but should not be called a fully held-out confirmation. Define any quantitative “substantial, reproducible suppression” threshold before results if it will determine which model advances automatically. Alternatively keep the next-step choice explicitly exploratory and do not present the threshold as preregistered afterward.

5. **Make identity-dependent scoring explicit.** Source0 tests score targets2..end, as specified. For the U/R/q-double-newline localization panel, state a common suffix score after the changed q token—for example predictors129..end−2, hence targets130..end−1—if any cross-input loss comparison will be made. Within-world intervention effects may use their declared span, but raw U/R losses over targets containing the manipulated p/q IDs are not matched causal effects. This panel currently measures localization descriptively; do not infer performance rescue from its native loss differences.

## Controls already well chosen

- Eight development passages and16 evaluation passages are separated by source/corpus split; candidate selection has no held-out loss criterion. Save all scores, not only the winner, so a negative result cannot quietly become a different candidate search.
- Natural/The/BOS-or-EOS start substitutions preserve the rest of the token IDs. The explicit no-special-token/no-chat-template convention is appropriate for a mechanistic contrast, provided the instruction-tuned model is not represented as being evaluated in its recommended chat-use format.
- Exact literal-token eligibility is preferable to guessing comparable token IDs. Single and double newlines can differ by tokenizer. Removing specified IDs does not prove removal of every token that may decode to a newline-containing string; keep the conclusion tied to the chosen entries.
- GQA does not affect the initial row/column deletion definition. It matters for any future head K/V manipulation, where multiple query heads can share one KV head. Preserve the warning and report the query-to-KV mapping before such experiments.
- Learned LayerNorm and RMSNorm summaries are not interchangeable geometric objects. Record whether each summary is before or after learned scale/bias, and define target-coordinate fraction as squared-coordinate share or another explicit formula. Do not transplant OLMo's centered-axis interpretation without accounting for the model's normalization.
- BF16 Llama versus FP32 GPT-2 is disclosed as a confound. The native-copy and repeated-forward controls are essential; differences cannot be uniquely attributed to architecture.

## Failure and resource criteria

The intended single-model, sequential execution with no download, quantization, offload or silent precision change is appropriately bounded. Save actual peak allocation/reservation and process RSS. Before a first forward, enforce local checkpoint completeness and the exact intended dtype/backend; on OOM, stop that model and preserve the failure record rather than modifying the protocol. A missing single-token literal makes only its affected panel ineligible, not the entire transfer study.

Other reasons to stop the affected comparison are nonfinite inputs/outputs/scores, parameter-hash or version changes, native-copy closure failure, repeated-native instability beyond a declared bound, and fewer than512 tokens in any required passage. Verify positive valid character starts and report duplicate/overlapping windows rather than silently increasing sample count. Short passages or missing literal entries should be recorded before evaluation effects are inspected.

The model hash, full source/token hashes, private token tensors, source/output captures and per-token metrics provide adequate material for subsequent independent CPU audit. Runtime implementation still needs static review before launch, especially GPT-2 Conv1D layout and the selected token output replacement. Four windows per corpus permit descriptive paired uncertainty, not a broad architecture ranking. Negative results concern the frozen selected candidate and tested intervention family, not every possible newline mechanism.
