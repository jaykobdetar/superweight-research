# Transfer implementation review

2026-10-21, pre-evaluation static and synthetic CPU review of `run_transfer.py` against `PROTOCOL.md`. **No remaining launch blocker for the two specified models with eligible required literals.** No pretrained model or GPU was loaded. Publication was not retried.

Two findings were fixed before sign-off: development now uses512 tokens as declared, and each layer's candidate truncation applies the complete descending-score/ascending-row/ascending-column tie order before the global ordering. The earlier score-only truncation could have omitted the correct row-tied candidate.

## Selection and inputs

Eight natural-start development inputs supply the post-activation down-projection inputs. Source0 mean absolute activation is multiplied by each column's maximum absolute output weight. Weight-row ties use the first row, and the global ordering is now deterministic. All layer scores and maxima are checked finite and saved, with raw/JSON hashes frozen in the manifest before evaluation. Evaluation tokenization occurs before selection, but no evaluation model metric enters the criterion. The selected layer excludes the final decoder layer, as prescribed.

Three controls sample an independent uniform row then column with the stated NumPy seeds in the selected layer; a selected-pair collision aborts. These controls are not magnitude matched. The protocol permits distinct seeds rather than promising distinct sampled controls; any realized overlap should be disclosed in the result audit.

The same fixed character passages are tokenized independently per model. Book wrappers are consistently stripped, input files and processed text/passages are hashed, and private tokens are saved. Length failure occurs before model loading. The/The-history eligibility has one portability limitation: a nonsingle-token `The` aborts preparation entirely, rather than leaving the natural/special source panels available. A missing BOS and EOS similarly aborts later. This does not invalidate runs where the literals are eligible; an unsupported literal must be reported as an eligibility limitation, not a negative causal result.

## Projection definitions and localization

GPT-2 analysis uses the transpose of its Conv1D weight as output-by-intermediate. A selected row contribution is zeroed at native stored weight[column,row]; a complete output column is native stored weight[column,:]. Its counterfactual uses flattened native addmm with the original bias, then restores the output shape. Llama uses native F.linear with the copied [output,intermediate] matrix and original bias. Hooks act before subsequent residual/dropout operations; evaluation mode disables dropout.

Only the chosen source token output is replaced. Row deletion changes one coordinate, whole-column deletion changes the full source output, and original parameters remain untouched. The copied-matrix oracle recomputes with an unmodified matrix. Direct inspection of the installed Conv1D forward confirms the same addmm/bias path.

`hook_cpu_test.py` exercises20 synthetic cases: GPT-2 FP32 and Llama BF16, source positions0 and128, selected and alternate row/column coordinates, and copied-weight oracle. All patches equal the independently constructed expected output, all other token positions remain exact, all parameter tensors remain unchanged, and every oracle is exact. Peak test RSS735,348KiB, below the delegated3GiB bound. These tests validate hook tensor operations, not native GPU kernels; the runtime closure anchors remain required.

## Scoring and counts

Loss indices correspond to predictor positions. Source0 `score_index=1` evaluates target positions2 through the last input token. History `score_index=129` evaluates target positions130 through the last input token, giving126 targets at length256 and382 at length512. The U/R/R_double suffixes beyond q remain unchanged and no retokenization occurs. Each source intervention uses its own input's native logits for forward KL and argmax agreement. History records are native within each input, so their self-reference KL and agreement are trivially0 and1; meaningful U/R comparisons must use matched suffix loss/contribution contrasts rather than those self-reference fidelity fields.

Expected full eligible run:8 development forwards,880 source-panel forwards/records,96 history forwards/records, for984 forwards and976 records. The source panel contains96 input variants×9 conditions plus16 repeat/oracle checks at the eight corpus/length anchors. If newline-history literals are ineligible, the history panel is skipped and the expected total is888 forwards/880 records.

The selected source and q contributions remain diagnostics of the frozen primary candidate even during random-control interventions. Downstream normalization summaries likewise use the primary selected row. They should not be mislabeled as the random candidate's own contribution/coordinate.

## Resource and inference limits

One checkpoint is loaded offline per process, with explicit dtype/backend, no training, no cache, no fallback offload or silent quantization. Chunked logits processing bounds temporary log-softmax memory. Full parameter hashes and version counters are checked before/after; failure prevents a completed manifest. An OOM or closure failure propagates and preserves the incomplete run for diagnosis. The current checkpoint-copy/hash procedure can still consume material CPU memory; measured peaks must be reported rather than inferred from parameter count.

The history signature is descriptive and the next head-local experiment requires fresh confirmation inputs. GQA causes no ambiguity for this initial neuron-output intervention, but any later head K/V patch must map shared KV heads explicitly. BF16 versus FP32 and instruction tuning remain confounds. A negative result pertains to this development-selected candidate, lengths, inputs and contribution-removal family; it is not proof that an architecture lacks massive activations or newline-sensitive mechanisms.
