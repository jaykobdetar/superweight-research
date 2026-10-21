# Native GPT-2 prefix numerical diagnostic

Prospective diagnostic, 21 October 2026. The frozen gate-subspace study failed
its native-C shared-prefix hidden-state guard: maximum absolute error
0.0022735595703125 exceeded 0.001. Its loss maximum was0.0001621246337890625
and scored argmax agreement was1. The failed study remains failed. This
separate experiment does not relax that protocol, refit maps, select new
passages, or evaluate gate replacements.

Use only the exact failed natural-start512-token input, its unchanged first128
tokens, the same pinned GPT-2 checkpoint, FP32, eval mode and no cache. No
weights, query biases, source values or residual writes change. Run one local
GPU model sequentially. Root executes only after independent implementation
review. No extra forwards are permitted as an automatic fallback.

The total is18 forwards, in a fixed order:

1. Default SDPA, hook-free128 and512 references (2 forwards).
2. Default SDPA, then forced SDPA MATH:128,128 repeat,512,512 repeat,
   512 with suffix rotated by one token,512 with suffix replaced by the saved
   BOS/EOS ID, and that constant-suffix512 input with attention-mask ones at
   positions0..127 and zeros at128..511 (7 per backend,14 forwards).
3. Default SDPA128 and512 using the exact frozen gate-study C hook classes
   (`LayerWrites` with no lesions and `GateSubspace` with native mask, no V
   zeroing and intact Q bias), in addition to the observer (2 forwards).

Every512 suffix modification preserves token IDs0..127 and absolute position
IDs. Rotated and constant suffixes retain the ordinary native causal-mask
path and tensor shape. Padded512 deliberately changes the attention-mask
representation and may alter backend selection; it is a separate mask/shape
diagnostic, not a pure suffix comparison. All cases request hidden states
through the same model API. Hook-free references have no registered hooks.
Observer hooks copy tensors to CPU and return None. Their identity is checked
against the hook-free references; the original study-hook path receives its
own identity comparison. None of these identities is silently assumed.

Record actual Python/torch/transformers/CUDA versions, GPU identity, TF32
flags, float32 matmul precision, determinism and enabled SDPA flags. Preserve
these settings; only the MATH backend context changes, and it must restore
them afterward. CPU profiler event names record the observed SDPA operators
for every forward. Enabled flags alone are not evidence of the selected
backend. If profiler names are unavailable, record that explicitly without
guessing or adding inference. Timings include observation/profiling and are
not speed benchmarks.

Save each case separately: input hash and private input tokens, first128
logits and final-normalized hidden states, per-token losses/predictions, and
all12 layers' input/output residual states, attention-layernorm output,
combined Q/K/V, actual SDPA context before `c_proj`, attention output and MLP
output. Save effective prefix masks and check explicit masks expose no future
keys to the first128 queries. The source position is included. No gate
calibration or prior research files are rewritten.

Compare25 fixed pairs: within each backend short and long repeats,128 versus
512, original512 versus each unmasked suffix alteration,128 versus padded512,
and constant512 versus padded512 (14); corresponding backend cases (7);
hook-free versus observer128/512 (2); observer versus frozen-study hooks
128/512 (2). All hidden/logit comparisons report absolute maximum, RMS error,
reference RMS, relative L2, maximum-error location and values, its relative
error and number of entries exceeding the original0.001 threshold. This
threshold is descriptive here; no drift result is discarded for exceeding it.
Finite tensors, input/hash/parameter integrity and mask safety remain checks.

The scored shared queries1..126 predict targets2..127 (126 targets), matching
the failed study. Compute FP32 chunk32 losses identically to that study, and
report their differences, FP64 KL(reference||trial) and argmax agreement.
Source-query0 and query127 logits/hidden states receive separate comparison
fields; scored top-two-logit margins and any argmax mismatch IDs are retained.
Query127 logits and all128 hidden states are retained and separately compared;
the diagnostic does not score a nonexistent target128 for the short input.

For every pair with two observed layer traces, independently reconstruct
causal prefix attention scores and probabilities from saved Q/K/V in FP64
and compare them (276 layer-pair records). For every observed case/layer,
compare actual SDPA context with an FP64 prefix-only causal reconstruction
(192 records). These localize numerical divergence; they do not identify a
unique arithmetic operation as causal without further controlled isolation.

Compare new128/512 default, math, hook-free and study-hook states with the
saved failed-C states and metrics (8 alignment records). The correct saved
native hidden tensors are `failure_state.prefix_pairs.C`; its separate
`current_hidden` is P512 because that forward was computed before C recording
failed. Never use it as the native reference.

Interpretation: repeated within-shape stability plus fixed-shape suffix
invariance supports numerical shape/backend sensitivity within these tested
inputs; it does not prove universal absence of leakage. A math-backend change
may or may not reduce length drift, and it also changes arithmetic kernels.
Unchanged suffix comparisons alone cannot attribute length drift to SDPA:
linear/MLP/normalization operations also vary with tensor shape. Inspect the
first trace divergence, native-model scale and relative errors alongside
absolute error. Native magnitude and tolerance failure alone imply neither
a meaningful causal suffix effect nor harmlessness in all later interventions.

Preserve raw outputs, unsuccessful hypotheses, fixed source hashes, observed
operator names and resource usage. Failure saves partial traces and completed
case files with a failed manifest. This is one-input numerical diagnosis, with
no corpus generalization or change to the earlier failed study's status.
