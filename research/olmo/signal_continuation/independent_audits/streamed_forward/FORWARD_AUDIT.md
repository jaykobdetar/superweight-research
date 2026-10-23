# Streamed OLMo forward: architecture and numerical audit

Audit date: 23 October 2026, 05:36 UTC.

Verdict: the reviewed StreamedOlmo FP32 implementation is mathematically consistent with the official transformers v4.40.2 OLMo forward for this pinned model, single unpadded sequences, positions starting at zero, no cache, and evaluation/dropout zero. No architectural defect was found. This is source review plus small synthetic tests, not a native PyTorch comparison or a full pretrained-model execution.

Pinned model: allenai/OLMo-1B-0724-hf, revision d7cbab742d80589e714b1a2d7f838dcd21cbe143.

Reviewed implementation: /workspace/shared/olmo_computational_role/downstream/streamed_olmo.py
SHA256: 6cc4a613a411db7e011ba86f5b284985646b24e53c37c43abe52299949342f78.
A frozen copy is included here as reviewed_streamed_olmo.py. Later edits require checking the diff before applying this verdict to them.

## Source contract

The official reference is Apache-2.0 licensed Hugging Face transformers source:
https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/olmo/modeling_olmo.py
https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/olmo/configuration_olmo.py

Downloaded reference files are saved here. modeling_olmo SHA256 is 22e31514232522622f20eb898c0537440a931896c691b05e2b917b1bf53bd519; configuration_olmo SHA256 is 45c981d249c608ac454050c7e46876ca9bb8e466fbca2a2f08f48bc3e2be107e.

Line references below refer to the downloaded raw modeling_olmo_v4.40.2.py, not rendered web line numbering.

### Normalization (lines 75–86)

For each token vector x of width D=2048, calculate c=x-mean(x), then c/sqrt(mean(c*c)+1e-5). The variance uses divisor D, not D-1. There are no trainable gamma/beta parameters, no RMSNorm substitution, and no normalization over positions or batch. The official operation computes in FP32 and casts back to the incoming dtype. StreamedOlmo.ln and the recovered prefix implement this mathematical operation in FP32.

Population-variance semantics are documented by PyTorch: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html . The documented estimator has correction=0. NumPy's reduction order may differ from a native LayerNorm kernel, so source equivalence is not a bitwise guarantee.

### Embedding, dimensions and heads (lines 893–918, 960–980)

The input is the unscaled embedding lookup. No sqrt(D) embedding multiplier or additional learned position embedding is applied. The pinned configuration has 16 layers, 16 attention heads, 16 KV heads, head width128, MLP width8192, vocabulary50304, no attention biases, no dropout, and untied input/output embeddings. RoPE base is10000 with no scaling. Noncached positions are0…N-1.

### QKV, clipping and RoPE (lines 93–142, 177–209, 318–335)

Q, K and V are separate linear projections of normalized input. Clip all three coordinatewise to[-8,8] immediately after their projections. Then reshape each2048-vector into16 consecutive128-coordinate heads, retaining token/head order.

RoPE applies to Q and K only. Frequency i is10000^(-2i/128), i=0…63. Duplicate the64 frequencies into two contiguous halves. The rotation pairs coordinate i with i+64: (a,b) becomes(a cos−b sin,b cos+a sin). This is not adjacent/even-odd pairing. Trigonometric computation is FP32 in the official implementation; the coefficients are cast to input dtype afterward.

Do not clip rotated Q/K again: legitimate rotated coordinates can exceed8. The synthetic check produced magnitude11.1198 despite pre-RoPE clipping. V is clipped but not rotated.

### Attention and masks (lines 342–362; SDPA alternative575–662; mask1039–1113)

Attention uses QK^T/sqrt(128), additive causal/padding masks, softmax along keys, then multiplication by V. Official eager softmax computes in FP32 and casts back to query dtype. Heads are transposed back to token-major order and concatenated before the output projection. There is no additional normalization of Q/K.

For the current unpadded/noncached scope, key j is allowed precisely when j<=query i, unless a specific intervention removes it. StreamedOlmo applies this mask correctly. Its source-removal intervention suppresses key s only for destinations s+1 onward, preserving the self edge and preventing fully masked rows. This is the requested intervention, not an architectural default.

Official eager masks use dtype minimum, whereas StreamedOlmo uses negative infinity. With finite logits and at least one allowed key, both give zero blocked probabilities at the working precision here. This equivalence must not be generalized to fully masked rows or padding cases without explicit testing. HF SDPA/Flash kernels can have different rounding while computing the same ideal attention.

If query/key chunks or cache are added later, preserve global position indices, global causal comparisons, and separate sequence/world identities. Do not restart RoPE or the causal triangle at chunk boundaries. This implementation currently uses neither cache nor query chunks.

### Residual order and MLP (lines 212–225 and674–737)

Each decoder layer implements two sequential residual blocks:
1. z = x + Attention(LN(x)).
2. x_next = z + Down(SiLU(Gate(LN(z))) * Up(LN(z))).

The second normalization uses post-attention z, not the original x. SiLU applies only to gate, then multiplies up, then down projects. All three MLP projections have no biases. StreamedOlmo and the recovered prefix agree with this ordering.

### Final normalization, logits and loss (lines1018,1120–1125,1210–1225)

After all16 blocks, apply the same non-affine LN once, then the untied LM-head linear map. There is no logit scaling or bias. Official v4.40.2 casts logits to FP32 AFTER the head matmul; this distinction matters when using half weights, even though it is a no-op for the pinned FP32 run.

Every logsumexp denominator must include all50304 output rows, including padded/unusual token IDs. StreamedOlmo's complete-vocabulary loop does so. It pairs hidden position t with input token t+1 and excludes the final hidden row, matching causal next-token scoring. A later scored-span selection must select the corresponding loss positions while retaining the entire prefix context. Perplexity is exp(mean scored NLL), not mean per-token perplexity.

The implementation's float64 running logsumexp is mathematically correct and improves aggregation accuracy. It is not bitwise equivalent to HF's default FP32 CrossEntropyLoss. It should be described as loss computed from FP32 logits with higher-precision aggregation.

## Recovered-prefix comparison and precision limits

The recovered phase2/checks/prefix_model.py is a NumPy port labeled transformers4.57.6. Its FP32 LN, clip, RoPE, attention, residual and SwiGLU equations match the relevant v4.40.2 branches for this configuration. It stops at selected L1 neuron outputs rather than a full model. No extra mathematical transformation is needed to extend those blocks through layers2–15 and the final head.

Its half=True mode is not an exact FP16 oracle: it retains FP32 intermediates where eager half execution can round RoPE coefficients/products, scores, and softmax probabilities separately. StreamedOlmo improves the RoPE half rounding but still intentionally approximates half attention. Neither establishes CUDA/SDPA bit equivalence. Scientific conclusions from a pinned FP32 run should not be presented as exact reproduction of previously published FP16 losses.

PyTorch and transformers are not installed in this execution environment; no native-framework run was performed or claimed. No pretrained weights were downloaded or read by this auditor. Actual full-model baseline/cached-prefix validation remains the execution worker's responsibility.

## Tests performed

1. check_math_contract.py uses tiny random matrices (width32, heads4, length11) and independent block/unchunked formulas. Population-LN analytic check is exact; RoPE norm error is5.68e-14; streamed attention differs by1.42e-14; chunked full-vocabulary NLL differs by1.78e-15. Blocked/future probabilities are exactly zero. Results: math_contract_checks.json.
2. check_concrete_stream.py imports the reviewed StreamedOlmo class without calling its weight-loading constructor. It tests actual attention and logits_summary against independently written per-head/per-query and dense-vocabulary references, using only synthetic QKV and a tiny synthetic LM-head file. Attention max error is1.86e-5, source-probability error7.83e-8, streamed NLL error4.68e-6, and argmax is identical. Source0/source4 masked later-key probabilities are exactly zero; source0 self-probability is exactly1. Results: concrete_stream_checks.json.

These tests validate code paths and mathematical identities, not the learned model or its downstream findings.

## Minor recommended hardening

Before freezing the execution version, assert hidden_act=silu, rope_theta=10000, rope_scaling=null, intermediate_size=8192, vocab_size=50304, attention_bias=false, and attention_dropout=0 in addition to the existing shape/config assertions. Add finite-value assertions for hidden states, logits and loss, so unexpected overflow cannot silently produce invalid summaries. These were sent to the execution worker; no parent files were edited.

Update05:38 UTC: config assertions for activation, rope base, MLP/vocabulary width, biases and dropout, plus finite hidden/logit guards were added by the execution worker. The concrete synthetic tests were rerun successfully on this revision. No numerical forward equations changed. rope_scaling=null remains verified in the pinned config, though not asserted in this version.

## Native-prefix amendment and runner review, 05:43 UTC

The final reviewed model hash is6cc4a613a411db7e011ba86f5b284985646b24e53c37c43abe52299949342f78. The pre-amendment snapshot is preserved separately. The amendment adds mask_query_start and per-layer prefix-state hashes; default masks are unchanged. For source s, the first suppressed query is max(s+1,mask_query_start). Setting mask_query_start=p+1 for all compared sources preserves positions0..p and therefore their K/V through every layer. Later query states still evolve endogenously. This distinguishes native-prefix access from the global source0 cut, which can alter how later source p is constructed.

Independent cheap synthetic projections exercised the actual16-block control flow: every post-cut prefix hash and every pre-cut NLL matched native exactly, while post-cut NLL changed; global source0 masks altered the later prefix, as expected. Reused maskp also preserved both invariants. No pretrained weights were used for these tests. See check_post_cut.py and post_cut_checks.json. The execution worker separately supplied a four-real-layer invariant check; that is not an independent full-model validation by this auditor.

The first affected prediction query is p+1, which predicts target p+2. The runner's target-span mask and pre-cut NLL[:p+1] comparisons are correct. The runner checks both post_ conditions and reused maskp against native prefix hashes and null-loss invariants.

A resume-path bug was caught before scientific execution: prefix_hashes were read from NPZ although stored only in the JSON receipt. The worker corrected this and added NPZ checksum validation. Current run_factorial.py SHA256 is8d6a6aa0fdeea87c745279ac5f7ccdb3670a8a46802ea4d162bf64e371726204. The remaining reviewed resume and scoring code is consistent with the frozen protocol. The protocol/amendment is an execution-worker artifact; this audit does not authorize broader experiments.
