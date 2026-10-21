# Query-source implementation review

Status: **ready for the predeclared run; no remaining launch blocker**. Independent CPU-only static and synthetic review, 21 October 2026. This is implementation evidence, not an audit of measured pretrained-model outcomes.

Final approved runner SHA256: `c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5`. Protocol SHA256: `fabb87c17196403c439fdf22982216861245c73e4d2e3eb0d7d358f5bf748922`. Complete dependency hashes and measurements are in [gpt2_query_source_cpu_test.json](gpt2_query_source_cpu_test.json); reproducible tests are in gpt2_query_source_cpu_test.py (`gpt2_query_source_cpu_test.py`, private raw bundle). The synthetic test ran runner `59b9fec5acf98eed5ab8103b8024fa74f5dbca3e60f248ac6f405f29e01a7b04`. The final delta adds exactly one metadata line recording the saved bias-controls SHA256 and writing the manifest. Independently removing that line recovers the tested hash byte-for-byte; no hooks, conditions, or scoring changed. No inference rerun was necessary for this metadata-only change.

## Implementation and estimand

The native GPT-2 combined projection is split into Q/K/V at the correct widths. The intervention subtracts the learned Q bias, or the fixed per-head signed permutation, only at positions 1 onward. It leaves source Q, all current K, and non-source V unchanged at that hook. Literal zeroing of source V occurs before native attention. The intervention covers all twelve heads in layers 4–11; no evaluation-driven head selection occurs. There is no RoPE or GQA in this GPT-2 architecture. Learned/permuted bias controls use the frozen seed and exactly preserve per-head absolute-value multisets.

The inherited layer hooks implement C/E/B in their specified order: source-coordinate erasure after layer 2, followed in B by suppression of those source-coordinate MLP writes at layer 3. Query modifications start at layer 4. Hook locality does not imply that subsequent non-source representations or K/V projections stay fixed: those can change endogenously. Source Q/K/V equality across the zero-V query/mask arms is valid because source0 is causally isolated from later query changes and retains self-attention.

Explicit masks retain causality, preserve query0 self-attention, and block only source0 access for later queries when requested. Both boolean and additive existing-mask restrictions are retained. Native-mask zero-V references separately measure dispatch/numerical differences. Diagnostic probabilities use the actual post-intervention Q/K and effective mask. Full weighted-V reconstruction is compared against the actual native attention context entering c_proj. Source and non-source vectors are unnormalized contributions; zero source values imply an exactly zero source contribution even when source probability mass is high.

The 24 unique conditions comprise 3 native worlds, 3 native-mask zero-V references, and 18 zero-V query-by-mask arms. With 32 corpus/window/start inputs and 8 anchors this gives 776 forwards and records. Shared scoring begins at loss index1, targeting tokens2–511. Natural and BOS use the same suffix. Actual token-span preflight independently passed for sixteen fresh windows against all twenty-four prior used spans per corpus and within the new panel. Long-tokenization warnings refer to text tokenized for span discovery; only verified 512-token crops enter the experiment.

## Independent synthetic evidence

Tests used randomly initialized GPT-2 with the actual attention width (768), twelve heads of width64, twelve layers, short contexts, and a small MLP/vocabulary. Q biases were deliberately made nonzero before hashing so bias-removal controls were not vacuous. No pretrained model was loaded and no GPU was used.

- 52 synthetic forwards, covering C/E/B, natural/special-token starts, native and explicit masks, learned/random bias removal, literal zero-V, and repeat/identity controls.
- 416 exact combined-projection locality checks and 288 source Q/K/V equality checks.
- 72 independent FP64 reconstructions of saved Q/K/V/mask snapshots, including probability and unnormalized non-source norm comparisons.
- Four exact full-logit repeat/identity closures; four direct boolean/additive mask-merging fixtures.
- Maximum native CPU SDPA versus implementation diagnostic error: `5.960464477539062e-7`. Maximum independent FP64 versus actual-context discrepancy: `3.162926434097102e-7`. Native/explicit-mask maximum logit drift in this synthetic test was zero.
- Parameters and version counters unchanged; all review/experiment hooks removed after testing. Peak RSS: 892,016 KiB, below 3 GiB.

The actual 512-token pretrained GPU calculation must still meet the frozen `1e-4` reconstruction tolerance. Synthetic CPU agreement does not guarantee GPU agreement.

## Failure evidence and interpretation limits

The initial implementation discarded the complete offending reconstruction outside the frozen snapshot subset. This was reported before launch and corrected by the parent: Q/K/V/mask are retained until the context check, successful checks release them, and a failure preserves actual/expected context plus full reconstruction inputs, layer, error, and threshold. An injected mismatch stopped as required; the exception-handler fixture saved those diagnostics and a failed manifest without a completed timestamp. The threshold was not relaxed.

The learned-bias-by-source-blocking NLL interaction identifies effect modification by source access in these zero-V worlds. It does not isolate mediation through attention mass: query-bias subtraction changes logits against every attended key, and later non-source computations remain endogenous. One fixed random direction is a matched control, not a randomization significance test. Allocation correlations, near-native NLL, or small average effects must not be described as exact output preservation, universal source dispensability, a unique repair mechanism, or discovery of the previously documented GPT-2 query-bias circuit.

Remaining work is runtime assertion verification and independent numerical audit of the completed batch. No publication action was attempted.
