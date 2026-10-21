# Independent implementation review: GPT-2 source K/V after layer-3 regrowth

**Verdict: passed for execution.** No pretrained-model numerical findings are established by this review. Review scope was the frozen protocol, `gpt2_regrowth_kv.py`, inherited `LayerWrites`/`Writes`/`Probe` hooks, the inherited `KV.attn` mask helper, input preparation and the declared analyzer contrasts. No main experiment code was edited by the reviewer. Two failure-preservation issues were reported to the main worker and fixed before this verdict.

The final CPU test is `gpt2_regrowth_kv_cpu_test.py` (`gpt2_regrowth_kv_cpu_test.py`, private raw bundle), with results in [`gpt2_regrowth_kv_cpu_test.json`](gpt2_regrowth_kv_cpu_test.json). Its final peak RSS was 802,640 KiB (0.765 GiB). It used two small randomly initialized 12-layer GPT-2 models, no pretrained weights and no GPU. The real local tokenizer and source files were used only for input preflight. The test forces offline operation and one CPU thread.

## Frozen implementation

The final synthetic test asserts that all reviewed source files retain their starting hashes through the end of the test. Complete hashes are saved in the JSON. The principal identifiers are:

- `gpt2_regrowth_kv.py`: `b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8`
- `GPT2_REGROWTH_KV.md`: `322098a58710bb61db619271100caca186e3acf7bf5fd1c4ed338b3677ff5750`
- Inherited `llama_source_kv.py` mask helper: `84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2`

The runner declares 19 conditions per input/start: three worlds, eight recipient/key/value factorial conditions, four zero-value conditions and four mask conditions. Sixteen passages with two starts give 608 records, with eight native/identity anchors bringing the total to **616**. The expected check count is **392**: 32 × (one first boundary + four diagonal + four recipient + two mask-reference drift + one blocked-recipient check), plus eight anchors. Duplicate recipient and diagonal records are closure controls, not additional independent samples.

## Hook and mask findings

`E` zeros only residual coordinates `[447,138]` at source position 0 after layer 2. `B` performs the same erasure and additionally zeros those two coordinates of the layer-3 source MLP output before addition. Their first difference is therefore after layer 3's MLP. The first possible source K/V transmission boundary is layer 4. The code captures the layer-3 block output after the writer hook and the layer-4 query projection; it tests later-position equality between E and B and source layer-3 attention equality.

GPT-2 `c_attn` has combined `[Q,K,V]` output. The patch indexes source position 0 and exactly the K and/or V slice in layers 4–11. Queries and every later-position projection are asserted unchanged; a non-patched K/V slice is also asserted unchanged. Donors are stored after projection patching. Full E and B source K/V states are collected separately for the two donor worlds. Every factorial recipient uses the intended independently chosen K and V world.

The inherited mask helper adds an explicit triangular causal mask. In the blocked condition it additionally excludes source key 0 only for queries 1 onward; query 0 retains its causal self attention. Existing bool or additive masks are preserved. Zero source values retains the source key in the attention normalization, whereas the blocked mask removes that source option for later queries. Their difference cannot be interpreted as a pure value-content or additive mediation effect.

The first layer-4 query is not modified by a source K/V patch or by the attention mask, because it is computed before either change can feed back into the residual stream. Source queries may differ between E and B; only later-position E/B query equality is required. The runner also checks first queries against each recipient's own unpatched world.

## Executed CPU checks

The final test performed **84 forwards** across two independently seeded tiny GPT-2 models and natural/BOS input variants. All tests passed:

- 16 full donor diagonal closures and 16 cross-recipient closures, exact on scored per-token loss, KL and argmax arrays.
- Four blocked-source E/B closures, exact on those same arrays.
- Eight native-repeat/identity closures.
- Four first-boundary checks: complete layer-2 outputs identical; layer-3 attention outputs identical at all positions; layer-3 residuals and layer-4 queries identical at later positions; the source layer-3 residual differs only at the two deliberately blocked coordinates.
- Eight standalone combined-projection tests, including real GPT-2 projection width 768 and tiny width 24, covering no patch, K only, V only and K+V. Original projection tensors remain unmodified, donor copies are exact, and caches/reset behavior are correct.
- 972 explicit mask-entry checks across both widths, both causal/blocked modes, and absent/bool/additive prior masks. Future positions and caller-excluded entries remain blocked; query-0 self attention remains available.
- All parameters and parameter version counters remain unchanged in both tiny models. All installed hooks are removed on close.

The tiny CPU models had zero measured causal-reference loss drift. This is a synthetic implementation result and does **not** justify assuming zero mask-reference drift for the pretrained GPU run; its separate drift measurements remain required.

The actual fifth-panel preparation succeeded for all **16 fresh 512-token passages**, with 16 earlier used spans per corpus. Actual token-character spans are checked against all four earlier panels and one another, with source hashes and the matching tokenizer. Fractions remain `[0,.05,.9,.95]`; no passage was selected after inspecting experimental effects. These fractions can include front matter or code headers and should retain that characterization in the eventual report. The tokenizer's context-length warning is harmless for this preparation: it encodes an 8,192-character extraction and keeps 512 tokens, without passing the longer extraction to a model.

## Failure preservation and analyzer inspection

The initial implementation lost the in-memory failed closure evidence before the next item checkpoint. The main worker added an exception handler that writes partial raw arrays, traces, check records, donors, operation counts and the immediate writer/projection/query/boundary state. A subsequent review found that a final invariant failure could retain `completed_utc`; the main worker added its removal in the failure handler.

The final CPU test executes the actual exception-handler save statements against a small failing fixture, excluding only the terminal re-raise. It verifies the saved state, failed assertion type/message, partial counts, and absence of `completed_utc`. This covers the handler's logic, not catastrophic storage failures or process termination that cannot execute a handler. Model initialization failures before the experiment try block still lack this partial-forward evidence, because no experiment forward has run.

Static analyzer inspection confirms the declared interaction sign, `EE − EB − BE + BB`, and both simple key and value effects. It uses the E recipient for the unique factorial analysis. The raw blocked-minus-zero contrast is `blocked − zero`; the adjusted contrast is `(blocked − causal_reference) − (zero − native_world)`. It retains mask-reference drift separately. It also computes B-minus-E, uses the native C run for all condition loss ratios, and retains per-condition native KL and agreement.

No implementation blocker remains. The pretrained run still needs independent numerical review: exact condition/key coverage, 616 records, 392 checks, expected patch counts and donor equality, per-token reductions, full-vector locality/addition checks, actual source-file/hash consistency, all corpus/start results, and the signed factorial and adjusted contrasts. Closure success would validate the architectural intervention and transfer of the chosen source state; it would not alone establish a unique mechanism, additive mediation, linguistic value content, native restoration or a whole-network replacement.
