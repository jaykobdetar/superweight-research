# GPT-2 source-value geometry: implementation review

Independent CPU static/synthetic review, 21 October 2026. **Pass: no launch-blocking defect found.** The experiment code was not edited. No pretrained model or GPU was loaded. Numerical research conclusions remain subject to the completed-run audit.

Reviewed runner SHA256: `e9d9f44f1e6f4cb0c952912b6fbb5d545e9af23bfb16351eebe110c8b9b038c8`.
Reviewed protocol SHA256: `7324637a224c69de3af7317c4c71a126d3fde50cc8bdec31f5ba8ea260e3f117`.
All dependency hashes are preserved in `gpt2_value_geometry_cpu_test.json`; they were checked unchanged before/after the tests.

## Implementation findings

- `value_patch` uses complete native projected V, including projection bias, split into 12 contiguous 64-dimensional heads. Diagonal E/E and B/B vectors are copied exactly. Mixed cells use the intended positive per-head norm ratio. Donor source tensors remain unmodified. The four conditions share E keys and B recipients.
- Both E-recipient controls correctly replay their corresponding B-recipient cells. Only E/E is asserted to restore native E; B/B under E keys is correctly treated as a mixed world. C K/V identity is separately checked.
- Signed permutations/signs are drawn once with a local CPU generator at seed20261023, outside the input loop. Every input and both E/B control vectors share the same maps. They preserve absolute-value multisets exactly; norm reductions use an appropriate relative FP32 tolerance.
- Inherited `GPT2KV.combined` modifies only source0 K/V slices of combined QKV output in L4..11. It checks Q, all later-position projections and every unpatched slice against the pre-patch tensor, and captures post-patch donors. First-boundary queries remain unchanged within a recipient world. The prior `LayerWrites` hooks retain E's L2 erasure and B's L3 source-output-coordinate block.
- Hook order is correct: `LayerWrites` is constructed before the K/V boundary-capture hook, so the saved post-L3 boundary includes the source intervention. Per-forward state is reset. Full later-token equality follows from fixing all downstream source K/V, without requiring the source residual itself to become identical.
- All records use `Runner.rec`'s score index1: targets2..511, excluding predictor0. KL remains relative to native C, including when the loss contrast is conditional on E/B.
- Conditions are 16 per input plus eight natural-window0 anchors: 520 forwards/records. The native C pass is reused rather than rerun for its record, and three native passes also generate the donors. No hidden fitting/calibration forwards occur.
- The failure handler checkpoints current metrics, donors, operation/value checks, writer/KV state and a failed manifest without a completion timestamp. It re-raises the error and removes hooks in `finally`.

Expected successful-run bookkeeping for the later audit: 520 operation rows; 2,048 layer-level value-check rows (six factorial/recipient calls plus two randomized calls, eight layers, 32 inputs); 168 exact-output closure rows plus 32 first-boundary rows. These do not constitute independent statistical samples.

## Executed evidence

`gpt2_value_geometry_cpu_test.py` imports the actual transformation/hook implementations. It uses deterministic synthetic tensors and a randomly initialized short-context GPT-2 with the real 768-wide, 12×64 attention layout, 12 layers, a narrow MLP and a tiny vocabulary. It does not call `from_pretrained` for a model.

- 384 per-head magnitude/direction checks, including FP64 reference reconstruction, exact diagonal copies and direction alignment.
- 192 signed-permutation per-head checks; repeat generation is deterministic and independent of global RNG state. Zero-norm donors are rejected.
- 36 synthetic forwards across two inputs, all native/factorial/zero/random/identity arms; 14 exact later-logit closures and 288 instrumented QKV locality-layer checks. Original parameters and version counters are unchanged, and all hooks are removed.
- Actual locally installed tokenizer/input preflight independently yields 16 fresh 512-token windows, each disjoint from 20 prior used spans per corpus and the other fresh windows. The tokenizer warning concerns the longer extraction buffer; only the first512 tokens are used, and no long sequence was sent through a pretrained model.
- The actual failure-handler AST was executed against a small saved-state fixture and produced the required evidence files and failed manifest.

Peak reviewer RSS: **887,828 KiB**, below the 3GiB bound. Runtime checkpoint/device/dtype-specific closure remains a required part of the research run; synthetic passing does not replace it.

The three interpretation clarifications from design review are now in the protocol: joint layer/head norm profile, mixed-world B/B cell under E keys, and separate matched permutation controls rather than one random direction at two scales. No additional arm or implementation change is requested.
