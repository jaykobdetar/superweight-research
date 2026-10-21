# Query mass × conditional mixture: independent implementation review

**Passed for the frozen 520-forward protocol.** No remaining implementation blocker was found. This is a prospective hook/arithmetic/failure review, not a numerical endorsement of results from the pretrained checkpoint.

Reviewed hashes:

| File | SHA256 |
|---|---|
| `gpt2_query_mixture.py` | `0ee05a000051e34636211f10b04b290b78f0c475d650f558e503ba35e788b8ed` |
| `GPT2_QUERY_MIXTURE.md` | `0e3cd228104e1473deeb4c52e52c1b379d9fde586c5214713e3234b60ab47028` |
| `analyze_query_mixture.py` | `c5d23460628ae3d2e606d7f45a46965909775e0e20e7c09411474486816a0c8e` |

All inherited dependency hashes are in [CPU test evidence](gpt2_query_mixture_cpu_test.json). The test script (`gpt2_query_mixture_cpu_test.py`, private raw bundle) uses a random tiny-vocabulary GPT-2 configuration on CPU, with the native12-head/64-dimensional layout,12 layers, short lengths9/13 and narrow MLPs. Nonzero learned Q biases and output-projection biases were assigned before the immutable-parameter reference was taken. No pretrained model or GPU was used. Peak RSS was **910,760 KiB**, about0.87GiB.

## Design and implementation agreement

The context hook captures actual native SDPA output before `c_proj`, and the Q/K/V hook zeroes source0 V while subtracting only learned Q bias at positions1+. Immediate Q/K/V locality and whole-source trajectory identities were tested independently. Conditional mass is the direct sum of permitted non-source probabilities, not `1-p0`. The stable mixture is calculated by softmax restricted to the allowed non-source positions. Query0 is excluded from every division and every context replacement.

Main patch arithmetic is exactly `c_D*(a_G/a_D)` at each head/query. The diagonal branch copies `c_D` directly; off-diagonal donors come from unpatched P/R trajectories. Head concatenation into the projection input is correct, the actual context is captured before replacement, and donor caches are not mutated. The current-context control copies the actual tensor without rescaling. Learned output-projection bias and residual/MLP execution remain active.

The runner verifies strictly positive finite masses, finite contexts/mixtures/gains/errors, unchanged query0 context, and all declared maximum errors at `1e-4`. Diagonal copying does not bypass normalized-message or reconstruction diagnostics. Every full-output closure also requires the SHA256 of the complete final normalized hidden state, including all positions/features, alongside per-token loss/KL/prediction equality. This is stronger than relying on the metrics alone. The synthetic test compares actual full hidden tensors and full logits directly.

The final protocol writes the factorial indices as **mass first, mixture second**. This differs from the notation in the earlier design-review table, but the runner names (`G` for mass, `D` for mixture), protocol and analyzer agree. Specifically, `mass_at_DP` is `Y_RP-Y_PP`, `mass_at_DR` is `Y_RR-Y_PR`, `mixture_at_GP` is `Y_PR-Y_PP`, `mixture_at_GR` is `Y_RR-Y_RP`, and the interaction is `Y_RR-Y_RP-Y_PR+Y_PP`. Recipient P is the analysis copy; recipient R is an exact closure, not an additional independent sample.

The final condition list includes one P native-mask reference rather than both P/R native-mask references suggested in the preliminary review. This is adequate here: both donors and all factorial arms share the same explicit-mask dispatch, and the shared mask implementation was independently reviewed/tested previously. The P reference reports native-versus-explicit drift. The source-blocked P/R controls are comparison arms, not claimed closures of a gate-one donor patch.

## Independent synthetic results

Across **72 forwards**, spanning two sequence lengths and natural/BOS starts:

- **576** independent immediate projection-locality checks passed, along with576 exact supplied-context arithmetic/layout checks and576 FP64 reconstructions from saved actual post-hook Q/K/V/masks.
- **480** source Q/K/V trajectory equalities passed across zero-V arms. Query0 contexts were always unchanged by the context patch.
- **16 donor-diagonal**, **16 cross-recipient**, **8 current-context**, and **8 native repeat/identity** closures were exact for both full logits and complete final hidden tensors. Both mixed donor pairs were included in cross-recipient checks. Learned-bias removal changed outputs in the nonzero-bias fixture, so these are not vacuous zero-bias tests.
- Donor tensors and all model parameters/version counters remained unchanged. All hooks were removed after the test.
- Maximum native SDPA/manual context discrepancy was `7.1525574e-7`; maximum independent FP64 context discrepancy was `5.1290624e-7`; maximum mixed-context stable-reference discrepancy was `7.1525574e-7`. Explicit/native-mask full-logit drift was zero on these synthetic inputs.

These small CPU tests exercise the implementation but cannot guarantee numerical behavior for length512 on the pretrained GPU checkpoint. The stronger runtime guards must remain active, and any failure must stop that batch without adjusting its frozen tolerance.

## Failure checks and fixes before execution

Two issues were reported during the static review and corrected before the frozen CPU test:

1. The inherited context diagnostic removes its cached reconstruction inputs. A subsequent mixture-normalization failure could therefore have lost the Q/K/V/mask needed for diagnosis, and a failure in R before writing its input donor file could have lost the completed P donor. The final runner saves the immediate raw reconstruction inputs in `failure_mixture`, and the outer handler saves completed input donors and current snapshots as well as current partial donors.
2. A Python `max` over diagnostics can overlook a later NaN. The final implementation explicitly checks finite donor contexts/stable mixtures and every error/gain before acceptance.

**Ten negative fixtures passed:** zero, negative and NaN current mass; excessive normalized-message error; excessive inherited context reconstruction error; NaN donor stable mixture; NaN donor context; zero donor mass; NaN mass donor; and excessive mixed-context reconstruction error. Each was rejected rather than silently clipped or accepted. Mixture failures retained raw Q/K/V/mask and actual context. A separately executed copy of the outer `BaseException` handler retained completed donors/current snapshots/current donors and failure evidence, wrote partial records, marked the manifest failed, and removed a preexisting `completed_utc`.

## Fresh inputs, records and audit scope

Using the matching local tokenizer and private input configuration on CPU, the reviewer independently regenerated **16 length512 windows** at the declared eighth-panel fractions. They exclude all **28 prior used spans per corpus**. The tokenizer warns when initially encoding8192-character extraction passages beyond its nominal1024 positions; these are truncated to512 tokens before any model forward. No long sequence was fed to the model during the review.

The main list has16 distinct conditions per input;32 inputs plus8 native repeat/identity anchors give **520 forwards/records**. Static check accounting agrees with the analyzer:480 source-identity records,64 current closures,128 donor diagonals,128 recipient closures,32 mask references and8 anchors = **840 checks**. Fifteen zero-V conditions ×32 inputs ×8 layers = **3,840 mixture checks**. The analyzer's conditional-geometry summaries exclude the final unscored query, retaining510 matching scored queries.

Donors are sharded by input, permitting sequential numerical review below3GiB. They preserve actual contexts, masses and stable mixtures for both donors at all eight layers. The fixed raw snapshot subset covers both donors and all8 factorial arms at L4/L11, including actual and delivered contexts. Model/source/protocol/condition identities and immutable-parameter checks are recorded. Full-array numerical aggregation and actual completed-run closure verification remain future work; they were not inferred from synthetic success.

The scientific scope remains exchanging mass and conditional-mixture properties of complete donor trajectories under literal zero source V. It does not identify a unique direct Q-bias pathway, unit direction, additive mediated fraction, native trajectory restoration, universal mechanism or deployable computational replacement. No external action, credential change or GitHub retry occurred.
