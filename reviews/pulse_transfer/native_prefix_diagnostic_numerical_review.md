# Independent audit: native-prefix numerical diagnostic

**The separate18-forward native diagnostic completed and its saved numerical outputs pass independent audit. The original gate study remains failed.** This audit used saved shards only, with1129096KiB peak RSS; no pretrained model or GPU inference. Evidence is in `native_prefix_diagnostic_numerical_audit.py/.json`.

All18 case hashes, frozen sources, input transformations, masks, precision settings, recorded parameter identity and the failed-run alignment were verified. Independently recomputed25 complete pair reports,276 FP64 layer-score/probability comparisons,192 actual-context reconstructions, and8 alignments to failed native C. Across45139 checked fields, the largest discrepancy from saved summaries was1.42e−14. Independent CPU FP64 target losses differed from the saved GPU FP32 losses by at most8.37e−7. Maximum actual-context discrepancy from FP64 prefix-only reconstruction was5.81e−6.

## Findings

Both backends have exact repeated outputs at each length. Within512 shape, rotating or replacing the suffix preserves every captured prefix layer state, final hidden state, logit and scored loss exactly. The padded-mask comparison is also exact within each backend. Hook-free/observer comparisons and observer/original-study-C-hook comparisons are exact. These are14 independently verified exact pairs. Default128/512 states and metrics reproduce the original failed native-C pair exactly.

Profiler records identify12 memory-efficient SDPA dispatches per default forward and12 math SDPA dispatches per forced-math forward. Matrix-multiply TF32 is disabled and float32 matmul precision is `highest`; the settings restore after the math context. These are observed backend operators, not an inference from enabled flags.

|128 versus512 shared prefix|Default memory-efficient|Forced math|
|---|---:|---:|
|Maximum final-hidden absolute drift|0.002273560|0.001060486|
|Final-hidden RMS drift|1.376497e−5|6.260740e−6|
|Final-hidden relative L2 drift|1.623847e−6|7.385768e−7|
|Maximum per-query relative L2 drift|1.915251e−5|7.310816e−6|
|Maximum shared-target NLL drift|0.000162125|0.000163078|
|Mean signed NLL drift, long minus short|+7.545043e−7|+8.807651e−8|
|Mean FP64 KL(short||long)|6.565076e−10|6.542608e−10|
|Shared-target argmax agreement|100%|100%|
|Hidden entries exceeding original0.001|5|1|

**Math does not pass the old hidden guard either.** It lowers hidden drift while leaving maximum NLL drift and mean KL similar. Neither result supports choosing a backend solely to make the old test pass. The smallest scored top-two logit gaps are about0.00032; no argmax changes occur in this input. Low KL and unchanged argmax do not prove equivalence for other inputs or gate interventions.

The first measured divergence in **both** length pairs is `L0.qkv`, maximum3.814697e−6. `L0.input` and `L0.ln1` are exact. Thus the first observed difference appears at the combined affine Q/K/V projection, before SDPA. Later discrepancies are larger: layer11 attention-output maxima are0.001371384 default and0.000732422 math. This observation rules out describing the observed length effect as originating solely in SDPA. It does not identify a unique GEMM algorithm or assign additive shares of downstream error to particular operations.

Taken together, exact same-shape suffix invariance, exact repeats, exact hook controls, and a first pre-attention projection divergence provide strong evidence for length/shape-dependent floating-point behavior in these configurations. They provide no evidence of a causal effect of the changed suffix on this tested prefix. This remains one input and two backends: it is not a universal proof of causal isolation or numerical harmlessness.

The worker used19.274148 seconds,656467968 bytes peak GPU allocation and2118256KiB process peak RSS. Time includes observers/profiling and analysis; it is not a speed benchmark.

## Prospective V2 assessment

Retaining the default backend is reasonable. Keep the failed protocol, sources and raw run immutable, and issue a separately versioned amendment with explicit disclosure that this diagnostic and the first gate-evaluation input have already been seen. Preserve the original calibration/map bytes, ranks, QR seed, conditions and comparison family; do not select a rank, gate control or calibration from partial outcomes.

Separate two questions that the old cross-length absolute guard conflated:

1. **Causal isolation at fixed shape.** Add a prespecified changed-suffix control while keeping length512, positions0–127 and dispatch unchanged. Require exact prefix hidden states, gate states, logits and scored losses for the tested same-shape pair, with same-shape repeats distinguishing nondeterminism. Cover gate replacements, not only native C. The simplest complete coverage is one fixed rotated-suffix companion for each of21 conditions on each of32 long input/start combinations:672 additional forwards using already-required original512 references. A smaller sentinel panel is possible, but its limited coverage must be explicit. A second constant-suffix check can be a separately declared bounded sentinel; do not choose cases after seeing failures.
2. **Numerical comparability across shapes.** Retain the original1e−3 maximum NLL guard and1e−4 attention/conditional-mixture reconstruction guards. Always report the original absolute hidden maximum and its old-threshold exceedance count. Add per-query scaled hidden diagnostics, so large learned coordinate magnitudes do not silently dominate the interpretation.

Two complementary scale-aware statistics are useful. For shared query q, let delta_q be long-minus-short final hidden state, and define `s_q=max(1,RMS(h_short,q),RMS(h_long,q))`. Report `max_q ||delta_q||∞/s_q` and `max_q RMS(delta_q)/s_q` (a per-query vector-relative measure when the floor is inactive). Preserve sourceq0, scored queries1–126 and unscoredq127 separately. The maximum avoids averaging away a localized error; the RMS statistic records distributed error. An elementwise allclose criterion is another option, but near-zero coordinates and its chosen absolute floor create a different estimand. Avoid a single global denominator that can hide a bad query.

**This one-input diagnostic does not validate a universal new numerical tolerance.** Any numeric budget for these scaled statistics should be declared prospectively as a research equivalence criterion, justified by its units and intended downstream precision, not picked as the smallest threshold above the observed maximum. A bounded native-only check on predeclared calibration inputs could assess that budget without fitting gates or consulting additional held-out gate outcomes. It would still be a numerical-policy validation, not a new map fit. Keep KL, argmax margins and losses visible; relative hidden closeness alone cannot guarantee functional equivalence.

A defensible alternative is to make cross-length hidden errors descriptive and reserve hard causal isolation claims for exact fixed-shape suffix controls, while retaining the NLL/reconstruction safety guards. If V2 instead uses a hard scaled-hidden budget, freeze its precise formula, floors, denominators, source/scored scope, threshold and failure policy before a new run. Preserve any subsequent failure without silently widening the budget or switching kernels.

This review supports proceeding to a concrete V2 design, not a blanket approval of an unspecified amendment. No completed gate-subspace scientific finding follows yet.
