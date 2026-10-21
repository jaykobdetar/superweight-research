# Proposed V2: calibrated gate subspaces with separate numerical controls

**Status: proposal only, pending root and independent scientific review. No V2
implementation or inference is authorized by this document.** This is a new
protocol and run, not a relabeling or continuation of the failed original.

## Why an amendment is proposed

The original [gate protocol](GPT2_GATE_SUBSPACE.md) stopped at its declared
absolute final-hidden prefix tolerance. The failed run and frozen sources are
preserved at `gpt2_gate_subspace_run/`; its status remains failed. It consumed
33 forwards: eight calibration and25 evaluation forwards, with24 evaluation
records saved. The extra P512 forward had completed before recording native
C512 triggered the guard. It must remain in resource accounting.

The separate [18-forward native diagnostic](GPT2_PREFIX_DIAGNOSTIC.md) found
exact repeat, observer-hook, original-study-hook and fixed-shape changed-suffix
closures on this input. Default128/512 final-hidden maximum absolute drift
was0.00227356, relative whole-prefix L2 drift1.62385e-6, maximum shared-token
loss drift0.000162125, KL approximately6.56508e-10 and argmax agreement100%.
The first trace divergence appeared at L0 Q/K/V projection, before attention.
Forced math changed arithmetic but still exceeded the old absolute hidden
threshold (0.00106049). These observations support numerical shape sensitivity
for this input; they do not prove universal suffix independence or identify
SDPA as the sole origin. Default attention used the observed memory-efficient
operator; the math operator was separately verified.

The following tolerances are therefore an explicit **post-failure engineering
amendment**, informed by native diagnostics. They are not the original
preregistration, a floating-point theorem, a statistical confidence bound or
evidence that every later intervention will be numerically stable. No rank,
map, evaluation input or scientific contrast is selected by gate-arm results.
The supplied diagnostic has passed independent saved-output audit; policy
approval for this amendment is a separate decision.

## Recommended numerical policy

Keep the same checkpoint, FP32, default SDPA selection and original precision
settings. Do not force math, change TF32 flags, or modify model parameters.
Record versions, precision settings and an observed backend operator sample
without extra model forwards. Preserve the existing mask-reference checks.

For each existing128/512 pair, let `s_q` and `l_q` be the final-normalized
hidden vectors at query `q=0,...,127` for short and long inputs. Compute in
FP64:

```
local_q = ||l_q - s_q||_2 / max(||s_q||_2, ||l_q||_2)
global  = ||L - S||_F / max(||S||_F, ||L||_F)
```

Require finite tensors and strictly positive finite denominators; there is no
epsilon replacement or clipping. Require:

- `max_q(local_q) <= 1e-4` (0.01% of each query's hidden-vector norm).
- `global <= 1e-5` (0.001% of the full prefix's hidden-state norm).
- The **original** maximum per-token shared-prefix loss drift `<=1e-3`.

The local criterion prevents a small set of queries from disappearing into a
global average; the tighter global criterion bounds broadly distributed
error. Both scales have a direct interpretation and avoid a coordinate-size
assumption about final layer normalization. For transparency, the already
observed native default input has maximum local relative L2 drift1.91525e-5
at query110 and global drift1.62385e-6. These values describe the exposed
diagnostic; they do not validate the bounds across the remaining inputs.

Report absolute maximum, RMS, relative L2, maximum-error coordinates/values,
query0/query127 errors, argmax agreement and top-two margins. Continue counting
entries and pairs that exceed the **old**0.001 absolute threshold. Do not hide
those failures or call a V2 pass a pass under V1. Large-norm hidden vectors can
admit larger absolute differences under V2; retained loss/context guards and
the separate exact fixed-shape controls address different failure modes.

All other guards stay unchanged: native/manual SDPA, conditional-mixture,
rescaled-context, full-rank logit/mass/context reconstruction and placement
errors each remain `<=1e-4`; current denominator mass remains finite and
strictly positive; no clipping; finite projected sigmoid endpoints0/1 remain
allowed and counted. Exact current/full-rank/native-repeat/identity closures,
source-state identity, parameter integrity and immutable map checks remain
required. Native explicit/default-mask drift remains reported. A new guard
failure remains a failure with preserved partial outputs; no further automatic
tolerance adjustment, kernel switch, input deletion or arm deletion follows.

## Fixed-shape suffix controls for every longer input and mode

Add exactly one suffix-rotation control to **all32 longer inputs** (16 passages
times natural/BOS), for **all21 original conditions**, including native C and
all control/active maps. This adds672 forwards. Rotate token positions128..511
by one, leaving0..127 unchanged. Require the suffix actually changes. Keep
length512, absolute positions, starting-token identity, the condition's mask
representation and all other settings unchanged. No new passages are chosen.

Compare each rotated run with the same condition's original512 run. Require
bitwise equality for:

- Final-normalized hidden states at all first128 positions.
- Prefix logits at queries0..127, and scored losses/predictions at queries
  1..126 predicting targets2..127.
- Each affected layer's current `b`, current mass, proposed logits when
  present, and delivered target mass at queries1..127.
- The recorded source Q/K/V states and the current/delivered prefix contexts
  used by the hook. Source context is included.

The controls use current endogenous computation and the identical frozen map,
not a donor patch. Their altered suffix metrics are not scientific outcomes;
do not aggregate or inspect scores for changed suffix targets. Preserve the
original FP32 chunk32 arithmetic on the shared scoring rows so the comparison
does not inadvertently introduce a different reduction shape.

A failed equality is a numerical/protocol failure to investigate, not proof of
semantic leakage by itself. Passing establishes invariance only for this fixed
family of changes. Static query-locality review and the raw controls complement
each other; neither licenses a universal claim.

These controls cover every main512 input and condition and are **not additional
held-out samples**. They do not replace the680 original cross-length checks.
There are still1360 main evaluation records, plus672 separately labeled
numerical-control records. Keep the original1488 main runtime checks, with672
additional suffix-check records; do not merge controls into the n=4 statistics.
There are15360 new gate-context checks (10240 main plus5120 suffix controls),
and2032 new forwards, each with eight projection/query/context checks.

## Calibration: reuse authoritative frozen maps; provide a rebuild path

For the primary V2 run, **reuse the exact completed calibration artifacts**.
The native diagnostic provides no reason to change their data, algorithm,
backend or fit. Refit after this exposure would create an unnecessary new
degree of freedom. Reuse makes the intervention definitions identical to V1
and avoids counting the same eight passages as new independent evidence.

Load only verified copies of the frozen calibration tokens, raw `b`/mass,
maps, spectra/LOO summaries, input manifest and code provenance. Verify the
existing byte hashes and tensor digest before any V2 inference. In particular:

```
calibration_raw.pt SHA256:
d41d5a91c699d30d5414a2916f7350a9093eda4b0004d8bb7b2fb458ebb31935
calibration_maps.pt SHA256:
ed62f089168abeec63434e5f6dfa83ff8e7e139bc362e952456ea34859b5a098
map tensor digest:
fbf127a1a618e709b0d8e39625a156f11402c9703ac4b55ae3dcfd3711571c81
```

A dependency manifest must identify the preserved parent, original eight
calibration forwards and copied files. Do not require the old failure to
reoccur as a prerequisite for reproduction.

Provide a separate, deterministic **calibration-only rebuild** command for
independent replication: the same eight natural512 passages and exclusions,
same zero-V P calibration, same4080 scored vectors/layer, FP64 centered sample
covariance/SVD, QR seed20261025, sign conventions, ranks and descriptive LOO
outputs. This costs eight forwards and performs no evaluation or model/map
selection. It must be explicit, not an automatic fallback after a mismatch.

The exact frozen maps remain the reference for strict V2 replay. A rebuild
on another platform may differ numerically; preserve its hashes, tensor and
subspace comparisons and environment. Do not silently substitute its maps
into the primary V2 run or merge that run with reference-map results. Such a
run is a separately labeled replication with a fixed rebuild algorithm, not
a search for a better fit. A portable reproduction bundle should supply the
reference numeric maps and retrieval instructions for restricted inputs.

## Scientific design and exposure accounting

All21 conditions, ranks1/3/6, retain/remove actions, mean/position/global-PC1
controls, fixed QR basis, four corpora,16 passages, natural/BOS replacement,
128/512 prefixes, original scored targets and32 contrasts remain unchanged.
No input replacement or outcome-dependent subset is allowed. All main
1360 records are rerun in the new V2 directory; saved incomplete V1 records
are neither silently spliced in nor treated as an independent replicate.

Mark the entire first Wiki-validation passage (window0) as **exposed** for
both lengths and both starting-token conditions. All natural128 arms and
native/P512 states were computed before the original stop, and its native
input then drove the diagnostic and protocol amendment. BOS was not thereby
run, but shares the exposed passage and must not be described as wholly fresh.

Retain the original336 n=4 group tables,504 segment tables and32 paired
contrasts per input, explicitly labeling their exposure. In addition, report
the predeclared unexposed-passage subset: Wiki windows1..3 (n=3) and all four
windows of each other corpus (n=4), for both identities and lengths. Use the
same2000 bootstrap resamples/seed20261021 with the actual group size. Show all
ranks/conditions/contrasts in both views. These are conditional descriptive
intervals; do not pool the subset as an independent dataset, use it to choose
a winning rank, or claim correction for multiple comparisons.

The remaining15 passages are unexposed to the native-debugging amendment in
this sense, but the study still uses the same checkpoint/corpus family and
frozen calibration. Report that scope rather than calling V2 a wholly fresh
confirmatory experiment. Dimensionality remains descriptive until the fixed
directional interventions are evaluated; low rank alone is not one mechanism.

## Storage, resources and stop policy

Use new V2 runner/analyzer/protocol filenames and a new output directory;
retain all V1 and diagnostic sources/outputs byte-for-byte. Preserve main raw
metrics, gates and prefix-hidden pairs as before, and store controls in their
own shards. Save control hidden states and gate tensors. Exact prefix-logit
checks can save separately computed original/control SHA256 hashes and equality
results, with full compared logits preserved on failure, avoiding roughly17GB
of redundant successful prefix logits. Current/delivered prefix contexts may
likewise save independently computed original/control hashes and exact flags,
with the fixed raw snapshot subset and complete compared arrays on failure;
do not duplicate all successful contexts (roughly4GB). Document both audit
limits: hash checks are not independent recomputation of absent logits or
contexts. Keep the original40 main
attention snapshots and add20 matching suffix snapshots for the fixed
Wiki-window0 natural512 ten-mode L4/L11 subset.

The primary budget is **2032 new forwards** (1360 main plus672 controls), zero
new calibration forwards and one local GPU model. The existing eight
calibration forwards are an inherited dependency, not newly executed work.
The explicit calibration-only rebuild adds eight forwards if separately run.
Count the original failed33 and native diagnostic18 in the cumulative ledger:
2083 forwards from the start of V1 through completed primary V2, or2091 with
one separately executed rebuild. These figures exclude earlier research.

No speedup claim is made. Keep two current-input reference logits, shard
controls/snapshots, avoid a second model load or concurrent GPU work, and use
CPU reviews under3GiB where feasible. Require independent prospective policy
and implementation review before V2 inference, and independent numerical
review before scientific interpretation. Save every numerical failure and
partial record. A report boundary does not establish that the broader causal
research is complete.
