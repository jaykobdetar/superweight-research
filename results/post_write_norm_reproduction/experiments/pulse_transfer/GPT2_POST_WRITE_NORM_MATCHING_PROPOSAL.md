# T20: match the actual L10 attention-write change

Prospective protocol draft, 21 October 2026. **This document does not approve
implementation, a model load, or execution.** Independent design review comes
first. Source, input, numerical-fixture and complete storage/failure reviews
must then pass before a separate literal launch freeze. No T20 evaluation
inputs or model outcomes have been inspected to choose this protocol.

## Question and fixed scope

T19 found different loss responses to the frozen F leading-direction and
fixed orthogonal gate edits, including at the intact `000` boundary. They
match gate-logit change norm, but their context changes do not match norm.
The learned output projection can further change their relative magnitudes.
T20 asks whether the tested PC1-versus-orthogonal loss difference remains
when their **actual L10 attention writes into the residual stream** have
approximately matched change norms, within the FP32 rules below.

Keep the GPT-2 checkpoint, tokenizer, four corpus definitions, physical length,
native SDPA mask behavior, frozen F center/directions and original full edits
from T19. There is no calibration, direction search, loss fitting, nonlinear
root search, probability clipping, epsilon in a denominator, or outcome-based
choice of norm, amplitude, passage or tolerance. This tests one fixed fitted
operation against one fixed control. Native-specific calibration and a control
direction ensemble are separate questions.

The matching location is the output of L10 `attn.c_proj`, before the residual
addition; evaluation-mode dropout is an identity. Bias and all FP32 projection
rounding are included in the observed write. This is **attention-write-space
interpolation**, not a claim that an FP32 allocation/context interpolation
produces the same array. The original full edits still use T19's context
formula before `c_proj`. Matching pre-projection context norm is not an
additional intervention pair in this protocol; record that norm as a diagnostic.

## Model and operation pins

Use `openai-community/gpt2`, revision
`75e09b43581151bd1d9ef6700faa605df408979f`, with the original checkpoint-file
hashes and model parameter digest
`2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527`.
No prescribed operation edits parameters. Every completed process must pass
its own before/after parameter hash and parameter-version checks.

F is the audited T18 fresh four-Wiki fit, with no refitting:

| Binding | SHA256 |
|---|---|
| F map digest | `8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8` |
| T18 reference manifest | `f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060` |
| Map NPZ | `f75cf7582402ce5b999758c70477e6cf2740f317a9ed8e508b350501397ef14a` |
| Direction NPZ | `53d686645f431a47f5e22ec9c5ec64c5cfb0ed1086c2172b8bb6922a532be8dc` |

Retain the verified full-edit arithmetic from `gate_native_value.py`, SHA256
`b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd`:
for non-source query q, compute the frozen coefficient
`s = (b - log(q) - alpha) dot u`; propose `b - s*u` for PC1 and `b - s*v`
for the fixed orthogonal control. The full context change uses the actual
native context plus `(target_mass - diagnostic_mass) * (mixture - V0)`, with
the unchanged-gate exact bypass. Preserve the source query and the model's
native mask argument. This does not replicate T18's old explicit-mask execution.

Pin Python 3.12.3, Torch 2.11.0+cu130, Transformers 4.57.6, NumPy 1.26.4 and
CUDA runtime 13.0, along with all T19 precision/backend flags. Verify the
observed SDPA operator, no competing GPU computation before each load, no
cache, FP32 model inference and exactly 512 physical positions. Record hardware
identity and actual runtime settings. Changes need review before execution.

## Fresh panel and exposure closure

Select 16 new passages: four each from Wiki validation, Utilitarianism, Ethics
and the pinned optimizer-code sources. Each has the unchanged natural start
and a BOS replacement of token zero only, giving **32 input identities**, not
32 independent passage pairs. Do not insert or retokenize a BOS token. Keep
the physical 512-token window and all remaining tokens identical between starts.

Reuse only the hash-pinned deterministic selection function from
`gate_calibration_factorial_inputs.py`, SHA256
`810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a`:
preferred grid numerators 125/375/625/875 over denominator 1000, candidates
ordered by distance then numerator, first admissible window per slot, code
used span inside a single source component. Preserve pinned source bytes,
tokenizer files, offset semantics, corpus bounds and complete rejection ledger.
Do not reuse T19's hardcoded history scan as the complete T20 inventory.

Before selection, make a new pinned full-history scan. Carry forward every
inherited exclusion interval and token hash, including complete historical
calibration/development buffers and all four retained 8192-character F buffers.
Append all T19 selected natural used-token character spans and **all 32
attested natural/BOS 512-token hashes**, regardless of original, replay,
recovery or composite alias. Use hash attestations without exposing token IDs.
Include available interrupted/failed and
tokenizer-only preparation manifests under the same conservative exposure
policy. Retain origin paths and hashes even when intervals/hashes coincide;
deduplicating a set for membership must not erase provenance or shorten a
buffer. The old 577-hash inventory plus 16 new T19 windows gives a minimum
593-hash inherited union before the additional BOS attestations; the complete
union must also contain all 32 T19 identity hashes and any other discovered
history. Record the realized set size rather than substituting this minimum.

The selector must reject overlap with that complete union, duplicate old or
new token windows, out-of-bounds spans and code-component crossing. If a
source cannot provide four admissible windows, preparation fails; do not
weaken exclusion rules. Tokenizer-only preparation performs zero model
forwards and loads no pretrained weights. Independently reconstruct accepted
and rejected choices, source/offset/token hashes, all retained buffers and
paired-start identities before freezing the new input hashes. Check both
fresh natural and BOS window hashes against the full historical union and
within the selected panel, preserving deterministic candidate traversal and
an explicit ledger for any additional duplicate-start rejection. Preserve every
preparation failure. Freshness concerns this research's recorded exposures;
overlap with original GPT-2 pretraining remains unknown.

Historical first-token membership in F/T18 is not a selection factor or an
estimand here. Omit that optional legacy diagnostic prospectively, labeling
it unavailable/not used rather than inventing seen/unseen flags or reopening
old raw token tensors. Retain the actual new natural/BOS identity check and
paired suffix. The study makes no balanced-start or seen-versus-unseen-token
population claim. Publication exports must retain hash/provenance checks
without leaking restricted historical token IDs or host paths.

## Boundaries, arms and order

All layer numbers are zero-based. `000` keeps native projected source V at
layers 4–11. `111` zeros that slice at layers 4–11. Values elsewhere follow
each condition's live trajectory. The two boundaries have different incoming
states; there is no cross-boundary shared-state or pure-local-V claim here.

At each boundary use these six arms, in this fixed order:

| Arm | Delivered L10 write |
|---|---|
| `baseline` | Unedited write for that boundary |
| `current` | Exact no-change observer/control |
| `remove_full` | Original full frozen-F PC1 edit |
| `orth_full` | Original full fixed-orthogonal edit |
| `remove_write_matched` | Interpolated PC1 candidate, defined below |
| `orth_write_matched` | Interpolated orthogonal candidate, defined below |

For each corpus, window 0–3 and natural/BOS identity, run `000` then `111`.
For each arm run its main forward, then its fixed-512 suffix-control forward.
The suffix control rotates tokens 128–511 by one position and preserves the
first 128 tokens. Full candidates from **both** directions and their own
baseline must exist for each variant before either matched arm executes.

After `000` baseline main on natural window zero of each corpus, run the
unhooked native-repeat and historical writer-identity anchors, as in T19.
Remove every T20 intervention/observer hook for native-repeat and reinstall
cleanly afterwards. Do not use an unhooked native anchor as an identity check
for `111`, because unhooking would remove the source-value boundary.

Counts are fixed: 32 inputs × 2 boundaries × 6 arms = 384 main forwards;
each has one suffix forward, giving another 384; four corpora × two native
anchors adds eight. **776 planned model returns, zero calibration forwards.**
Model intents, actual invocation attempts, returned calls and fully processed
records have separate hash-linked journal events. Count invocation immediately
before the model call and return before scoring/capture. An exception is not
permission to silently add model attempts or reinterpret planned counts.

## Actual-write interpolation

For a fixed input, boundary, main/suffix variant and query q, let `y0`, `yu`
and `yv` be the actual FP32 768-coordinate `c_proj` outputs from baseline,
`remove_full` and `orth_full`. These include the projection bias. They are
cached actual arrays, not `delta_context @ W` approximations or donor states.

For i in {u,v}, define in CPU FP64:

```
di = float64(yi) - float64(y0)
ni = sqrt(sum(di * di, over 768 coordinates, in float64))
target = min(nu, nv)
lambda_i = 0                         if ni == 0
lambda_i = target / ni               otherwise
```

Require finite values and `0 <= lambda_i <= 1`. If the target is zero, both
matched arms deliver `y0` exactly. If `lambda_i == 0`, copy `y0`; if
`lambda_i == 1`, copy `yi`. These are exact comparisons, not near-zero tests.
For the remaining queries, multiply `lambda_i * di` in CPU FP64, add `y0`
in a separate FP64 operation, then cast once to FP32 round-to-nearest-even.
Transfer those FP32 bytes to the original device without arithmetic and
verify returned-device bytes by a CPU round trip. No GPU FP64 reduction,
fast-math fusion, flush-to-zero assumption, clipping or inverse sigmoid is
part of this definition. Record CPU precision/runtime behavior and test
subnormal conversion before weights. The device synchronization is acceptable:
this is a causal probe, with no speedup claim.

This computation is independent for every query across its 768 coordinates.
Never pool across queries to choose a scale. That would make an early write
depend on future suffix content. Source query zero must already agree
bit-exactly in baseline and both full candidates; its returned write remains
bit-exact baseline. The matching target and scales can depend on the actual
incoming state, but never on loss, labels, predictions or later-layer outputs.

Before interpolating, require the matched arm's incoming L10 Q/K/V, native
mask, unedited context and full proposed context to match its cached full-arm
reference exactly. Require its actual pre-interpolation `c_proj` output to
match the corresponding cached full write exactly. This establishes that
the cached candidate is the current same-state result.

Cache keys bind corpus/window, start identity, boundary, main-versus-rotated
variant, actual 512-token fingerprint, frozen F digest and execution/source
freeze. Main and suffix variants need separate baseline/full-candidate
caches; neither boundary may borrow the other's arrays. Reset caches for
every boundary/input. Capture actual post-interpolation writes separately
from the proposed full contexts and full writes; never label the proposal's
gate/context norm as the matched arm's actual delivered write norm.

## Frozen numerical acceptance rule

These rules are chosen from floating-point analysis and synthetic arrays
before any T20 data. Matching is approximate in FP32, not an exact ideal norm
equality on every query. The operation contains no numerical epsilon; the
following constants are comparison/error budgets only.

Let D=768, `u32=2^-24`, `eta32=2^-149` and `u64=2^-53`. Set
`kappa64 = (32*D*u64)/(1 - 32*D*u64)` (about 2.7284841e-12).
For each query/arm compute this conservative FP64 rounding budget:

```
Bi = u32 * norm2(abs(y0) + lambda_i * abs(di))
     + 0.5 * sqrt(D) * eta32
     + kappa64 * (norm2(y0) + ni + target)
```

Here `y0` and `di` in the budget are FP64 views/differences and every norm
uses the same explicit rowwise FP64 sum-of-squares/square-root convention.
The FP32 term follows the coordinate round-to-nearest bound
`abs(round32(z)-z) <= u32*abs(z) + eta32/2`; the triangle inequality gives
a norm-error bound. The conservative FP64 term covers the difference,
row norm, ratio, multiply and add reductions under the specified finite
arithmetic. All inputs/outputs must be finite. FP32 inputs squared in FP64
stay within FP64 range, including subnormal FP32 inputs. Independent design
review must check this argument and the actual implementation convention.

After actual device delivery, let `n_hat_i` be the FP64 norm of the actual
returned FP32 write minus actual baseline. Apply all of the following:

1. Per query q=0–511: `abs(n_hat_i - target) <= Bi` for each matched arm,
   and `abs(n_hat_u - n_hat_v) <= Bu + Bv`. Exact zero/endpoint and source
   branches must additionally satisfy their bit-exact array checks.
2. For the non-source domain q=1–511, define `RMS_target` and
   `RMS_gap = RMS(n_hat_u - n_hat_v)` using FP64. If `RMS_target == 0`,
   both actual norm vectors must be exactly zero. Otherwise require
   `RMS_gap <= 1e-5 * RMS_target`.
3. Repeat that RMS quality check on the primary scored-query domain q=1–510
   (targets 2–511). Its zero-denominator case has the same exact branch.
   Apply the rules to every main and suffix variant, not selected examples.

The relative RMS limit 1e-5 is a fixed operational match-quality criterion,
not a theorem that every rounded ray can meet it. It prevents a large
rounding envelope from being mistaken for a useful norm match. Individual
tiny targets may have larger relative discrepancies while satisfying all
rules. Retain them and report their maximum positive-target relative gap,
quantized-to-zero counts, and the distribution of absolute/relative errors;
use `null` for undefined ratios at exact zero, without an epsilon. Report the
all-query and scored-query RMS denominators/gaps separately. Do not claim
an exact per-query magnitude control or infer equal downstream sensitivity.

Any hard numerical or exact-control failure halts the accepted batch and
preserves evidence; no passage/query exclusion or tolerance relaxation is
allowed after seeing its loss. A proposed implementation correction requires
independent review and explicit accounting of every prior attempt. If the
scientific rule changes, preserve the failed study and freeze a revised
prospective protocol before another fresh-panel claim.

The preserved design fixture is
`review/t20_write_norm_numeric_design_fixture.py`, SHA256
`c42387e28b75e4487376529f68e08b751fd6839b8f033ae1aa9dd2537e3b3d68`;
its PASS JSON has SHA256
`33b7e992cb4c3c22371e41cee0c7c9c23fda01dab8b81446de3d53e5481847e0`.
It uses only synthetic arrays, zero pretrained weights/forwards/GPU, and
81,772 KiB peak RSS. Tested regular scales had relative RMS gaps at most
1.565e-7. Exact-zero, lambda-one, anisotropic projection and causal-prefix
cases pass. A deliberately quantized ray has target one FP32 ULP and actual
other-arm norm sqrt(2) ULP: the analytical bound alone passes, but the 41.4%
relative discrepancy correctly fails the frozen RMS quality test. Incorrect
delivered norms, nonfinite/wrong-precision inputs and pooled-query scaling
are rejected. This is arithmetic-design evidence, not production-hook or
whole-writer validation. Production fixtures must also exercise the separate
primary-domain RMS check, near-zero mixtures, mass saturation/overshoot,
cancellation, signed values, actual cache misuse and hook ordering.

## Exact controls and evidence

Baseline/current metrics, full hidden states and logits must agree exactly
at each boundary. Retain all source Q/K/V and block-output q0 references at
all 12 layers, per-layer projection fingerprints proving that Q/K and
non-source V are unchanged by source slicing, and exact source-q0 invariance
among each boundary's arms. Preserve the native mask argument and fixed
shape, with no caching or padding substitution.

For every main/suffix arm, require exact equality on the unchanged prefix:
logits q=0–127, hidden and actual post-write states q=0–127, losses for
targets 1–127, source states, masks, relevant gate/norm arrays and matching
targets/scales. Suffix matching must be recomputed from its own candidate
states. Check incoming and cached full-arm identity before interpolation.
Endpoint exactness is a **local write** identity; one query with lambda zero
or one does not imply its later hidden/logit matches an otherwise different
whole forward. CPU fixtures must additionally demonstrate whole-array
all-zero/all-one identity behavior. No T19 direct-local-V factorial identity
is inferred from just the two T20 endpoints.

Save every per-token NLL, KL, prediction and agreement array, natural/BOS token
identity, per-query target/own/actual norm, lambda, rounding budget and residual,
both pre-context and post-write norm diagnostics, actual source references,
all exact/hard-control records and hash-linked invocation journals. Norms
include q=1–511, with q0 separately checked; analyses use the scored views
specified below. Store proposed full contexts/full writes and actual matched
writes under distinct names and fingerprints.

Predetermine full raw snapshots on Wiki window zero, natural start, for all
six arms at both boundaries: 12 states. Include L10 Q/K/V/native effective
mask, gates, actual unedited/full proposed contexts, actual baseline/full/
returned projected writes and interpolation arrays, full final hidden, and
actual full-vocabulary logits at q=1,126,127,510 with targets 2,127,128,511.
Record complete runtime fingerprints elsewhere and clearly state that
ordinary full contexts/hidden/logits were not independently reconstructed
from unsaved arrays. An offline pinned-embedding readout audit can later
check these rows without a model forward; it needs its own reviewed code.

## Estimands and uncertainty

The primary readout is teacher-forced NLL on targets 2–511 (510 targets).
Early targets 2–127 and later targets 128–511 are secondary partitions of
the same 512-position execution. They are not different context-length
experiments. Ratios to intact C and to the arm's own boundary baseline are
separate; report exponentiated paired mean NLL differences and intervals,
not a mean of per-passage perplexity ratios. Include KL and argmax agreement
with C and with the own-boundary baseline, without a fidelity/generation claim.

At each boundary, predeclare these nine linear NLL contrasts, with B the
baseline, Gu/Gv the full candidates and Mu/Mv the matched candidates:

```
Gu-B, Gv-B, Gu-Gv,
Mu-B, Mv-B, Mu-Mv,
Mu-Gu, Mv-Gv,
(Mu-Mv)-(Gu-Gv)
```

The primary direction comparison is `Mu-Mv` at native `000`, reported by
corpus and start; no universal sign is required for acceptance. The other
eight per-boundary contrasts and all `111` results are prespecified
secondary estimands. Also retain the `111`-minus-`000` difference of each
of the nine contrasts. That gives 27 definitions, 648 corpus/start/view
contrast rows and 324 paired BOS-minus-natural interaction rows. An endpoint
difference combines source-value and upstream-state changes; no mediation
fraction or individual U/L/D attribution follows.

Preserve four passage clusters per corpus and paired starts, with one shared
2000-by-4 PCG64 bootstrap-index matrix, seed 20261021, across arms/contrasts.
Use the same indices for natural/BOS differences. The 12 conditions yield
1,152 main window/view rows and 288 corpus/start/view/condition summaries,
plus 24 anchor/view rows. Intervals condition on the frozen F calibration;
they do not estimate population calibration uncertainty or make deterministic
passages a random sample. Contrasts/views are correlated, with no multiplicity
correction. Report actual norm matching and remaining error alongside loss;
do not treat an interval exclusion as a count of independent replications.

## Processes, storage and failure preservation

Use two planned sequential processes with no concurrent model loads. Segment A
contains Wiki and Utilitarianism; B contains Ethics and code. Each includes
both boundaries for all of its inputs: 192 main + 192 suffix + four native
anchors = **388** forwards per segment. Each has all prechecks and final
parameter/version, source/input/calibration and precision checks. Both use
the same exact freeze. B starts only after A completes successfully. This
planned segmentation is part of the protocol; it is not interruption recovery.
There is no additional compatibility replay in the 776 count. An unexpected
interruption requires a separately reviewed recovery/accounting proposal.

Commit one complete boundary/input shard after its six main/suffix arm pairs
and applicable anchors. This gives 64 shards: 32 per segment. Smaller commit
units bound an uncommitted snapshot payload to six arms rather than 12. A
separate merger may copy byte-identical shards into an explicitly labeled
complete segmented study only after both segments' final guards pass.
Original segments, logs, caches saved as evidence and failed attempts remain
unchanged. No hardlink replacement, deletion, cleanup or deduplication is
authorized. Repository visibility remains user-controlled.

Retain a 512 MiB normal-artifact cap per process/composite, a 256 MiB raw
failure cap, and a 2 GiB free-disk reserve. Before starting, conservatively
reserve room for both segment normal caps, the composite cap and one failure
cap in addition to the 2 GiB reserve (3.75 GiB total if nothing is already
materialized). Continue checking reserve plus all outstanding planned writes.
No source/output size, peak RSS, wall time or capacity PASS is assumed from
T19. CPU peak RSS must remain at most 3 GiB; no paid compute or unrelated job
changes are part of this proposal.

Before model approval, serialize and round-trip a complete high-entropy
64-shard layout, all 12 snapshots/readouts, versioned metadata, journals and
source/input/calibration copies through the actual writer. Measure total
normal bytes and peak RSS. Measure the worst failed operation with both
actual compared sides, current tokens/hidden, all per-query matching errors,
actual baseline/full/returned writes, bounded offending vocabulary rows and
the complete uncommitted six-arm input payload. Test both successful separate
partial-shard preservation and the conditional fallback where that write
fails. The complete fallback capsule must fit 256 MiB and round-trip exactly.
Do not omit offending arrays or merely retain fingerprints to fit the cap.
If the layout fails, stop for design review before weights; these proposed
limits do not authorize loosening evidence or numerical requirements.

All output directories and fixtures are new and exclusively created; preserve
existing and race-created paths unchanged on rejection. Use immutable files
and versioned checkpoints. On a hard failure, stop accepted execution and
preserve actual arrays, condition/counter/source metadata and any partial
serialization error. Every attempted/returned call remains counted. Neither
the merger nor analysis may relabel a failed or interrupted process complete.

## Interpretation and approval boundary

A persistent matched difference would show that this particular PC1/control
comparison is not explained solely by their scalar attention-write norms
under the defined approximate-FP32 match. A reduced or absent difference
would be evidence that this intervention's dose/path matters; it would not
prove that magnitude caused the earlier effect or identify a mediated fraction.
Matching Euclidean write norm does not match headwise allocation, feature
direction, LayerNorm response, downstream Jacobian sensitivity or semantic
disruption. A single orthogonal vector cannot establish uniqueness. Two
boundaries do not locate a minimal circuit, and no result here establishes
faithful replacement, faster inference, generation quality, longer-context
generalization or the original OLMo single-weight mechanism in GPT-2.

The parent must independently review this protocol and its mathematical
guard before author implementation. Then require independent tokenizer/history
audit, production-hook and hostile CPU fixtures, complete normal/failure
storage measurements, analyzer/readout review, exact source/input/model/
precision bindings and a separate launch freeze. This document supplies
neither a ready flag nor permission to load weights or run T20.
