## What the gate experiment tests

The earlier transfer screen did not find GPT-2 dependence on its selected
single large weight. Subsequent coordinate lesions identified conditional
regrowth and source-value pathways. This study isolates a related attention
allocation question in otherwise native upstream GPT-2. It does not apply
the original OLMo pulse or establish transfer of that exact mechanism.

After source values are zeroed in layers 4–11, a later query's head context
has the form `c = a × u`: `a` is total probability on non-source positions,
and `u` is their conditional value mixture. The preceding donor factorial
separated these factors. The present question is whether a simpler fixed map
can retain useful variation in the existing gates.

At each current query `q >= 1`, the model supplies 12 head allocation logits
`b = logsumexp(s_1, ..., s_q) - s_0`. In exact arithmetic, `a = sigmoid(b)`;
the code computes `b` from scores, avoiding a logit of rounded probabilities.
Here `q` also counts available non-source keys. A control subtracts `log(q)`
before centering. Eight natural-start calibration passages from Wiki test
and Hume determine the means and centered FP64 covariance/SVD. The 4,080
query vectors per layer come from eight passages and are correlated.
Leave-one-passage-out spectra and principal-angle cosines describe stability;
they do not select a rank.

Each layer centers the detrended vector by its own calibration mean `alpha`,
retains or removes PCA directions at the predeclared ranks 1, 3 and 6, and
restores `alpha + log(q)` before the sigmoid. Rank-matched QR controls use a
fixed random seed. Other controls use constant mean mass, constant mean
logits, a position-only map, and raw-logit PC1. These are separate maps per
layer, not a single shared scalar for the whole network.

The patch `c′ = c × (a_target/a_current)` preserves the current conditional
mixture locally. Earlier changes can alter later states and mixtures, so this
is not a fixed donor-trajectory decomposition. Gate projection preserves the
source query relative to zero-V baseline P. Zeroing V itself can change that
query relative to native C. Tiny denominator masses remain unclipped and are
checked against a stable conditional-softmax reconstruction.

![Gate intervention and calibration boundary](figures11/gate_intervention.png)

The panel has four passages each from Wiki validation, Du Bois, Hobbes and
optimizer code. Natural/BOS variants replace only token 0. Calibration uses
disjoint Wiki-test/Hume passages and excludes earlier full extraction buffers;
evaluation used-token spans exclude 32 earlier spans per corpus. These splits
do not establish independence from pretraining data.

## Two failed execution-length checks remain part of the evidence

The first implementation passed independent CPU checks but stopped on the
pretrained model's first native 128/512 comparison. It consumed eight
calibration and 25 evaluation forwards, with 24 evaluation records appended.
Maximum final-normalized hidden drift was 0.00227356, above the original
absolute 0.001 limit. Maximum shared-target loss drift was 0.000162125;
argmax predictions agreed. This run remains failed, with frozen code and raw
partial outputs preserved.

Independent audit found five of 98,304 hidden entries above 0.001. The
largest was query 62, coordinate 430: 118.330078 versus 118.327805.
Whole-prefix relative L2 error was 1.62385e-6. Those small relative values
did not override the failed criterion or identify the discrepancy's cause.

A separately reviewed 18-forward native diagnostic crossed default and forced
math attention with repeated lengths, changed future suffixes, a padding-mask
control, and hook-free versus study-hook paths. All repeat, same-shape suffix,
padding and hook comparisons were exact in saved prefix states, logits and
losses. Default execution reproduced the failure. Forced math still exceeded
the old absolute hidden threshold, at 0.00106049. Mean cross-length FP64 KL
was 6.56508e-10 under default attention and 6.54261e-10 under math; predictions
agreed in both cases.

![Block-output numerical drift; relative errors use the short reference](figures11/prefix_numerics.png)

Both length pairs first diverged at layer 0's combined Q/K/V projection
(maximum 3.81470e-6), with identical input and preceding normalization.
Observed operators confirmed memory-efficient attention by default and math
attention under the explicit control. This supports execution-shape numerical
sensitivity in these configurations; it does not uniquely blame attention or
prove universal suffix invariance. PyTorch documents possible differences
between full-tensor and sliced computations and between attention backends.
Those facts motivate controls rather than establish this diagnosis alone.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html),
[SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).

Independent audit verified all 18 diagnostic shards, 25 pairs, 276 FP64 score
comparisons, 192 context reconstructions and eight alignments to the failed
run; maximum saved-summary discrepancy was 1.42e-14.

A separately declared V2 retained all maps and conditions, added exact suffix
controls, and used post-failure relative hidden limits: 1e-4 per query and
1e-5 globally, with the original loss/context limits. It also failed, on the
first Wiki passage's BOS native baseline. Query 110 had relative L2 drift
1.0108874e-4, exceeding the local limit by 1.09%; it was the only query above
that limit. Global drift was 2.44156e-6, maximum loss drift 0.000191212, and
scored predictions agreed.

V2 consumed 90 forwards and saved 68 main records plus 21 suffix records in
10.591 seconds. Independent audit confirmed the failing C pair, 23 completed
natural-start prefix comparisons, all 21 completed suffix controls and 60
FP64 snapshot states; maximum reconstruction component error was 2.9832e-5.
The run did not reach its final parameter-integrity check. Its most recent
forward was P512, whereas the failing saved comparison was C; the audit used
the explicitly saved C pair. Neither failed run supplies a completed gate
result. There was no further threshold increase or automatic restart.

## The narrower fixed-shape study

The independently reviewed new design uses exactly 512 tokens in every model
forward. It retains all 21 conditions, ranks, maps and 32 input/start variants.
It reports targets 2–511 overall, early targets 2–127, and later targets
128–511. The early segment is a score view of a 512-token execution, not a
standalone 128-token run, an independent sample, or a matched-target test of
changing available history. Cross-length numerical validation remains open.

All 672 fixed-shape suffix controls remain: each condition on every longer
input rotates positions 128–511 while preserving positions 0–127. Prefix
hidden states, logits, gates, source Q/K/V, contexts and shared scoring must
be exact. Changed suffix targets are never scored. Attention, gate and stable
mixture reconstruction retain the original 1e-4 limits. Exact copy controls,
source identity relative to P, immutable maps and completed parameter checks
remain required. Any new failure stays a failure.

The first Wiki passage is exposed at both starts. Original n=4 tables are
accompanied by a fixed unexposed-passage view: the remaining three Wiki
passages and all four passages in each other corpus. Both views retain every
condition and contrast. Paired 2,000-resample bootstrap intervals use seed
20261021 and actual group sizes; they condition on the fit and selected
passages, with no multiple-comparison or population claim. The two views
are not independent datasets.

Successful prefix logit/context comparisons retain separate runtime equality
flags and hash/shape/dtype records; full arrays are saved only for fixed
snapshots and failures. An independent audit cannot recompute absent arrays.
Raw gate, source, prefix hidden and per-token metric tensors remain available.

The separate calibration-only rebuild performed eight forwards, with no
evaluation inference, in 3.614 seconds. Raw calibration and fitted maps were
byte-identical to the original; independent audit verified all 176 tensors
and 64 operation checks. This is same-machine reproducibility, not external
or cross-platform replication. The primary study uses the original maps.
