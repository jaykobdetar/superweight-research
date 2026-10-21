## Causal gate results

The fixed512 study completed all 1,352 forwards: 680 main records (including
eight repeat/identity anchors) and 672 changed-suffix controls. Independent
saved-output review verified the per-token reductions, full and unexposed
tables, paired intervals, source identities and saved suffix arrays. The
first exposed Wiki passage is excluded from the main tables below; Wiki has
three passages and the other corpora four. Full n=4 results remain available.

The centered, position-detrended calibration vectors have substantial shared
variance: PC1 explains 40.47–61.12% across the eight layers, the first three
directions 70.02–87.04%, and the first six 88.14–95.44%. These percentages are
descriptive calibration measurements. The interventions show why they cannot
be read as percentages of a single causal mechanism.

Retaining only PC1 in each layer raises perplexity to 1.148–1.292 times the
zero-source-value baseline P, with only 80.07–87.40% argmax agreement. Keeping
six directions improves the range to 1.0105–1.0954, with 90.39–95.20%
agreement. These are ranges of eight corpus/start means, not uncertainty
intervals. Six directions per layer are already half the head space; this
does not demonstrate a faithful scalar replacement or a single shared
network-wide factor. Native-C comparisons are also retained: rank-six ratios
are 1.0156–1.1102 and agreement 88.92–94.56%.

The matched controls separate fidelity from loss. PCA retention has higher
agreement with P than the single fixed QR control in all 24 rank × corpus ×
start mean comparisons. Yet PCA rank-one retention has worse loss than QR
rank one on both book corpora at both starts; all four paired intervals for
that contrast are positive. Wiki and code show the opposite sign, with all
four paired intervals negative. At rank six, six of the eight mean loss
comparisons favor QR, but five of those six intervals include zero. A
single random basis and small samples do not establish a distribution over
random subspaces.

Removing PC1 is also informative. It worsens Wiki perplexity by 5.70–6.06%
relative to P, with paired delta-NLL intervals entirely positive. On Du Bois
it lowers perplexity by 2.72–3.24%, with both intervals entirely negative.
Hobbes and code have slightly lower mean loss but intervals spanning zero.
Despite these loss improvements, removing PC1 reduces agreement more than
removing one QR direction in all eight mean comparisons. Useful prediction
fidelity and lower empirical loss are distinct objectives here.

Simple frozen replacements are inadequate on this panel. Constant mean mass
produces ratios 1.102–1.226 relative to P; the position-only map gives
1.148–1.298. The latter's early-target ratios are 1.392–1.822, compared with
1.078–1.192 on later targets. Removing PC1 has early losses above P on all
eight means, whereas its later losses are below P on books and code. These
are within-execution position views of the same passages; they do not show
that changing context length or earlier history causes those differences.

P itself is close in average loss to native C on this panel (0.9956–1.0135
perplexity ratio), but only 94.84–98.19% of argmax predictions agree. Its
closeness in loss does not erase the zero-value intervention or justify
calling subsequent maps native-model replacements.

## What is established, and what remains a hypothesis

The intervention results establish that the specified calibrated gate
directions affect the outputs of this pinned checkpoint, with corpus- and
position-dependent loss effects. Centered variance alone fails to identify
a faithful low-dimensional replacement. The combination of high variance,
better prediction fidelity, and sometimes worse loss is a measured result;
it is not evidence that a direction is intrinsically good or harmful.

Several explanations remain open. Calibration-domain differences could make
a fixed mean or basis unsuitable elsewhere. Applying maps at eight layers
could compound small local changes. Large-variance directions could carry
native preferences that are imperfect on these passages. The results do not
separate these explanations or identify a single semantic feature. Fresh
calibration and layer-isolated interventions are the next discriminating
tests; no rank is to be selected and retested on the same held-out passages.

No speedup was tested: the patch still computes native attention and then
modifies its output. Teacher-forced loss and agreement do not establish
generation fidelity, an efficient replacement, or transfer of OLMo's exact
single-weight pulse to GPT-2.

## Numerical conditioning and audit limits

All 744 main checks and 10,240 main/control gate checks passed. Independent
review recomputed all 680 raw-record reductions, 672 saved suffix closures,
640 source identities and 40 predetermined FP64 attention snapshots. Maximum
FP64 snapshot reconstruction error was 2.9832e-5; the saved FP32 delivery
formula was exact. The largest runtime stable-conditional reconstruction
error was 4.7684e-5, below the unchanged 1e-4 guard. Current/fullrank and
repeat/identity closures remained exact, and final parameter/map integrity
checks completed.

This success has a material conditioning limit. The smallest current
non-source mass was 5.11798e-11 and the largest target/current gain was
1.9538970624e10 (Wiki window3, BOS, constant-mean-logit map, layer4).
That worst case lies outside the predetermined raw-QKV snapshot input; it
has passing runtime reconstruction checks, not a separate FP64 snapshot
reconstruction. No denominator clipping was used. Some target sigmoid outputs round to one;
the main active interventions contain 244,318 such head/query endpoints and
zero target zeros. Passing these checks on this checkpoint and machine does
not certify arbitrary precision, new inputs or kernels.

The reviewer used 1,211,524 KiB peak RSS and no model/GPU. Successful full
prefix logits and contexts are partly represented only by runtime equality
flags and independent hash/shape/dtype records. Saved raw snapshots verify a
fixed subset; absent arrays cannot be reconstructed by an audit. Bootstrap
intervals condition on the eight calibration passages and the small evaluation
panel, with no multiple-comparison correction or pretraining-independence claim.
