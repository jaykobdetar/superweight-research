## What the new experiment establishes

The layer-10 result survives a fresh calibration on these passages. Removing
the leading gate direction worsens Wiki validation loss and improves loss on
Utilitarianism, Ethics and optimizer code. That sign pattern holds for both
natural starts and a BOS replacement, for all four crossed Wiki-calibration
choices, and for the original mixed-calibration anchor. All 40 corresponding
conditional intervals exclude a perplexity ratio of one. These are correlated
comparisons on 16 passages, not 40 independent replications.

For natural starts, switching from old Wiki calibration to fresh Wiki
calibration changes the L10 perplexity ratios from **1.003560 to 1.003240** on
Wiki, **0.982135 to 0.982692** on Utilitarianism, **0.979451 to 0.980979** on
Ethics, and **0.985271 to 0.986016** on code. The reference, P, has source-token
values zeroed in layers 4–11; it is not the intact model. A ratio below one is a
loss improvement under this intervention boundary, not faithful output
preservation or evidence of better generated text.

The reference distinction changes the interpretation. On Ethics, fresh-fit
L10 removal gives perplexity ratios **1.000043 / 1.000783 relative to native
C** for natural/BOS starts, with both intervals including one. Its apparent
improvement over P mostly returns the mean loss toward C. Some other groups
beat C in loss, but those runs still combine source-value zeroing with the
gate change; they do not establish the effect of a gate-only change in the
intact model.

The matched orthogonal control matters. Across all 24 non-Wiki corpus/start/
calibration groups, L10 PC1 removal beats the orthogonal change, with each
paired interval below zero. The same is true for all 24 joint-layer groups.
Wiki is less decisive: only one of eight L10 and two of eight joint-layer
matched comparisons exclude zero in that direction. The control matches
gate-logit perturbation norm, not context-vector norm or semantic disruption.
This is evidence about the tested direction relative to this fixed control,
not proof that PC1 is unique among possible directions.

Jointly changing layers 4–11 gives a different result. All four Wiki-calibrated
PC1-removal choices harm Wiki and help Ethics for both starts, with intervals
excluding one. Utilitarianism and code retain intervals crossing one for all
four choices. Thus “removing this direction helps books” remains too broad.
For natural-start code, OO gives a ratio of 0.999304 and FF gives 1.002882,
while their argmax agreement with P is only 91.96% and 91.37%. Near-zero mean
loss change does not make either transformation an output-preserving
replacement.

## What changing the calibration actually changes

O and F each use four Wiki-test passages. We independently cross their
12-head centers and leading directions. In OO/OF/FO/FF, the first letter
selects the center and the second selects the direction. M8 is the earlier
four-Wiki/four-Hume fit, retained as a secondary anchor. OO is not M8, and M8
contrasts confound both corpus mixture and calibration sample count.

The old and fresh leading axes are close: absolute cosine ranges from
**0.932976 to 0.998835** across layers 4–11, and is **0.996647 at L10**.
These two fits have resolved leading eigenvalue gaps. This does not estimate
calibration-population stability; it is one comparison between two small,
fixed calibrations. A stable axis is also not a count of causal mechanisms.

The crossed interventions reveal a systematic difference between local and
joint effects. For L10 PC1 removal, changing only the center increases mean
loss at either direction in all eight corpus/start groups; all 16 intervals
exclude zero. The mean increases range from **0.000233 to 0.000729 nats per
target**. Changing centers jointly across layers 4–11 decreases mean loss in
all 16 comparisons, with mean changes from **−0.005942 to −0.001780**.
Changing directions jointly increases mean loss in all 16 comparisons; 12
intervals exclude zero. The eight L10 and eight joint center-by-direction
interaction means are positive, with intervals excluding zero. These are
finite interventions on a small panel, without multiple-comparison correction.

At a common incoming L10 state, the center swap has a simple identity:
the change in the proposed gate logits is
`((α_F − α_O) · u) u` for PC1 removal, or
`((α_F − α_O) · u) v` for the orthogonal control. It is a constant 12-head
offset across queries at that shared state. The runtime checks verify this
identity. The observed loss interactions can arise from the sigmoid,
downstream network and loss calculation; they do not identify an internal
feedback pathway. Joint-layer trajectories are different incoming states and
must not be substituted into this common-state identity.

## Findings, rejected explanations and remaining hypotheses

| Statement | Current status |
|---|---|
| The L10 Wiki-versus-other-source sign pattern requires the exact original mixed calibration. | Rejected on this panel: both four-Wiki calibrations and their crossed choices preserve it. |
| Calibration choice has no causal effect. | Rejected for these finite center/direction changes; their paired losses differ. |
| PC1 removal is interchangeable with an equal-logit-norm orthogonal change. | Rejected on the non-Wiki panel; downstream/context norm is not matched. |
| Joint PC1 removal is an output-preserving simplification. | Rejected: substantial argmax and distribution changes remain, even when loss is nearly unchanged. |
| A single leading covariance direction identifies one circuit. | Unsupported. Covariance, these interventions and their interactions do not establish uniqueness. |
| The L10 pattern also holds with native source values restored. | Open and the next prioritized causal test. |
| The result generalizes to generation, arbitrary prefixes, longer contexts or other checkpoints. | Open. No such claim is made by this batch. |

The next informative experiment restores source values while retaining a
precisely defined gate intervention. With native source value `v₀` and
conditional non-source mixture `u`, the local output is
`c = a u + (1−a) v₀`. Changing allocation to `a′` can be implemented as
`c′ = c + (a′−a)(u−v₀)`, with explicit rounding and identity checks.
Comparing that intervention with the current zero-value boundary would test
whether the scoped L10 effect survives in the intact computation. Local
source restoration and earlier-layer restoration should be separated; a
single all-at-once comparison would confound them. This is a prospective
proposal, not a result of the current study.
