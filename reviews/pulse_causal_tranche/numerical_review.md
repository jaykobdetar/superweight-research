# Independent completed-result review

**Scope:** After the three global-deletion replication anchors, the new causal,
direction, replacement, and generalization experiments delete the first target
only at position 0. Later-position uses of that scalar remain intact. Conclusions
about these experiments do not automatically apply to global scalar deletion.

The completed run manifest is timestamped 2026-10-21T06:58:51Z. Independent CPU
replay passed for all 2,652 observations, 255 summary rows, and 642 paired bootstrap
contrast rows. Raw per-token means match the window file; paired token hashes
match; model/parameter invariance flags and the frozen runner hash match. All K/V
identity controls are bitwise equal in loss and prediction. Every directional
patch preserves recipient losses and predictions at predictors 0–8 exactly.
Final-block 15 full rescue and damage reproduce donor losses and predictions at
predictors 9+ exactly, resolving the practical full-control precision concern for
these measured conditions. Peak review RSS was approximately 782MiB; no model or
GPU was loaded. Scripts and machine-readable evidence accompany this note.

## Findings that survive independent replay

- L2/H15 position-zero K/V damage is strongly nonadditive: K-only+0.048855,
  V-only+0.672986, jointly+5.455206 nats against clean. The paired interaction
  `joint−K−V+clean` is+4.733365, pointwise95% interval[4.554718,4.911274].
  Rescue is asymmetric: K-only−3.244339, V-only−4.264962, jointly−4.909794 against
  deletion. Its interaction is+2.599508[2.291791,2.892599], indicating overlapping
  rather than additive rescue benefit. Neither calculation is a mediation fraction.
- Late-head position-zero effects are small compared with replacing full contexts.
  Joint K0/V0 damage is+0.008476 for L12/H5 and+0.001295 for L15/H11; full-context
  damage is+0.154832 and+0.299576. Full-context clean-donor rescue at L12/H5 actually
  worsens deletion by+1.078740 nats, while L15/H11 improves it by−0.233229. A clean
  component transplanted into a disrupted recipient is not guaranteed to help.
- Covariance arithmetic, sampled entries, orthonormality, spectrum/trace equality,
  singular-value scaling, and selection-threshold replay passed. The mean accounts
  for 91.64%, 59.71%, 32.43% of development delta energy at layers 2, 12, 15. Held-out
  *centered* PC8 retention is 40.30%, 14.51%, 13.75%; PC32 retention is 47.48%, 24.79%,
  21.89%. These do not imply that the total perturbation is low rank. PC patches
  use raw delta projected onto centered-covariance directions, so their causal
  effects and centered retention are distinct measurements.
- Mean-only rescue illustrates that difference: layer 2 improves deletion by
  −4.194435 nats, whereas layer 15 worsens it by+1.710809. Mean-plus-PC8 and PC8
  alone are different interventions; their effects should not be added.
- Selected constant K/V0 replacement approximately preserves clean NLL
  (−0.000411 nats) but worsens pulse-state NLL by+0.001561. Replacing selected
  complete head contexts loses+0.047555 nats in clean and+0.048864 in pulse runs;
  same-layer controls lose+0.015270 and+0.014832. This supports narrowly localized
  constancy, not an exact redundant-head claim or measured compute savings.
- Fixed-token fresh calibration has zero observed dispersion and identical values
  across the two corpora. Fresh-pulse mean NLL matches clean at lengths 128/512;
  small EOS/period differences appear at 1024, while `The` remains matched. Natural
  starts retain residual mean NLL differences. Shape-dependent FP16 computation
  is a plausible explanation for the 1024 discrepancies, not an independently
  isolated mechanism. Do not claim exact universal restoration.

All intervals above bootstrap paired windows with 2,000 fixed-seed resamples.
They are pointwise and conditional on chosen corpora, nominees, calibration,
development fits, and model. The generalization panel reuses eight held-out starts;
fixed-token calibration repeats are not 16 independent contexts at position 0.
The continuation/reset diagnostic also changes the intervention source's identity
and distance, so it cannot isolate reset alone.

No blocking implementation or arithmetic issue remains for the measured result
set. Remaining limits are scientific scope and precision of interpretation.
