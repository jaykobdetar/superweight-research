# Completed baseline qualification

All64 recipient baseline/source-access runs completed after the parent granted a CPU slot. No donor forward or intervention has run. The slot was released immediately on completion.

- Correct-versus-foil discrimination:16/16 native and16/16 post-boundary-mask0 recipients favor the correct label. All8 reciprocal pairs pass in both worlds.
- Native newline-p access is not generally necessary here. Mean correct-label NLL cost of removing p is−0.00584nats; only3/8 pair means pass the predeclared descriptive>0.01nat p-usage gate.
- After masking source0, the average p-removal cost is−0.06282nats, so an all-pair donor null would still be weak evidence. However, Mira red/blue and Emma summer/winter have clear reciprocal p usage in the mask0 world:
  - Mira: +0.297/+0.442nats, mean+0.370.
  - Emma: +0.603/+0.398nats, mean+0.501.
  - Emma's summer correct-minus-foil margin changes from+0.780 under mask0 to−0.256 under mask{0,p}.
- These two pairs are candidates for a baseline-qualified maskworld test, not evidence of a nativeworld semantic mechanism. Selection uses only frozen baseline/source-access measurements, not donor effects. All8 pairs and every result remain archived.

## Integrity and resource record

64forwards took156.39seconds summed model time; peak RSS424.13MiB (444731392bytes). RLIMIT_AS and model RSS cap were both4,000,000,000bytes, with2BLASthreads. Maximum manual observed-target logit discrepancy was7.63e−6. All64 NPZ cache checksums were independently verified. All16 recipient groups had exact per-layer prefix hashes across all4source-access worlds. Independent audit found no mathematical bug in scoring, mask boundaries, donor orientation, full routing replay or the layer8-only alternative.

Frozen forward code: run_content_donors.py SHA256 29a76fc2e8635b9d5e8b94bfde52107cf936152a089f56d22dcfb3f408749501. Existing results bind that code, donor algebra, original input manifest, protocol, explicit pre-output amendment and the passing10-case FP32 validation artifacts. No runner edit was made after evaluation began.

## Suggested bounded next stage, only after another CPU grant

A54-forward priority study would execute6donor baselines plus48interventions over the two qualified pairs, both reciprocal labels, both native/mask0 worlds and6conditions. The native world remains a comparison rather than being pooled with the maskworld. At the observed baseline mean this is about132seconds; intervention timing is not measured and may differ.

    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase donors --pairs mira_color,emma_season --cpu-slot-authorized
    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase interventions --pairs mira_color,emma_season --conditions identity,matched,mismatched,source0,p_from_source0,ordinary_raw --cpu-slot-authorized

The primary question is label-specific transfer, not damage. The unscaled fact-token control tests whether this fixed-routing assay can transfer content at an obvious carrier. Source0→p tests whether context-free values can supply p's contribution. Identity and cross-content source0→source0 are exact-null controls. Matched versus mismatched donors are reciprocal and independently phrased.

All344forwards of the expanded default study remain runnable;64are complete,280remain. An all-pair run should retain the separate discrimination and p-usage stratifications. No negative donor result should be treated as evidence that newline values lack payload when access to p is not useful in that case.

## Files

- frozen_manifest.json and contrast_protocol.json: original input/contrast freeze.
- preoutput_amendment_01.json: source-access baseline controls, ordinary_raw condition and interpretation clarifications, frozen before outputs.
- results/preregistered_v1/baseline_analysis.json: complete audit/resource summary.
- results/preregistered_v1/baseline_gates.json: separate discrimination and p-usage gates.
- results/preregistered_v1/baseline_metrics.csv: every baseline result.
- results/preregistered_v1/source_access_effects.csv: each recipient's p-access costs and source0×p interaction.
- results/preregistered_v1/native_routing_diagnostics.csv: per-layer final-query source mass and source0/p V cosine, descriptive only.
- delivery_receipt.json: machine-readable status, code hashes and exact counts.
