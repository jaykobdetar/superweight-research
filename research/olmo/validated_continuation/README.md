# Completed implementation validation and synthetic-prompt baseline cutoff

Cutoff: 2026-10-23 06:22 UTC. Exactly 64 recipient baseline/source-access forwards are included. Zero donor baselines and zero content-donor intervention forwards are included. The 24-window natural-text factorial is pending and intentionally has no outcomes here. This snapshot must not be described as a donor null or as a completed fresh factorial.

The exact pinned FP32 full forward passed all 10 predeclared archived-output validation comparisons. Independent source/math/masking audits passed. The initial uncentered-versus-centered prefix comparator failure is retained along with the corrected comparator. FP32 output is close to archived FP16, not bitwise identical.

All 16 synthetic recipients favor the correct label in native and source0-masked worlds. Only 3/8 native pairs and 2/8 source0-masked pairs satisfy the separate predeclared newline-access-usage gate. See content_baselines/BASELINE_FINDINGS.md. Those outcomes justify a qualified follow-up, not a semantic payload conclusion.

No model weights, natural-text token arrays, tokenizer vocabulary, full corpora or hidden/value/probability caches are included. Cache SHA receipts and independent checks are retained, but omitted cache bytes cannot be independently rehashed from this compact export. Synthetic prompts were created for this study and are retained in the audited frozen manifest.

Some code preserves original cloud workspace paths for audit fidelity and requires relocation plus privately acquired inputs/weights to run. The core source SHA, protocols, reviewed runner/algebra snapshots and receipt identities are authoritative. Official Transformers source is referenced by the audit but not redistributed here; see https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/olmo/modeling_olmo.py . Model: https://huggingface.co/allenai/OLMo-1B-0724-hf/tree/d7cbab742d80589e714b1a2d7f838dcd21cbe143 .
