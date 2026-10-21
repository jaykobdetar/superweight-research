# Research checkpoint and continuation

Checkpoint: 21 October 2026. The user requested a clean handoff to GPT-6.1 Sol to conserve usage. T18 is complete and reviewed; no T19 model experiment has started. This is a continuation checkpoint, not a claim that the mechanism is fully understood.

## Current evidence

Start with [Report14](GPT2_GATE_WIKI_CALIBRATION.md), the [audit/replay quickstart](GPT2_GATE_WIKI_CALIBRATION_QUICKSTART.md), and the [living hypothesis ledger](HYPOTHESES_WIKI_CALIBRATION.md). Earlier reports, the coordinate correction and primary-literature comparison remain linked from the repository README.

T18 completed 1,548 forwards: four fresh calibration, 776 main and 768 fixed-shape suffix controls. A fresh four-passage Wiki fit preserves the tested L10 pattern under the P boundary: source-token V is zeroed in L4–L11. L10 removal worsens Wiki loss and improves the three other sources across all four crossed center/direction choices and the M8 anchor. These are correlated comparisons on 16 passages, not 40 independent replications. With fresh FF calibration, natural-start PPL/P is 1.003240 on Wiki, 0.982692 on Utilitarianism, 0.980979 on Ethics and 0.986016 on code.

The native-model reference matters. Ethics FF L10 PPL/C is 1.000043 for natural starts and 1.000783 for BOS starts; both intervals include one. Its P-relative gain mostly returns loss toward C. Gate-only efficacy with native values remains untested. Joint L4–L11 effects differ, and near-zero loss changes do not preserve predictions. Old/fresh L10 PC1 cosine is 0.996647; this is neither a mechanism count nor population calibration stability.

The preceding T17 study decomposed finite output nonadditivity from nonlinear loss, including opposing components. It does not establish feedback, internal compensation or mediation. Earlier OLMo and Llama findings are scoped causal case studies, not transfer of a single universal mechanism. Gate algebra, attention sinks, massive activations and super-weight sensitivity already have primary-literature precedents; the candidate contribution is the particular controlled causal account and its limits. Preserve the documented second-coordinate layer correction.

## Validation and failure status

T18 has independent saved-state, numerical, limited readout, safe-export and preparation reviews. The NumPy auditor checks all exported numerical conditions. The raw numerical audit checked 4,032 summary groups, 912 paired contrasts and 162,432 norm reductions; the separate readout audit checked 44 saved rows using one embedding tensor. Neither is a second pretrained T18 replay.

The original saved-state audit failed because NPZ did not preserve QR storage strides. A separately reviewed layout-only adapter restored exact direction results without changing tolerances or frozen model-experiment code. The original failure remains preserved. The original mixed-corpus recalibration design failed its Hume capacity preflight before inference; the revised Wiki-only estimand is explicit. See [the failure ledger](GPT2_GATE_WIKI_CALIBRATION_FAILURES.md).

The standalone preparation reconstructs the required 20 windows with only GPT-2 tokenization and an existing validated M8 reference. Eight canonical files remain byte-identical, six portable metadata files change explicitly, and the 37-source staged runner closure is checked. Its execution path has not been exercised as a complete second T18 replay, and a standalone numerical replay comparator is still missing. The older fixed512 study did complete an independently executed 1,360-forward replay with exact raw/numeric comparison; do not rerun it unnecessarily.

## Next scientific experiment

The [T19 prospective protocol](GPT2_NATIVE_VALUE_BOUNDARY_PROPOSAL.md) tests L10 with native values. Keep the fresh T18 F center/direction fixed; do not recalibrate or search new axes. Cross three bits, each 0=native and 1=zero: upstream source V at L4–L9, local L10 V, downstream L11 V. Native V is computed on each live trajectory, not borrowed from clean C.

At each of eight boundaries, compare baseline, exact current/no-op, PC1 removal and the fixed orthogonal control. Add two111 legacy-scaling bridges. The proposed total is 2,184 forwards on 16 fresh passages with paired starts and fixed512 suffix controls. No T19 tokenizer selection, implementation freeze, model execution or scientific result is claimed by this checkpoint. The revised protocol addresses the independent design review but still requires implementation and launch review.

Use the stable context delta `c_SDPA + (a_target-a_diag)*(mixture-v0)`. The reconstructed softmax mass is diagnostic; SDPA does not expose its internal probabilities. Preserve exact no-op/source/causal-order/suffix controls. Freeze and test bridge limits before inference; a failed bridge halts the accepted batch rather than inviting a relaxed tolerance on the same exposed data. Retain every failure. See the [design review](../reviews/gate_native_value_design_review.md).

Before any launch: verify no existing worker, audit fresh source/history exclusions, implement in separate files without editing frozen T18 sources, test CPU fixtures and hostile identity cases, measure complete normal/failure storage, independently review code/input/source freeze, then use one GPU worker. Review workers should remain below 3 GiB RSS. Full factorial controls should not be removed simply to fit disk.

## Reproduction and resources

```bash
python scripts/audit_wiki_calibration_results.py
```

This needs Python plus NumPy 1.26.4, approximately 117 MiB peak RSS and no private inputs. Full preparation/execution commands are in the [reproduction guide](GPT2_GATE_WIKI_CALIBRATION_REPRODUCING.md).

GPT-2 checkpoint: `openai-community/gpt2`, revision `75e09b43581151bd1d9ef6700faa605df408979f`; full model tensor hash `2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527`. Primary runtime: Python3.12.3, PyTorch2.11.0+cu130, transformers4.57.6, NumPy1.26.4. Seed20261021; exact intervention/control seeds and precision settings are preserved in manifests and frozen source.

T18 used one8GiB RTX3060Ti, 262.6 seconds, 2.163 GiB peak host RSS and approximately1GiB reserved GPU memory. These measurements exclude initial checkpoint/runtime retrieval. The local working machine is nearly full; preserve a2GiB free-space reserve. Private raw evidence remains outside the committed payload. A proposed post-audit storage deduplication has only a read-only plan and synthetic tests; no real raw files were modified or removed. Its physical-redundancy and timestamp limitations require separate review before any use.

## Publication and continuation boundaries

Repository visibility is controlled by the user. The user intentionally made it public and subsequently authorized this research-only update at its current visibility. An earlier assistant restoration to private overrode that choice; the correction record preserves the mistake. Do not change visibility again or investigate the intentional setting. Keep weights, corpora, token payloads, credentials and unrelated files excluded from every future update.

The local continuation handoff contains machine-specific paths, Library identities, exact raw-output locations, pending storage plan and current process state. It is deliberately not published. No paid compute, credential/security changes, external messages, unrelated job interference or restart of paused UIT work is authorized. Use the existing reviewed artifacts and inspect saved jobs before restarting anything. The next agent should complete the pending resource/input/implementation work, then test the native-value boundary; the research remains open.
