# Independent fixed512 implementation review

**Ready for the frozen 1,352-forward run. No launch blocker found.** This is prospective implementation approval, not a completed scientific result. V1 and V2 remain failed and unchanged.

Frozen files reviewed:

- `gpt2_gate_fixed512.py`: `23c9104c4b80f72c13daee601373a4f026b6967bc21124411a85ec5f79cff1d0`
- `analyze_gate_fixed512.py`: `3a76791f0fec956765c7304a1b8459b101cff88c0d1e4a7710a73ddd11d9c6bd`
- `GPT2_GATE_FIXED512.md`: `75a567b57927bf15e2f04b34be4d9ab6ddd94c3ebddfb47573fd525150cba2d7`

The implementation matches the approved fixed-shape proposal. Every forward asserts length512; there is no short execution, padding, relative-prefix call or hidden-error tolerance. The runner imports the exact previously reviewed GateV2 and control-scorer functions, without calling their original main entry points. Frozen map/raw hashes, input identities, selection/checkpoint revision, source provenance, default precision settings and final model/map integrity remain enforced. There is no new calibration fit or selection.

All21 conditions,32 natural/BOS inputs, original gate directions/ranks/controls and both C/P fidelity references remain. Native-C versus P distinctions are preserved: source V zeroing changes query0; subsequent gate edits preserve query0 relative to P. The same512 main metrics score targets2..511; early targets2..127 and later128..511 partition the510 full targets into126 and384. They are overlapping-data views of one execution, not independent samples or a changed-history experiment. All672 suffix controls retain exact prefix comparisons and the unchanged1e-4 reconstruction guards; changed target128 and later are not gathered by their scorer.

Counts agree statically and in the full fabricated analyzer fixture:680 main records,672 separate suffix controls,1,352 forwards/attempts on success,744 main runtime checks,10,240 gate checks,40 predetermined snapshot states and680 saved original-prefix references. Summaries have168 full groups and336 early/later groups;32 contrasts per input produce1,024 full raw contrasts and256 grouped contrasts. The unexposed view retains actual n3 for Wiki and n4 elsewhere, with960 raw contrasts. Controls do not enter scientific summaries. Both full and unexposed views use the frozen2000-resample seed; these intervals condition on calibration and do not adjust for multiple comparisons.

Failure artifacts now separate latest model-forward identity, active record target and active check target. Pending record/reference bundles are retained where applicable, while current hook tensors retain their actual forward identity. Resource fields, failed status, attempted/returned counts and partial records are saved; no automatic fallback follows. Successful full logits/contexts remain partly hash-only, with raw snapshot coverage and explicit saved-audit limitations.

## Focused independent evidence

`gpt2_gate_fixed512_independent_flow.py/.json` passed. It confirms the actual hook/scorer object aliases are unchanged, no cross-length evaluator is imported, and full/early/later target slices partition correctly. Three separately injected cases passed:

1. C recording fails after the P forward:2 returned/attempted,0 records; current tensors identify P while the preserved pending record bundle identifies C.
2. The first suffix hidden comparison fails:3 returned/attempted,1 main record,1 failed suffix record; both original/control logits and hidden states are retained, and the active check is the suffix comparison.
3. The suffix forward raises before returning:3 attempted versus2 returned,1 main record,0 suffix records; original pending bundle/logits remain, and no nonexistent current hidden output is claimed.

These orchestration fixtures adapt the author's credited fake model/probe classes; added failures and assertions are independent reviewer work. They load no model/tokenizer and use no GPU. Peak RSS was800,720 KiB. The CPU profiler emitted a device-enumeration warning with CUDA disabled; no GPU inference or model loading occurred.

`gpt2_gate_fixed512_independent_analyzer.py/.json` passed the full fabricated680-main/672-control schema and checks all stated counts. Independent added adversarial cases reject failed status despite completion metadata, a physical-length list containing128, reintroduced hidden tolerance, wrong score bounds, and disagreement between separately recorded original/control logit fingerprints. Existing scalar and raw-closure corruptions also reject and stale passed audit state is cleared. Actual-n3 bootstrap is numerically verified. This fixture credits the author's full-panel generator. Peak child RSS was876,848 KiB (parent832,820 KiB).

The unchanged attention arithmetic already passed independent random-weight Transformers CPU integration in `gpt2_gate_v2_independent_tests.py/.json`:21 model forwards, nonzero QKV/output biases, ten representative exact suffix-mode pairs, current/fullrank copies, all20 gate maps' query-locality, independent FP64 KL and hook cleanup. No repeated pretrained/GPU test was performed for this orchestration-only change.

Independent numerical audit remains required before any gate efficacy, rank or fidelity interpretation. This approval does not predict that the pretrained run will satisfy every frozen exact/reconstruction guard. No repository publication or GitHub action occurred.
