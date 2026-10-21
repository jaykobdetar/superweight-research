# Independent GPT-2 writer-factorial numerical review

Status: passed. `gpt2_writes_numerical_audit.py/.json` independently verifies360 records/forwards, eight exact native/identity anchors,88 summary groups,192 individual conditional contrasts,48 paired summaries and10,800 geometry rows. Model parameter hashes/versions are unchanged. Peak RSS865,716 KiB (about0.83GiB), CPU-only, no model load.

Private-source processing and tokenizer offsets were replayed independently. All16 third-panel inputs and stored manifests reproduce exactly, with each used512-token span disjoint from both earlier panels' eight spans per corpus and from other new spans. The fixed fractions were not changed. Scoring uniformly covers targets2..511; source predictor0 is excluded. All condition keys and saved per-token NLL/KL/argmax reductions agree.

Full source vectors provide stronger checks than normalized summaries. Every captured component modification matches its declared coordinate operation. All3,240 later-block full-vector residual additions reconstruct exactly as `(previous residual + attention output) + MLP output`. All640 E+A+M target-coordinate zero closures pass, including the random-pair control. The352 operation records each contain30 successful later-token locality checks (10,560 runtime assertions). Saved source vectors independently verify source-coordinate specificity; later-token vectors are not retained, so that part relies on runtime assertions plus static/synthetic tests. Rescale replay uses a2e−6 relative tolerance for CPU/GPU norm-reduction differences; observed norm preservation error is at most2.38e−7. Other vector operation/addition checks are exact.

The capture-scope wording correction is present: source residuals are retained for all12 layers, attention/MLP outputs for layers3..11. The actual erasure is afterL2; subsequent A/M operations zero those coordinates in layer3..11 output writes before residual addition.

The evidence supports conditional MLP-mediated **regrowth/reappearance of erased raw coordinates**, not restoration of the original vector or trajectory. After E, both coordinates are zero afterL2 and become positive afterL3 in every input: coordinate447 ranges457.95–557.91 and138 ranges54.85–68.38. Natural-start mean L3 values are approximately[535,66], versus native[2653,771]. The mean final-L11 values after E are approximately[−1121,−341], versus native[378,6]. Native trajectories are therefore not recovered. All-natural L3 MLP write means are approximately[540,66], while attention contributes[−5.47,0.23], directly locating early regrowth in the measured MLP output. The intervention spans all nine later layers, so its loss effect cannot be attributed uniquely to L3.

Natural-start PPL ratios relative to the same input's native condition:

| Intervention | Across four corpus means |
|---|---:|
| Erase E only |1.0454–1.0994|
| E plus MLP-write block |5.4643–37.6870|
| E plus attention-write block |1.0439–1.1021|
| E plus both write blocks |3.2481–22.2404|
| MLP-write block without E |1.0011–1.0037|
| Random pair E+A+M |0.999972–1.000057|
| E then rescale remaining vector |3.8534–6.8918|

BOS is more sensitive: E+MLP-write block gives31.29–549.39 and E+both gives20.77–262.94; E alone1.073–1.349. Report both start conditions rather than silently generalizing the natural range. The paired E×M difference-in-differences is positive in all32 input comparisons, with every corpus/start paired-bootstrap lower bound positive. This supports conditional dependence on the MLP writing operations after erasure, not their necessity in the clean recipient.

Blocking attention in addition to MLP after E **improves** loss relative to MLP-only blocking in all32 inputs (paired changes−0.805 to−0.086 nats). The combined lesion is thus not additively interpretable. The random pair is a frozen negative control, not a magnitude-matched coordinate set or proof of unique necessity.

Rescaling after E increases loss in every input despite broadly similar early MLP coordinate writes. For natural starts the mean L3 MLP writes are[511,69] under rescaling versus[540,66] under E alone. This control changes all remaining source components' magnitude and their later residual trajectory; it does not isolate a universal LayerNorm principle. Similar early regrowth alone is insufficient for fidelity.

The conditional effects are well supported within this prescribed family, but they do not identify a minimal repair circuit, an individual responsible neuron, or a unique general mechanism. Four passages per corpus, teacher-forced512-token contexts, one FP32 checkpoint and fixed start-token choices limit generalization. Broader self-repair and massive-activation claims require explicit prior-art attribution; novelty should be scoped to the specific coordinate/writer factorial and its cross-model comparison.
