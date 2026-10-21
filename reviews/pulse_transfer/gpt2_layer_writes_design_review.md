# Independent GPT-2 layer-writer design review

No conceptual blocker in `GPT2_LAYER_WRITES.md`. This finite layer-set family follows the previous result without selecting a neuron or attributing the nine-layer effect to layer3 alone. The fourth input panel has a prespecified disjointness failure rule; no new fit or test-dependent selection is proposed.

The set count is consistent: eight E×block3×block4..11 cells, eight additional single-unblocked-layer cells, six additional blocked-prefix cells after deduplication, and two random-pair controls =24 conditions.32 inputs plus eight anchors gives776 forwards/records. Saving aliases and exact blocked-layer sets before evaluation is important for independently verifying that count.

The only-layer11 versus all-later-MLP-writes-blocked scored-output equality is an appropriate architectural negative control. The final-layer source0 MLP output has no subsequent cross-token operation; predictor0 is excluded. This equality must not be presented as a discovered empirical lack of importance. Attention writes remain native, so all-MLP-blocked does not require the coordinates to remain identically zero.

Report all layers and paired NLL interactions, as planned. A successful single-unblocked-layer condition demonstrates sufficiency only within this specific coordinate-write blocking family, with the rest of the model and attention still operating. It does not establish a standalone repair circuit or independence from earlier layers' other coordinates. The design already avoids native-trajectory and neuron-identity claims.

Implementation should state capture scope precisely: preserve full source residual vectors at all blocks and attention/MLP vectors at the layers actually instrumented, rather than repeating an ambiguous all-components/all-layers promise. Verify MLP-output hooks precede residual addition, apply only source0, and preserve all unselected coordinates; independently reconstruct saved residual additions. CPU-only implementation and numerical review remain pending. No model/GPU was loaded for this design review.
