# Independent value-interaction contrast review

`value_interactions_audit.py/.json` independently reconstruct all 16 derived contrasts from raw token losses and replay their paired bootstrap intervals. No model/GPU was loaded; peak RSS 513,112 KiB. No numerical correction required.

With every source key fixed to its development mean, joint all-layer value removal exceeds the sum of 14 individual-layer removal effects in every evaluated window (32/32). Corpus means [95% paired bootstrap interval], in nats, are Wiki 4.63601 [4.19297, 5.01561], Hume 0.16886 [0.13640, 0.19889], Plato 0.17798 [0.12396, 0.23695], and Python 1.24221 [0.34078, 2.90352]. Python's broad interval reflects one especially large window (6.93278 nats).

The eight-corner third-order contrast is correctly signed for **value-presence coding** (each region present = 1). Switching to removal indicators reverses its sign. Presence-coded means are 0.33859, 0.01118, 0.01305, and 0.10701 nats in Wiki/Hume/Plato/Python; all corpus-level intervals are positive, although only 4/8 Python windows are positive.

These are finite intervention interactions in a fixed-key world. They demonstrate nonadditivity of the measured losses; they do not identify a unique compensating circuit, establish semantic content in the constants, or provide a fraction of the natural deletion effect mediated by a region. Reused eight-window panels and exploratory analysis scope should remain explicit.
