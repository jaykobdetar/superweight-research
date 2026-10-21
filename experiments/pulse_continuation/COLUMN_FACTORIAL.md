# Frozen column-component factorial

MLP factor interventions change the entire down-projection column1710, whereas
super-weight deletion changes only its row1764 contribution. Preliminary scalar
factor experiments show this distinction may matter strongly at a later induced
pulse. Directly test the column split, without changing any numerical weight.

Use existing16 held-out512-token windows (four/corpus, fresh-confirmation first4).
At native source0 use natural/The510/period15 identities, score targets2..511.
At a later boundary use The/period, prefix128/256 and the same suffix/254 targets
as upstream. Boundary worlds: ordinary history, L0/L1 boundary self-only, and
ordinary history with isolated L1/H12 donor context. Original source0 stays intact.

For each world evaluate clean, row1764 contribution removed, complementary2047
coordinates removed, and whole neuron1710 column contribution removed. Implement
all using native F.linear with a copied matrix whose column1710 is zero, selecting
only the specified token's output rows. No model parameters are mutated. This
2x2 factorial distinguishes loss from one coordinate's removal versus the full
neuron's output. Whole-column deletion does not eliminate the entire MLP.

Verify row-removal against the established Source deletion implementation, and
retain original pre-down-projection activations unchanged. Capture L2 normalized
states and clipped source/boundary keys/values for descriptive interpretation.
The source0 and boundary panels have different histories/targets and are not a
paired causal estimate of position itself. Reused evaluation panels are
exploratory. Preserve all conditions and paired uncertainty; no selection.
