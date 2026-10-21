# Frozen upstream pulse-production experiment

21 October 2026. No upstream head screen outcomes have been inspected.

Question: which boundary-token attention head outputs in layers 0 and 1 mediate
ordinary-history versus self-only-history pulse production? This is distinct
from the already-studied downstream first-token interface.

World O uses ordinary causal attention. World I masks earlier keys ONLY for the
boundary query in layers 0 and 1. Capture each world's 32 head contexts at that
boundary, before output projection. In each recipient world transplant selected
head contexts from the other world, changing no other token's contexts directly.

Development: eight existing Wiki development windows (18432 through 25600 step
1024), 512 tokens, boundary256, first and boundary ID The510. Rank 32 single-head
O->I patches by mean signed fraction of the pulse gap recovered. Preserve all
screen results. Freeze top1/2/4/8/16/32 and one per-layer-count-matched random set
per size (seed20261021), independently sampled without excluding selected heads.

Evaluation: first four saved fresh-confirmation windows from Wiki validation,
Du Bois, Hobbes and optimizer code. Those are held out from this screen, but
have been examined in other intervention families; no claim of untouched global
confirmation. Shared Gutenberg final windows are not included. Fix the suffix
to original tokens256..511, with its first ID replaced by The510 or period15.
Use preceding128 or256 real tokens and set each available source0 to the same
ID. No retokenization. Score predictor boundary+1 onward: the same254 target IDs.

Run both donor directions for all frozen sets. Include both worlds, all32 closure,
layer-only sets, and deletion of ONLY the boundary super-weight contribution
within both worlds and selected top2/top8/all32 O->I patches. These deletions
measure conditional dependence on the induced pulse; head patches change other
features too. Source0 remains intact. Record full pre-down-projection8192D
intermediates at source0/boundary, pulse amplitudes, per-token loss/argmax/KL,
raw donor contexts, source hashes, model integrity, and resource accounting.

Expected exact controls: source0 and boundary intermediates match in world I
(same ID, self-only early attention); all32 donor patches reproduce the donor
world boundary intermediate and later model outputs. If these fail, stop and
resolve before scientific interpretation. Natural head necessity or a unique
mechanism does not follow from patch sufficiency. Interactions may be nonadditive.
