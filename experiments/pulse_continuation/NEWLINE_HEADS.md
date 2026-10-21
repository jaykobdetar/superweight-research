# Frozen development/held-out newline pathway test

The preceding controlled-history batch establishes that an earlier token187
suppresses the pulse at a later187, and removing only L0 attention edge(q,p)
restores it. It does not identify a head, separate routing from value content,
or establish whether the restored pulse can compensate for deleting the earlier
one. This batch tests those distinctions.

Development: eight old Wiki and eight old Hume development windows, never the
confirmation texts. At p64/q128 put187/187, remove IDs187/535 elsewhere through128,
set source0=The510, preserve suffix129+. Screen each of16 L0 heads by deleting
only its(q,p) edge. Include clean and all-head edge deletion. Rank heads by mean
signed fraction of the clean-to-all-head q-pulse change. Freeze the top singleton;
control H0 unless top isH0, then H1. This selection uses pulse, not held-out loss.

Held-out: sixteen new512-token windows, four/text. Non-code starts16384,18432,
20480,22528; code4096,5120,6144,7168. Verify disjointness from all confirmation
calibration/evaluation and later-use windows. Source files/hashes and token counts
must match their existing manifests. These are new windows, not new corpora.

Use two worlds: U has space209 atp and187 atq; R has187 atp and187 atq.
Construct the same way as development. An additional R input with q535 supplies
only a layer0 query donor. All interventions score identical targets130..511.
Each world uses its own clean baseline for output effects.

In each world delete(q,p) edge for selected head, control head, all heads, or
selected head sham(q,p-1). Reconstruct the selected/control head's q context via
native SDPA as numerical oracles. Separately swap K, V, orboth at p from the
opposite world's raw L0 projections, but apply the resulting attention output
only at q in the selected/control head. Other queries and all other heads stay
native. These are local computational interventions, not globally changed token
identities. Also swap only q's query from the q535 donor. Retain exact checks that
L0 queries atq and captured source0/q projections match across U/R.

In R, delete target weight row at p, q, orboth, with and without the selected
edge removal. This tests whether inducing a second pulse compensates for the
earlier row deletion, and whether compensation requires the induced q row.
Neither restoration of pulse amplitude nor lower loss alone establishes complete
output fidelity. Preserve KL, argmax, per-token loss, pulse/factor traces, masks,
oracle errors, source identities, hashes and resource accounting.

Expected880 forwards:16development*18 +16heldout*37. Paired-window intervals are
pointwise exploratory n4/text, conditional on the frozen development selection.
No weight sweep, parameter mutation, duplicate GPU model or concurrent GPU job.
