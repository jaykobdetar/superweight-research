# Coordinate attribution correction — 2026-10-21

Independent literature review found that Yu et al., *The Super Weight in Large
Language Models*, v2 Table 2 lists **both OLMo-1B weights in layer1**:
`layers[1].mlp.down_proj.weight[1764,1710]` and `[1764,8041]`.
Source: https://arxiv.org/html/2411.07191v2#S3.T2 (checked2026-10-21).

The earlier `two_units.py`/`two_units_run` actually intervened at first
`[1,1764,1710]` and second `[2,1764,8041]`, exactly as its executable code and
manifest record. Its description as two *known* or *paper-listed* super weights
is incorrect. Its measured conditional dependence at the layer2 coordinate is
an exploratory finding about that coordinate, not a replication of the paper's
pair. References in report5 and any later summaries to a known layer2 second
weight must be read with this correction. Immutable historical Library reports
and archives are preserved; current reports and repository must link this erratum.

No change to the main target `[1,1764,1710]`, H14 newline-history interventions,
source K/V tests, or row-versus-column measurements follows from this correction.
`same_layer_pair.py` separately tests the correctly cited layer1 pair. Because
these two columns share an MLP, removing one at its down projection cannot alter
the other's intermediate activation in that same forward pass. The old layer2
activation response must not be attributed to same-layer adaptation.
