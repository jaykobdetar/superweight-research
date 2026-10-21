# Unedited-text validation of the newline route

Use the original sixteen newline-heads held-out512-token windows without any ID
edits or retokenization. Eligibility is fixed from tokens alone: at least two
ID187 occurrences with second position<=255. Setp/q to the first/second187.
Twelve book/code windows qualify; all four Wiki windows have no187 and are
reported ineligible, not silently replaced. Two DuBois windows start with187;
their p0 cases are source-position cases, not later first-occurrence cases.

Freeze L0/H14 and controlH0 from the prior development selection. Compare clean,
onlyH14 edge(q,p) removed, controlH0 edge removed, andH14 nearby nonnewline sham
(p-1 whenp>0, otherwiseq-1; verify sourceID!=187). Only the selected head/query
edge changes. Standard explicit causal masks match all conditions.

To test whether L1/H12 carries the upstream H14 effect, swap its q context
between clean and H14-edge-removed worlds in both directions. Separately replace
that context with its current clipped value atp orq in both worlds. Context
patches can affect more than one neuron; do not treat pulse as the sole mediator.

In each clean/edge world remove the target row atp, atq, orboth. Also remove the
whole neuron column atp/q in clean andatq in edge world. These test natural-text
conditional dependence and distinguish one-coordinate from whole-neuron effects.
For each deletion assert the first-neuron intermediate is unchanged from its
corresponding world. Capture L0 selected-head routing, H12 current/applied contexts
and values, and first-neuron gate/up/activation/pulse traces.

Score predictorsq+1..510, targetsq+2..511, matched within each window across all
conditions. Target counts vary across windows; aggregate window means with
paired bootstrap n4/text. These same held-out windows have already informed prior
mechanism panels; this is an unedited-input validation, not a wholly untouched
confirmation. No fitting or selection from output effects. Expected228forwards
(19 per eligiblewindow). Retain all negative/heterogeneous results.
