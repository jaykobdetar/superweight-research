# T19 draft: does the L10 gate effect survive native source values?

Status: prospective protocol revised after independent design review; not approved for model execution. No T19 outputs have informed these changes. T18 independently
established a scoped L10 pattern under zero source values. Its Ethics improvement
largely returns loss toward the intact C baseline. The next causal question is
whether the gate-only change has the same function when native values remain.
No T19 outputs have been generated or used to select this design.

## Fixed intervention and factorial boundary

Use the pinned GPT-2 FP32 checkpoint/backend and exactly512 physical positions.
Keep the T18 fresh Wiki4 center/directions frozen; no new calibration or axis
selection. This explicitly tests transfer of that fitted operation to another
value boundary, not whether a separately calibrated native operation can work.

Cross three binary settings, independently of the L10 gate intervention:

| Factor | bit1: zero | bit0: native |
|---|---|---|
| U: upstream source values | Zero projected V at position0 in L4–L9 | Retain native V |
| L: local source value | Zero projected V at position0 in L10 | Retain native V |
| D: downstream source value | Zero projected V at position0 in L11 | Retain native V |

Each bit is0 for retaining the native projected V slice and1 for zeroing it at position0 for every head. U/L/D=000 is the intact boundary;111 is T18's P boundary. Native V is computed on that boundary's live trajectory, not transplanted from C. Parameters are never
changed. The hook changes only the named source V entries. Q/K/non-source V
can respond downstream; they are not directly patched. Preserve original
source identity within each gate trial relative to its own boundary baseline.

At each of the eight boundaries, run baseline, a zero-delta current control,
FF PC1 removal and its fixed matched orthogonal control at L10 only. For q>0:

```
b = logsumexp(non-source attention scores) − source score
a_diag = sum(reconstructed_softmax[...,1:]) # SDPA does not expose internal probabilities
a_ideal = sigmoid(b)                    # separately compare with a_diag
s = (b − log(q) − alpha_F) dot u_F
b' = b − s*u_F                        # removal
b' = b − s*v_F                        # orthogonal control
c_ideal = a_diag*u + (1−a_diag)*v0                    # local attention decomposition
c' = c_actual + (sigmoid(b')−a_diag)*(u−v0)
```

Here u is the conditional non-source value mixture, not the PC1 direction u_F.
Compute u from a non-source-only normalized score distribution for numerical
stability. Leave q0 unchanged by the gate edit. Source-V boundary changes may
change q0 relative to C. The current control must use zero delta and return the actual context without arithmetic, reproducing its boundary baseline exactly. If a scientific map is exactly unchanged (torch.equal for all b), it uses the same exact no-change bypass; no near-zero threshold applies. Verify the native decomposition, conditional
mixture, actual SDPA context and delivered context under prospectively frozen
bounds before any outcome interpretation. Direct source probability and1−a_diag are separately measured, not assumed identical in FP32. Finite saturated a_diag at0or1 is allowed by the delta formula; no clipping/epsilon substitution. Legacy scaling still requires strictly positive mass.

The delta formula is algebraically equivalent to T18's scaling formula at111,
but has different FP32 rounding. Add two explicit111 bridge arms using the
unchanged T18 scaling arithmetic for FF removal and orthogonal control. Compare
all per-target losses/KLs/predictions, local context errors and actual output
differences; do not call them byte-identical unless demonstrated. Freeze a
reasonable numerical acceptance policy through CPU fixtures and review before
the pretrained launch, not after observing output drift. Preserve any failure.

### Prospective arithmetic bridge acceptance

Adopt the independent review's following policy before any T19 model output. Every input runs its two111 bridge arms early. Require manual-context and conditional-reconstruction max absolute error≤1e−4; bridge delivered-context max error≤1e−4; all scored target-loss differences≤1e−3 nats; and absolute per-passage mean loss difference≤1e−5 nats. These are scientific continuity thresholds, not sensitivity guarantees. Report both KL directions, hidden/logit max/RMS/relative drift and all argmax mismatches with margins. Do not demand argmax equality at near ties. Any hard criterion failure halts the planned accepted batch, retaining both sides; no post hoc tolerance relaxation or input exclusion.

Random CPU fixtures must cover tiny masses, saturation, large gains and cancellation. In the zero-V idealization, let e=c_SDPA−a_diag*u and g=a_target/a_diag; then c_scale−c_delta=(g−1)e. This diagnostic explains why algebraic equality alone is insufficient. Current/observer/source/suffix/causal-order checks remain exact and cannot use these approximate bridge tolerances.

At each shared upstream state also verify the direct delta identity: delta_c_native−delta_c_zero=−(a_target−a_diag)*v0_native, with FP64 diagnostics and separately bounded reconstruction residuals. Save pre-/post-boundary fingerprints so a local V edit cannot hide an incoming-state change.

## Identification checks and estimands

All local-zero/native conditions with the same U must have identical incoming
L10 Q/K, non-source V, b and intervention coefficient: changing L changes only
V0, and D occurs later. All D variants share the same L10 delivered state for
the same U/L/gate operation. These are causal-order checks, not estimated
correlations. At fixed U and gate coefficient, the direct difference between
native and zero source contributions is prescribed by the formula; downstream
responses remain live and must not be labeled a mediation fraction.

For each family and boundary, the primary within-passage estimand is intervention
NLL minus its own boundary-baseline NLL. Compare000 with111, then compute the
local L, upstream U and downstream D interaction contrasts at every setting of
the other two factors. Also compare removal to the matched orthogonal change.
Report native-C loss, KL and argmax fidelity separately from boundary-relative
loss. A negative loss effect is not output preservation or generation quality.
Do not infer a unique circuit, feedback or universal native-value harmlessness
from any factorial interaction.

The clearest competing predictions are: (a) the L10 sign pattern persists at000,
extending the scoped operation to intact GPT-2; (b) it depends on local zeroV,
supporting a boundary-specific allocation/value interaction; (c) earlier or
later zeroing controls the sign, implicating a broader altered state. Partial
or corpus-specific outcomes remain informative. The orthogonal control still
matches only gate-logit norm at each own incoming state, not context norm.

## Fresh panel, counts and review gates

Select16 new passages before any run: four each Wiki validation, Utilitarianism,
Ethics and optimizer code, with exact source/processed hashes and full current
history including T18. Use the existing deterministic selection/rejection rules;
do not relax exclusions if capacity fails. Natural/BOS token0 replacement is
paired without insertion or retokenization. Record start identities rather than
asserting generic unseen-token transfer. All2,040 scored targets per stratum
remain clustered in four passages. The prespecified paired bootstrap is2000
draws,seed20261021, with the same passage-index draws across arms and paired starts; no population calibration uncertainty or multiplicity
correction is claimed.

Eight boundaries times four operations plus two bridge arms make34conditions.
Thirty-two input/start pairs give1088main+1088suffixforwards, plus8 native
repeat/writer-identity anchors: **2,184 forwards**, no calibration. Every suffix
control executes512positions with tokens128–511 rotated; unchanged-prefix
checks remain exact. Main scores target2–511; early/later views share execution
and do not substitute for length interventions.

Preserve per-targetmain/suffix data, all operation/identity checks, local
coefficient/logit/mass/context norms and boundary source fingerprints. Preselect
one natural Wiki input for bounded full attention and actualGPUreadout snapshots;
final member counts and complete output/failure sizing must be measured and
reviewed. Review workers stay below3GiB; a single GPU model worker runs only
after checking other jobs. Maintain2GiB disk reserve and a separately measured
failure allowance. No paid compute, credential changes or unrelated jobs.

Required before launch: independent design review; separate implementation
without changing frozen T18 sources; random CPU hook fixtures and hostile
identity tests; exact source/input freeze; explicit bounds for the arithmetic
bridge; complete storage accounting; independent final implementation review.
No new model run is authorized by this proposal itself. The existing user
authorization permits proceeding through these local scientific checks.
