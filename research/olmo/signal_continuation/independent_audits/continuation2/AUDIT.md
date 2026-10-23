# Alternate-writer downstream runner and feature-feasibility audit

Reviewed 2026-10-23 05:48 UTC. No downstream/model forward was run. The only weight reads were six existing 2048-element gate/up row slices to authenticate the saved compact row matrix (49,152 bytes total); no new weights or full matrix projections were acquired/executed.

## Downstream runner and frozen panel

The frozen four-panel, six-condition design is implemented correctly: two archived natural newline-source windows, each paired with a source-token510 replacement while all suffix tokens remain identical. Original512-token window hashes and truncated256-token input hashes independently match. The first later newline is at16 for DuBois and4 for optimizers. Every hook edits only L1 output at position0; no attention masks are passed.

The scientific2x2 comprises the native primary/group state, primary1710 removal, group-only removal, and joint removal. Untouched native versus reconstruct_native is a numerical control. Group same-world add-back after joint removal is compared against drop1710, not against native. Six conditions per panel give24 forwards. Fixed hook/runner hashes match the reviewed frozen protocol.

For256 input tokens, the loss vector contains255 next-token losses. Entry t predicts ids[t+1]. Printed loss[1:] means query indices1…254, target indices2…255, exactly the specified primary span. Query0 is available separately in saved arrays. The stored layer2 source0 K/V diagnostic is correct: apply LN to actual post-L1 source0 state, project and clip. K requires no nonidentity RoPE at position0. This diagnostic is not an intervention on K/V, and differences in it alone are not causal mediation evidence.

The runner validates per-panel reconstruction/add-back NLL discrepancies against max<1e-3 and mean absolute<1e-4, using all255 entries including query0. This is more conservative than testing only the primary span. It refuses partial result pairs and verifies result hashes/run identity when resuming.

Execution safeguard gap in reviewed version: execute() does not read or enforce the completed10-case validation artifact, its all_pass result, or equality of validation code SHA with current model SHA. Its docstring relies on coordinator approval. This was flagged before CPU release. Recommended remedy: enforce those conditions before model execution, preserving the frozen scientific conditions via a documented pre-output runner/protocol revision. This is an execution gate issue, not an error in the2x2 conditions or indexing.

Reviewed runner SHA256: a2cd6082523c7f082e935a7370c748ebfe113233f71859781b12c75576cc0c8b.
Reviewed frozen protocol SHA256: 14e31dc8822fec9bae98a6cd26ae6df0af445e569b7258df8c94216edb524f4c.
See runner_audit.json for input and condition checks. Any later gate amendment should be recorded as a new reviewed version.

## Six-feature LayerNorm-domain feasibility

The mathematical argument is valid. Center each of the six gate/up rows for1710,3950,4786 to form full-row-rank A. Every ideal non-affine LayerNorm output h has mean zero and ||h||²<2048 for fixed positive epsilon. For exact feature target b, the minimum squared norm is bᵀ(AAᵀ)⁻¹b. The minimizing vector is mean zero because it lies in the centered row span. Adding a centered nullspace component can increase the norm while retaining the features.

An independent thin-SVD pseudoinverse, rather than the script's Gram solve, reproduces all minima within5.01e-12. The row matrix exactly matches six existing FP32 weight-row slices after FP64 centering. Its Gram condition number is56.4184, with smallest singular value0.202755, so these findings are not near rank deficiency.

- E-primary/R-backup exact six-factor targets require minimum squared norm2235.5763…2349.0280. These exceed the universal ceiling by187.5763…301.0280 and are impossible for any ideal LayerNorm output.
- R-primary/E-backup exact targets require1834.3120…1916.6218, below the chosen E-state squared radii2045.1021…2045.4753. The saved constructions satisfy the six targets to5.33e-14, their chosen norms to1.14e-12, and mean zero to7e-18.
- The explicit preimage x=h sqrt(epsilon/(1−||h||²/2048)) correctly yields h under ideal LayerNorm; independently checked errors are below9e-16.
- Matching only the two E-primary factors, keeping R orientation in their nullspace, and scaling to the E radius is also valid. The backup pair is positive+1.8990…+2.3931 while primary axis is strongly negative. Primary constraints hold within3.35e-14.

The script defines donor targets by applying centered FP64 rows to recentered cached FP32 directions, not by taking stored rounded FP32 gate/up entries verbatim. Substituting the latter independently changes minimum squared norms by at most0.00124, far below the feasibility margins; conclusions are unchanged.

## Wording and boundary cautions

1. Impossibility concerns these exact six donor factor values. It does not mean an E-like primary pulse forces a negative backup pair, nor that all weak-backup states are impossible. The positive-pair constructions explicitly show why that broader statement is false.
2. Feasible in the ideal LayerNorm domain means some real pre-normalization vector exists. It does not establish a token prefix, an attainable upstream residual state, a natural intervention, or downstream behavioral significance.
3. Shares from projecting the R-to-E change onto the primary feature span are signed linear decompositions, not causal mediation percentages. Negative backup-up shares are legitimate and cannot be described as a positive fraction explained.
4. For an arbitrary exact chosen radius, feasibility is minimum_norm²<=radius², not strictly less. The code's strict comparison would mishandle equality at a sub-ceiling radius, but none of these cases is close to that boundary. The strict universal LayerNorm ceiling itself is correct for positive epsilon.

Independent outputs: check_feasibility.py and feature_feasibility_audit.json. No substantive numerical or algebraic error was found in the actual reported feasibility cases.
