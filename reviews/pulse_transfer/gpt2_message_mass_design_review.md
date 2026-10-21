Historical design review for the completed `gpt2_query_mixture_run` batch. Its exploratory message-first notation is superseded by the frozen protocol and Report10 mass-first notation. This is not the subsequent unexecuted gate-subspace study.

# Query-bias donor message × non-source mass: independent design review

**The factorial is scientifically well defined, with the numerical guards and closure definitions below frozen before execution.** Copying the exact donor SDPA context on diagonal arms is appropriate. Finite positive mass alone, however, does not validate the conditional-message interpretation of `c/a`. Implementation review and numerical review are still required; this is not permission to skip those checks.

This review read the completed query/source numerical review, its runner and protocol, the shared attention-mask helper, and the fresh-span preparation code. It did not load a model or use a GPU. Small synthetic algebra checks and four saved prior-panel snapshots are in `gpt2_message_mass_design_checks.py` and `.json`. Peak RSS was 813,380 KiB (about 0.78 GiB). No runner was edited and no external action was attempted.

## What is being intervened on

For each treated layer, head and query position `q>=1`, with literal source value `V0=0`, let

`a = sum(j>=1) p_j`, `u = sum(j>=1) softmax(non-source logits)_j V_j`, and `c = a*u` in exact arithmetic.

The proposed implementation captures the model's actual context immediately before `c_proj`, defines the operational donor message as `u_D = c_SDPA,D / a_D`, and supplies `c_D*(a_G/a_D)` for independently selected message donor D and mass donor G. P retains learned query bias; R removes it at later queries. Both donors are generated from the same input with zero source V in all treated layers4–11.

This is a valid intervention on an internal tensor, including when a mixed context is not naturally produced by either network trajectory. The factor called **message** contains its conditional magnitude, direction and composition. It is not a unit direction, and this experiment should not be described as another norm-versus-direction factorial. The mass factor multiplies each head/query's entire 64-dimensional context before the learned output projection.

At layers after4, donor messages and masses incorporate prior effects of query-bias removal on the donor's residual stream, Q, K and V. They therefore test exchanging two properties of whole donor trajectories. They do not isolate a direct, same-state effect of learned Q bias, produce a unique additive mediation decomposition, or identify independently deployable replacements. Both donors use the same evaluation input; this is a diagnostic intervention rather than held-out prediction by an input-independent approximation.

## Necessary numerical safeguards

1. Compute `a` by summing permitted non-source probabilities, not by subtracting source mass from one. Require finite strictly positive values for every patched query/head, including the final unscored query. Exclude query0 entirely from the division: its mass is zero and its context must remain untouched. Permit only declared floating-point normalization roundoff in the upper bound near1; do not clamp or silently repair values.
2. Capture `c` from the actual SDPA `c_proj` input **before** the context patch and retain the raw `c` and `a`. Do not reconstruct the causal patch from a separate manual context while calling it an SDPA donor. Freeze tensor layouts, data types and the evaluation order `c_D*(a_G/a_D)`; `(c_D/a_D)*a_G` need not be bitwise identical.
3. Add the stable conditional non-source softmax diagnostic on rows `q>=1`, using the actual post-hook Q/K/V and applied mask. Verify the actual context against `a*u_ref` and the operational `c/a` against `u_ref`, explicitly rejecting NaN/Inf. For every off-diagonal pair, additionally check its final scaled context against `a_G*u_ref,D` with the frozen `1e-4` maximum absolute tolerance. Keep the existing full-context reconstruction guard. Record output/message scales, minimum mass, maximum gain `a_G/a_D`, and maximum normalization and mixed-context errors by layer/arm; summary means are insufficient.
4. Preserve a predeclared CPU-audit snapshot subset for both P and R donors and both mixed arms, including raw Q/K/V, mask, `a`, unpatched actual context and final supplied context. The suggested existing Wiki-natural-window0, L4/L11 subset is economical. Runtime guards must cover all rows/layers; independently replaying the subset does not audit unretained tensors. On failure, preserve the current donor/mix/diagnostic and partial outputs, mark the run failed without `completed_utc`, and stop. Do not enlarge tolerances, clip gains or change the formula during that run.
5. On diagonal arms copy `c_D` directly. This gives a precise finite-precision closure and avoids making the diagonal depend on a divide/multiply round trip. Still calculate and record the diagonal algebra/reconstruction residual as a diagnostic; a successful copied diagonal does not establish the accuracy of mixed arms.

These are material guards rather than a hypothetical need for an arbitrary small-mass cutoff. In the **previous**, frozen Wiki-natural P/L4 snapshot, the FP64-reference minimum non-source mass is `2.8175994e-7`; changing to R mass can multiply that donor context by as much as `385,291.55`. The captured SDPA context nevertheless agrees well with the stable reference: the P-message/R-mass maximum context discrepancy is only `5.7019e-6` there. Across all four inspected L4/L11 mixes the discrepancy is below `9.3e-6`, and the largest normalized-message discrepancy is `1.8790e-5`. This is encouraging prior-panel evidence, not a guarantee for the new panel or an audit of its GPU mass calculation.

The synthetic check also shows why the old context guard alone is insufficient: an allowed absolute context error of `1e-6` at donor mass `1e-8`, rescaled to mass0.5, permits a context error of50. That example is a mathematical stress case, **not an observed SDPA failure**. Synthetic exact-algebra factorization and all four donor combinations pass to below `7e-16` in FP64.

## Required closures and controls

- Use the same explicit causal mask in donors and mixed arms. Keep native C and native-mask P/R zero-V references, and report explicit/native-mask drift separately. Source-blocked P and R controls are useful for fresh-panel replication of the prior access contrast. Blocking source should leave query0 self-attention intact.
- Add an actual current-context copy/no-op hook for each recipient and compare it with that recipient's unpatched explicit-mask zero-V run. This checks capture/return layout and hook placement without conflating them with donor transfer or rescaling arithmetic.
- For each of the four `(message,mass)` pairs, run both P and R recipients. They should agree exactly, because all eight later attention contexts are supplied externally and query-bias removal only affects attention at those sites. At L4 the recipient residual histories agree; copied contexts then give the same residual/MLP history inductively. This equality applies to **mixed pairs as well as diagonals**. Treat the second recipient as a closure check, never a second independent observation.
- For PP and RR, additionally require equality to their respective unpatched P/R donor outputs in each recipient. Per-token loss/KL/argmax equality is a necessary measured closure but does not logically establish equality of full logits. If the report says full outputs are exact, compare the complete final normalized hidden states or logits transiently; the former adds little storage. Persist equality counters and failure tensors. The same standard applies to cross-recipient output claims.
- Require exact query0 context preservation at every patch; source0 Q/K/V trajectories must agree across the zero-V donor/recipient arms. Native C with original nonzero source V is not the reference for this source-state equality. Check immediate-hook locality for Q, K and non-source V, with zero source V exact. Later-position Q/K/V may legitimately differ across donor trajectories.
- Validate each patched head/query against the declared donor formula; include shape and head-concatenation checks. Verify that no source tensor, donor cache or parameter is mutated. Hash the bias vectors, donor files, source files, protocol, parent manifests and conditions; retain unchanged model hashes/version counters.

A source-blocked run is **not** expected to equal an all-layer donor-message-with-mass-one patch. It changes subsequent live states and conditional mixtures, while the donor patch clamps those mixtures to a different trajectory. A mass-one donor arm can be a useful optional diagnostic, but it must not be claimed as a blocked-source closure. It is not required to answer the proposed P/R factorial question.

## Frozen contrasts and reporting

Write `Y_MG` for one input's mean scored NLL, with the first letter the message donor and the second the mass donor. Choose one recipient as the analysis copy after verifying both-recipient closure. Prespecify:

| Contrast | Formula |
|---|---|
| Change mass at P message | `Y_PR − Y_PP` |
| Change mass at R message | `Y_RR − Y_RP` |
| Change message at P mass | `Y_RP − Y_PP` |
| Change message at R mass | `Y_RR − Y_PR` |
| Signed interaction | `Y_RR − Y_RP − Y_PR + Y_PP` |
| Total P-to-R donor difference | `Y_RR − Y_PP` |

Report all four cells, all simple effects, the signed interaction, and NLL/KL/argmax agreement to native C. A small NLL difference does not establish restored outputs. Exponentiate within-window NLL differences for PPL ratios; keep signed NLL as the primary factorial scale. Report all corpus/start groups, per-window values and fixed paired-bootstrap intervals at `n=4`. Natural and BOS share suffixes, so neither the two starts nor duplicate recipients increase the independent passage count. Do not convert a favorable average into a universal statement about every head/layer or corpus, and do not call these contrasts percentages of unique causal mediation.

The fresh eighth-panel fractions `.125,.325,.525,.725` are reasonable prospective choices, conditional on actual token-span verification. Include all seven earlier used panels, including the newly completed query/source panel: the prior-span total should be28 per corpus. Keep the start substitution procedure and scored targets2–511 fixed, while patching queries1–511. Report attention averages over the matching scored queries1–510; query511 is an intervention/guard row but has no next-token target in this window. Freeze the full condition list and expected forward/check counts before running; no condition-count claim is possible from the current informal list alone.

Store donor context/mass sufficient statistics, not all attention score matrices. The two full context donors across32 inputs and8layers are about768MiB in FP32; masses add about12MiB. Derived `c/a` is reproducible and need not be stored as another full persistent copy. Shard or stream donors/checkpoints and CPU audits if necessary to keep review below3GiB. Preserve negative results and failed guards. The experiment remains a conditional extension of the query/source findings; it does not justify a priority claim, attention-sink elimination, native-trajectory restoration, whole-network sufficiency or a deployment recommendation.

**Disposition:** proceed to implementation after incorporating the necessary guards and unambiguous contrasts above. No access blocker was encountered. GitHub remains outside this review and was neither accessed nor retried.
