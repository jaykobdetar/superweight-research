Export note: This is the completed archived-output reanalysis, not the later fresh-model study. Selected source-text examples have been omitted in this numeric-only export. The historical next-study/blocker section describes the original cutoff; see ../validated_continuation for later implementation validation only.

# What useful native computation does OLMo's source pulse support?

## Answer in brief

The strongest supported functional account is a **query-dependent attention reference**, not a stored summary of the later text. A nearly constant source can provide both a value contribution and a gate on the other tokens' contribution. The precise linguistic operation implemented by that reference is still unidentified.

The new reanalysis adds a more specific hypothesis: **the first later single-newline source can make the initial source largely redundant in normal prediction.** Source-0 removal is damaging throughout the sampled Wiki windows, which contain no single-newline token 187, but its cost almost vanishes after the first such newline in the sampled book/code windows. A suffix-matched first-token edit supports this interpretation. A clean two-source masking factorial is still required to establish mediation.

Catastrophic scalar deletion is a poor way to discover the native task. It creates a much more broadly damaging state than either deleting the whole producing neuron output or removing access to an intact source. Lesion repair and native usefulness must be kept separate.

## Scope and what was actually run

- Model identity: allenai/OLMo-1B-0724-hf, revision d7cbab742d80589e714b1a2d7f838dcd21cbe143. The archived full-model experiments used FP16/native SDPA.
- Repository snapshot: 0bf1747a8d1b4e6a0adfa3aab72005f7b13df096.
- Acquired 42 public result files from seven OLMo experiments and verified their exact Git blob hashes. Recomputed 4,108 per-window loss/KL reductions, with maximum scalar error below 5e-16, and checked 96 exact output closures.
- Joined stored losses to actual public input tokens. Both books and optimizer code match the archived full text and corpus token hashes. Wiki acquisition stopped at a rate limit after 2,800 rows; all four selected 512-token window hashes nevertheless match exactly. No claim of full Wiki-corpus reconstruction is made.
- A small standard-library byte-level BPE implementation was accepted only after these exact token-hash checks. No tokenizer package, model package or new model weights were installed for this work.
- Main numerical reduction peaked at 77,324 KiB RSS. All computation was CPU-only. No new pretrained-model forward pass, training, GPU use, public repository edit or upload occurred.
- This is new exploratory analysis of old model outputs, plus a proposed experiment and synthetic algebra tests. It is not independent model replication.

Primary evidence: [research report](https://github.com/jaykobdetar/superweight-research/blob/0bf1747a8d1b4e6a0adfa3aab72005f7b13df096/docs/REPORT.md), [raw OLMo exports](https://github.com/jaykobdetar/superweight-research/tree/0bf1747a8d1b4e6a0adfa3aab72005f7b13df096/results/pulse_continuation), [input provenance](https://github.com/jaykobdetar/superweight-research/blob/0bf1747a8d1b4e6a0adfa3aab72005f7b13df096/manifests/input_sources.json).

## 1. Native dependence is real, but unlike catastrophic scalar sensitivity

The first table uses the first four natural windows per corpus, exactly matching the core source-0 row/column panel and avoiding the known duplicated Gutenberg footer in the older eight-window aggregate. Each window has 510 scored targets, positions 2-511. PPL ratios are exp(mean window NLL difference); a target cost above 1 nat means its observed-token probability falls by more than a factor of e.

| Corpus | Remove intact source 0 from L2-15 attention: PPL | Targets costing >1 nat | Fixed mean source K/V: agreement | Mean-K/V targets costing >1 nat |
|---|---:|---:|---:|---:|
| Wiki validation | 3.227 | 41.52% | 99.46% | 0 |
| Du Bois | 1.223 | 7.65% | 99.12% | 0 |
| Hobbes | 1.048 | 1.52% | 99.26% | 0 |
| Optimizer code | 1.045 | 1.62% | 99.75% | 0 |

Mean K/V remains an approximation, not exact output replacement. Here none of 8,160 targets loses more than 1 nat, but some predictions change. The source-removal run supplies mean K/V at source 0 while masking it; because no later query can access that source at any downstream layer, those supplied source values cannot create a later-token path. Separate source-mask controls in the quiet-anchor experiment confirm the architectural closure.

In contrast, source 0 scalar deletion costs >1 nat on 89.66-96.72% of these targets. It is not merely a magnified version of a small, sharply identified linguistic dependency.

The discrepancy is still clearer for later sources:

| Existing panel | Scalar-deletion targets costing >1 nat | Less pathological comparison |
|---|---:|---|
| Controlled later newline, four corpora | 76.11-81.61% | Blocking the intact newline source costs >1 nat on 0 of 6,112 scored targets; PPL 1.002-1.007 |
| Natural later positions, Du Bois | 59.12% | Whole-neuron output deletion: 0.196% |
| Natural later positions, Hobbes | 80.93% | Whole-neuron output deletion: 0.147% |
| Natural later positions, code | 55.69% | Whole-neuron output deletion: 0.196% |

Thus a huge deletion effect can expose an incompatible representation rather than the size or identity of a native computation. The later-whole-output controls are not completely harmless, and the report retains their nonzero errors.

Sources: [confirmation implementation](https://github.com/jaykobdetar/superweight-research/blob/0bf1747a8d1b4e6a0adfa3aab72005f7b13df096/experiments/pulse_continuation/confirmation.py), [later-use outputs](https://github.com/jaykobdetar/superweight-research/tree/0bf1747a8d1b4e6a0adfa3aab72005f7b13df096/results/pulse_continuation/later_uses_run), [newline-broadcast outputs](https://github.com/jaykobdetar/superweight-research/tree/0bf1747a8d1b4e6a0adfa3aab72005f7b13df096/results/pulse_continuation/newline_broadcast_run).

## 2. New localization: native source 0 dependence usually ends at a later first newline

For a source at position p, its first strictly later attention query is p+1, predicting target p+2. The following split uses that architectural cutoff, not a cutoff fitted to loss.

| Corpus/window | First single newline p | Source 0 removal: mean target cost before p+2 | Cost from p+2 onward |
|---|---:|---:|---:|
| Du Bois 1 | 15 | 1.014 | 0.009 |
| Du Bois 2 | 8 | 2.071 | 0.013 |
| Du Bois 3 | 9 | 0.076 | 0.019 |
| Hobbes 0 | 13 | 1.477 | 0.037 |
| Hobbes 1 | 10 | 2.699 | 0.037 |
| Hobbes 2 | 17 | 0.266 | 0.018 |
| Hobbes 3 | 2 | -0.101 | -0.002 |
| Code 0 | 3 | 1.733 | -0.005 |
| Code 1 | 12 | 1.640 | 0.006 |
| Code 2 | 18 | 2.348 | 0.020 |
| Code 3 | 8 | 0.962 | 0.011 |

All entries are nats per observed target. Before-period samples contain only 2-18 targets per window, so the contrasts are descriptive and confounded by sequence position and target composition. Tokens are not independent replicates. The large afterward spans contain 492-508 targets each.

The four Wiki windows contain no token 187 at all, and source 0 removal costs 0.986-1.378 nats per target across the window. The exceptional Du Bois 0 window starts with token 187 itself; its source 0 removal cost stays substantial, at 0.703 nats overall and 0.669 after the first 32 targets. This is consistent with the previously established circuit in which an earlier single newline suppresses the later pulse.

Sensitivity check on additional archived windows: nine more non-footer book/code windows with a later first token 187 give pre-broadcast costs of 0.640-1.848 nats and afterward costs of 0.003-0.042. A second Du Bois window beginning with 187 retains 0.963 nats of source 0-mask cost overall. These are additional existing outputs, not fresh experiments or a pre-registered confirmation set.

### A matched existing edit strengthens the hypothesis

Du Bois 0 has an initial newline at position 0 and another at position 16. The confirmation experiment also changes only source 0 to The510, leaving every suffix token unchanged.

| Source 0 identity | Native mean NLL | Source 0-mask PPL | Mask cost after target 18 onward |
|---|---:|---:|---:|
| Native newline 187 | 3.27915 | 2.020 | 0.67193 |
| The510 | 3.28477 | 1.083 | 0.03696 |

The native-loss change is just +0.00562 nats, while the post-newline source 0 dependency changes much more. This is a genuine recorded input intervention, but not a completed mediation test: changing token 0 also changes L0/L1 computations beyond the proposed later source.

The same matched contrast repeats in Du Bois window 6, whose next newline is at 17: source 0 replacement by The510 changes native NLL by +0.01976, but reduces the post-next-newline masking cost from 0.93029 to 0.02410 nats. Its whole-window mask PPL falls from 2.620 to 1.077.

The best current hypothesis is availability of a second usable reference, not a claim that an induced later pulse repairs a poisoned earlier source. The latter rescue was already tested and failed. Removing a source's downstream access and leaving a malformed source accessible are different conditions.

## 3. Whole-output exceptions align with source identity and available alternatives

Input identities were independently recovered and joined to 32 natural windows: the original 16 source 0 windows and the 16 fresh later-use windows.

- All three windows whose first token is 187 have source 0 whole-column deletion PPL 7.030-10.621.
- All other 21 book/code windows have a later first token 187 at positions 1-30 and source 0 whole-column deletion PPL 0.998-1.108.
- The eight Wiki windows have no-187 and give PPL 1.021-2.697.

This does not prove token 187 is universally special or sufficient for damage. It identifies a concrete, previously aggregated source of heterogeneity. In the original Du Bois 0 case, source 0=187 is verified by the full 512-token hash, not guessed from its geometry.

The separate upstream analysis finds a plausible contributor: other L1 MLP writes cancel the surviving coordinate-axis signal for source 0 token 187, unlike several other initial token identities. That finding and the possible availability of an alternate later source must be separated experimentally. One can change the residual geometry while keeping the source identities/routing fixed, or mask the alternate source while keeping the native geometry intact.

## 4. Which ordinary predictions depend on the source?

The recovered Wiki targets show broad lexical and punctuation effects, not only newline or boundary predictions. Among 2,040 targets in the four windows, 41.79% of alphabetic pieces and 41.83% of punctuation pieces cost more than 1 nat when source 0 is inaccessible. This coarse classification is descriptive, not a discovered linguistic circuit.

Two selected examples illustrate the kind of prediction involved:


These are observed target probabilities, not reconstructed argmax tokens. The public exports do not contain the intervention's predicted token identities. Top-error examples were selected after observing effects and cannot validate a semantic interpretation.

Unseen-in-prefix target tokens show larger average costs than repeated tokens in this small Wiki sample, but their native surprisal also differs. That correlation does not establish a copy/suppression function and should not be elevated into the explanation.

## 5. The relevant computation is a coupled attention operation

For one head and one query after source s, let alpha be attention mass on s, v_s its value, and mu the conditional attention-weighted mean over the other tokens. Then

    head_output = (1 - alpha) * mu + alpha * v_s.

This identity makes three distinctions explicit:

1. A fixed v_s still produces a dynamic contribution because alpha depends on the query and competing keys.
2. Zeroing v_s leaves attenuation of the other tokens in place: output=(1-alpha)*mu.
3. Masking s renormalizes the other tokens: output=mu.

Consequently, zero-value failure does not show that the source carries document semantics. A useful nonzero default vector, a useful attenuation gate, and their coordinated balance can produce the same failure. Existing constant-gate failures also do not refute a bias/reference account; a constant vector need not have a constant gate.

For source 0, causal masking gives an additional exact constraint: with token 0 and position 0 fixed, its state cannot depend on later suffix content. It may encode its own token identity and learned parameters, and its downstream usage can depend on the suffix through later queries. A source 0 cross-suffix donor with the same token/position should therefore be an exact mathematical negative control in deterministic evaluation, up to implementation roundoff. It cannot demonstrate later-content transfer.

Later sources can attend preceding content, so a boundary-information carrier remains possible there. That possibility needs an information-transfer test, not a magnitude or lesion-rescue argument.

These concepts have substantial prior art. [Massive Activations](https://arxiv.org/html/2402.17762v2) studies near-constant attention biases; [When Attention Sink Emerges](https://arxiv.org/abs/2410.10781) emphasizes softmax competition; [The Spike, the Sparse and the Sink](https://arxiv.org/html/2603.05498v1) distinguishes persistent near-constant states from local attention routing. None establishes the precise useful task of this pinned OLMo source.

## 6. The discriminating next study

### Study A: clean source 0 x first-newline access

First test the newly localized redundancy hypothesis in otherwise native models. Use 16 eligible unedited 256-token windows, selected only by input criteria: first single-newline p between 16 and 96, no earlier 187, multiple book/code sources, no duplicated footer. Add 8 no-187 windows as negative controls. Freeze sampling and all scoring spans before observing interventions.

Four core conditions on each eligible input:

1. Native.
2. Mask source 0 for strictly later queries in L2-15.
3. Mask the first later newline p for strictly later queries in L2-15.
4. Mask both sources with the same rule.

Score exactly the same targets p+2 through 255. Also retain the pre-p span as a secondary check. The primary statistic is the window-level interaction in NLL:

    (both_masked - p_masked) - (source 0_masked - native).

A positive interaction means loss from removing source 0 increases when the later source is absent. This is evidence of functional substitutability under the tested cuts, not an additive fraction of “the mechanism.” Require source/p-mask closure, a matched ordinary-token mask control, and the no-187 controls. On the suffix-matched Du Bois newline0/The510 pair, repeat the same factorial at p=16; this tests whether the edited source 0 dependence is mediated by access to that later token.

This is not the already-failed later-pulse compensation experiment: no poisoned scalar-deleted source remains accessible in this design.

### Study B: reference versus information-bearing value

Use minimal pairs that differ in a one-token fact before p, have the same token at p and absolute positions, and share the exact suffix/query. For example, counterbalance two one-token labels assigned to an entity before the first newline, then query the assigned label afterward. Choose tokenizer-valid pairs first. Include ordinary text-completion controls and require that native prediction distinguishes the correct labels before using this as a functional task.

Cache donor states from:

- a content-matched independent prefix with the same label;
- a content-mismatched prefix with the opposite label;
- a structurally matched prefix with irrelevant content changed;
- the identical recipient (oracle).

Patch source V only while preserving recipient K and source attention mass. At a single intervened layer, preserve the complete recipient attention row, so the local effect is exactly alpha_recipient*(V_donor-V_recipient). For multi-layer interventions, explicitly distinguish frozen native source-gate replay from endogenous downstream routing; do not call “K held fixed” equivalent to “routing held fixed.” Run reciprocal donor directions.

Measure the signed change in correct-versus-foil log probability, not only perplexity. A mismatch that systematically transfers the donor label, unlike the matched/irrelevant donor, would support a source-value information carrier. A matched and mismatched donor that behave alike, while source access still matters, favors a reusable reference on these tasks. Failure on one task would not rule out all possible content carried at later sources.

Mandatory negative controls: same-token source 0 cross-suffix donor; an ordinary source at a matched distance; same-donor reconstruction; donor permutation; matched score positions; and donor-content counterbalancing. Do not infer native function from a random direction merely being harmful.

### Study C: distinguish attenuation, value injection, and normalization

At an identified native-use layer/head, compare the 2x2 local branches using the same recipient alpha and mu:

- Native: (1-alpha)*mu + alpha*v_s.
- Attenuation only: (1-alpha)*mu.
- Value injection only: mu + alpha*v_s.
- Neither: mu.

The new informative branch is value injection without attenuation, interpreted together with the other branches, not another claim that zeroV or constant gates have not been tried. Apply W_O and continue the model endogenously. Final loss interactions need not be additive even though the local vector decomposition is.

For a normalization-only account, compare actual residual-space writes after W_O, not just raw V norms. A control should preserve write mean, norm, and dot product with the recipient residual. Then resulting residual mean and centered variance are identical while its direction changes. If such native-world matched controls preserve function broadly, normalization calibration remains plausible; selective content transfer surviving these controls would support a more specific representation. These are diagnostic counterfactuals, not guaranteed on-manifold states.

The supplied small NumPy reference implements the branch identities, fixed-routing value swap and mean/radius-matched residual-write control. Eleven synthetic closure/error checks pass below 2e-15. This tests the algebra only, not an OLMo hook or any model claim.

## 7. Bounded execution plan and remaining blocker

The current compact caches stop around the early MLP and do not contain the full downstream model. They cannot answer the proposed label-transfer or joint-source-loss tests.

Preferred next execution: an authorized environment already containing the exact pinned model, with the four clean masks above, saved per-target logits/losses, source attention mass, source values, and the relevant post-W_O writes. Start with the matched Du Bois pair plus two eligible and two no-187 controls, then run the frozen held-out panel if closure and baseline checks pass.

If new cloud acquisition is authorized instead, the checkpoint index reports exactly 5,119,148,032 bytes of model tensors, not merely a 2GB “1B” download. A straightforward full FP32 load exceeds the 4GB process ceiling and is unacceptable. Use a streamed/memory-mapped implementation: one block is 268,435,456 bytes; the embedding and output matrices are each 412,090,368 bytes. Process one 256-token window/condition at a time, retain only compact activation caches, and project logits in vocabulary chunks. Target RSS below 1GiB, abort before 4GiB, limit BLAS threads, and benchmark one 128-token native pass before estimating the run duration.

This route entails a new large download and new full-model replay and has NOT been executed or authorized here. A CPU FP32 replay must be labeled a different numerical backend, with baseline and intervention robustness checked before interpreting agreement with original FP16 results. Do not silently promise bitwise reproduction or a runtime estimate without the benchmark. No paid GPU or training is needed for the scientific question.

## Conclusion

The useful native computation is currently best described as **a dynamically accessed, approximately reusable attention reference whose availability can shift to a later first-newline source**. The second clause is a new, well-localized hypothesis supported by existing output data and a matched edit, not yet a completed causal explanation.

The most valuable next step is not another catastrophic scalar ablation. It is a clean two-source interaction test, followed by recipient-routing-preserved content donors. Those tests can distinguish an interchangeable reference, a genuine boundary information carrier, and normalization-related dependence without assigning a semantic label prematurely.
