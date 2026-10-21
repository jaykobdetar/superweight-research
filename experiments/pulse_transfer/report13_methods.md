## Question and exact comparison

The preceding layer screen found positive joint-minus-singleton loss residuals. That alone cannot tell whether output changes fail to add, or whether the nonlinear cross-entropy calculation converts additive output changes into nonadditive losses. This experiment measures both terms on fresh passages. It does not identify an internal mediator or demonstrate feedback.

The checkpoint remains GPT-2 revision `75e09b43581151bd1d9ef6700faa605df408979f`, FP32 with SDPA. All executions have physical length 512; layer, head and token indices are zero-based. Native C is the unchanged model. Reference P zeroes the position-0 value at L4–L11. Every scientific arm retains that boundary. At each patched head/query, the current context is multiplied by `sigmoid(b′)/a`, where a is non-source attention mass and b is its logit. This changes the mass while preserving the current conditional non-source value mixture at that boundary; subsequent network computation remains live.

The frozen eight-passage calibration supplies α and the first centered direction u of `b−log(q)` per layer. A seeded orthogonal direction v provides a matched logit-norm control. Three patch families are evaluated at each of eight layers separately and at all eight jointly:

- PC1 removal: `b′ = b − s u`, with `s = (b − log(q) − α) · u`.
- Orthogonal control: `b′ = b − s v` using the coefficient at that arm's own incoming state.
- Position replacement: `b′ = α + log(q)`.

All 27 scientific arms, native C, reference P and two exact-copy controls are retained. Gate maps act on queries 1 onward and preserve the source-query context relative to P; they do not undo P’s source-value lesion. Singletons must receive the same incoming logits, masses and contexts as P at their patched layer. Matching logit norm does not match delivered context norm, and one orthogonal control does not sample a distribution of random directions.

Let z denote the actual full-vocabulary output logits. For each family, build the diagnostic `z_add = z_P + Σ_l (z_l − z_P)` in fixed L4…L11 order, using CPU FP64 on actual FP32 model logits. For next-token cross-entropy L, define:

```
S = Σ_l [L(z_l) − L(z_P)]
N = L(z_joint) − L(z_P) − S
J = L(z_joint) − L(z_add)
R = L(z_add) − L(z_P) − S
N = J + R
```

All component losses in this identity use the same FP64 calculation. The inherited FP32 losses are a separately checked numerical readout. R is target-independent: the target-logit terms cancel, leaving `LSE(z_add) + 7 LSE(z_P) − Σ LSE(z_l)`. R can have either sign; convexity does not require it to be nonnegative for this extrapolation. J is a loss contrast between two outputs, not a path-specific causal interaction. Direct centered logit discrepancies, both KL directions and top-1 agreements accompany it: a small J alone does not imply similar output distributions. The additive output uses eight singleton executions and is a diagnostic, not a cheaper replacement.

## Fresh panel, controls and uncertainty

There are four passages each from WikiText-2 validation, Mill's *On Liberty*, Wollstonecraft's *A Vindication of the Rights of Woman*, and local optimizer source (Adagrad, RMSprop and LBFGS). The original tokenizer-only preflight scans 84 historical manifests and excludes previously exposed intervals. The fresh Wiki grid fractions are 119/379/620/877; the other sources use 125/375/625/875. One code passage each comes from RMSprop and LBFGS and two from Adagrad. Full source hashes, selected intervals, candidate rejections and exact token hashes remain in the local manifest. These are exposure-separated research inputs, not a claim that the model never saw the text during pretraining.

A later provenance review found that the inherited inventory had not explicitly mapped the original main/smoke/readme report windows. A supplementary tokenizer-only audit validated their original OLMo tokenizer, revision, source and saved token arrays, mapped 102 exact source ranges, and checked both latest studies' eight Wiki passages using GPT-2 offsets. It found zero overlaps. This audit occurred after T17 execution; no passage was changed or replaced, and it is disclosed as supplementary verification rather than part of the original prospective selection.

Natural/BOS pairs reuse exactly the same 512-token sequence and replace token 0 only. Wiki window 1 and Mill window 1 begin with token 68, which appeared among calibration starts. Therefore the present panel tests fresh passages and varying natural starts; it is not uniformly a held-out starting-token identity test. BOS is absent from calibration starts. There are no new calibration forwards.

Every condition receives a fixed-shape suffix control: positions 128–511 rotate by one token, leaving the first 128 unchanged. The unchanged prefix's logits, hidden states, gates, source projections and contexts must match exactly. Changed suffix targets are not scored. Targets 2–511 form the primary view; targets 2–127 and 128–511 are correlated early/later views of that same 512-token execution, not different context lengths. Query 0 remains an unscored diagnostic.

The plan specifies 1,000 main records, including eight native repeat/identity anchors, and 992 suffix controls: 1,992 forwards. Four paired passages per corpus/start yield 2,000 bootstrap draws with seed 20261021. J, R and N share resampling indices, preserving their identity within every draw. Intervals condition on this fixed calibration and small nonrandom panel. Starts, views and families are correlated; there is no multiplicity correction, population-wide generalization or mediation percentage. All declared conditions and all groups are reported.

## Preserved evidence and numerical limits

The actual tensor entering `lm_head` after final LayerNorm is saved at all 512 positions for 29 distinct trajectories per input, totaling 928 hidden tensors. The exact tied unembedding identity, shape, dtype, bias absence and parameter version are recorded. Main gate states and prefix controls, source states, per-query losses/distribution metrics, operation guards and four fixed attention snapshots are preserved.

The prospective storage test found the original eight sampled full-vocabulary rows left too little room for analysis. Before any pretrained output, independent review accepted six rows `{0,1,126,127,254,510}` for all 31 conditions and all additive families on natural window 0 of each corpus. All-query hidden states and runtime metrics stayed unchanged. The original proposal, input preflight and eight-row sizing evidence remain intact, alongside the reviewed storage addendum. Raw successful full-vocabulary model outputs are independently available at those 744 condition/query rows only. Complete successful source/context arrays exist only where explicitly saved; other closures rely on runtime comparisons and fingerprints.

An independent CPU audit can reconstruct logits approximately from every saved hidden state and the pinned unembedding tensor without loading the whole model. That reconstruction is a numerical cross-check, not an exact replay of the actual GPU readout or a replacement for runtime primaries. Sampled raw outputs allow direct error checks at six positions only. No complete second pretrained replay of this new batch is claimed.
