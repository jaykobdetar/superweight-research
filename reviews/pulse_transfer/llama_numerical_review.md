# Independent Llama transfer numerical review

Status: passed, CPU-only. Evidence: `llama_numerical_audit.py/.json`, plus `llama_column_row_contrasts.py/.json`. Peak main audit RSS729,496 KiB; only one down-projection matrix read, no model instantiated or GPU used.

All984 forwards/976 evaluation records match the expected panel, including880 source records,96 history records,16 repeat/copied-matrix exact controls and eight development forwards. The216 source summary groups and eight history groups replay from saved per-token arrays; their paired bootstrap intervals also replay. Script/protocol hashes match the frozen implementation, parameter hashes match before/after, and parameter version counters are unchanged. Saved losses and KL are finite. This checks reductions of saved KL values; full-vocabulary logits are not saved, so it does not independently recompute KL from logits.

All development contribution scores and global top-ten ordering were reconstructed from the saved eight-window activation arrays. The actual selected layer1 down-projection matrix independently verifies every column maximum and maximizing row in that layer, candidate/control weights, and selected column energy. The candidate is L1/output588/input1419, score279.65625, weight−0.61328125, column energy fraction0.30167854. The source activation selection is development-only and precedes evaluation. Other layers' maximum arrays were checked internally but not reread from checkpoint matrices.

All880 source records preserve selected intermediates and pre-intervention source outputs versus native; all384 row-removal records preserve every other source output coordinate. Selected row and full-column interventions give exactly the same changed output588 value. BF16 native copied-matrix arithmetic differs from analytic saved scalar-product subtraction by up to1.34375: do not define the experiment as exact real-arithmetic subtraction. The native matrix-recomputation control itself is exact. Source-only hook localization at other positions is established by inspected code and prior CPU synthetic tests; full all-token output traces are not saved.

Natural-start512-token PPL ratios (row / column) are:

| Corpus | Row | Column |
|---|---:|---:|
| Wiki validation |1.010602|1.449770|
| DuBois |1.016497|1.498632|
| Hobbes |1.005491|1.689357|
| Optimizers |1.018868|1.449170|

The paired column-minus-row NLL contrast is positive in all96 individual window/length/start comparisons and all24 group means; all24 four-window paired-bootstrap lower bounds are positive. This gap persists for natural, The and explicit BOS starts, at256 and512. It is a clear reversal of the earlier OLMo row-catastrophe/column-tolerance pattern for this selected intervention family. It is not proof that Llama lacks other superweights. Scalar row removal also changes predictions: natural512 agreement is90.34–94.12%; small NLL change is not exact output preservation.

No corpus at either length meets the prespecified history signature. At512 the R/U q-contribution ratio ranges0.708–1.023 and U contribution/MLP output norm0.0153–0.0627, failing the required suppression and pulse size. This limits the selected candidate under the specified artificial-token construction, not all newline-sensitive mechanisms. The full protocol compares token IDs without retokenizing the common suffix. Llama is an instruction-tuned3B model evaluated without a chat template; comparison with GPT-2 and OLMo is confounded by model size, training, tokenizer, architecture and precision.

A fresh held-out source K/V donor factorial is a reasonable next test of whether this column-deletion dependence is transmitted through source attention states. Full-donor and blocking closures must exclude source predictor0 uniformly. Llama's24 query heads and8 K/V heads require correct grouped-query semantics (three query heads per K/V head); a source K/V projection hook should patch the native8-head projection, before RoPE and repetition. No OLMo-specific clipping operation should be imported. The current result does not establish a semantic-content mechanism, nor a universal difference between scalar and whole-column deletion.
