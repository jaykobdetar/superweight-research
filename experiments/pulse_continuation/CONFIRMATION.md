# Frozen fresh-text confirmation design

Written after the value-layer, global-gate and gate-subspace exploratory panels,
before running gate-path or fresh-text confirmation outcomes. No model parameter
selection or fitting will use confirmation evaluation windows.

The core claims to check are: frozen source mean K/V transfers closely; replacing
all source values by zero is worse than removing source attention; arithmetic
mean source gates remain imperfect even with fresh corpus-specific calibration;
retaining more development-fitted gate directions improves fidelity but low
static-template rank does not imply one dynamic gate direction.

New local texts: WikiText validation Arrow (the earlier continuation used test),
Du Bois's The Souls of Black Folk, Hobbes's Leviathan, and three installed PyTorch
optimizer source files (SGD, Adam, AdamW). This is experimental separation, not a
claim that the checkpoint never encountered these documents during training.
Use exact local files and retain hashes, tokenizer settings and token IDs. Two
new books extend text scope but do not make the collection representative.

Fresh calibration: eight disjoint 512-token windows per new corpus. For books
and Wiki, development starts8192,9216,...,15360; evaluation starts are eight
evenly spaced windows from32768 through the final full512-token window. For the
shorter code corpus, calibration starts0,512,...,3584, evaluation starts from8192
through the last full window. Assert no overlap. Native source gate masses are
measured in the fixed old Wiki mean-KV world; fresh scalar gates are their
arithmetic average. Old source means and PCA bases remain frozen.

Evaluation: eight natural-source512-token windows per corpus, plus the same
eight suffixes with first token The510 edited directly (no retokenization).
For each identity: original clean/pulse-deleted controls, frozen mean K/V,
fixed-meanK/zeroV, source attention removal, global Wiki/Hume/fresh-corpus gates,
PCA ranks4 and12 using the already frozen pooled development basis, and native
gate reconstruction. Preserve matched source-token and scoring conventions.
Score predictor1 onward equally (510 next-token targets); fixed-ID conditions
share all scored targets with their natural-ID condition.

Fresh gate calibration uses natural source IDs. Fixed-The evaluation is a
transfer check, not a separately fitted gate. No winner will be fitted after
looking at these results. Report every prespecified condition and corpus,
pointwise paired-window intervals, per-token agreement/KL, numerical controls
and resource accounting. A CPU-only independent review will audit this panel.
