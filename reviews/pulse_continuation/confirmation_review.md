# Independent confirmation review

Reviewed `confirmation.py` and `CONFIRMATION.md` and completed data without model/GPU loading. No blocking implementation issue found. One sampling caveat requires disclosure below. Replay script and evidence are `confirmation_audit.py/.json`; peak RSS approximately 513 MiB. Final replay verifies all 88 summary rows and 176 paired contrast rows in `summary.csv` and `contrasts.csv`.

The script implements all eleven prespecified conditions for eight windows, two source identities, and four corpora (704 evaluation forwards plus 32 calibration forwards). Fresh corpus gate means are fitted on eight natural-ID calibration windows per corpus, and every calibration span is asserted disjoint from every evaluation span. Evaluation spans are also pairwise disjoint. Source mean K/V and pooled development PCA bases remain frozen. Fixed-The is explicitly an identity-transfer check using the natural-ID calibration.

Each identity starts from a fresh cloned input; only source ID 0 changes to 510 for The. All later input IDs and all 510 scored targets stay matched. The baseline is recomputed for each identity, preventing a baseline/source mismatch. Uniform score_start=1 excludes predictor 0, including original pulse deletion, source removal, and oracle reconstruction. Source-column removal occurs at L2..15 while other rows remain causal; mean-KV and zero-V conditions otherwise use the established cut hooks. Projection mode is disabled before each fresh baseline and enabled only for ranks 4/12.

The data preparation retains file hashes, token IDs, start offsets, and no-special-token settings. New Wiki validation, Du Bois, Hobbes, and installed PyTorch optimizer sources provide experimental text separation, not proof of absence from checkpoint training. Eight windows per text and two new books do not establish representative corpus coverage. Pointwise paired-window intervals condition on the frozen constants and do not account for calibration estimation uncertainty or repeated exploratory choices made before this confirmation design.

Numerical audit should check all source identities share suffixes, fitted fresh means reconstruct from saved calibration masses, original constants match their recorded hashes, full family inclusion, and current-recipient oracle errors before interpreting effects. No confirmation outcome fitting or condition selection is present in the inspected code. The claimed pre-outcome design chronology is recorded by the parent workflow; code inspection alone cannot independently certify when a design was first frozen.

All 704 raw records across 88 groups and 64 reconstruction checks replay. All recorded source-file and frozen-fit hashes match, calibration/evaluation starts reproduce the design and are disjoint within each corpus, and fresh gate means reconstruct exactly. Recorded input hashes match the saved natural windows or the same suffix with source ID 510. Identity changes can propagate through layers 0/1 before the fixed source cuts begin at L2; equality of resulting outputs across identities is not expected.

**Shared boilerplate:** Du Bois evaluation window 7 and Hobbes evaluation window 7 are identical 512-token windows, SHA256 `6c1ad0c9d8e1c0f375012ae06e6d93fc711a90e458f0ff8d449bd364e97dc879`. Their files end with the same Gutenberg licensing/footer text. There are therefore 31 unique natural evaluation windows, not 32 mutually distinct text samples. No duplicate calibration/evaluation token windows were found otherwise. Preserve the frozen primary panel; do not silently replace these windows after seeing results. Seven-window book sensitivity estimates excluding index 7 are saved alongside the primary replay.

| Natural source | Mean KV | Mean K, zero V | Remove source | Fresh global | PCA4 | PCA12 |
|---|---:|---:|---:|---:|---:|---:|
| Wiki validation | 0.99972 | 231.485 | 3.32794 | 2.04177 | 1.12955 | 1.01781 |
| Du Bois | 1.00086 | 3.55920 | 1.26952 | 1.14725 | 1.04638 | 1.00706 |
| Hobbes | 0.99998 | 2.34293 | 1.04411 | 1.07834 | 1.03115 | 1.00424 |
| Optimizers | 0.99995 | 1.68693 | 1.05705 | 1.07056 | 1.01974 | 1.00327 |

Entries are PPL ratios against each condition's matched native identity baseline. The-source conditions preserve the same broad pattern. Mean-KV argmax agreement is 99.36–99.66% for natural identities, not exact recovery. Natural Wiki PCA12 agreement is 93.21% despite its 1.01781 ratio. Current-recipient reconstruction against native mean KV has absolute corpus/identity mean drift at most 7.16e-5 nats, at most 6/4,080 prediction mismatches, and at most 0.04554 individual loss difference.

Excluding shared boilerplate changes natural Du Bois zero-V from 3.55920 to 2.35901 and natural Hobbes from 2.34293 to 1.46283. In both books, for both identities, all seven-window paired intervals still support zero-V being worse than source removal, fresh globals being worse than mean-KV, and PCA12 improving over PCA4. These sensitivity checks retain the qualitative conclusions while showing that the boilerplate materially affects magnitudes and cross-book independence.

`confirmation_excluding_shared_footer_summary.csv` provides all 88 original condition groups and the original summary metric columns. Book groups exclude window 7 (n=7); Wiki and optimizer groups retain all eight windows. The primary experiment tables remain untouched.
