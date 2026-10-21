# Independent CPU review: frozen source caches through 4,096 tokens

The implementation and completed results pass review. The audit reconstructs all 896 observations, 224 summaries and 448 contrasts, including paired intervals. Fourteen dictionary identities account for the 476 forwards: 448 scored passes plus two single-token cache constructions per identity. Peak audit RSS was approximately 514 MiB. No model/GPU was loaded and no experiment artifacts were modified.

## Controls and alignment

Frozen prior cache means and the prior static rank-two residual template are hash-verified. No new fitting or selection occurs. Each corpus has four mutually disjoint 4,096-token windows; the 512/1,024/2,048/4,096 crops share the same starts. Some text was used earlier, as declared, so this is a length-extension panel rather than fresh independent validation.

Every bypass receives tokens 1 onward with cache positions 1 onward. The source and normalization hooks are reset before suffix execution. Cache copies are independent and every recorded final cache length equals its input length. The rank-two dictionary preserves exact single-token caches at layers 0 and 1, while layers 2–15 use the frozen normalized source template.

All primary scores exclude predictor zero, including the cached first-token logit used to align arrays. The tail begins at predictor n−513 and contains exactly the last 512 valid predictions, except the 512-token case where both panels contain all 510 available predictions. Tail losses, predictions and KL are exact slices of the primary arrays. Cross-length averages cover different target sets; interpret each replacement against its own matched clean crop.

Full-context donor caches reproduce **every saved loss, prediction and KL exactly** on all 64 input/crop combinations. Independent single-token dictionary caches are numerically close but not exact: maximum absolute group mean NLL difference is 0.000147804 nats, maximum individual token loss difference is 0.101711 nats, and minimum group prediction agreement is 99.7070%. Those controls should accompany small replacement effects.

The common first 510 targets also expose finite-precision shape dependence in clean execution: 512 versus 2,048 is exact in this panel, while 1,024 or 4,096 can differ. The largest observed clean common-target token loss change is 0.0372965 nats. This does not indicate future-token causal influence and should not be described as exact invariance across lengths.

## Results and limits

At 4,096 tokens, using all scored predictors after predictor zero:

| Corpus | Wiki mean, all layers: PPL ratio | Wiki mean, layers 2–15: PPL ratio | Static rank-two residual plus axis: PPL ratio | Rank-two prediction agreement |
|---|---:|---:|---:|---:|
| Wiki | 1.000413 | 0.999467 | 1.003858 | 97.0506% |
| Hume | 1.000495 | 1.000034 | 1.001023 | 98.7665% |
| Plato | 1.000822 | 1.000908 | 1.003140 | 98.0948% |
| Python | 1.002370 | 1.000030 | 1.000241 | 99.1939% |

These are exp(mean paired NLL difference), not mean per-window ratios. Near-baseline loss does not imply exact predictions. For the last 512 predictions at length 4,096, rank-two ratios are 1.005743, 1.001522, 1.005671 and 0.997719 respectively. Wiki and Plato show positive tail loss differences; Python's negative difference is specific to these four fixed windows, not proof of a generally improved model.

Original source-zero deletion remains catastrophic through the longest tested context; frozen cache replacements avoid that collapse while leaving small distribution/prediction changes. This supports persistence of the approximation through the configured 4,096-token limit on these panels. It does not establish arbitrary-length behavior, free-running generation preservation, task quality, or a practical speed advantage. The rank-two label still refers to the static across-layer source residual fit, not a rank-two downstream mechanism.

Each corpus interval resamples only four windows, and the multiple lengths/tail panels overlap extensively. No multiplicity correction or independent-document inference is provided. There is no remaining implementation or numerical-review blocker.
