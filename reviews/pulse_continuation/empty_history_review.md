# Independent CPU review: later boundary with empty early attention history

The code and saved raw results pass review. No model or GPU was loaded. Audit peak RSS was about 540 MiB. Experiment artifacts were not modified. The companion `empty_history_audit.py` and JSON contain reproducible checks and per-cell paired intervals.

## Intervention and scoring

The hook supplies a standard four-dimensional additive causal mask at every layer. At the selected layers only, it sets the boundary query's entries for keys before the boundary to negative infinity. Its self entry remains zero and future entries remain masked. Every other query row retains the standard causal mask. CPU tests verify every mode/layer combination and verify that cloned isolation masks never mutate the cached causal control.

Consequently, this removes the boundary token's own early attention history. It does **not** remove the prefix from later queries, reset the model's entire accessible context, or alter position IDs. The native SDPA implementation uses an explicitly supplied mask and does not add an inconsistent causal offset.

All comparisons use the same tokens, with source zero and boundary set to the same identity. Within-mask deletions use that mask's clean logits for KL and baseline loss. The cross-mask panel separately uses the explicit causal control. The 511-target panel predicts the suffix after its first forced token; the 510-target panel additionally excludes the boundary predictor. The latter is exactly the former with its first loss/prediction/KL entry removed. Same-suffix targets are preserved across prefix lengths and identities.

The scalar deletion occurs at the L1 down-projection output, after the saved intermediate, and affects only the specified output coordinate at position zero, the boundary, or both. Runtime assertions verify that deleting the scalar does not change its own earlier MLP input. Directly recording the intermediate is therefore a valid pre-intervention pulse measurement.

## Raw numerical checks

The audit verifies all 3,456 observations in 864 groups, 1,536 recorded forwards, 384 saved intermediate pairs and pulse observations, script/source-manifest hashes and unchanged model hashes/version counters. Every row's NLL, agreement and KL matches its saved token metrics. The reconstructed scalar coefficient is exactly 0.63232421875 throughout.

The explicit causal mask and native automatic mask produce identical saved token loss and predictions on all 64 tested inputs: maximum token loss difference zero, no prediction mismatches. Their pulse traces also agree. All source-zero traces remain unchanged under boundary-mask edits within a given input.

With both L0 and L1 restricted to boundary-self attention, the **entire 8,192-dimensional pre-down-projection intermediate** at the boundary equals source zero exactly in all 64 cases. Restricting all layers gives the same early intermediate. The induced The pulse is −258.620605 throughout. Period pulses are approximately −253.878174 or −254.036255 depending on sequence length, with exact within-sequence boundary/source equality. Do not generalize this to bitwise equality across lengths.

L0-only isolation induces partial pulses: roughly −57.98 to −5.16 for The, and −54.38 to −21.16 for period. L1-only leaves pulses small (absolute magnitude at most 0.07793), and ordinary causal boundary pulses are below 0.00154 in absolute magnitude.

## Causal interpretation

| Boundary attention restriction | Range of mean later-only NLL increase after boundary scalar deletion, across 16 corpus/identity/prefix cells |
|---|---:|
| Ordinary explicit causal mask | −0.000121 to +0.000072 nats |
| L0 only | 2.096952 to 5.854395 nats |
| L1 only | −0.000118 to +0.000109 nats |
| L0 and L1 | 4.910896 to 6.878489 nats |
| All layers | 4.908932 to 6.875705 nats |

The two-early-layer condition therefore recreates the source intermediate at a later position, and its scalar contribution is causally important to later predictions under that intervention. L0-only already creates a smaller, consequential contribution. These results rule out a requirement that this intermediate can occur only at absolute position zero in the tested model/computation, but they do not identify a natural reset mechanism or uniquely isolate which internal features normally suppress the pulse in real history.

The induced pulse does not function as an interchangeable backup. With L0/L1 isolation, deleting the original source-zero scalar still adds 6.958830–8.284058 nats to later NLL, despite the boundary pulse remaining present. Joint deletion is strongly nonadditive: the paired interaction (both minus source-zero-only minus boundary-only plus clean) ranges from −6.903777 to −4.932698 nats. Calling these additive contributions, a recovery fraction, or proof of redundant sources would be incorrect. Earlier damaged-prefix states remain available to later queries.

Each interval uses only four matched windows. This exploratory panel reuses suffixes from the residual-path evaluation and adds real prefixes that may overlap prior text; no fitting is performed. It is an intervention sufficiency test over these fixed corpora, identities and lengths, not independent confirmation across broad natural contexts.

The completed table replay verifies all 864 summary rows and 864 contrast rows against the saved token metrics, including paired bootstrap intervals. The final status is recorded in `empty_history_audit.json`. No numerical discrepancy or remaining review blocker was found.
