# Independent gate-subspace implementation review

**Status: approved for the predeclared run at the frozen source identities below.** Static, math, input and actual Transformers CPU integration checks passed; the author confirmed the final freeze. No pretrained checkpoint or GPU was loaded. The Transformers integration uses a newly initialized random CPU model only.

Reviewed the prospective protocol, design review, `gate_inputs.py`, frozen `gpt2_gate_subspace.py`, and inherited QuerySource/GPT2KV/LayerWrites/Runner hook paths. The design has21 main conditions,64 evaluation inputs and16 anchors, hence1360 evaluation records plus8 calibration forwards =1368 forwards. The expected1488 run checks and680 paired-prefix checks agree with the loops.

The combined-QKV hook zeroes only source V in layers4–11. The attention-output prehook changes query1 onward before `c_proj`, preserves source-query context exactly, and derives each gate from the current causal Q/K state. Cross-head projection operates separately at each query: no evaluation means, suffix pooling, future-layer donor, or evaluation-dependent fitting enters the map. Earlier interventions can change later endogenous states and conditional mixtures; these are joint gate-map interventions, not isolated additive pathway effects.

Calibration occurs entirely before evaluation. Eight equal-length passages provide4080 scored query vectors per layer. Centered FP64 covariance uses N−1; SVD and eigenspectrum agree. Detrending subtracts log(q), and the descriptive position-centered spectrum subtracts the calibration passage mean at each query. The latter does not enter evaluation. Frozen maps include separate raw-b/global-PC1, detrended means/bases, mean mass, and one deterministic nested QR basis. Removing k retains12−k dimensions; PCA/QR comparisons match action and rank. Fullrank copies actual context exactly and separately checks reconstructed logits/mass/context rather than interpreting the bypass as proof of projection precision.

Source masks exclude future positions before both probability and conditional-mixture calculations. The implementation directly sums non-source mass and uses stable non-source-only softmax. It validates finite positive denominators, finite target/gain, native and rescaled reconstruction against1e−4, and retains diagnostic tensors on failure. Sigmoid targets are not clipped. Source-state identity, immutable maps/parameters, input/code hashes, and copy controls are runtime assertions. Both C and P reference metrics are retained; score index1 correctly evaluates targets2..127 or2..511. The inherited query128 scalar is explicitly null at length128.

Input-helper review agrees with the saved no-inference preflight:8 calibration passages exclude8 complete old8192-character buffers per calibration corpus;16 evaluation passages exclude32 earlier used-token spans per corpus.128-token inputs are exact prefixes of512; BOS changes only token0. Corpus/start/length groups share passages and must not be treated as independent sample expansions.

Independent evidence:

- `gpt2_gate_subspace_synthetic.py/.json`:102 CPU math/direct-hook fixtures, no model instantiation; centered fits, complementary projections, query-locality, all gate modes, boolean/additive masks, exact source/copy controls and failure rejection. Peak745364KiB RSS.
- `gpt2_gate_subspace_integration.py/.json`:16 actual Transformers random-model CPU forwards,12 layers/12 heads/width768/inner16/vocab97, nonzero query/output biases, lengths128/512. Checks dual-reference KL against FP64, exact P/native-mask/current/fullrank controls, source equality, prefix losses/hidden states, length metadata and hook cleanup. Peak982676KiB RSS, maximum KL-P error4.35e−7 and prefix hidden error1.19e−6.

These synthetic results do not predict pretrained projection conditioning or prove the selected snapshot panel covers all numerical tails. Runtime failures must remain failures under the frozen tolerances. Low loss alone is not faithful recovery; report KL/agreement to both P and C, fixed ranks and negative controls. Calibration tokens are correlated, the QR seed is singular, and paired n=4 intervals remain conditional summaries.


The final analyzer’s scoring and grouping were independently checked:336 condition summaries,504 segment summaries,2048 contrast records and512 paired summaries; shared targets2–127 and later targets128–511 are partitioned correctly. Ten separate synthetic scoring/bootstrap fixtures passed. It rejects stale-success reuse by writing a checking state at startup and directly rechecks current/fullrank/repeat/identity raw metrics and final-hidden hashes. The full-run artifact audit remains to be performed after inference.

An initial provenance omission was corrected before freeze: `analyze_transfer.py`, which supplies the shared aggregation/bootstrap implementation, is now included in the runner’s frozen source hashes. Runtime Python/torch/transformers/numpy versions are also recorded. The finalized protocol explicitly allows/counts finite sigmoid endpoints, preserves the1e−4 context guards, and fixes the1e−3 shared-prefix loss/hidden tolerance while treating gate/argmax drift descriptively.

The independent tokenizer-only preflight regenerated the inputs and exactly matched the parent preflight; no model was loaded. See `gpt2_gate_inputs_independent.py/.json` (peak755732KiB RSS). Calibration manifest SHA256 `24b1bdf8c1a0744bbd70e0936d24cbf741835aab691bfe496bd9eb809100557e`; evaluation manifest SHA256 `7f7abe557c9dd67df1d66c98499cdba258de8f021164950040a989bb553ec32c`.

Reviewed source identities:

| File | SHA256 |
|---|---|
| `gpt2_gate_subspace.py` | `6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e` |
| `analyze_gate_subspace.py` | `0b65d05ec9afec48f182ab7fecff2cd5cc35e3a47ae41df51ac410215f47cdce` |
| `GPT2_GATE_SUBSPACE.md` | `84cb95cf7ef48f6d70ea44663ab3559ef16d38396449de31d0ebe36edff59b40` |
| `gate_inputs.py` | `437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d` |

No pretrained/GPU experiment, publishing attempt, or runner edit was performed by this reviewer. No mathematical, hook-placement, calibration-separation, scoring, count or implementation blocker was identified at these source bytes. The recorded synthetic fixtures test implementation behavior, not pretrained numerical safety or empirical scientific outcomes.
