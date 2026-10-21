# Reproduce the layer-local gate study

This entry point needs no knowledge of UIT and does not run UIT. It uses the
same pinned GPT-2 checkpoint, local input sources and environment documented in
[the gate reproduction guide](GPT2_GATE_REPRODUCING.md). Obtain inputs under
their own terms; the repository contains hashes and retrieval instructions,
not model weights, source corpora, token sequences or activation donors.

Use the private configuration format from that guide. The GPT-2 weights are
required; historical Llama **tokenizer files only** reconstruct calibration
exclusion metadata. No Llama or OLMo weights are loaded. The recorded historical
source-exclusion inventory is inherited, with new tokenization/offset/hash
checks; this route does not regenerate all old exposure evidence from old
model runs.

```bash
python scripts/run_gate_localization.py \
  --config /absolute/path/config.private.json \
  --work /absolute/path/new-localization-reproduction \
  --prepare-only

python scripts/run_gate_localization.py \
  --config /absolute/path/config.private.json \
  --work /absolute/path/new-localization-reproduction
```

The first command stages code and historical metadata without model forwards.
The second performs eight new calibration forwards, reconstructs the 15 frozen
inputs and control directions with a tokenizer, then runs 1,868 study forwards
and the saved-output analyzer. Calibration may instead be copied from an
explicit `--calibration-reference /absolute/path/verified-calibration` whose
hashes match the frozen reference, in which case there are zero new calibration
forwards. The wrapper never automatically overwrites or resumes a partial run.

The input helper checks source/tokenizer hashes, exact offsets, 1,192 historical
interval exclusions, token hashes and all seven reconstructed artifact hashes.
The original failed 16-passage preflight is preserved: only three eligible code
windows exist on the frozen grid, so actual n=3 is used for code and n=4 for each
prose corpus. This is not a random sample or proof about all character starts.

Require at least 7 GiB of free disk before the study: 4 GiB estimated output
plus 3 GiB reserve. A separate guard checks the remaining 3 GiB before each
input group. The estimate is not an enforced output-size cap. Keep one GPU
worker at a time. The tested machine has an 8 GiB RTX 3060 Ti and 16 GiB RAM;
actual measured usage is in the report and run manifest. Review processes are
bounded below 3 GiB RSS. A new machine/runtime may differ numerically; preserve
failures rather than relaxing tolerances. No paid compute is required.

The runner writes exact frozen sources, model/input/map hashes, observed SDPA
backend, source-state closures, main/suffix raw gate shards, per-token metrics,
prefix hidden arrays, norm vectors and the 52 predeclared attention snapshots.
Successful full output logits are not saved. Full attention contexts are
saved only for the fixed snapshots; other context equality and all full-logit
equality use runtime comparisons and shape/dtype/content hashes. Both sides are retained on an actual comparison failure.

`audit.json` must end with `status: passed`. Independently inspect per-passage
values and every predeclared arm; do not report only the strongest layer.
Recompute paired intervals with n=3/n=4 and preserve natural/BOS correlations.
Joint-minus-sum is descriptive nonadditivity, not mediation. Audit the exported
reference arrays and every predeclared summary without loading a model:

```bash
python scripts/audit_localization_results.py --output /absolute/path/export-audit.json
```

The independent numerical audit script in `reviews/gate_localization/` is
historical audit evidence, not a turnkey auditor for a new workspace. It imports
`gpt2_gate_fixed512_numerical_audit` and uses the original private failed-run and
fixed512 evidence in addition to the localization tensors. Changing paths alone
does not recreate those dependencies. The portable NumPy audit above validates
the exported reference arrays; the production analyzer checks a new run. A
separate replay-to-reference comparator is not supplied, and the wrapper does
not claim exact comparison. Neither an analyzer pass nor a same-machine exact
comparison is an external independent scientific replication.

The new localization batch has not yet had a complete second pretrained replay
in the originating environment. Its inputs have been independently rebuilt
and audited, and its implementation and primary outputs receive separate
reviews. The earlier 1,360-forward fixed512 replay is distinct evidence.
