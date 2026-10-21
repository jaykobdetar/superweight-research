# Standalone additive-study preparation and reproduction

Commands in this guide assume the repository root as the working directory.

The T17 preparation path uses the pinned GPT2 tokenizer and an explicitly supplied frozen calibration reference. It does not run an earlier experiment, fit a new map, or load a Llama/OLMo tokenizer or model. Preparation defaults to zero model forwards; `--execute` explicitly requests the 1,992 T17 forwards and its saved-data analyzer. A successful run is not an exact reference comparison or an independent external replication.

## Inputs and identities

Provide six lawful local source files: WikiText validation Arrow, Mill's *On Liberty*, Wollstonecraft's *A Vindication of the Rights of Woman*, and the frozen `adagrad.py`, `rmsprop.py`, and `lbfgs.py` versions. Their filenames, byte hashes, processed-text hashes, extraction offsets and token-window hashes are in the safe metadata. The GPT2 checkpoint is `openai-community/gpt2`, revision `75e09b43581151bd1d9ef6700faa605df408979f`; five tokenizer/config files are checked before tokenization. No download, credential change or model purchase occurs in these scripts.

`prepare_additive_replay_inputs.py export-metadata` creates a small shareable bundle from the locally verified canonical input directory. Its `identity.json` SHA256 is `13cca2b5ec8ef9993e01df36f0d86aecfc0bab96c4b598fd94a887fec0605a98`. The bundle contains no source text, token arrays, starting-token IDs, original absolute paths or model weights. It retains the 84-manifest historical inventory, exact exclusion coordinates, selection/rejection decisions, and original file hashes as provenance. The original historical files are not re-audited by a new reconstruction. In particular, inherited OLMo offsets are not recomputed and no claims about model training-data overlap are made.

The rebuild replays the frozen GPT2 selection policy, including all rejected candidates, and requires exact canonical hashes for `tokens.pt`, both direction files, starting-token summary and the historical scan. It derives directions from the exact frozen maps. Private token files and starting-token IDs are generated only inside the caller's reproduction workspace.

Three metadata files intentionally have new hashes: `data_manifest.json`, `exclusion_inventory.json`, and `preflight.json`. Original source paths and prior localization first-token IDs cannot be both omitted from a package and reproduced byte-for-byte. The original hashes remain explicit provenance; the new files describe the private reconstruction and inherited sanitized history. The wrapper changes **only the literal `INPUT_HASHES` assignment in its staged copy** of the runner, proves all other AST nodes identical, saves the original source copy and both hash sets, and keeps the production candidate untouched. Any input tensor, direction, starting-identity, intervention or other code change is rejected.

## Preparation with an existing verified calibration

The caller supplies a config containing only `models.gpt2` (`model_id`, `revision`, `path`) and `sources` entries for `wiki_validation`, `mill`, `wollstonecraft`, `adagrad`, `rmsprop`, and `lbfgs`. Keep this config private. It is not a repository artifact.

The wrapper also requires a reviewed source inventory JSON with a `files` map of basenames to SHA256 values, the directory containing those code/protocol files, the safe metadata directory, and the canonical `gpt2_run/manifest.json` and `selection.json` metadata seeds. The source inventory must cover the runner's explicit provenance list, transitive local imports, both new helpers, and the chosen calibration manifest's source dependencies. No earlier raw run directory is needed.

```bash
python experiments/pulse_transfer/reproduce_additive.py \
  --config additive_inputs.local.json \
  --metadata additive_replay_metadata \
  --calibration-reference VERIFIED_CALIBRATION_DIRECTORY \
  --parent-metadata GPT2_PARENT_METADATA_DIRECTORY \
  --source-dir EXPORTED_EXPERIMENT_SOURCE_DIRECTORY \
  --source-manifest REVIEWED_SOURCE_INVENTORY.json \
  --work NEW_PRIVATE_REPLAY_WORKSPACE \
  --prepare-only
```

Use a new empty workspace; failed or existing work is never overwritten or automatically resumed. Preparation checks the six substantive calibration files and accepts exactly two reviewed operation-ledger schemas:

- Original study operations SHA256 `a26e553b554fe9f155dd2b67cb1663177ba4a458a444919d4f787c3df75f191e`.
- Explicit eight-forward rebuild operations SHA256 `e76e5efc9e097edabe67643c1159d6a22d4d585a28514396927cdde042389f2f`.

The second requires a completed `calibration_only_rebuild` manifest with exactly eight calibration forwards and zero evaluation forwards. Its six other required files, including maps, raw calibration tensors and calibration tokens, must match exactly. Each actual operations file is preserved unchanged; no old counters or schema fields are fabricated. Both references require model closure and their original source hashes. Unknown variants fail.

The measured tokenizer-only preparation staged about 6.7 MB, including approximately 6 MB of frozen calibration data. It retains the large model and raw research outputs in place. The normal T17 resource limits remain in the study protocol and storage addendum; this wrapper does not override them. `--execute` replaces `--prepare-only` when an explicitly reviewed model run is intended. It verifies checkpoint hashes before invoking the same staged runner and analyzer. Its final bookkeeping records verified completed counts; after a failed child it reads available child manifest counts and explicitly leaves unavailable facts unknown.

## Obtaining the frozen calibration without the private reference

The existing reviewed eight-forward entrypoint is `rebuild_gate_calibration.py` (SHA256 `43172c7ee68af83d186859e8a3a29a92eb702682a8c53e4bb0f2454fc57b4b30`). It uses the original eight WikiText-test/Hume passages and frozen fit procedure; it does not evaluate T17 inputs or select a new rank. The complete existing bootstrap route is:

```bash
python scripts/run_gate_study.py \
  --config gate_inputs.local.json \
  --work NEW_CALIBRATION_BOOTSTRAP_WORKSPACE \
  --prepare-only

python NEW_CALIBRATION_BOOTSTRAP_WORKSPACE/pulse_transfer/rebuild_gate_calibration.py \
  --config NEW_CALIBRATION_BOOTSTRAP_WORKSPACE/pulse_transfer/inputs.local.json \
  --out NEW_CALIBRATION_BOOTSTRAP_WORKSPACE/pulse_transfer/additive_calibration_rebuild
```

The first command stages the historical source/selection metadata and reconstructs exclusion metadata without model forwards. This **existing calibration-bootstrap route requires the pinned Llama tokenizer files** to reconstruct the original historical calibration metadata, along with the original local sources and GPT2 checkpoint identified by the repository's input/model manifests. It does not load Llama weights or rerun a Llama experiment. The second command performs exactly eight GPT2 calibration forwards. These commands are documented, not executed by the present metadata-preparation task.

Then supply the resulting `additive_calibration_rebuild` directory as `--calibration-reference` in the T17 command. This route is strict: byte-identical maps/raw/tokens and the reviewed operations schema are required. A numerical mismatch on another platform must be preserved and reported; it is not permission to substitute different maps. A wholly Llama-tokenizer-free, first-principles calibration bootstrap is not implemented here. Once a verified reference exists, the T17 wrapper itself requires only GPT2 and the six T17 source files.

## Review and remaining scope

Before a public or private repository is marked reproducible, review the safe bundle, canonical-versus-portable hash distinction, AST-only staging change, source inventory, both calibration schemas and the zero-forward preparation evidence. The optional 1,992-forward branch has bookkeeping stub tests but has not been executed as a model reproduction in this preparation task. A saved-result comparator and full replay remain separate work.

Author evidence is in `review/additive_replay_author_tests.py/.json`, `review/additive_wrapper_outcome_tests.py/.json`, and `review/additive_wrapper_flow_tests.py/.json`. The first failed helper attempt is retained separately: its privacy checker initially mistook a JSON pointer for an absolute filesystem path; the corrected checker permits only the two declared pointer roots. No scientific input or historical selection changed.
