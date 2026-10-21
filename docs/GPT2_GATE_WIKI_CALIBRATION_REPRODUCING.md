# Reproducing the T18 Wiki calibration comparison

Status: independent root review passed for preparation, source closure, layout-auditor integration and execution accounting. The tokenizer-only author preparation passed; no T18 model replay has been launched. The primary study and failed historical studies remain unchanged. This guide supersedes the proposal's unimplemented-status statements for preparation only.

## What the scripts do

`prepare_wiki_calibration_replay_inputs.py` exports a fixed safe metadata package or rebuilds four retained calibration and16 evaluation windows from local source files using only the pinned GPT-2 tokenizer. It verifies all seven calibration and58 evaluation rejection decisions, exact source/offset identities and the561-hash historical duplicate set. History is an inherited attestation; this tool does not rerun historical OLMo/Llama investigations or assert that those private source files were re-audited.

`reproduce_wiki_calibration.py` defaults to preparation only. It stages reviewed source files, two original parent metadata records and an existing validated M8 calibration reference. It reconstructs inputs and changes only the staged runner's literal `INPUT_HASHES` assignment. Six metadata hashes intentionally change to remove private source locators and distinguish current reconstruction from historical evidence. Eight files, including all input tensors and M8 directions, must remain byte-identical. Original and staged source hashes and AST-equivalence evidence are saved side by side.

The original `prior_T16_T17_evaluation_first_token_ids` list is not distributed. Its relevant membership booleans are labeled inherited evidence. All current/BOS and W_old4/W_fresh4/M8 identities are reconstructed locally and checked against hash commitments, so removing private fields cannot silently permit changed starting tokens or BOS IDs.

The safe metadata identity is `7e3a43808ee51ab8a44526d3cb9e9f545e12ac771a1723ca2d8ca9f360a8a0b6`. The package contains14 files including `identity.json`, totaling1,197,246bytes. No corpus text, input-token arrays, explicit token IDs, private absolute paths or model weights are included. The prepared workspace contains private local paths and regenerated token arrays and must not be published.

## Local requirements

Use the pinned `openai-community/gpt2` revision `75e09b43581151bd1d9ef6700faa605df408979f` and the original runtime/dependency versions. The model tensor hash is `2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527`. Preparation verifies only the five tokenizer/config files; actual execution verifies all pinned checkpoint files before loading the model. No automatic downloads are performed.

Supply seven exact source files: WikiText test and validation Arrow files, Mill's *Utilitarianism*, Spinoza's *Ethics*, and the local Adadelta/Adamax/ASGD Python source versions. Filenames, byte hashes, processed-text hashes and character counts are in `data_manifest.template.json`. The code files are joined in that order with two newlines. Books use the frozen Gutenberg delimiter stripping, and WikiText uses the frozen two-newline row join. The helper stops on any mismatch instead of choosing a replacement source.

A private config has `models.gpt2` with `model_id`, `revision` and local `path`, plus `sources` keys `wiki_test`, `wiki_validation`, `utilitarianism`, `ethics`, `adadelta`, `adamax`, `asgd` with local filenames. No Llama/OLMo model or tokenizer entry is needed for this T18 preparation route.

An existing M8 reference must include the seven `calibration_*` files and its completed manifest. The wrapper verifies exact raw/maps/tokens/input/spectra/metric identities, model closure and source bindings. It recognizes the original calibration ledger and the separately reviewed eight-forward rebuild ledger; their operation JSONs differ and are never relabeled as identical. The failed original gate-study manifest is rejected as an independently completed calibration reference; the completed fixed512 bundle or completed calibration-only rebuild provides valid provenance.

## Preparation command

Run the following command from the **private repository root**. The package source directory is `experiments/pulse_transfer`; its safe metadata, two parent records and source inventory are under `results/gate_wiki_calibration_reproduction`. These explicit package paths replace the private author-workspace paths. Supply your own `T18_INPUTS_LOCAL_JSON`, a completed `M8_CALIBRATION_REFERENCE`, and a new workspace path:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python \
  experiments/pulse_transfer/reproduce_wiki_calibration.py \
  --config T18_INPUTS_LOCAL_JSON \
  --metadata results/gate_wiki_calibration_reproduction/metadata \
  --calibration-reference M8_CALIBRATION_REFERENCE \
  --parent-metadata results/gate_wiki_calibration_reproduction/parent_metadata \
  --source-dir experiments/pulse_transfer \
  --source-manifest results/gate_wiki_calibration_reproduction/source_inventory.json \
  --work NEW_EMPTY_WORKSPACE \
  --prepare-only
```

The placeholder config and calibration reference are user-supplied local inputs, not files shipped in the repository. Each attempt needs a new/empty workspace. The source inventory is itself part of release review; edited or missing source dependencies fail before model loading. A source inventory must cover the declared dependencies of the chosen calibration reference as well as T18.

Preparation writes `wiki_calibration_preparation.json`, private `pulse_transfer/inputs.local.json`, reconstructed `calibration_wiki_inputs/`, reference files and canonical/staged source hashes. A passed preparation reports zero forwards and `model_loaded=false`. It is not a successful model replay. The recorded author preparation before the auditor-adapter addition used 8,688,514 bytes, 3.23 seconds for reconstruction and 762,424 KiB peak child RSS. These are measurements from this machine, not universal limits. The small adapter and its provenance entries are additional staging files; no preparation repeat was needed for this audit-only change. The wrapper still permits at most 64 MiB staging and keeps 2 GiB free; independent review/CPU jobs should stay below 3 GiB RSS.

## Obtaining M8 without rerunning earlier studies

The existing repository bootstrap is already available. Run these commands **from the private repository root**, replacing the local config and workspace placeholders:

```bash
python scripts/run_gate_study.py \
  --config GATE_INPUTS_LOCAL_JSON \
  --work NEW_BOOTSTRAP_WORKSPACE \
  --prepare-only

python NEW_BOOTSTRAP_WORKSPACE/pulse_transfer/rebuild_gate_calibration.py \
  --config NEW_BOOTSTRAP_WORKSPACE/pulse_transfer/inputs.local.json \
  --out NEW_BOOTSTRAP_WORKSPACE/pulse_transfer/t18_calibration_reference
```

This fallback performs eight GPT-2 calibration forwards and zero evaluation forwards. **Its preparation currently requires the pinned Llama tokenizer/config files**, plus the old source metadata and source files named by `input_sources.json`; it never loads Llama weights or runs a Llama model. Do not omit `--prepare-only` from the first command: that would also run an older gate study. Use the completed `t18_calibration_reference` with the T18 wrapper only if all strict canonical hashes pass. Hardware-dependent mismatches are preserved and reported; the tool does not substitute different maps or recalibrate to get a pass.

A wholly GPT-2-only eight-forward bootstrap remains deferred. It is not required when an already validated M8 reference is available, and it has not been implemented or claimed complete here.

## Model replay, audit and limits

Only after independent implementation approval should a researcher replace `--prepare-only` with `--execute`, in another fresh workspace. Omit `CUDA_VISIBLE_DEVICES=''` for that explicit model invocation. The wrapper then runs exactly 1,548 T18 forwards: four fresh calibration, 776 main/anchor and 768 suffix. It runs the CPU analysis and layout-aware saved-state auditor afterward. With a separate new M8 bootstrap, the total is 1,556. The frozen runner retains its 176 MiB normal-output limit, 48 MiB failure allowance, 2 GiB free reserve and per-group disk guards. Use one GPU worker; the reference setup was an 8 GiB RTX 3060 Ti. Actual resource requirements remain recorded in the primary manifest, and staging costs must be included before launching.

The wrapper invokes `audit_gate_wiki_results_layout.py`, not the frozen auditor directly. NPZ serialization preserves tensor values but loses the original QR strides. The approved adapter restores the original column-major layout of exactly 24 `basis_qr` matrices before calling the unchanged frozen auditor. Its SHA is `c837ab07243683e08dcfd0288fcf76a7ca8372dfdad06d605a01a80fe23f3974`; the frozen auditor SHA is `ebae29f3c04c937aa8abf7dcbb608af91ee0a957f3ffface128e914e6b5f1686`. Completion requires both exact identities, unchanged map values/digests, exact directions/geometry and `tolerances_changed=false`. The original failed direct-auditor result remains preserved. This adapter changes no model output, intervention, calibration value or tolerance.

The execution path is implemented but **not exercised by this author preparation**. The wrapper's completed/partial child counters were tested with explicitly synthetic fixtures, without model loading. A later scientific replay still needs independent comparison against primary per-target metrics, calibration maps/raw values, norms, snapshots/readouts and derived statistics. The numerical comparator and a separate readout audit are not implemented by this preparation task. Successful local execution alone is not an independent external replication or a confirmed numerical match.

Failure states preserve the exact phase and known child counters. Missing/unreadable child manifests remain unknown; no successful completion or zero-forward claim is invented. No automatic retry, old-output deletion, credential change or publication is part of this route.

## Frozen source-relative review links

Frozen protocols retain the original local layout so their recorded hashes remain valid. Their two source-relative review links resolve in this repository as follows; the protocols themselves have not been edited.

| Frozen source and original relative link | Repository review |
| --- | --- |
| `GPT2_GATE_WIKI_CALIBRATION_REVISION.md` → `review/gate_calibration_factorial_failure_review.md` | [Failure review](../reviews/gate_wiki_calibration/gate_calibration_factorial_failure_review.md) |
| `GPT2_WIKI_CALIBRATION_REPLAY_PROPOSAL.md` → `review/gate_wiki_replay_contract.json` | [Replay contract](../reviews/gate_wiki_calibration/gate_wiki_replay_contract.json) |

Some historical audit programs and evidence contain local paths or token identities and remain private. Exported review prose identifies those filenames and original hashes through the [omission manifest](../manifests/wiki_calibration_review_export.json). The [link adaptation record](../manifests/wiki_calibration_review_link_adaptations.json) preserves original and exported review hashes and every changed link. Their absence does not affect the standalone NumPy audit or the reviewed replay source closure, but the omitted historical checks are not independently runnable from this package alone.
