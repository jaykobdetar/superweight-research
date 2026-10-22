# Report 18 audit and reproduction

This guide accompanies the [reviewed report](GPT2_L11_KEY_VALUE_DONOR.md).
The immutable study contains 776 retained GPT-2 forwards. Two preserved failed
prefixes add 27 acquired returns, so acquisition was 803. Neither the technical
26-forward overlap check nor the report rebuild is a second full-panel replay.

## Completed reanalysis and input checks

The 72-file public numerical package contains 25,748 permitted arrays and 32
control ledgers. The independent public audit reproduced every published
statistic exactly. Its optional private-raw mode also compared all 25,748
arrays and all 32 ledgers to the immutable originals. Neither mode loads a
model or reconstructs omitted activations or prediction IDs. The public-only
run took 6.65 seconds and peaked at 190,404 KiB RSS; private-raw comparison took
15.04 seconds and peaked at 440,200 KiB. The repository convenience wrapper
was separately run once after copying the exact packages and passed.

The separately reviewed 96-file metadata package preserves 77 of 80 original
source members exactly and supplies three narrowly sanitized certificate
views. All 49 local-locator occurrences, including 11 dictionary keys, were
reviewed; all 74 measured integer values and their order remain unchanged.
Its tokenizer-only reconstruction reproduced the canonical token artifact
byte for byte, all 831 candidate decisions (16 accepted and 815 rejected),
16 natural/32 paired BOS identities, 777 inherited history hashes and the
four full 8,192-character calibration buffers. This checks the recorded
exposure chain; it is not a host-wide or pretraining-overlap guarantee.
Input reconstruction took 5.50 seconds and peaked at 852,944 KiB RSS.

Source-only replay staging and its independent binding review completed without a model. It prepares A/B/merge
commands; it does not execute them. The six certificate-DAG edges and four
runner provenance-binding assignments are separately recorded, preserving
the scientific functions, guard predicates, ordering and scoring AST.
Staging took 2.93 seconds and peaked at 748,592 KiB RSS.
The historical 26-return gate still needs an explicit portable interpretation,
and a new source/input-aware independent raw comparator remains unperformed.
**A second complete 776-forward model replay has not been run.** Prepared
commands alone do not establish replay readiness or independent replication.

The report, all 128 primary comparisons, 1,368 contrasts, 684 paired-start rows,
negative and reversed results, figures and original report-build provenance
are in [the report artifact directory](../results/l11_key_value_donor_report/).
Its original candidate badge is retained as historical provenance; the linked
report above carries the reviewed presentation. The self-contained
[HTML report](../reports/report18.html) contains its data downloads.

## What the checks establish

The saved-array numerical audit checked 33,933 array occurrences, all frozen
statistics and the required controls. Projection and readout audits separately
checked 28 projection relations and 12 readout states/48 selected vocabulary
rows. The selected-row precision check does not establish whole-panel numerical
resolution or a downstream sensitivity bound. Agreement is an observed result;
omitting prediction IDs prevents independently reconstructing it from IDs.

The local report builder was executed once without a model and reproduced 33
deterministic files exactly. Its private JSON inputs have the commitments in
`results/l11_key_value_donor_report/build_manifest.json`. Publishing the original
builder does not make those private inputs publicly available. Public numerical
reanalysis, metadata rebuilding and a pretrained model replay are separate tasks.

## Run the public numerical audit

From the repository root, with Python and NumPy 1.26.4:

```bash
python -B scripts/audit_l11_key_value_donor_results.py --report t22_new_public_audit.json
```

Choose a new report path outside the immutable packages. This uses the pinned
public auditor and frozen analysis helpers; it is exact reanalysis rather than
a second independent implementation of the statistics. Original independent
N/P/R audits remain distinct from this packaging check.

## Reconstruct inputs and prepare a replay workspace

Use a private configuration with `models.gpt2.{path,model_id,revision}` and
`sources.{wiki_validation,utilitarianism,ethics,adadelta,adamax,asgd}`. The
[configuration example and acquisition details](GPT2_QUERY_WRITE_TIMING_REPRODUCING.md#lawful-external-inputs-and-private-configuration)
apply unchanged. Exact physical source bytes, processing, tokenizer identity,
component boundaries and decisions are asserted; a text-equivalent but
byte-different Arrow container is not silently accepted. Do not track your
configuration or rebuilt inputs.

```bash
python -B results/l11_key_value_donor_reproduction/experiments/pulse_transfer/l11_key_value_donor_portable.py rebuild \
  --package results/l11_key_value_donor_reproduction \
  --config config.local.json --out .reproduce-t22/inputs

python -B results/l11_key_value_donor_reproduction/experiments/pulse_transfer/l11_key_value_donor_portable.py stage-replay \
  --package results/l11_key_value_donor_reproduction \
  --inputs .reproduce-t22/inputs --config config.local.json \
  --out .reproduce-t22/replay
```

Each output must be fresh. These are reader commands; the local reconstruction
and staging have already been performed once. Staging does not load weights
or execute the prepared model commands. The staged workspace and its absolute
private configuration paths are intentionally excluded from this repository.
No old GPT-2 or Llama experiment is rerun to reconstruct the metadata.

## Model, data and protocol

Use `openai-community/gpt2` revision
`75e09b43581151bd1d9ef6700faa605df408979f`, safetensors file SHA256
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
The observed environment was Python 3.12.3, Torch 2.11.0+cu130,
Transformers 4.57.6, NumPy 1.26.4 and CUDA runtime 13.0. Runs used FP32 SDPA,
batch 1, physical length 512, no cache, seed 20261021, highest FP32 matmul
precision and CUDA matmul TF32 disabled. Shared conditional bootstrap draws
used PCG64 seed 20261021 and 2,000 repetitions.

Obtain the lawful checkpoint, tokenizer and source files separately, using the
existing [source/model manifest](../manifests/transfer_models.json) and
[source retrieval guide](TRANSFER_REPRODUCING.md). Required evaluation sources
are WikiText validation, Gutenberg 11224 (Mill), Gutenberg 3800 (Spinoza), and
PyTorch optimizer files `adadelta.py`, `adamax.py`, `asgd.py` at commit
`70d99e998b4955e0049d13a98d77ae1b14db1f45`. Match physical and processed hashes,
component boundaries and tokenizer identities; a fresh download is not assumed
to have the historical bytes. Model weights, tokenizer payloads, source text,
input/prediction IDs and private configuration files are excluded from Git.

The experiment reuses frozen T18 F, fitted from four Wiki calibration passages
with source-position values zeroed at L4–L11 and an explicit causal mask. T22
evaluation is native000, with the model's attention-mask argument preserved.
It does not refit in native values. Each donor changes only earlier L10 actual
projected writes at queries 1–126. Its recipients transplant earlier L11 keys,
values or both into baseline while keeping every query and source/later K/V at
baseline bytes. Both paths are not naturally dose matched at L11. Consult the
report for the exact scoring, closure, suffix and conditional-uncertainty scope.

## Observed resources

Use one GPU process at a time. The observed RTX 3060 Ti has 8 GiB; host memory
was 16 GiB. Completed segments each took about 309 seconds, with maximum
observed process RSS 2,461,524 KiB and peak allocated GPU memory 1,145,661,440
bytes. These are observations, not minimum requirements or a speed guarantee.
The reviewed primary process ceiling is 4 GiB with at least 6 GiB host memory
available before loading; reviewers remain below 3 GiB. Preserve the 2 GiB
disk reserve and all separate normal/failure allowances. Budget additional
space for lawful inputs and fresh A/B/composite outputs rather than deleting
earlier evidence. No Llama load or prior experiment rerun is needed to rebuild
the recorded input-history metadata.

## Unfinished continuation

A calibration-world comparison is designed but has acquired zero model calls
at this checkpoint. Its first gate is a four-input exact replay of archived
calibration b/mass arrays and the frozen fit identity before collecting native
value calibration. It is a hypothesis and implementation task, not a Report 18
finding or independent calibration replication.
