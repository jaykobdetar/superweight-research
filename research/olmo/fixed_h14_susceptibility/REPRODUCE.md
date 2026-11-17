# Reproducing the fixed-H14 evidence

## 1. Audit the exported numbers (no GPU, inputs or network)
From this directory run:
```sh
python3 -B audit.py
python3 -B -m unittest -v test_audit
```
The standard-library audit verifies the SHA-256 manifest, vector byte hashes, the 24-window × three-query panel, shared endpoints between independent runs, actual write-dose match, per-query activation/functional gates, and paired full-cost and dose reductions. It reads, never overwrites, the published CSVs. It does not independently regenerate model outputs. Python 3.11+ for the optional replay.

## 2. Independently reproduce one literary model window (optional)
Bring a local compatible OLMo FP16 safetensors directory and a separately obtained NLTK Gutenberg archive. No weights, corpus text, or input/prediction token sequences are published here and the script does not download them. The original release identifier is `allenai/OLMo-1B-0724-hf@d7cbab742d80589e714b1a2d7f838dcd21cbe143`; the measured runtime used exact recovered FP16 parameters, not the original higher-precision file representation. Both full named-parameter and tokenizer hashes are checked. A matching model name alone is insufficient. Archive SHA-256: `2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611` ([official package index](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml)). Original GPU: RTX 3060 Ti; PyTorch 2.8.0+cu128, Transformers 4.57.3, Accelerate 1.12.0, eager attention, FP16 activations and FP64 metric reductions. CUDA kernel/hardware changes need not be bitwise equivalent.

```sh
python3 -B replay.py --model /path/to/local/olmo --corpus /path/to/gutenberg.zip --book austen-emma --anchor 20
# Explicitly opt in on an idle GPU with sufficient RAM/VRAM:
python3 -B replay.py --model /path/to/local/olmo --corpus /path/to/gutenberg.zip --book austen-emma --anchor 20 --run
```
Without `--run` only the source window is verified using the local tokenizer. With it, the script verifies the model's FP16 parameter digest, applies the hooks, checks bitwise identities and pre-source logits, and compares activation and C/V/J/KL/NLL/top-1 readouts with saved results for that window (default absolute tolerance 1e−8). Select each manifest book and anchor to cover the full panel. Use local resource limits; the script requires at least 6 GiB free VRAM at startup. Opt-in replays are separate executions, not additional independent evidence.

## Protocol and source lineage
- [G1](protocols/G1.md), [U1](protocols/U1.md), [U2](protocols/U2.md), [N1](protocols/N1.md), [L1](protocols/L1.md), [E1](protocols/E1.md) preserve their pre-outcome designs. E1's existing full-dose endpoints were not a fresh data split; its newly measured C and J endpoints and fractional-dose results are separate.
- [Template source metadata](data/template_sources.json): eight CPython stdlib member hashes and exact source/window offsets, no source code or token sequences. Original 66-token windows had original query newline at offset 32. Replace only position 8 with newline ID187. Compute H14 context before the layer-0 attention output projection at 32 in each history, subtract earlier-newline from original, mean the eight 128-dimensional differences in FP64. No target-specific adjustment. [Frozen vectors](data/vectors.json) include all controls; U2 defines their generator.
- [Literary manifest](data/literary_manifest.json): per-work archive-member hashes and three selected windows' token hashes and offsets, including decoded-line-ending protocol. The exact report units are eight works, not 768 independent targets. Use the published window hashes and offsets for replay.
- [Literary query CSV](data/literary_queries.csv): 720 rows = 24 windows × three current queries × native/template/eight matched controls. `same_world_scalar_lesion_KL` always compares the intact arm with its own lesion. Gate columns describe the template for that query, repeated for convenience; they are not per-random-direction successes.
- [Literary dose CSV](data/literary_doses.csv): 312 rows = 24 windows × (five frozen amplitudes + eight full-dose random controls). C = native vs intact arm; V = intact arm vs its own lesion; J = native vs lesioned arm. These directional KLs need not add.
- [Precursors](data/precursor_summaries.json) give aggregate or per-source numeric evidence for B–H1, not complete forward traces. [Earlier controls](data/earlier_controls.csv): 1,280 G1/U1/U2/N1 measurements. G1 donor-head rows have no focal KL if the focal perturbation wasn't run; U1 d0..d7 are other document donors and r0..r7 are tangent controls. U2/N1 r0..r7 are shared frozen equal-write-length vectors. In G1/U1/U2 orig/newline denotes recipient history; in N1 newline/neighbor/substitute denotes current query. The file-level gates refer respectively to joint history transfer, direction specificity (technically inconclusive also counts false), template transfer and natural-newline functional induction. A blank field is not zero.

No inference on a second checkpoint, generation benchmark, change in global task performance, or training counterfactual is part of this release.

**Release validation:** the portable script was run on the original local FP16 parameter identity at *Emma*, 20% anchor: 69 model forwards, 237 exported scalar comparisons, maximum absolute discrepancy **0**. This is a public-script smoke replay of one window, not a new literary holdout. Protocol files have trailing whitespace normalized; [source hashes](protocols/SOURCE_HASHES.json) distinguish original and published bytes.
