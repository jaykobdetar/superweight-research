# Explicit gate-calibration rebuild

This command performs exactly eight calibration forwards and no evaluation
forwards. It uses the fixed passages and unchanged algorithm from
`GPT2_GATE_SUBSPACE.md`. It is a separate replication artifact, not an automatic
fallback or a way to select calibration maps from evaluation results.

From the workspace containing `pulse_transfer/`:

```bash
/tmp/superweight-venv/bin/python pulse_transfer/rebuild_gate_calibration.py \
  --config pulse_transfer/inputs.local.json \
  --out pulse_transfer/gpt2_gate_calibration_rebuild \
  --reference pulse_transfer/gpt2_gate_subspace_run
```

The environment path is local to this workstation; another researcher should
use their environment with the pinned dependencies and lawful input/checkpoint
retrieval described in the transfer reproduction guide. The config names the
local GPT-2 checkpoint and corpus files. Existing input/selection manifests
from earlier stages are prerequisites because they define exclusions and
checkpoint identity. This command does not require the failed gate study to
execute or fail: `--reference` is optional, and can point to a separately
supplied reference calibration directory containing `calibration_maps.pt` and
`calibration_raw.pt`. Output must be new or empty.

Eight natural-start 512-token passages, four each from Wiki test and Hume,
use the same fixed fractions and exclusions. Source values are zeroed in
L4–11, all query biases stay intact, and the explicit causal mask is used.
The model remains FP32 with default SDPA. Parameters and precision flags
are checked for changes. Saved calibration KL is self-reference zero; it is
not an evaluation against unmodified native C.

The original `prepare_gate_inputs` helper also tokenizes the fixed evaluation
panel and writes private token files. No evaluation passage is forwarded,
scored, fitted, or used to select a map. Do not export those token files.

The unchanged `fit_maps` implementation consumes queries 1–510, 4,080
correlated vectors per layer. It preserves FP64 centered sample covariance,
SVD sign conventions, log(q) detrending, QR seed 20261025, ranks 1/3/6, and
leave-one-passage-out descriptive statistics. No optimizer or search is used.

Outputs include raw calibration gates, maps, spectra, self-reference metrics,
operation checks and a manifest of source/input/model hashes, versions,
forward counts and resources. With `--reference`, a separate comparison
records byte and tensor equality, maximum tensor differences, and principal
cosines/projector differences for the fixed subspaces. A mismatch is recorded
without overwriting the reference, adjusting calibration, or automatically
substituting rebuilt maps into the main experiment. Cross-platform differences
must remain disclosed in a separately labeled replication.

Failures retain partial calibration outputs and the active diagnostic state.
No thresholds or kernels change automatically. Use one local GPU model at a
time; the main experiment and this rebuild must run sequentially. This command
adds eight actual forwards to resource accounting even if its maps match the
reference exactly. Successful local byte equality is a same-machine check,
not independent external replication.
