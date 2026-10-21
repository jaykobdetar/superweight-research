# Independent calibration rebuild implementation review

Status: ready for the separately accounted eight-forward calibration rebuild. No launch blocker found. This is an implementation signoff, not a claim that rebuilt pretrained outputs already match the reference or that V2 is ready to execute.

Reviewed frozen bytes:

- `rebuild_gate_calibration.py`: `43172c7ee68af83d186859e8a3a29a92eb702682a8c53e4bb0f2454fc57b4b30`
- `GATE_CALIBRATION_REBUILD.md`: `4a9b3aa8cd0c442cfb295a1f035de95421ad66fb8edf59635bfed26878204981`

The script forwards only the eight original calibration passages. The helper also prepares sixteen evaluation input files, which are neither forwarded nor used in fitting. The original GateSubspace, GateRunner, input helper and FP64 fitting functions remain authoritative; the model path requires local FP32 GPT-2, SDPA, the original checkpoint revision and parameter hash. Source values are zeroed in L4–11 with intact query bias and no coordinate lesions. Calibration KL is explicitly self-reference, not native-C fidelity.

Reference comparison is read-only and optional. It reports exact raw/map tensors, serialized bytes, and sign-invariant subspace comparisons without replacing maps, adjusting parameters, rerunning calibration or selecting against evaluation. A numerical mismatch can coexist with a completed rebuild; completion means execution finished, not reference equivalence. The primary V2 study must continue to name its explicit calibration reference and verify its hashes. Deliberately supplying a separately rebuilt reference is a distinct provenance choice, never an automatic fallback.

The latest metadata additions were included in this review: start identity summary, evaluation input-manifest hash, calibration freeze time and NumPy version. Source hashes cover the documentation and imported experiment/fitting dependencies. Unused analyzers need not be included in this script's dependency list. Parameter hashes/version counters, source hashes and precision settings are checked at completion; failures preserve partial calibration metrics/raw gates and active diagnostic state. The documentation accurately distinguishes same-machine equality from independent replication and adds eight forwards to resource accounting.

Independent CPU evidence: `rebuild_gate_calibration_tests.py` and `.json` passed twelve tests. These cover identical bytes/tensors; sign-changed bases with the same subspace; a finite mismatch without fallback; shape/dtype/NaN/Inf rejection; unchanged reference files; exactly eight calibration-only stub calls; explicit reference comparison without substitution; failure on the third calibration item retaining the first two records without retries or evaluation; and occupied-output rejection. The real fitting function ran on synthetic gates. No model or tokenizer was loaded, no GPU was used, and peak RSS was 754,508 KiB. The three orchestration fixtures made eighteen successful stub forwards total, not model forwards.

Limits: the fixtures verify control flow, comparisons and failure handling, not pretrained numerical reproducibility or measured GPU resources. Those require the separately authorized actual rebuild. Original V1 failure and frozen artifacts remain untouched. No experiment source edits were made by this reviewer.

The fixture was refreshed to supply the newly required literal/start-identity and data-manifest metadata, then all twelve checks were rerun against the final hash above. The earlier fixture metadata omission was a test-stub mismatch, not an experiment failure.

## Completed rebuild audit

The actual eight-forward rebuild passed the independent saved-artifact audit (`rebuild_gate_calibration_numerical_audit.py/.json`). Raw calibration gates and maps are byte-identical to the original reference, whose previously frozen hashes remain unchanged: raw `d41d5a91…`, maps `ed62f089…`. All 176 corresponding tensors match exactly; eight passage identities, eight map layers, 64 operation checks, source hashes, input manifests and calibration token hashes pass. Independent FP64 mean and residual-covariance checks agree (largest eigenbasis off-diagonal residual 1.33e-14). Maximum recorded context reconstruction error is 2.91e-5, below the retained 1e-4 guard.

Manifest accounting is eight calibration forwards and zero evaluation forwards; measured runner time is 3.61379385 seconds, peak GPU allocation 645,195,776 bytes and process RSS 1,828,584 KiB. Checkpoint hash before/after and parameter-version closure are recorded by the runner. The independent reviewer did not reload the model to remeasure those claims. Review used no GPU/model and peaked at 495,292 KiB RSS. The original gate-study manifest remains failed. This establishes same-machine calibration reproduction, not independent external replication or V2 implementation approval.
