# Independent four-passage fitter review

**Passed for use in the prospective T18 implementation**, at fitter SHA256 `9cc6ea695f6c3753f8878a5d5146fa9a371be264685c2e26d8f831cd2b147c7e`. The original eight-passage fitter remains unchanged at `6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e`.

The independent CPU audit reran the generalized n=8 fit and the unchanged legacy fit against the exact frozen raw calibration file. Every saved map tensor and the complete spectra/leave-one-out reports match exactly; the map digest remains `fbf127a1a618e709b0d8e39625a156f11402c9703ac4b55ae3dcfd3711571c81`.

The four-passage subset is exactly `wiki_test__0` through `wiki_test__3` in original order. Its new maps, spectra and directions reproduce the retained `wiki_calibration_preparation/old_wiki_*` artifacts exactly; the W_old4 map digest is `e59fab6ed96e92d34a3286680b989f8ca12608edee7923baaf088f5d1b40573d`. It is a new Wiki-only fit, not a renamed mixed-eight object. Raw and output file hashes are bound in `gate_wiki_fit_independent_audit.json`.

Independent formulas checked all 32 full layer/space means, centered covariances and eigenvalues; 128 leave-one-passage-out spaces; and 384 reported principal-cosine values. All differences were zero. Full fits pool 2,040 query vectors with covariance denominator 2,039; leave-one-out fits pool 1,530 with denominator 1,529. Position-centered statistics are correctly recomputed after each omission. An independent symmetric eigensolver also agrees with the SVD eigenvalues within the inherited tolerance. QR matrices remain identical to M8, and PC1/basis signs satisfy the inherited pivot convention. Invalid expected sample counts were rejected rather than duplicating observations.

The eight W_old4 relative leading gaps range from 0.1934103 to 0.7099472, above the fixed 1e-12 floor. These are geometry diagnostics, not independent calibration uncertainty or evidence of general stability. Four-passage leave-one-out variation is descriptive; evaluation intervals will still condition on the fitted maps.

The reviewed revision document is `GPT2_GATE_WIKI_CALIBRATION_REVISION.md` SHA256 `0b437fcdd53ade96f9bf3562bf307edeae7bf62b9da002823179f8a58dfe9765`. It clearly distinguishes W_old4, retained W_fresh4 and M8, preserves the failed Hume attempt, and keeps the intended 24-mode/1,548-forward scope. Any final storage amendment, fresh-input outcome, hook implementation or model launch remains a separate review gate.

Audit resource use: 0.760 seconds, peak 756,472 KiB RSS, zero model/tokenizer loads, zero model forwards and no GPU. Only saved small calibration tensors were loaded. No production source, preparation artifact or original failed study was edited.
