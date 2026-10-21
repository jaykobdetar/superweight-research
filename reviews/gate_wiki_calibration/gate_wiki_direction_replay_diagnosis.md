# T18 saved-direction replay diagnosis

The frozen saved-state auditor failed after exact fresh map/spectrum reconstruction, when rebuilding directions from NPZ-decoded maps. No experiment or fitting defect was found in this diagnostic.

The lossless NPZ codec preserves tensor values, shape and dtype, but makes tensors contiguous. All three original map sets have `basis_qr` stride `(1,12)`; decoded maps have `(12,1)`. Other map tensors retain their original layout. The direction algorithm selects a QR column and computes its orthogonal residual. Replaying this calculation on the contiguous QR matrices changed 13 saved FP64 `v` vectors by at most 1.1102230246251565e-16. All affected FP32 vectors were exactly equal. Derived hashes and some FP64 diagnostic scalars changed; the geometry report remained exactly equal.

Two independent replay routes recovered exact equality of every direction tensor, hash, scalar and geometry field:

1. Refit O and F from their original raw calibration data and load M from its original PT file.
2. Use only the run's decoded maps, restoring each `basis_qr` as `v.T.contiguous().T`.

The second route changes storage layout only. Every map value remains exactly equal, and no threshold relaxation is required. A separately versioned auditor may use this restoration after validating map values/digests, while preserving the original frozen auditor and its failure log. The primary run, raw inputs, maps, intervention vectors and outputs must remain unchanged.

Evidence: `gate_wiki_direction_replay_diagnosis.py`, `.json`, and `.log`. CPU only, no model/GPU, peak RSS786064KiB. This diagnosis is not a full numerical-result audit; that remains pending the repaired saved-state audit.
