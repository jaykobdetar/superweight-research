"""Run the unchanged frozen T18 auditor with exact QR storage-layout restoration.

NPZ records tensor values but not strides. The production QR returns column-major
matrices; an exact-valued row-major replay can change FP64 reduction rounding.
No tolerance, model output, direction or frozen auditor is changed here.
"""
import argparse
from pathlib import Path
import torch
import audit_gate_wiki_results as frozen
from gpt2_gate_subspace import maps_digest
from run_transfer import sha, dump

FROZEN_SHA = 'ebae29f3c04c937aa8abf7dcbb608af91ee0a957f3ffface128e914e6b5f1686'

def restore_qr_layout(maps):
    assert set(maps) == {'O', 'F', 'M'}
    before = {label: maps_digest(value) for label, value in maps.items()}
    count = 0
    for label, layers in maps.items():
        assert set(layers) == set(range(4, 12))
        for layer, values in layers.items():
            qr = values['basis_qr']
            assert qr.dtype == torch.float64 and qr.shape == (12, 12)
            assert torch.isfinite(qr).all() and qr.stride() == (12, 1)
            restored = qr.T.contiguous().T
            assert restored.stride() == (1, 12) and torch.equal(restored, qr)
            values['basis_qr'] = restored
            count += 1
    assert count == 24
    assert before == {label: maps_digest(value) for label, value in maps.items()}
    return maps

def audit(run):
    assert sha(frozen.__file__) == FROZEN_SHA
    original = frozen.read_npz
    restored_paths = []
    def read_with_layout(path):
        result = original(path)
        if Path(path) == run / 'calibration_maps.npz':
            assert not restored_paths
            result = restore_qr_layout(result)
            restored_paths.append('calibration_maps.npz')
        return result
    frozen.read_npz = read_with_layout
    try:
        result = frozen.audit(run)
    finally:
        frozen.read_npz = original
    assert restored_paths == ['calibration_maps.npz']
    result['layout_replay'] = {
        'adapter_sha256': sha(__file__), 'frozen_auditor_sha256': FROZEN_SHA,
        'field': 'O/F/M L4-L11 basis_qr only', 'matrices': 24,
        'serialized_stride': [12, 1], 'original_stride': [1, 12],
        'map_values_and_digests_unchanged': True,
        'directions_and_geometry_exact': True,
        'tolerances_changed': False,
        'reason': 'NPZ loses tensor strides; preserve original QR reduction order.'}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    target = args.run / 'saved_state_audit.json'
    assert not target.exists()
    result = audit(args.run)
    dump(target, result)
    print(result['status'], result['totals'])
