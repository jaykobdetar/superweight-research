"""Prepare T18 with safe metadata and an existing validated M8 calibration.

Default is zero-forward preparation. --execute explicitly requests only the
1,548 T18 forwards, never a bootstrap or an older model study. Draft pending
independent implementation review; no invocation here authorizes publication.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUPPORT_SHA = '453e808115b2d8d3cc8c72be06e7d34a83c012db61bf2b9dfe6cf2e00394f156'
METADATA_SHA = '7e3a43808ee51ab8a44526d3cb9e9f545e12ac771a1723ca2d8ca9f360a8a0b6'
CHANGED = {'calibration_inputs.json', 'data_manifest.json', 'exclusion_inventory.json',
           'prior_exclusion_inventory.json', 'start_identity_summary.json', 'preflight.json'}
PARENT_SHA = {'manifest.json': '60909cdc2af2240ba99bf103c3cbf62146bddffd43c4f8f6810d092632fb5515',
              'selection.json': '56bbb1076fc342ae1f504bf8bca0d72b521ff1b69269e51575375a144eba50ae'}
OLD_WIKI_DIGEST = 'e59fab6ed96e92d34a3286680b989f8ca12608edee7923baaf088f5d1b40573d'
FROZEN_AUDITOR_SHA = 'ebae29f3c04c937aa8abf7dcbb608af91ee0a957f3ffface128e914e6b5f1686'
LAYOUT_ADAPTER_SHA = 'c837ab07243683e08dcfd0288fcf76a7ca8372dfdad06d605a01a80fe23f3974'
PHYSICAL_SOURCES = ('wiki_test', 'wiki_validation', 'utilitarianism', 'ethics', 'adadelta', 'adamax', 'asgd')
STAGING_LIMIT = 64 * 1024**2
RESERVE = 2 * 1024**3


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def dump(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def support():
    """Reuse already reviewed stdlib-only reference and failure validators."""
    assert sha(ROOT / 'reproduce_additive.py') == SUPPORT_SHA
    return importlib.import_module('reproduce_additive')


def copy_exact(src, dst, expected):
    assert sha(src) == expected, ('Source hash mismatch', str(src))
    assert not dst.exists(), 'Never overwrite a staged file'
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    assert sha(dst) == expected


def assigned(text, name):
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1 and len(nodes[0].targets) == 1, ('One plain assignment required', name)
    return nodes[0], ast.literal_eval(nodes[0].value)


def bind_input_hashes(source, bindings):
    node, original = assigned(source, 'INPUT_HASHES')
    assert original == bindings['original'] and len(original) == 14
    assert bindings['kind'] == 'explicit_T18_metadata_binding_only'
    assert bindings['metadata_identity_sha256'] == METADATA_SHA
    assert set(bindings['changed_metadata_files']) == CHANGED and len(bindings['changed_metadata_files']) == 6
    assert set(bindings['exact_canonical_files']) == set(original) - CHANGED and len(bindings['exact_canonical_files']) == 8
    assert set(bindings['regenerated']) == set(original)
    for n, h in bindings['regenerated'].items():
        assert isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
        if n not in CHANGED: assert h == original[n], ('Canonical tensor/history changed', n)
    lines = source.splitlines(keepends=True)
    changed = ''.join(lines[:node.lineno-1]) + 'INPUT_HASHES = ' + repr(bindings['regenerated']) + '\n' + ''.join(lines[node.end_lineno:])
    def normalized(text):
        tree = ast.parse(text)
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INPUT_HASHES' for t in n.targets):
                n.value = ast.Constant(value='ONLY_INPUT_HASH_BINDING')
        return ast.dump(tree, include_attributes=False)
    assert normalized(source) == normalized(changed), 'Intervention/source semantics changed'
    return changed


def inventory_valid(source_dir, files):
    required = {'gpt2_gate_wiki_calibration.py', 'gate_wiki_calibration.py', 'gate_wiki_storage.py',
        'fit_gate_calibration.py', 'gate_direction_geometry.py', 'gate_factorial_algebra.py',
        'analyze_gate_wiki_calibration.py', 'audit_gate_wiki_results.py', 'audit_gate_wiki_results_layout.py',
        'prepare_wiki_calibration_replay_inputs.py', 'reproduce_wiki_calibration.py', 'reproduce_additive.py'}
    assert required <= files.keys()
    for name, h in files.items():
        assert Path(name).name == name and Path(name).suffix in ('.py', '.md', '.json')
        assert isinstance(h, str) and len(h) == 64 and sha(source_dir / name) == h, ('Source inventory', name)
    _, owns = assigned((source_dir / 'gpt2_gate_wiki_calibration.py').read_text(), 'SOURCE_NAMES')
    assert set(owns) <= files.keys()
    for name in files:
        if not name.endswith('.py'): continue
        for node in ast.walk(ast.parse((source_dir / name).read_text())):
            imported = ([node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module
                        else [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [])
            for module in imported:
                if (source_dir / (module+'.py')).is_file():
                    assert module+'.py' in files, ('Unstaged local import', name, module)
    assert files['reproduce_additive.py'] == SUPPORT_SHA
    assert files['audit_gate_wiki_results.py'] == FROZEN_AUDITOR_SHA
    assert files['audit_gate_wiki_results_layout.py'] == LAYOUT_ADAPTER_SHA


def space_guard(work, projected=STAGING_LIMIT):
    free = shutil.disk_usage(work).free
    assert free >= RESERVE + projected, ('Insufficient staging space', free, projected)
    return free


def verify_completed_child(out, files):
    inherited = support(); m = read(out / 'manifest.json')
    assert m['status'] == m['phase'] == 'completed' and m['study'] == 'gate_wiki_calibration'
    assert m['forward_passes'] == m['forward_attempts'] == 1548 and m['records'] == 776
    assert m['calibration_forwards'] == 4 and m['evaluation_forwards'] == 776 and m['suffix_forwards'] == 768
    assert m['model_hash_before'] == m['model_hash_after'] == inherited.MODEL_HASH
    assert m['parameter_versions_unchanged'] is True and m['calibration_maps_unchanged'] is True
    assert m['reference_map_digest'] == inherited.MAP_DIGEST and m['old_wiki_map_digest'] == OLD_WIKI_DIGEST
    assert m['calibration_map_digests']['M'] == inherited.MAP_DIGEST and m['calibration_map_digests']['O'] == OLD_WIKI_DIGEST
    analysis = read(out / 'wiki_factorial_analysis.json'); audit = read(out / 'saved_state_audit.json')
    assert analysis['source_sha256'] == files['analyze_gate_wiki_calibration.py']
    assert len(analysis['window_metrics']) == 2328 and len(analysis['summary']) == 576 and len(analysis['contrasts']) == 912
    assert audit['status'] == 'passed' and audit['source_sha256'] == files['audit_gate_wiki_results.py'] == FROZEN_AUDITOR_SHA
    assert audit['fresh_fit_and_geometry_exact'] is True
    layout = audit['layout_replay']
    assert set(layout) == {'adapter_sha256', 'frozen_auditor_sha256', 'field', 'matrices',
        'serialized_stride', 'original_stride', 'map_values_and_digests_unchanged',
        'directions_and_geometry_exact', 'tolerances_changed', 'reason'}
    assert layout['adapter_sha256'] == files['audit_gate_wiki_results_layout.py'] == LAYOUT_ADAPTER_SHA
    assert layout['frozen_auditor_sha256'] == files['audit_gate_wiki_results.py'] == FROZEN_AUDITOR_SHA
    assert layout['field'] == 'O/F/M L4-L11 basis_qr only' and layout['matrices'] == 24
    assert layout['serialized_stride'] == [12, 1] and layout['original_stride'] == [1, 12]
    assert layout['map_values_and_digests_unchanged'] is True and layout['directions_and_geometry_exact'] is True
    assert layout['tolerances_changed'] is False
    assert layout['reason'] == 'NPZ loses tensor strides; preserve original QR reduction order.'
    assert audit['totals'] == {'inputs':32,'records':776,'suffix_records':768,'checks':1128,
                             'gate_checks':11808,'snapshot_states':11,'norm_layer_states':3136}
    return {'status': 'completed', **inherited.child_outcome(out)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'metadata', 'calibration-reference', 'parent-metadata', 'source-dir', 'source-manifest', 'work'):
        ap.add_argument('--'+name, type=Path, required=True)
    modes = ap.add_mutually_exclusive_group(); modes.add_argument('--prepare-only', action='store_true'); modes.add_argument('--execute', action='store_true')
    args = ap.parse_args(); started = time.time(); work = args.work.resolve()
    assert not work.exists() or not any(work.iterdir()), 'Use a fresh workspace; preserve all attempts'
    work.mkdir(parents=True, exist_ok=True); dest = work / 'pulse_transfer'; dest.mkdir()
    out = dest / 'gpt2_gate_wiki_calibration_run'
    state = {'status': 'checking', 'model_forwards': 0, 'calibration_forwards': 0, 'model_loaded': False,
             'gpu_used': False, 'requested_execute': args.execute, 'wrapper_sha256': sha(__file__)}
    dump(work / 'wiki_calibration_preparation.json', state)
    inherited = None
    try:
        state['free_bytes_before'] = space_guard(work)
        files = read(args.source_manifest)['files']; inventory_valid(args.source_dir, files)
        assert files['reproduce_wiki_calibration.py'] == sha(__file__)
        inherited = support(); calmeta, calfiles = inherited.calibration_valid(args.calibration_reference, files)
        assert sha(args.metadata / 'identity.json') == METADATA_SHA
        identity = read(args.metadata / 'identity.json')
        assert set(p.name for p in args.metadata.iterdir()) == set(identity['metadata_sha256']) | {'identity.json'}
        assert identity['kind'] == 't18_wiki_safe_metadata_v1' and identity['canonical_calibration_maps_sha256'] == inherited.CAL_SHA['calibration_maps.pt']
        parents = {}
        for name, h in PARENT_SHA.items():
            assert sha(args.parent_metadata / name) == h; parents[name] = read(args.parent_metadata / name)
        for name, h in files.items(): copy_exact(args.source_dir / name, dest / name, h)
        for name, h in PARENT_SHA.items(): copy_exact(args.parent_metadata / name, dest / 'gpt2_run' / name, h)
        cal = dest / 'wiki_replay_calibration'
        for name, h in {**calfiles, 'manifest.json': sha(args.calibration_reference / 'manifest.json')}.items():
            copy_exact(args.calibration_reference / name, cal / name, h)
        metadata = dest / 'wiki_replay_metadata'
        for name, h in {**identity['metadata_sha256'], 'identity.json': METADATA_SHA}.items():
            assert Path(name).name == name; copy_exact(args.metadata / name, metadata / name, h)
        supplied = read(args.config)
        cfg = {'models': {'gpt2': copy.deepcopy(supplied['models']['gpt2'])},
               'sources': {name: str(Path(supplied['sources'][name]).expanduser().resolve()) for name in PHYSICAL_SOURCES}}
        cfg['models']['gpt2']['path'] = str(Path(cfg['models']['gpt2']['path']).expanduser().resolve())
        config = dest / 'inputs.local.json'; dump(config, cfg)
        inputs = dest / 'calibration_wiki_inputs'
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        def invoke(script, *arguments, cpu=False):
            subprocess.run([sys.executable, str(dest / script), *map(str, arguments)], cwd=dest,
                           env={**env, 'CUDA_VISIBLE_DEVICES': ''} if cpu else env, check=True)
        invoke('prepare_wiki_calibration_replay_inputs.py', 'rebuild', '--config', config,
               '--metadata', metadata, '--calibration', cal, '--out', inputs, cpu=True)
        rebuilt = read(inputs / 'reconstruction.json')
        assert rebuilt['status'] == 'passed' and rebuilt['model_forwards'] == 0 and rebuilt['model_loaded'] is False and rebuilt['gpu_used'] is False
        invoke('prepare_wiki_calibration_replay_inputs.py', 'verify-reconstructed', '--metadata', metadata, '--out', inputs, cpu=True)
        bindings = read(inputs / 'input_bindings.json'); canonical = (dest / 'gpt2_gate_wiki_calibration.py').read_text()
        portable = bind_input_hashes(canonical, bindings)
        original = dest / 'staging_source/gpt2_gate_wiki_calibration.canonical.py'; original.parent.mkdir()
        original.write_text(canonical); assert sha(original) == files['gpt2_gate_wiki_calibration.py']
        (dest / 'gpt2_gate_wiki_calibration.py').write_text(portable)
        staged = {n: sha(dest / n) for n in files}
        assert {n for n in files if staged[n] != files[n]} == {'gpt2_gate_wiki_calibration.py'}
        staged_bytes = sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
        assert staged_bytes + 16384 <= STAGING_LIMIT and shutil.disk_usage(work).free >= RESERVE + 16384
        state.update(status='prepared', staged_bytes=staged_bytes, source_manifest_sha256=sha(args.source_manifest),
            canonical_source_sha256=files, staged_source_sha256=staged, metadata_identity_sha256=METADATA_SHA,
            calibration_files_sha256=calfiles, calibration_reference_manifest_sha256=sha(args.calibration_reference / 'manifest.json'),
            calibration_reference_kind=calmeta.get('kind', calmeta.get('study', 'original_reference')),
            parent_metadata_sha256=PARENT_SHA, input_bindings_sha256=sha(inputs / 'input_bindings.json'),
            reconstruction_sha256=sha(inputs / 'reconstruction.json'), ast_only_input_hashes_changed=True,
            scope='20 current windows rebuilt with GPT2 only; historical exposure and prior-start membership are inherited attestations. Six metadata hashes intentionally differ. No older model study/bootstrap invoked.')
        dump(work / 'wiki_calibration_preparation.json', state)
        if not args.execute:
            print('PREPARED_T18_REPLAY', dest, 'zero model forwards'); return
        for name, record in parents['manifest.json']['checkpoint_files'].items():
            p = Path(cfg['models']['gpt2']['path']) / name
            assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256']
        for name, h in staged.items(): assert sha(dest / name) == h
        for name, h in bindings['regenerated'].items(): assert sha(inputs / name) == h
        state.update(status='execution_requested', model_forwards=None, calibration_forwards=None, model_loaded=None, gpu_used=None)
        dump(work / 'wiki_calibration_preparation.json', state)
        invoke('gpt2_gate_wiki_calibration.py', '--config', config, '--inputs', inputs,
               '--calibration-reference', cal, '--out', out)
        invoke('analyze_gate_wiki_calibration.py', out, cpu=True)
        invoke('audit_gate_wiki_results_layout.py', out, cpu=True)
        state.update(verify_completed_child(out, staged))
        dump(work / 'wiki_calibration_reproduction.json', {'status': 'completed', 'new_model_forwards': 1548,
            'new_calibration_forwards': 4, 'study_manifest_sha256': sha(out / 'manifest.json'),
            'analysis_sha256': sha(out / 'wiki_factorial_analysis.json'), 'saved_state_audit_sha256': sha(out / 'saved_state_audit.json'),
            'saved_state_layout_adapter_sha256': staged['audit_gate_wiki_results_layout.py'],
            'frozen_saved_state_auditor_sha256': staged['audit_gate_wiki_results.py'],
            'scope': 'Successful local run with saved-state audit; not an independent external replication, numeric reference comparison or separate readout audit.'})
    except BaseException as error:
        if args.execute and state.get('model_forwards') is None and inherited is not None:
            state.update(inherited.child_outcome(out))
        state.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        state.update(wall_seconds=time.time()-started, peak_wrapper_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(work / 'wiki_calibration_preparation.json', state)


if __name__ == '__main__': main()
