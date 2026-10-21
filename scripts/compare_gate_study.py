"""Exact CPU comparison of fixed-512 gate results, without loading a model.

Numeric scope compares exported JSON results. Private scope additionally loads
the original and replay tensor files, one shard pair at a time, and independently
reduces each saved per-token metric. A match is not proof of run independence.
No tolerance is selected after observing a replay; cross-platform drift fails
this exact comparator and must be investigated separately.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_COUNTS = {
    'windows.jsonl': 680, 'summary.json': 168, 'segment_summary.json': 336,
    'unexposed_summary.json': 168, 'unexposed_segment_summary.json': 336,
    'paired_contrasts.json': 1024, 'paired_summary.json': 256,
    'unexposed_paired_contrasts.json': 960, 'unexposed_paired_summary.json': 256,
    'gate_geometry.json': 5120, 'checks.json': 744,
    'operation_checks.json': 680, 'suffix_checks.json': 672,
    'suffix_operations.json': 672, 'calibration_operations.json': 8,
    'calibration_spectra.json': 8, 'conditions.json': 21,
}
RAW_FILES = ('calibration_raw.pt', 'calibration_maps.pt', 'calibration_metrics.pt',
             'calibration_tokens.pt', 'tokens.pt', 'per_token.pt', 'traces.pt',
             'source_states.pt', 'attention_snapshots.pt')
MANIFEST_FIELDS = ('study', 'protocol_version', 'model_id', 'revision', 'dtype',
    'attention', 'seed', 'qr_seed', 'layers', 'ranks', 'conditions_sha256',
    'calibration_inputs_sha256', 'evaluation_inputs_sha256',
    'calibration_raw_sha256', 'calibration_maps_sha256',
    'calibration_maps_tensor_digest', 'context_tolerance',
    'physical_execution_lengths', 'cross_length_comparison', 'score_views',
    'snapshot_modes', 'exposed_passages', 'suffix_rule',
    'precision_settings_before', 'precision_settings_after',
    'model_hash_before', 'model_hash_after')
CRITICAL_SOURCES = ('gpt2_gate_fixed512.py', 'analyze_gate_fixed512.py',
    'GPT2_GATE_FIXED512.md', 'gpt2_gate_subspace_v2.py',
    'gpt2_gate_subspace.py', 'analyze_gate_subspace.py', 'analyze_transfer.py',
    'gpt2_query_source.py', 'gpt2_regrowth_kv.py', 'gpt2_layer_writes.py',
    'gpt2_writes.py', 'llama_source_kv.py', 'run_transfer.py')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    if path.suffix == '.jsonl':
        return [json.loads(s) for s in path.read_text().splitlines()]
    return json.loads(path.read_text())


def exact(a, b, path, stats, torch=None):
    """Compare actual values; no hash-only shortcut for saved tensors."""
    if torch is not None and isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), (path, 'tensor type')
        assert a.shape == b.shape and a.dtype == b.dtype, (path, 'shape/dtype')
        # Matching infinities can represent a causal mask; NaN is never a pass.
        if a.is_floating_point():
            assert not torch.isnan(a).any() and not torch.isnan(b).any(), (path, 'NaN')
        assert torch.equal(a, b), (path, 'tensor content mismatch')
        stats['tensors'] += 1
        stats['tensor_elements'] += a.numel()
        return
    assert type(a) is type(b), (path, 'type mismatch')
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, 'keys mismatch')
        for key in a:
            exact(a[key], b[key], path+'/'+str(key), stats, torch)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (path, 'length mismatch')
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path+'/'+str(i), stats, torch)
    else:
        if isinstance(a, float):
            assert math.isfinite(a) and math.isfinite(b), (path, 'nonfinite scalar')
            stats['float_scalars'] += 1
        assert a == b, (path, 'value mismatch')
        stats['leaves'] += 1


def completed(path):
    m, a = read(path/'manifest.json'), read(path/'audit.json')
    assert m['status'] == m['phase'] == 'completed' and m['completed_utc']
    assert m['study'] == 'gate_fixed512' and m['physical_execution_lengths'] == [512]
    assert m['cross_length_comparison'] is False
    assert m['score_views'] == {'full': [2,511], 'early': [2,127], 'later': [128,511]}
    for key, count in {'forward_passes':1352, 'forward_attempts':1352,
        'records':680, 'evaluation_forwards':680, 'suffix_forwards':672,
        'calibration_forwards':0, 'inherited_calibration_forwards':8}.items():
        assert m[key] == count, (path.name, key)
    assert m['parameter_versions_unchanged'] is True and m['calibration_maps_unchanged'] is True
    assert m['model_hash_before'] == m['model_hash_after']
    assert len(m['model_hash_before']) == 64
    assert a['status'] == 'passed' and a['records'] == 680 and a['forwards'] == 1352
    assert a['total_new_gate_context_checks'] == 10240 and a['suffix_audit']['records'] == 672
    return m, a


def indexed_rows(rows):
    result = {r['key']: r for r in rows}
    assert len(result) == len(rows) == 680, 'Duplicate or missing window records'
    return result


def reduce_raw(path, torch):
    raw = torch.load(path/'per_token.pt', weights_only=True, map_location='cpu')
    rows = indexed_rows(read(path/'windows.jsonl'))
    assert set(raw) == set(rows)
    count = 0
    for key, z in raw.items():
        r = rows[key]; stem = key.rsplit('__', 1)[0]
        c, p = raw[stem+'__C'], raw[stem+'__P']
        assert r['length'] == r['input_tokens'] == 512 and r['score_index'] == z['score_index'] == 1
        assert r['scored_tokens'] == 510 and r['target_start'] == 2
        assert z['token_sha256'] == r['token_sha256'] == c['token_sha256'] == p['token_sha256']
        for name in ('loss', 'kl', 'kl_p', 'pred', 'agreement_p'):
            assert z[name].shape == (511,) and torch.isfinite(z[name]).all()
        assert torch.equal(z['agreement_p'], z['pred'].eq(p['pred']))
        values = {'nll': z['loss'][1:].double().mean(),
                  'baseline_nll': c['loss'][1:].double().mean(),
                  'p_baseline_nll': p['loss'][1:].double().mean(),
                  'kl': z['kl'][1:].double().mean(),
                  'kl_p': z['kl_p'][1:].double().mean(),
                  'agreement': z['pred'][1:].eq(c['pred'][1:]).double().mean(),
                  'agreement_p': z['pred'][1:].eq(p['pred'][1:]).double().mean()}
        for field, value in values.items():
            assert float(value) == r[field], (key, field, 'raw reduction mismatch')
            count += 1
    return count


def compare_exported_vectors(reference, candidate, torch):
    """Check safe archived arrays against freshly loaded replay raw tensors."""
    import numpy as np
    index = read(reference/'per_token_index.json')
    rows = indexed_rows(read(reference/'windows.jsonl'))
    assert len(index) == 680 and {r['index'] for r in index} == set(range(680))
    assert {r['key'] for r in index} == set(rows)
    raw = torch.load(candidate/'per_token.pt', weights_only=True, map_location='cpu')
    assert set(raw) == set(rows)
    fields = ('loss', 'kl', 'kl_p', 'matches_native', 'matches_p')
    with np.load(reference/'per_token_numeric.npz', allow_pickle=False) as archive:
        assert set(archive.files) == {f'{i}_{f}' for i in range(680) for f in fields}
        for entry in index:
            key, i = entry['key'], entry['index']; z = raw[key]
            stem = key.rsplit('__', 1)[0]
            assert entry['baseline_key'] == stem+'__C' and entry['p_baseline_key'] == stem+'__P'
            assert entry['score_index'] == rows[key]['score_index'] == z['score_index'] == 1
            assert entry['token_sha256'] == rows[key]['token_sha256'] == z['token_sha256']
            current = {f:z[f][1:].numpy() for f in ('loss','kl','kl_p')}
            current['matches_native'] = z['pred'][1:].eq(raw[stem+'__C']['pred'][1:]).numpy()
            current['matches_p'] = z['pred'][1:].eq(raw[stem+'__P']['pred'][1:]).numpy()
            for field, actual in current.items():
                saved = archive[f'{i}_{field}']
                assert saved.shape == actual.shape == (510,) and saved.dtype == actual.dtype
                assert np.isfinite(saved).all() and np.array_equal(saved, actual), (key, field, 'safe per-token mismatch')
    return 680*5*510


def calibration_operations(rows):
    """Accept only the two frozen calibration record schemas, without invention."""
    common={'corpus','window','gate_checks','max_context_error'}
    original={'phase':'calibration','length':512,'identity':'natural',
        'trial':'calibration','projection_checks':8,'query_checks':8,
        'context_checks':8,'writer_locality_checks':30}
    out=[]
    for r in rows:
        assert set(r) in (common,common|set(original)), 'Unknown calibration-operation schema'
        if set(r)!=common:
            for k,v in original.items():assert r[k]==v,('Original calibration metadata/counter',k)
        assert len(r['gate_checks'])==8
        out.append({k:r[k]for k in common})
    return out


def compare(reference, candidate, scope='numeric'):
    assert scope in ('numeric', 'private')
    assert reference.resolve() != candidate.resolve(), 'Use distinct reference and replay directories'
    ma, aa = completed(reference); mb, ab = completed(candidate)
    stats = {'float_scalars':0, 'leaves':0, 'tensors':0, 'tensor_elements':0}
    exact({k:ma[k] for k in MANIFEST_FIELDS}, {k:mb[k] for k in MANIFEST_FIELDS}, 'manifest/scientific_fields', stats)
    for name in CRITICAL_SOURCES:
        assert ma['source_hashes'][name] == mb['source_hashes'][name], ('source', name)
    # A rebuilt calibration reference may add dependency files and timestamps.
    for name in ma['source_hashes'].keys() & mb['source_hashes'].keys():
        assert ma['source_hashes'][name] == mb['source_hashes'][name], ('shared source', name)
    exact({k:v for k,v in aa.items() if k != 'peak_rss_kib'},
          {k:v for k,v in ab.items() if k != 'peak_rss_kib'}, 'audit/scientific_fields', stats)
    files = []
    for name, count in NUMERIC_COUNTS.items():
        a, b = read(reference/name), read(candidate/name)
        assert len(a) == len(b) == count, (name, 'count')
        if name == 'windows.jsonl':
            a, b = indexed_rows(a), indexed_rows(b)
        if name == 'calibration_operations.json':
            a,b=calibration_operations(a),calibration_operations(b)
        exact(a, b, name, stats)
        files.append({'name':name, 'reference_sha256':sha(reference/name), 'candidate_sha256':sha(candidate/name)})
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    raw_reductions = reduce_raw(candidate, torch)
    exported_elements = 0
    if scope == 'numeric':
        exported_elements = compare_exported_vectors(reference, candidate, torch)
        for name in ('per_token_index.json','per_token_numeric.npz'):
            files.append({'name':name, 'reference_sha256':sha(reference/name),
                          'candidate_source':'directly loaded per_token.pt'})
    if scope == 'private':
        raw_reductions += reduce_raw(reference, torch)
        tensor_files = list(RAW_FILES)
        for folder, inventory in (('gates','gate_files.json'), ('prefix','prefix_files.json'), ('suffix','suffix_files.json')):
            ia, ib = read(reference/inventory), read(candidate/inventory)
            assert len(ia) == len(ib) == 32 and ia.keys() == ib.keys()
            assert set(ia) == {p.name for p in (reference/folder).glob('*.pt')}
            assert set(ib) == {p.name for p in (candidate/folder).glob('*.pt')}
            for name in sorted(ia):
                assert Path(name).name == name, 'Shard must be a basename'
                assert sha(reference/folder/name) == ia[name] and sha(candidate/folder/name) == ib[name]
                tensor_files.append(folder+'/'+name)
        for name in tensor_files:
            a = torch.load(reference/name, weights_only=True, map_location='cpu')
            b = torch.load(candidate/name, weights_only=True, map_location='cpu')
            exact(a, b, name, stats, torch)
            files.append({'name':name, 'reference_sha256':sha(reference/name), 'candidate_sha256':sha(candidate/name)})
            del a, b
            gc.collect()
    return {'status':'passed', 'scope':scope, 'exact':True, 'records':680,
        'new_study_forwards_per_run':1352, 'scalars_reduced_from_raw':raw_reductions,
        'safe_export_per_token_elements_compared':exported_elements,
        'counts':stats, 'files':files,
        'runtime_versions':{'reference':ma.get('runtime_versions'), 'candidate':mb.get('runtime_versions')},
        'sdpa_profiler_events':{'reference':ma.get('sdpa_profiler_events'), 'candidate':mb.get('sdpa_profiler_events')},
        'noncompared_fields':'Run timestamps, wall/memory, calibration reference provenance, environment versions and profiler events (last two reported). Original calibration-operation metadata/counters are validated against their frozen constants; rebuild records omit those fields. All four shared fields, including all64 layer gate checks, are compared. Missing counters are not invented or claimed remeasured.',
        'limitations':('Numeric scope compares safe archived metrics with replay raw per-token tensors; it does not inspect reference activations or prediction IDs. ' if scope == 'numeric' else
            'Saved tensors compared directly; successful full logits and all full contexts were not retained, only runtime hashes and fixed snapshots. ')+
            'Exact agreement does not prove independent execution or validate scientific interpretation. No physical-128 claim.',
        'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, default=ROOT/'results/pulse_transfer/gpt2_gate_fixed512_run')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--candidate', type=Path)
    group.add_argument('--work', type=Path, help='Workspace containing pulse_transfer/gpt2_gate_fixed512_run')
    ap.add_argument('--scope', choices=('numeric','private'), default='numeric')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    candidate = args.candidate if args.candidate else args.work/'pulse_transfer/gpt2_gate_fixed512_run'
    if args.output:
        assert args.output.resolve() not in {p.resolve() for d in (args.reference,candidate) for p in d.rglob('*')}, 'Do not overwrite a run artifact'
        args.output.write_text(json.dumps({'status':'checking', 'scope':args.scope})+'\n')
    try:
        result = compare(args.reference, candidate, args.scope)
    except BaseException as error:
        if args.output:
            args.output.write_text(json.dumps({'status':'failed','scope':args.scope,
                'error_type':type(error).__name__,'reason':str(error)},indent=2)+'\n')
        raise
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
