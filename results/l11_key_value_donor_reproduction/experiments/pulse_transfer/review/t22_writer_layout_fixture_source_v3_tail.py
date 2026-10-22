"""T22 tail storage-layout witness SOURCE CANDIDATE; execution remains on HOLD.

This file does not run a fake/pretrained forward. A separately approved, passed
small changed-ABI fixture supplies the actual normal/failure containers. Every
array occurrence then needs a literal reviewed shape-map entry bound to a
specific resource-inventory leaf. It is a storage/memory witness, not a claim
that the independent entropy values are full-width model activations.

Only stdlib imports occur before the explicit execution/review gates. No
automatic cleanup, replacement, hardlinks, alias codec, or retries are used.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RSS_LIMIT = 3 * 1024**3
NORMAL_LIMIT = 512 * 1024**2
FAILURE_LIMIT = 1024**3
RESERVE = 2 * 1024**3
PREPARATION_ALLOWANCE = 32 * 1024**2
OTHER_METADATA_ALLOWANCE = 32 * 1024**2
SIDECAR_ALLOWANCE = 3 * 64 * 1024
ANCHORS = ('native_repeat', 'native_identity')
CORPORA = ('wiki_validation', 'utilitarianism', 'ethics', 'optimizers_new')
RAW_GROUPS = ('snapshots', 'l11_panels', 'readouts', 'timing_samples')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def fresh(path, protected):
    path = Path(path).absolute()
    assert not path.exists() and not path.is_symlink(), 'Fresh output required'
    assert all(not item.is_symlink() for item in path.parents), 'Symlink ancestor'
    for source in protected:
        source = Path(source).resolve()
        assert not path.is_relative_to(source), ('Output inside source artifact', source)
    path.mkdir(parents=True, exist_ok=False)
    return path


def path_key(path):
    # JSON arrays distinguish integer sequence indices from dictionary strings.
    return json.dumps(path, separators=(',', ':'))


def check_manifest(path):
    """Pin source/artifact/map files before imports, output creation or allocation.

    The review file is an explicit root-reviewed execution contract. It contains
    exact dependency hashes, successful small-fixture report, both templates,
    both descriptor lists and the fully enumerated production shape map.
    No editable path is allowed to substitute an unreviewed template or map.
    """
    contract = json.loads(Path(path).read_text())
    assert contract['status'] == 'approved_T22_layout_fixture_execution'
    assert contract['fixture_source_sha256'] == sha(__file__)
    required = ('small_report', 'normal_template', 'normal_descriptors',
                'failure_template', 'failure_descriptors', 'shape_map', 'first_failure_report')
    for key in required:
        item = contract[key]
        assert Path(item['path']).is_file() and not Path(item['path']).is_symlink()
        assert sha(item['path']) == item['sha256'], key
    report = json.loads(Path(contract['small_report']['path']).read_text())
    assert report['status'] == 'passed_changed_ABI_CPU_composite_after_fixture_failure'
    assert report['no_successful_schedule_rerun'] is True and report['first_failure_retained'] is True
    lineage_keys = ('first_attempt', 'inherited_successful_schedule', 'inherited_expected_rejections', 'continuation')
    assert set(contract['small_report_lineage']) == set(lineage_keys)
    assert all(report[key] == contract['small_report_lineage'][key] for key in lineage_keys)
    first_failure = json.loads(Path(contract['first_failure_report']['path']).read_text())
    assert first_failure['status'] == 'failed_changed_ABI_fixture', 'Preserved failure must not be relabeled passed'
    pins = contract['dependency_sha256']
    for name, digest in pins.items():
        file = ROOT/name
        assert file.is_file() and not file.is_symlink() and sha(file) == digest, name
    for key in ('normal_template', 'failure_template'):
        assert contract[key]['sha256'] in contract['small_artifact_sha256'].values(), key
    assert contract['normal_limit'] == NORMAL_LIMIT
    assert contract['failure_limit'] == FAILURE_LIMIT
    assert contract['rss_limit'] == RSS_LIMIT and contract['reserve'] == RESERVE
    return contract


def shape_map(contract):
    """Validate a literal per-occurrence map; no 24-to-768 dimension inference.

    Each item explicitly names source path/dtype/shape/runtime category,
    production dtype/shape, and a resource file/component/leaf. A resource upper
    leaf may exceed the actual query extent only with a separately reviewed
    literal explanation, e.g. actual127 suffix NLL versus an upper511 leaf.
    Heads, positions, width, head_dim, vocabulary and dtype are never guessed.
    """
    document = json.loads(Path(contract['shape_map']['path']).read_text())
    assert document['status'] == 'reviewed_explicit_T22_occurrence_shape_map'
    assert document['fixture_source_sha256'] == sha(__file__)
    assert document['small_report_sha256'] == contract['small_report']['sha256']
    all_maps = {}; inventories = {}
    for universe in ('normal', 'failure'):
        descriptors = json.loads(Path(contract[universe+'_descriptors']['path']).read_text())
        entries = document[universe]
        by_path = {path_key(item['path']): item for item in entries}
        assert len(by_path) == len(entries) == len(descriptors)
        assert set(by_path) == {path_key(item['path']) for item in descriptors}
        for source in descriptors:
            item = by_path[path_key(source['path'])]
            for key in ('path', 'shape', 'dtype', 'runtime_type'):
                assert item['source'][key] == source[key], (universe, key, source['path'])
            assert item['source_sha256_C_order'] == source['sha256_C_order']
            assert item['source']['runtime_type'] in ('torch.Tensor', 'numpy.ndarray')
            reference = item['inventory']
            if reference['file'] not in inventories:
                inventory_file = ROOT/reference['file']
                assert sha(inventory_file) == contract['dependency_sha256'][reference['file']]
                inventories[reference['file']] = json.loads(inventory_file.read_text())
            inventory = inventories[reference['file']]
            group = reference['component_group']
            assert group in ('components', 'new_typed_components')
            leaves = dict(inventory[group][reference['component']]['leaf_universe'])
            leaf = leaves[reference['path']]
            target = item['production']
            assert target['dtype'] == source['dtype']
            assert target['dtype'] == leaf['dtype'].removeprefix('torch.')
            assert len(target['shape']) == len(leaf['shape'])
            if target['shape'] != leaf['shape']:
                assert item['query_extent_upper_explanation'] in contract['allowed_query_extent_explanations']
                assert all(a <= b for a, b in zip(target['shape'], leaf['shape']))
            assert item['semantic_field'], 'Every expanded occurrence needs a named semantic field'
        # These totals are independently reviewed with the concrete map. They
        # prevent an unbounded repeated reference to an otherwise valid leaf.
        sizes = {'bool': 1, 'uint8': 1, 'int64': 8, 'float32': 4, 'float64': 8}
        raw = 0
        for item in entries:
            size = sizes[item['production']['dtype']]
            for dimension in item['production']['shape']: size *= dimension
            raw += size
        expected = contract['mapped_universe_bounds'][universe]
        assert len(entries) == expected['array_occurrences'] and raw == expected['raw_array_bytes']
        assert raw <= expected['resource_raw_upper_bytes']
        all_maps[universe] = by_path
    return all_maps


def check_inherited_layout(contract):
    """Read-only first-attempt bindings; never rewrite its FAILED status."""
    info = contract['inherited_first_layout']
    directory = Path(info['directory'])
    assert directory.is_dir() and not directory.is_symlink()
    files = {str(p.relative_to(directory)): p for p in directory.rglob('*') if p.is_file()}
    assert set(files) == set(info['file_inventory'])
    assert all(not p.is_symlink() for p in directory.rglob('*'))
    for name, item in info['file_inventory'].items():
        assert files[name].stat().st_size == item['bytes'] and sha(files[name]) == item['sha256'], name
    report = json.loads(Path(info['report']['path']).read_text())
    assert report['status'] == 'failed_T22_storage_layout_fixture' and report['error_type'] == 'MemoryError'
    assert report['source_sha256'] == '0493681434cc4fba989328af36d484ef44b28d3c010c002bf53fdddc8c88106e'
    assert sha(info['source']['path']) == report['source_sha256']
    chain = '0'*64; events = []
    for raw in Path(info['journal']['path']).read_bytes().splitlines():
        event = json.loads(raw); assert event['previous_sha256'] == chain
        chain = hashlib.sha256(raw).hexdigest(); events.append(event)
    assert len(events) == 3 and [e['event'] for e in events] == ['pre_encode_admission','committed','pre_encode_admission']
    assert events[0]['path'] == events[1]['path'] == 'inputs/wiki_validation__0__natural__000.npz'
    assert events[0]['buffered_admitted'] and events[0]['non_evidence_allowance_bytes'] == 896*1024**2
    assert events[1]['sha256'] == info['shard']['sha256'] and events[1]['bytes'] == info['shard']['bytes'] == 235656934
    assert events[2]['path'] == 'inputs/wiki_validation__0__bos__000.npz' and not events[2]['buffered_admitted']
    assert events[2]['non_evidence_observed_bytes'] == 1077941464 and not events[2]['non_evidence_budget_passed']
    assert not (directory/'normal_32shards/inputs/wiki_validation__0__bos__000.npz').exists()
    assert info['failed_unwritten_ordinary_raw_missing'] is True
    return {'status': report['status'], 'report_sha256': info['report']['sha256'],
            'source_sha256': report['source_sha256'], 'journal_sha256': info['journal']['sha256'],
            'shard': info['shard'], 'all_original_artifact_bytes': sum(x['bytes'] for x in info['file_inventory'].values()),
            'old_non_evidence_allowance_bytes': 896*1024**2,
            'failed_unwritten_ordinary_raw_missing': True,
            'original_peak_rss_kib': report['peak_rss_kib'],
            'original_entropy_seed': 20261022,
            'source_order_scope': 'Original source runs full member/expected-entropy verification before next normal admission. That next admission is durable; no original successful final report is invented.'}


def execute(directory, contract, maps):
    # No model, tokenizer, hook factory or forward is imported/called here.
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import l11_key_value_donor_storage as storage
    torch.set_num_threads(1)
    assert storage.NORMAL_LIMIT == NORMAL_LIMIT and storage.FAILURE_LIMIT == FAILURE_LIMIT
    assert storage.RSS_LIMIT == RSS_LIMIT and storage.FAILURE_JOURNAL_DISK_ALLOWANCE == SIDECAR_ALLOWANCE
    assert storage.NON_EVIDENCE_LIMIT == 1152*1024**2
    start = time.monotonic()
    report = {'status': 'running_T22_storage_layout_tail_after_preserved_failure', 'source_sha256': sha(__file__),
              'execution_contract_sha256': contract['_contract_sha256'],
              'model_forwards': 0, 'tokenizer_calls': 0, 'checkpoint_or_weight_access': False,
              'scope': 'Actual small ABI containers; explicit production shapes; independent entropy per occurrence. Dense noncontiguous Torch stride stress, not actual-stride or full-width arithmetic reconstruction.',
              'disk_free_before_bytes': shutil.disk_usage(directory).free,
              'mem_available_before_bytes': int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024}
    assert report['disk_free_before_bytes'] >= RESERVE+4*FAILURE_LIMIT+NORMAL_LIMIT
    assert report['mem_available_before_bytes'] >= 4*1024**3
    metadata_files = dict(contract['dependency_sha256'])
    for key in ('small_report', 'first_failure_report', 'normal_descriptors', 'failure_descriptors', 'shape_map'):
        metadata_files[contract[key]['path']] = contract[key]['sha256']
    # Whole normal admission charges32MiB prepared inputs plus32MiB all other
    # metadata, including sources/reference files, reviewed maps and this report.
    # The exact non-tensor closure is measured, while the allowance is retained.
    metadata_sizes = {}
    for name, digest in metadata_files.items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        assert sha(path) == digest
        metadata_sizes[name] = path.stat().st_size
    for item in contract['calibration_and_metadata_files']:
        assert sha(item['path']) == item['sha256']
        metadata_sizes[item['path']] = Path(item['path']).stat().st_size
    assert sum(metadata_sizes.values())+1024**2 < OTHER_METADATA_ALLOWANCE
    report['non_tensor_closure_measured_bytes'] = metadata_sizes
    report['mapped_universe_bounds'] = contract['mapped_universe_bounds']
    inherited = contract['_verified_inherited_layout']
    report['inherited_first_attempt'] = inherited
    report['tail_entropy_seed'] = 20261023
    report['original_largest_normal_not_repeated'] = True
    report['failed_original_ordinary_random_bytes_not_recreated'] = True
    rng = np.random.default_rng(20261023)

    def budget(stage):
        current = storage.current_rss_bytes()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert current <= RSS_LIMIT and peak <= RSS_LIMIT, (stage, current, peak)
        assert shutil.disk_usage(directory).free >= RESERVE, stage
        return {'stage': stage, 'current_rss_bytes': current, 'peak_rss_bytes': peak}

    def descriptor_tree(value, universe, path):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            item = maps[universe][path_key(path)]
            assert list(value.shape) == item['source']['shape']
            array = value.detach().numpy() if isinstance(value, torch.Tensor) else value
            assert str(array.dtype) == item['source']['dtype']
            digest = hashlib.sha256()
            for chunk in storage.c_order_chunks(array): digest.update(chunk)
            assert digest.hexdigest() == item['source_sha256_C_order']
            return {'__T22_entropy_array__': True, **item['production'], 'runtime_type': item['source']['runtime_type']}
        if isinstance(value, dict): return {key: descriptor_tree(child, universe, path+[str(key)]) for key, child in value.items()}
        if isinstance(value, list): return [descriptor_tree(child, universe, path+[index]) for index, child in enumerate(value)]
        if isinstance(value, tuple): return tuple(descriptor_tree(child, universe, path+[index]) for index, child in enumerate(value))
        return copy.deepcopy(value)

    def entropy(value):
        if isinstance(value, dict) and value.get('__T22_entropy_array__') is True:
            dtype = np.dtype(value['dtype']); shape = value['shape']
            # Dense permutation stress: same dtype/dimensions/nbytes/storage,
            # no holes or aliases. These are deliberately stressed strides,
            # not a reconstruction claim about actual model tensor strides.
            stress = value['runtime_type'] == 'torch.Tensor' and sum(d > 1 for d in shape) >= 2
            allocation_shape = list(reversed(shape)) if stress else shape
            if dtype.kind == 'b': array = rng.integers(0, 2, size=allocation_shape, dtype=np.uint8).view(dtype)
            elif dtype.kind == 'f': array = rng.standard_normal(allocation_shape, dtype=dtype)
            elif dtype.kind in 'iu': array = rng.integers(0, min(2**31-1, np.iinfo(dtype).max), size=allocation_shape, dtype=dtype)
            else: raise AssertionError(('Unreviewed entropy dtype', str(dtype)))
            # A new independent allocation is used even for equal/backing-shared
            # source occurrences. Runtime NumPy stays NumPy for copy admission.
            if stress:
                array = array.transpose(tuple(reversed(range(len(shape)))))
                assert list(array.shape) == shape and not array.flags.c_contiguous
            result = torch.from_numpy(array) if value['runtime_type'] == 'torch.Tensor' else array
            if stress:
                assert not result.is_contiguous() and result.storage_offset() == 0
                assert result.untyped_storage().nbytes() == result.numel()*result.element_size()
            return result
        if isinstance(value, dict): return {key: entropy(child) for key, child in value.items()}
        if isinstance(value, list): return [entropy(child) for child in value]
        if isinstance(value, tuple): return tuple(entropy(child) for child in value)
        return copy.deepcopy(value)

    def roundtrip(path, wanted):
        """T21 streamed oracle extended to NumPy; no second full decoded tree."""
        seen = set(); raw = count = 0
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist(); assert len(names) == len(set(names))
            with archive.open('metadata.npy') as stream:
                metadata = np.load(stream, allow_pickle=False)
            header = json.loads(metadata.tobytes()); assert header['schema'] == storage.CODEC_SCHEMA
            def verify(node, expected):
                nonlocal raw, count
                if isinstance(expected, (torch.Tensor, np.ndarray)):
                    array = expected.detach().numpy() if isinstance(expected, torch.Tensor) else expected
                    dtype = str(torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype)
                    assert node['type'] == 'tensor' and node['shape'] == list(array.shape) and node['dtype'] == dtype
                    name = node['array']; assert name not in seen; seen.add(name)
                    digest = hashlib.sha256()
                    for chunk in storage.c_order_chunks(array): digest.update(chunk)
                    assert digest.hexdigest() == node['sha256']
                    with archive.open(name+'.npy') as stream:
                        version = np.lib.format.read_magic(stream)
                        shape, fortran, actual_dtype = np.lib.format._read_array_header(stream, version)
                        assert shape == array.shape and not fortran and actual_dtype == array.dtype
                        actual = hashlib.sha256(); size = 0
                        for block in iter(lambda: stream.read(4*1024**2), b''):
                            actual.update(block); size += len(block)
                    assert actual.hexdigest() == digest.hexdigest() and size == array.nbytes
                    raw += size; count += 1
                elif isinstance(expected, dict):
                    assert node['type'] == 'dict' and len(node['items']) == len(expected)
                    for (nk, nv), (key, child) in zip(node['items'], expected.items()): verify(nk, key); verify(nv, child)
                elif isinstance(expected, (list, tuple)):
                    assert node['type'] == type(expected).__name__ and len(node['items']) == len(expected)
                    for child, expected_child in zip(node['items'], expected): verify(child, expected_child)
                else:
                    assert node['type'] == 'scalar' and type(node['value']) is type(expected) and node['value'] == expected
            verify(header['object'], wanted)
            assert names == ['a'+str(index)+'.npy' for index in range(count)]+['metadata.npy']
        return {'array_occurrences': count, 'raw_array_bytes': raw, 'metadata_bytes': len(metadata),
                'exact_uncompressed_members_and_scalar_tree': True}

    def verify_inherited_members(path, expected):
        # No old entropy regeneration: stream all existing NPY member hashes,
        # headers and scalar/container layout against the source descriptor tree.
        seen = set(); count = raw = 0
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist(); assert len(names) == len(set(names))
            with archive.open('metadata.npy') as stream: metadata = np.load(stream, allow_pickle=False)
            header = json.loads(metadata.tobytes()); assert header['schema'] == storage.CODEC_SCHEMA
            def visit(node, wanted):
                nonlocal count, raw
                if isinstance(wanted, dict) and wanted.get('__T22_entropy_array__') is True:
                    dtype = np.dtype(wanted['dtype']); tdtype = str(torch.from_numpy(np.empty(0, dtype=dtype)).dtype)
                    assert node['type'] == 'tensor' and node['shape'] == wanted['shape'] and node['dtype'] == tdtype
                    name = node['array']; assert name not in seen; seen.add(name)
                    with archive.open(name+'.npy') as stream:
                        version = np.lib.format.read_magic(stream)
                        shape, fortran, actual_dtype = np.lib.format._read_array_header(stream, version)
                        assert list(shape) == wanted['shape'] and not fortran and actual_dtype == dtype
                        digest = hashlib.sha256(); size = 0
                        for block in iter(lambda: stream.read(4*1024**2), b''):
                            digest.update(block); size += len(block)
                    assert digest.hexdigest() == node['sha256'] and size == dtype.itemsize*__import__('math').prod(shape)
                    raw += size; count += 1
                elif isinstance(wanted, dict):
                    assert node['type'] == 'dict' and len(node['items']) == len(wanted)
                    for (nk, nv), (key, child) in zip(node['items'], wanted.items()): visit(nk,key); visit(nv,child)
                elif isinstance(wanted, (list,tuple)):
                    assert node['type'] == type(wanted).__name__ and len(node['items']) == len(wanted)
                    for child, expected_child in zip(node['items'], wanted): visit(child,expected_child)
                else:
                    assert node['type'] == 'scalar' and type(node['value']) is type(wanted) and node['value'] == wanted
            visit(header['object'], expected)
            assert names == ['a'+str(index)+'.npy' for index in range(count)]+['metadata.npy']
        assert count == 1366 and raw == 253652704
        return {'array_occurrences': count, 'raw_array_bytes': raw,
                'all_members_headers_C_order_hashes_scalar_layout_exact': True,
                'scope': 'Existing artifact internal byte hashes and literal production shapes; original source-order attests the first in-RAM expected-entropy comparison.'}

    raw_normal = storage.read_npz(contract['normal_template']['path'])
    normal = descriptor_tree(raw_normal, 'normal', ['payload']); del raw_normal; gc.collect()
    assert set(normal['readouts']) == set(storage.ARMS) and len(normal['snapshots']) == 5 and len(normal['l11_panels']) == 9
    inherited_bytes = inherited['all_original_artifact_bytes']
    inherited_check_count = len(normal['checks'])
    report['inherited_member_verification'] = verify_inherited_members(inherited['shard']['path'], normal)
    writer = storage.EvidenceWriter(directory/'normal_31shards',
        normal_limit=NORMAL_LIMIT-inherited_bytes-PREPARATION_ALLOWANCE-OTHER_METADATA_ALLOWANCE,
        outstanding_other_caps=FAILURE_LIMIT)
    entries = []; record_count = suffix_count = check_count = 0
    for corpus in CORPORA:
        for window in range(4):
            for identity in ('natural', 'bos'):
                if (corpus,window,identity) == ('wiki_validation',0,'natural'):
                    continue  # Durable first large shard is inherited, never regenerated.
                logical = copy.deepcopy(normal)
                logical['metadata'].update(corpus=corpus, window=window, identity=identity)
                if (corpus, window, identity) != ('wiki_validation', 0, 'natural'):
                    for group in RAW_GROUPS: logical[group] = {}
                    logical.pop('l11_mask_reference', None)
                if window != 0 or identity != 'natural':
                    for group in ('metrics', 'gates', 'gate_norms', 'write_norms', 'l11_norms', 'operations'):
                        for anchor in ANCHORS: logical[group].pop(anchor, None)
                    logical['checks'] = [item for item in logical['checks'] if item['kind'] not in ANCHORS]
                high = entropy(logical); name = f'inputs/{corpus}__{window}__{identity}__000.npz'
                admission = storage.admission(high)
                entry = writer.save(name, high)
                entries.append({'name': name, **entry, 'admission': admission, **roundtrip(writer.out/name, high)})
                record_count += len(high['metrics']); suffix_count += len(high['suffix']); check_count += len(high['checks'])
                del high, logical; gc.collect(); budget('normal '+name)
    assert len(entries) == 31 and record_count == 378 and suffix_count == 372
    normal_charge = inherited_bytes+writer.used()+PREPARATION_ALLOWANCE+OTHER_METADATA_ALLOWANCE
    assert normal_charge <= NORMAL_LIMIT
    report['normal'] = {'new_entries': entries, 'inherited_entries': [inherited['shard']],
                        'new_main_records': record_count, 'new_suffix_records': suffix_count,
                        'combined_main_records': record_count+14, 'combined_suffix_records': suffix_count+12,
                        'combined_shards': len(entries)+1, 'combined_checks': check_count+inherited_check_count,
                        'inherited_entire_attempt_bytes_conservatively_charged': inherited_bytes,
                        'actual_new_normal_directory_bytes': writer.used(),
                        'preparation_allowance_bytes': PREPARATION_ALLOWANCE, 'other_metadata_allowance_bytes': OTHER_METADATA_ALLOWANCE,
                        'charged_total_bytes': normal_charge, 'normal_cap_passed': True}
    del normal; gc.collect()

    raw_failure = storage.read_npz(contract['failure_template']['path'])
    failure = descriptor_tree(raw_failure, 'failure', ['capsule']); del raw_failure; gc.collect()
    high = entropy(failure); del failure; gc.collect()
    measured = storage.admission(high, failure=True)
    assert measured['npz_upper_bytes'] <= FAILURE_LIMIT
    report['failure_admission_actual'] = measured
    report['failure_routes'] = []
    # Freeze the actual data once. Each stream writes all occurrences separately
    # in the same schema. Route forcing is explicit test control, never a claim
    # that actual measured RSS refused the other route.
    for route in ('frozen_header_stream', 'bounded_header_stream'):
        budget('before '+route)
        actual_route_admission = storage.admission(high, failure=True)
        assert actual_route_admission['bounded_stream_admitted']
        if route == 'frozen_header_stream': assert actual_route_admission['stream_admitted']
        failure_writer = storage.EvidenceWriter(directory/route,
            outstanding_other_caps=FAILURE_LIMIT if route=='frozen_header_stream' else 0)
        controlled = dict(actual_route_admission, buffered_admitted=False, stream_admitted=route=='frozen_header_stream')
        entry = failure_writer.save_failure_stream('failure_state.npz', high, controlled)
        witness = roundtrip(failure_writer.out/'failure_state.npz', high)
        report['failure_routes'].append({'route': route, 'route_control': 'Forced encoding path; actual admission recorded separately',
                                        'actual_route_admission': actual_route_admission, 'admission': controlled, **entry, **witness, 'resources': budget('after '+route)})
    assert report['failure_routes'][0]['raw_array_bytes'] == report['failure_routes'][1]['raw_array_bytes']
    # Exact logical members are each checked against the same retained entropy
    # tree; differing ZIP timestamps/headers/compression bytes are immaterial.
    del high; gc.collect()
    assert all(sha(ROOT/name) == digest for name, digest in contract['dependency_sha256'].items())
    report.update(status='passed_T22_storage_layout_composite_after_preserved_failure', wall_seconds=time.monotonic()-start,
                  final_resources=budget('completed'), disk_free_after_bytes=shutil.disk_usage(directory).free)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed-layout-fixture', action='store_true')
    parser.add_argument('--reviewed-contract', type=Path, required=True)
    parser.add_argument('--expected-contract-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.execute_reviewed_layout_fixture, 'SOURCE ONLY HOLD: separate root execution GO required'
    assert sha(args.reviewed_contract) == args.expected_contract_sha
    contract = check_manifest(args.reviewed_contract)
    maps = shape_map(contract)
    contract['_verified_inherited_layout'] = check_inherited_layout(contract)
    contract['_contract_sha256'] = args.expected_contract_sha
    protected = [Path(contract[name]['path']).resolve().parent for name in
                 ('small_report', 'normal_template', 'failure_template', 'shape_map')]
    protected.append(Path(contract['inherited_first_layout']['directory']))
    directory = fresh(args.out, protected)
    with (directory/Path(__file__).name).open('xb') as stream: stream.write(Path(__file__).read_bytes())
    write_json(directory/'execution_contract.json', contract)
    try: report = execute(directory, contract, maps)
    except BaseException as error:
        traceback.print_exc()
        report = {'status': 'failed_T22_storage_layout_tail_fixture', 'error_type': type(error).__name__,
                  'error': str(error), 'source_sha256': sha(__file__), 'execution_contract_sha256': args.expected_contract_sha,
                  'scope': 'All written bytes/directories remain preserved; no automatic retry. Parent review required.',
                  'inherited_first_attempt': contract.get('_verified_inherited_layout'),
                  'original_failed_status_unchanged': True, 'tail_entropy_seed': 20261023}
    report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(directory/'report.json', report)
    print(json.dumps({'directory': str(directory), 'status': report['status']}))
    return 0 if report['status'].startswith('passed') else 1


if __name__ == '__main__': sys.exit(main())
