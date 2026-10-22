"""Independent, stdlib-only T22 saved storage-witness audit; no entropy replay."""
import ast
import collections
import copy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import shutil
import struct
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'review/t22_storage_layout_composite_independent_audit.json'
RUN = ROOT / 't22_writer_layout_fixture_tail_run_v1'
CONTRACT = ROOT / 'review/t22_layout_execution_contract_RELEASED_v3_tail.json'
REPORT_SHA = 'd17b16a9b5585e43376f4313ba7bacbd32d0911f130598b116bada2344c8cfc1'
RSS = 3 * 1024**3
RESERVE = 2 * 1024**3
CORPORA = ('wiki_validation', 'utilitarianism', 'ethics', 'optimizers_new')
ARMS = ('baseline', 'current', 'remove_full', 'orth_full', 'remove_write_matched_early',
        'remove_l11_kv', 'orth_write_matched_early', 'orth_l11_kv', 'remove_l11_k',
        'remove_l11_v', 'orth_l11_k', 'orth_l11_v')
ANCHORS = ('native_repeat', 'native_identity')
RAW = ('snapshots', 'l11_panels', 'readouts', 'timing_samples')
DT = {'float32': ('<f4', 4), 'float64': ('<f8', 8), 'int64': ('<i8', 8),
      'bool': ('|b1', 1), 'uint8': ('|u1', 1)}
HASHES = {}


def need(value, message):
    if not value:
        raise AssertionError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    if path not in HASHES:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def npy_header(stream):
    need(stream.read(6) == b'\x93NUMPY', 'NPY magic')
    version = tuple(stream.read(2))
    need(version in ((1, 0), (2, 0), (3, 0)), 'NPY version')
    count = 2 if version == (1, 0) else 4
    size = struct.unpack('<H' if count == 2 else '<I', stream.read(count))[0]
    need(size + 8 + count <= 256, 'declared bounded NPY header')
    result = ast.literal_eval(stream.read(size).decode('utf8' if version == (3, 0) else 'latin1'))
    need(set(result) == {'descr', 'fortran_order', 'shape'}, 'NPY header fields')
    need(result['fortran_order'] is False, 'C-order NPY')
    return result


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    dtype: str
    runtime: str = 'torch.Tensor'

    @property
    def nbytes(self):
        return math.prod(self.shape) * DT[self.dtype][1]


def metadata(archive):
    with archive.open('metadata.npy') as stream:
        header = npy_header(stream)
        need(header['descr'] == '|u1', 'metadata uint8')
        data = stream.read()
        need(header['shape'] == (len(data),), 'metadata length')
    value = json.loads(data)
    need(set(value) == {'schema', 'object'} and value['schema'] == 'wiki_gate_lossless_npz_v1', 'codec schema')
    return value['object'], len(data)


def decode(node, path, *, archive=None, mapping=None, counters=None):
    kind = node['type']
    if kind == 'tensor':
        need(set(node) == {'type', 'array', 'dtype', 'shape', 'sha256'}, 'tensor metadata fields')
        dtype = node['dtype'].removeprefix('torch.')
        need(dtype in DT, 'declared tensor dtype')
        need(len(node['sha256']) == 64 and all(c in '0123456789abcdef' for c in node['sha256']), 'array SHA format')
        if mapping is not None:
            entry = mapping[tuple(path)]
            need(node['shape'] == entry['source']['shape'] and dtype == entry['source']['dtype'], 'template source shape/dtype')
            need(node['sha256'] == entry['source_sha256_C_order'], 'template source C SHA')
            return TensorSpec(tuple(entry['production']['shape']), entry['production']['dtype'], entry['source']['runtime_type'])
        spec = TensorSpec(tuple(node['shape']), dtype)
        name = node['array'] + '.npy'
        need(name == 'a' + str(counters['arrays']) + '.npy', 'ordered unique array member')
        with archive.open(name) as stream:
            header = npy_header(stream)
            need(header['shape'] == spec.shape and header['descr'] == DT[dtype][0], 'member shape/dtype')
            digest = hashlib.sha256()
            count = 0
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                digest.update(block)
                count += len(block)
            need(count == spec.nbytes and digest.hexdigest() == node['sha256'], 'member raw C bytes and SHA')
        counters['arrays'] += 1
        counters['raw_bytes'] += count
        counters['members'].append(name)
        return spec
    if kind == 'scalar':
        need(set(node) == {'type', 'value'}, 'scalar metadata fields')
        need(node['value'] is None or type(node['value']) in (bool, int, float, str), 'scalar type')
        return node['value']
    need(kind in ('dict', 'list', 'tuple') and set(node) == {'type', 'items'}, 'container metadata fields')
    if kind == 'dict':
        result = {}
        for key, value in node['items']:
            need(key['type'] == 'scalar', 'known scalar dictionary key')
            k = decode(key, path + ['<key>'])
            need(k not in result, 'unique dictionary keys')
            result[k] = decode(value, path + [str(k)], archive=archive, mapping=mapping, counters=counters)
        return result
    result = [decode(value, path + [index], archive=archive, mapping=mapping, counters=counters)
              for index, value in enumerate(node['items'])]
    return tuple(result) if kind == 'tuple' else result


def exact_tree(actual, expected, path=()):
    need(type(actual) is type(expected), 'exact container/scalar type at ' + str(path))
    if isinstance(expected, TensorSpec):
        need(actual.shape == expected.shape and actual.dtype == expected.dtype, 'literal production array spec at ' + str(path))
    elif isinstance(expected, dict):
        need(len(actual) == len(expected), 'dictionary length at ' + str(path))
        for (a, av), (e, ev) in zip(actual.items(), expected.items()):
            need(type(a) is type(e) and a == e, 'ordered dictionary key at ' + str(path))
            exact_tree(av, ev, path + (e,))
    elif isinstance(expected, (list, tuple)):
        need(len(actual) == len(expected), 'sequence length at ' + str(path))
        for index, (a, e) in enumerate(zip(actual, expected)):
            exact_tree(a, e, path + (index,))
    elif isinstance(expected, float):
        need(struct.pack('>d', actual) == struct.pack('>d', expected), 'scalar float bytes at ' + str(path))
    else:
        need(actual == expected, 'exact scalar at ' + str(path))


def leaves(tree):
    if isinstance(tree, TensorSpec):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from leaves(value)


def template(path, entries, label):
    with zipfile.ZipFile(path) as archive:
        node, _ = metadata(archive)
    return decode(node, [label], mapping={tuple(e['path']): e for e in entries})


def audit_archive(path, expected, entry):
    path = Path(path)
    need(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'complete archive hash/bytes ' + path.name)
    counters = {'arrays': 0, 'raw_bytes': 0, 'members': []}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'no duplicate archive members')
        node, metadata_size = metadata(archive)
        actual = decode(node, ['root'], archive=archive, counters=counters)
        need(names == counters['members'] + ['metadata.npy'], 'complete ordered archive member universe')
    exact_tree(actual, expected)
    counters.pop('members')
    counters['metadata_bytes'] = metadata_size
    if 'array_occurrences' in entry:
        need(counters['arrays'] == entry['array_occurrences'] and counters['raw_bytes'] == entry['raw_array_bytes'], 'reported archive counts')
        need(metadata_size == entry['metadata_bytes'], 'reported metadata bytes')
    return counters


def journal(path):
    previous = '0' * 64
    result = []
    for raw in Path(path).read_bytes().splitlines(keepends=True):
        need(raw.endswith(b'\n'), 'complete journal line')
        event = json.loads(raw)
        body = raw[:-1]
        need(body == json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), 'canonical journal')
        need(event['previous_sha256'] == previous, 'journal hash chain')
        previous = hashlib.sha256(body).hexdigest()
        result.append(event)
    return result


def check_admission(receipt, tree, *, failure=False, allowance=1152 * 1024**2):
    specs = list(leaves(tree))
    raw = sum(x.nbytes for x in specs)
    noncontig = [x.nbytes for x in specs if x.runtime == 'torch.Tensor' and sum(d > 1 for d in x.shape) >= 2]
    npcopies = sum(x.nbytes for x in specs if x.runtime == 'numpy.ndarray')
    z = lambda n: n + (n >> 12) + (n >> 14) + (n >> 25) + 13
    upper = sum(z(x.nbytes + 256) + 384 for x in specs) + (8 if failure else 1) * 1024**2 + 640
    need(receipt['array_occurrences'] == len(specs) and receipt['serialized_raw_array_bytes'] == raw, 'actual admission array universe')
    need(receipt['owned_CPU_backing_storage_bytes'] == raw, 'independent dense entropy backing bytes')
    need(receipt['numpy_normalization_bytes'] == npcopies and receipt['normalization_allowance_bytes'] == 32 * 1024**2, 'NumPy copy accounting')
    need(receipt['tensor_contiguous_retained_copy_bytes'] == sum(noncontig), 'retained per-occurrence Torch copies')
    need(receipt['tensor_identity_contiguous_temporary_bytes'] == max(noncontig, default=0), 'second identity contiguous temporary')
    need(receipt['tensor_identity_byte_string_upper_bytes'] == max(x.nbytes for x in specs), 'identity byte-string upper')
    need(receipt['npz_upper_bytes'] == upper and receipt['rss_limit_bytes'] == RSS, 'NPZ/RSS upper')
    base = max(0, receipt['current_rss_bytes'] - raw)
    header = receipt['current_rss_bytes'] + sum(noncontig) + max(noncontig, default=0) + max(x.nbytes for x in specs) + 64 * 1024**2 + max(32 * 1024**2, npcopies)
    buffered = receipt['current_rss_bytes'] + sum(noncontig) + math.ceil(2.25 * upper) + 64 * 1024**2 + max(32 * 1024**2, npcopies)
    bounded = receipt['current_rss_bytes'] + 65 * 1024**2
    need(receipt['non_evidence_observed_bytes'] == base and receipt['non_evidence_allowance_bytes'] == allowance, 'observed R0 and exact prospective allowance')
    need(receipt['non_evidence_budget_passed'] is (base <= allowance), 'R0 flag')
    need(receipt['normalization_budget_passed'] is (npcopies <= 32 * 1024**2), 'normalization flag')
    need(receipt['header_phase_upper_bytes'] == header and receipt['buffered_phase_upper_bytes'] == buffered, 'complete encoding phase arithmetic')
    need(receipt['encoding_peak_upper_bytes'] == max(header, buffered) and receipt['bounded_stream_phase_upper_bytes'] == bounded, 'phase maximum/bounded arithmetic')
    need(receipt['buffered_admitted'] is (header <= RSS and buffered <= RSS and base <= allowance and npcopies <= 32 * 1024**2), 'actual buffered decision')
    need(receipt['stream_admitted'] is (header <= RSS and npcopies <= 32 * 1024**2), 'actual frozen header decision')
    need(receipt['bounded_stream_admitted'] is (bounded <= RSS), 'actual bounded decision')
    return base


def normal_tree(tree, corpus, window, identity):
    result = copy.deepcopy(tree)
    result['metadata'].update(corpus=corpus, window=window, identity=identity)
    if (corpus, window, identity) != ('wiki_validation', 0, 'natural'):
        for name in RAW:
            result[name] = {}
        result.pop('l11_mask_reference', None)
    if window != 0 or identity != 'natural':
        for name in ('metrics', 'gates', 'gate_norms', 'write_norms', 'l11_norms', 'operations'):
            for anchor in ANCHORS:
                result[name].pop(anchor, None)
        result['checks'] = [x for x in result['checks'] if x['kind'] not in ANCHORS]
    need(set(result['metrics']) == set(ARMS) | (set(ANCHORS) if window == 0 and identity == 'natural' else set()), 'normal arm roster')
    need(set(result['suffix']) == set(ARMS), 'suffix roster')
    return result


def main():
    started = time.monotonic()
    base = {'status': 'failed', 'audit_source_sha256': sha(__file__),
            'audited_composite_report_sha256': REPORT_SHA,
            'audited_storage_source_sha256': '445db76be9121967e93577394b38f0fc809b434c461ed769d5dd91efbb0d2d3d',
            'audited_resource_inventory_sha256': 'b12aecf2f137cb424cd4709d64306739002f097aecf569480295770d41231502'}
    need(not OUT.exists() and not OUT.with_suffix('.md').exists(), 'exclusive new audit outputs')
    try:
        need(sha(RUN / 'report.json') == REPORT_SHA, 'exact actual composite report')
        report = load(RUN / 'report.json')
        need(report['status'] == 'passed_T22_storage_layout_composite_after_preserved_failure', 'actual passed composite')
        contract = load(CONTRACT)
        need(sha(CONTRACT) == report['execution_contract_sha256'] == 'b684e90ab5a53558e709ee2c95b480682e88697bb9cc19f5699281573788d1be', 'released contract')
        stored = load(RUN / 'execution_contract.json')
        verified_inherited = stored.pop('_verified_inherited_layout')
        need(stored.pop('_contract_sha256') == sha(CONTRACT) and stored == contract, 'stored execution contract exact plus explicit verified lineage')
        need(contract['status'] == 'approved_T22_layout_fixture_execution', 'released status')
        mapping = load(contract['shape_map']['path'])
        need(sha(contract['shape_map']['path']) == contract['shape_map']['sha256'], 'released map')
        need(mapping['status'] == 'reviewed_explicit_T22_occurrence_shape_map', 'reviewed map status')
        need(sha(RUN / 't22_writer_layout_fixture_source_v3_tail.py') == report['source_sha256'] == contract['fixture_source_sha256'], 'actual source copy')
        for name, digest in contract['dependency_sha256'].items():
            need(sha(ROOT / name) == digest, 'current source/reference pin ' + name)
        need(sha(ROOT / 'l11_key_value_donor_storage.py') == base['audited_storage_source_sha256'], 'audited storage')
        need(sha(ROOT / 'review/t22_symbolic_resource_inventory_v6_R0_retention.json') == base['audited_resource_inventory_sha256'], 'audited v6 resource')
        for key in ('small_report', 'first_failure_report', 'normal_template', 'normal_descriptors', 'failure_template', 'failure_descriptors'):
            need(sha(contract[key]['path']) == contract[key]['sha256'], 'actual lineage/template pin ' + key)
        small = load(contract['small_report']['path'])
        need(small['status'] == 'passed_changed_ABI_CPU_composite_after_fixture_failure', 'actual small composite provenance')
        need(all(small[k] == value for k, value in contract['small_report_lineage'].items()), 'all four small lineage objects')
        first = contract['inherited_first_layout']
        oldrun = Path(first['directory'])
        actual_files = {str(p.relative_to(oldrun)): p for p in oldrun.rglob('*') if p.is_file()}
        need(set(actual_files) == set(first['file_inventory']), 'entire original first-attempt inventory')
        need(not any(p.is_symlink() for p in oldrun.rglob('*')), 'no original symlink')
        for name, entry in first['file_inventory'].items():
            need(actual_files[name].stat().st_size == entry['bytes'] and sha(actual_files[name]) == entry['sha256'], 'original bytes unchanged ' + name)
        need(load(first['report']['path'])['status'] == 'failed_T22_storage_layout_fixture', 'first attempt remains failed')
        need(first['failed_unwritten_ordinary_raw_missing'] and verified_inherited == report['inherited_first_attempt'], 'missing original entropy explicit')
        need(not (oldrun / 'normal_32shards/inputs/wiki_validation__0__bos__000.npz').exists(), 'unwritten original BOS stays absent')
        normal = template(contract['normal_template']['path'], mapping['normal'], 'payload')
        failure = template(contract['failure_template']['path'], mapping['failure'], 'capsule')
        oldevents = journal(first['journal']['path'])
        need([e['event'] for e in oldevents] == ['pre_encode_admission', 'committed', 'pre_encode_admission'], 'failed first journal stages')
        oldbase = check_admission(oldevents[0], normal, allowance=896 * 1024**2)
        check_admission(oldevents[2], normal_tree(normal, 'wiki_validation', 0, 'bos'), allowance=896 * 1024**2)
        need(oldevents[0]['buffered_admitted'] and not oldevents[2]['buffered_admitted'] and not oldevents[2]['non_evidence_budget_passed'], 'original rejection not waived')
        normal_stats = [audit_archive(first['shard']['path'], normal, first['shard'])]
        need(normal_stats[0]['arrays'] == 1366 and normal_stats[0]['raw_bytes'] == 253652704, 'inherited exact member universe')
        roster = list(itertools.product(CORPORA, range(4), ('natural', 'bos')))
        expected_tail = roster[1:]
        entries = report['normal']['new_entries']
        expected_names = ['inputs/%s__%s__%s__000.npz' % x for x in expected_tail]
        need([x['name'] for x in entries] == expected_names, 'all31 new shards in exact order')
        events = journal(RUN / 'normal_31shards/journal.jsonl')
        need(len(events) == 62, 'exact31 admissions/commits')
        bases = [oldbase]
        main_count = len(normal['metrics']); suffix_count = len(normal['suffix']); check_count = len(normal['checks'])
        for index, (entry, role) in enumerate(zip(entries, expected_tail)):
            expected = normal_tree(normal, *role)
            admit, commit = events[2 * index:2 * index + 2]
            need(admit['event'] == 'pre_encode_admission' and commit['event'] == 'committed', 'normal event pair')
            need(admit['path'] == commit['path'] == entry['name'] and admit['failure_write'] is False, 'normal event role')
            need(commit['sha256'] == entry['sha256'] and commit['bytes'] == entry['bytes'], 'normal durable receipt')
            bases.extend((check_admission(entry['admission'], expected), check_admission(admit, expected)))
            need(admit['buffered_admitted'] and entry['admission']['buffered_admitted'], 'actual normal admission passed')
            normal_stats.append(audit_archive(RUN / 'normal_31shards' / entry['name'], expected, entry))
            main_count += len(expected['metrics']); suffix_count += len(expected['suffix']); check_count += len(expected['checks'])
        need((main_count, suffix_count, check_count) == (392, 384, report['normal']['combined_checks']), 'full combined scientific container counts')
        need(report['normal']['new_main_records'] == 378 and report['normal']['new_suffix_records'] == 372 and report['normal']['combined_shards'] == 32, 'new/combined counts')
        newnormalbytes = sum(p.stat().st_size for p in (RUN / 'normal_31shards').rglob('*') if p.is_file())
        inheritedbytes = sum(e['bytes'] for e in first['file_inventory'].values())
        charge = newnormalbytes + inheritedbytes + 64 * 1024**2
        need((newnormalbytes, inheritedbytes, charge) == (146240222, 235708454, 449057540), 'all actual charged normal bytes')
        need(charge == report['normal']['charged_total_bytes'] <= 512 * 1024**2, 'global normal cap')
        check_admission(report['failure_admission_actual'], failure, failure=True)
        routes = report['failure_routes']
        need([x['route'] for x in routes] == ['frozen_header_stream', 'bounded_header_stream'], 'both required failure routes')
        failure_stats = []
        for entry in routes:
            actual = entry['actual_route_admission']
            bases.append(check_admission(actual, failure, failure=True))
            need(actual['bounded_stream_admitted'] and (entry['route'] != 'frozen_header_stream' or actual['stream_admitted']), 'fresh fitting route admission')
            controlled = dict(actual, buffered_admitted=False, stream_admitted=entry['route'] == 'frozen_header_stream')
            need(controlled == entry['admission'], 'only declared forced flags changed')
            ev = journal(RUN / entry['route'] / 'journal.jsonl')
            need([e['event'] for e in ev] == ['failure_stream_admission', 'committed'], 'direct failure journal stages')
            need(ev[0]['admission_breach'] is False and ev[0]['bounded_header'] is (entry['route'] == 'bounded_header_stream'), 'actual route receipt')
            need(ev[1]['admission'] == controlled and ev[1]['sha256'] == entry['sha256'] and ev[1]['bytes'] == entry['bytes'], 'failure durable commit')
            need(entry['bytes'] <= 1024**3, 'full failure archive cap')
            failure_stats.append(audit_archive(RUN / entry['route'] / 'failure_state.npz', failure, entry))
            need(entry['resources']['current_rss_bytes'] <= RSS and entry['resources']['peak_rss_bytes'] <= RSS, 'actual route measured resources')
        need(routes[0]['sha256'] == routes[1]['sha256'] == '0ff572f787bc286f566f4d154a4aa7d269306c854bea3639ad8e20f50b0b8ff2', 'both complete archive bytes identical')
        need(all(x['arrays'] == 1482 and x['raw_bytes'] == 694044990 for x in failure_stats), 'both full failure universes')
        expected_files = {'report.json', 'execution_contract.json', 't22_writer_layout_fixture_source_v3_tail.py', 'normal_31shards/journal.jsonl'}
        expected_files.update('normal_31shards/' + x for x in expected_names)
        for route in ('frozen_header_stream', 'bounded_header_stream'):
            expected_files.update((route + '/journal.jsonl', route + '/failure_state.npz'))
        need({str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()} == expected_files, 'entire39-file actual output roster')
        need(not any(p.is_symlink() for p in RUN.rglob('*')), 'no output symlink')
        need(report['peak_rss_kib'] * 1024 == report['final_resources']['peak_rss_bytes'] == 2003042304 < RSS, 'whole actual peak RSS')
        need(report['disk_free_before_bytes'] >= RESERVE + 4 * 1024**3 + 512 * 1024**2 and report['disk_free_after_bytes'] >= RESERVE and shutil.disk_usage(RUN).free >= RESERVE, 'reported/current reserve')
        need(report['mem_available_before_bytes'] >= 4 * 1024**3, 'preallocation memory admission')
        need(report['tail_entropy_seed'] == 20261023 and report['original_largest_normal_not_repeated'] is True and report['failed_original_ordinary_random_bytes_not_recreated'] is True, 'prospective continuation scope')
        need(report['model_forwards'] == report['tokenizer_calls'] == 0 and report['checkpoint_or_weight_access'] is False, 'zero model/tokenizer/weight activity')
        metadata_sizes = dict(contract['dependency_sha256'])
        for k in ('small_report', 'first_failure_report', 'normal_descriptors', 'failure_descriptors', 'shape_map'):
            metadata_sizes[contract[k]['path']] = contract[k]['sha256']
        sizes = {}
        for name, digest in metadata_sizes.items():
            p = Path(name) if Path(name).is_absolute() else ROOT / name
            need(sha(p) == digest, 'metadata current hash')
            sizes[name] = p.stat().st_size
        for x in contract['calibration_and_metadata_files']:
            need(sha(x['path']) == x['sha256'], 'calibration metadata bytes')
            sizes[x['path']] = Path(x['path']).stat().st_size
        need(sizes == report['non_tensor_closure_measured_bytes'] and sum(sizes.values()) + 1024**2 < 32 * 1024**2, 'exact metadata allowance')
        base.update(status='passed', audited_contract_sha256=sha(CONTRACT), audited_map_sha256=sha(contract['shape_map']['path']),
                    counts={'normal_shards': 32, 'new_normal_shards': 31, 'main_records': main_count, 'suffix_records': suffix_count,
                            'checks': check_count, 'normal_array_occurrences': sum(x['arrays'] for x in normal_stats),
                            'normal_raw_array_bytes': sum(x['raw_bytes'] for x in normal_stats),
                            'failure_routes': 2, 'failure_array_occurrences_each': 1482, 'failure_raw_array_bytes_each': 694044990,
                            'all_streamed_array_occurrences': sum(x['arrays'] for x in normal_stats + failure_stats)},
                    resources={'combined_normal_charge_bytes': charge, 'normal_cap_bytes': 512 * 1024**2,
                               'failure_archive_bytes_each': 643900402, 'failure_cap_bytes': 1024**3,
                               'producer_peak_RSS_bytes': report['peak_rss_kib'] * 1024, 'producer_wall_seconds': report['wall_seconds'],
                               'max_actual_successful_observed_R0_bytes': max(bases), 'R0_allowance_bytes': 1152 * 1024**2,
                               'actual_failure_buffered_upper_bytes': [e['actual_route_admission']['buffered_phase_upper_bytes'] for e in routes],
                               'actual_failure_buffered_refused_both': all(not e['actual_route_admission']['buffered_admitted'] for e in routes),
                               'actual_frozen_header_admitted_both_bounded_path_explicitly_forced': all(e['actual_route_admission']['stream_admitted'] for e in routes),
                               'disk_free_after_bytes': report['disk_free_after_bytes'], 'reserve_bytes': RESERVE},
                    lineage={'first_report_status': 'failed_T22_storage_layout_fixture', 'first_report_sha256': first['report']['sha256'],
                             'all5_original_files_byte_exact': True, 'original_unwritten_entropy_remains_missing': True,
                             'independent_fresh_tail_seed': 20261023, 'completed_first_entropy_not_regenerated': True},
                    scope='Independent stdlib streamed archive/member/header/C-byte-SHA and complete scalar/container comparison against pinned semantic templates/maps; recorded admissions/resources independently recomputed. Entropy values were not regenerated. Actual original strides and sampler RNG outputs are source/runtime attestations, not recoverable from C-normalized NPZ. No Torch/NumPy/model/tokenizer/checkpoint/GPU or fixture rerun. This is finite storage-layout evidence, not a model launch or universal failure/allocator theorem.')
    except BaseException as error:
        traceback.print_exc()
        base.update(error_type=type(error).__name__, error=str(error), scope='Failed independent actual audit; evidence unchanged, no automatic repair/rerun.')
    base['audit_wall_seconds'] = time.monotonic() - started
    base['audit_peak_RSS_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with OUT.open('x') as stream:
        json.dump(base, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with OUT.with_suffix('.md').open('x') as stream:
        stream.write('# T22 actual storage composite audit\n\n' + base['status'].upper() + '. Bound to composite report `' + REPORT_SHA + '`.\n\n')
        if base['status'] == 'passed':
            stream.write('Independently streamed all32normal archives and both complete failure archives. Every ordered member, NPY header, raw C-byte SHA and scalar/container layout agrees with the pinned templates/maps. Counts392MAIN/384SUFFIX and global normal charge449,057,540 bytes agree. Both failure archives contain1,482 arrays/694,044,990 raw bytes and are byte-identical643,900,402-byte archives.\n\n')
            stream.write('Peak producer RSS2,003,042,304 bytes is below3GiB. Actual per-route admissions were fresh: buffered refusal was observed for both; bounded encoding was explicitly forced despite frozen-header admission fitting. Caps and2GiB reserve hold. The original first witness remains FAILED and byte-exact; unwritten original entropy remains missing.\n\n')
            stream.write('No entropy/model replay, Torch/NumPy import, checkpoint/GPU access, copy or deletion. Normalized archives cannot independently recover original strides or sampler state. This passes the finite storage witness, not production model launch. Exact bindings and audit resources are in the JSON.\n')
        else:
            stream.write('The audit stopped on ' + base['error_type'] + ': ' + base['error'] + '. No evidence was changed and no retry occurred.\n')
    print(json.dumps({'status': base['status'], 'result': str(OUT), 'sha256': sha(OUT),
                      'wall_seconds': base['audit_wall_seconds'], 'peak_RSS_kib': base['audit_peak_RSS_kib']}))
    return 0 if base['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
