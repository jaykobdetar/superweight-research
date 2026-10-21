"""Export safe T18 metadata or reconstruct its exact 20 windows with GPT-2.

Preparation only: no model inference, weights, old model runs or non-GPT2
tokenizer. Historical evidence is inherited and labeled, never re-certified.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse
import copy
import hashlib
import importlib
import json
import re
import resource
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'openai-community/gpt2'
REVISION = '75e09b43581151bd1d9ef6700faa605df408979f'
METADATA_IDENTITY_SHA256 = '7e3a43808ee51ab8a44526d3cb9e9f545e12ac771a1723ca2d8ca9f360a8a0b6'
CONTRACT_SHA = '6549a42eb870e425662447d1f430a5afb277df76d871800de5930d6f92557264'
AUDIT_SHA = 'ba7efb46473e2f99d76b230abfebe3056274d0c7613f77a97a1af470fd8fb632'
DEPENDENCIES = {
    'gate_calibration_factorial_inputs.py': '810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a',
    'gate_direction_geometry.py': '5ca31e5bb1b5bce8ea2e2af093ffdad15224ba200ee1e852d2695f94af582ea8',
    'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044',
}
EXPECTED = {'calibration_inputs.json': 'b58e0dfc03cc6125cf618c9d66bd38b375db102e418971bf0a48b4fe2a58df04', 'calibration_tokens.pt': 'dcdc07d717bac319889491f637e0c98fd8f4c05dc663fb58aef994a2ca7faae5', 'data_manifest.json': '9dc7298938b81c4f434f95efbb631a81d2854292d07c557dcff531e0ecdcd38f', 'exclusion_inventory.json': '4ac2a6107ea6c7f75b453e79954877a88ccbf9355c01651bf507bf020c49a35e', 'historical_manifest_scan.json': '6b55bdab8a64dd6d0d5789dd2cc76b420ebd394a76b3e914103a0f95f913af0e', 'mixture_directions.json': '0a27dcc857b221a9fdbd1265be678aa341d59f914eb3030cff5eb44690ff852e', 'mixture_directions.pt': '4bf8fc396fd57e2146933131b928ace0a49498b3d295d749b91626a1b70260b3', 'preflight.json': 'e579f67d97e8f8adcbb250857657e956c15861bdd71558df3ff8dbf5e5532c46', 'prior_exclusion_inventory.json': 'ce87628cbc3507e51d785b0eb39ae496886f1166683c283e7f3a92aa5f22f41c', 'prior_historical_manifest_scan.json': 'cf7ec1ca442d3f0aaad4dd89f82da6661fca9b044252099e9bfc1fa76618d7e7', 'prior_olmo_source_mapping.json': '7b5f39d5be210bf4997c684ba01249b711561878385997642f7db341fa787515', 'prior_preflight.json': '0106ad5125408c21dc6d37941afd77f3de80c74ebc13013a18b6f98e1738201a', 'start_identity_summary.json': '9b124da72c1ddf98ebfca33503c388c1ecee55c0f589bc5932ce7a6d2a0e14a2', 'tokens.pt': '78b5b9a3594bd04a0abfe902fadb5c62dae6ef3f760aa67d28c9ebe912a6111b'}
MAP_SHA = 'ed62f089168abeec63434e5f6dfa83ff8e7e139bc362e952456ea34859b5a098'
CAL_TOKENS_SHA = 'a99fa8403d600404b4725bdc8ab8e051e78e70a864a3483b614879d68ccc668c'
PHYSICAL_SOURCES = ('wiki_test', 'wiki_validation', 'utilitarianism', 'ethics', 'adadelta', 'adamax', 'asgd')
EVALUATION = ('wiki_validation', 'utilitarianism', 'ethics', 'optimizers_new')
CHANGED = ('calibration_inputs.json', 'data_manifest.json', 'exclusion_inventory.json',
           'prior_exclusion_inventory.json', 'start_identity_summary.json', 'preflight.json')
COPIED_EXACT = ('historical_manifest_scan.json', 'prior_historical_manifest_scan.json',
                'prior_olmo_source_mapping.json', 'prior_preflight.json')
FILES = ('data_manifest.template.json', 'calibration_inputs.template.json', 'exclusion_inventory.json',
         'prior_exclusion_inventory.json', *COPIED_EXACT, 'historical_preflight.json',
         'history_token_hashes.json', 'start_identity.template.json', 'capacity_attestation.json', 'sanitization.json')
RESERVE = 2 * 1024**3
ALLOWANCE = 16 * 1024**2
FORBIDDEN = {'path', 'tokens', 'input_ids', 'ids', 'id', 'first_token_id', 'bos_id'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def clean(v):
    if isinstance(v, dict):
        return {k: clean(w) for k, w in v.items()
                if k not in FORBIDDEN and not k.endswith('first_token_ids')}
    if isinstance(v, list):
        return [clean(w) for w in v]
    return v


def safe(v, filename='', path=()):
    """Only documented historical JSON pointer fields may start with slash."""
    if isinstance(v, dict):
        assert not (FORBIDDEN & v.keys()), ('Private field', path)
        assert not any(k.endswith('first_token_ids') for k in v), ('Prior/current token identities', path)
        for k, w in v.items():
            safe(k, filename, path + ('<key>',))
            safe(w, filename, path + (k,))
    elif isinstance(v, list):
        for i, w in enumerate(v):
            safe(w, filename, path + (i,))
    elif isinstance(v, str):
        if v.startswith('/') and path and path[-1] == 'pointer':
            assert filename in {'prior_olmo_source_mapping.json', 'exclusion_inventory.json', 'prior_exclusion_inventory.json'}
            if filename == 'prior_olmo_source_mapping.json':
                assert len(path) == 3 and path[0] == 'mapped_ranges' and isinstance(path[1], int)
            else:
                assert len(path) == 4 and path[0] == 'exclusions' and path[1] in {'wiki_test', 'hume', *EVALUATION} and isinstance(path[2], int)
            assert re.fullmatch(r'/(?:\d+|wiki|hume|development|excluded|prefix|suffix|natural|calibration|evaluation)(?:/(?:\d+|wiki|hume|development|excluded|prefix|suffix|natural|calibration|evaluation))*', v)
            return
        assert not v.startswith(('/', '\\', '~', 'file://')), ('Absolute locator', filename, path)
        assert not re.match(r'^[A-Za-z]:[/\\]', v), ('Windows locator', filename, path)


def empty_out(out):
    assert not out.exists() or not any(out.iterdir()), 'Use a fresh or empty directory; preserve old attempts'
    out.mkdir(parents=True, exist_ok=True)


def space_guard(out, allowance=ALLOWANCE):
    free = shutil.disk_usage(out).free
    assert free >= RESERVE + allowance, ('Insufficient preparation space', free, RESERVE, allowance)
    return free


def dependencies():
    assert EXPECTED and len(EXPECTED) == 14
    for n, h in DEPENDENCIES.items():
        assert sha(ROOT / n) == h, ('Helper dependency changed', n)


def export_metadata(inputs, contract, audit, out):
    empty_out(out); space_guard(out); dependencies()
    assert sha(contract) == CONTRACT_SHA and sha(audit) == AUDIT_SHA
    evidence = read(contract)
    assert evidence['canonical_inputs_sha256'] == EXPECTED
    assert all(sha(inputs / n) == h for n, h in EXPECTED.items())
    dm = read(inputs / 'data_manifest.json'); cm = read(inputs / 'calibration_inputs.json')
    objects = {'data_manifest.template.json': clean(dm), 'calibration_inputs.template.json': clean(cm),
               'exclusion_inventory.json': clean(read(inputs / 'exclusion_inventory.json')),
               'prior_exclusion_inventory.json': clean(read(inputs / 'prior_exclusion_inventory.json')),
               'start_identity.template.json': clean(read(inputs / 'start_identity_summary.json'))}
    history = evidence['history']
    sets = history['token_hash_family_sets']
    union = sorted(set().union(*(set(v) for v in sets.values())))
    assert union == history['selector_token_hash_union'] and len(union) == 561
    assert len(set(union) - set(sets['current_scan'])) == 301
    objects['history_token_hashes.json'] = {'families': sets, 'evaluation_union': union,
        'calibration_union': sorted(set(union) - set(sets['retained_fresh_calibration'])),
        'contract_sha256': CONTRACT_SHA,
        'scope': 'Frozen historical token-hash commitments; no historical source arrays rechecked by recipient.'}
    capacity = read(audit)
    assert capacity['status'] == 'passed' and capacity['capacity']['hume']['all_integer_8192_starts'] == 0
    objects['capacity_attestation.json'] = {'original_audit_sha256': AUDIT_SHA,
        'hume': capacity['capacity']['hume'], 'original_failed_preflight_sha256': EXPECTED['prior_preflight.json'],
        'scope': 'Historical audited exclusions; interval arithmetic is rechecked, historical source mapping is inherited.'}
    original_starts = read(inputs / 'start_identity_summary.json')
    original_starts.pop('prior_T16_T17_evaluation_first_token_ids')
    objects['sanitization.json'] = {'changed_metadata_files': list(CHANGED),
        'exact_canonical_files': sorted(set(EXPECTED) - set(CHANGED)),
        'private_value_commitments': {
            'current_start_summary': value_sha(original_starts),
            'calibration_first_ids': value_sha([r['first_token_id'] for r in cm['passages']]),
            'evaluation_first_ids': value_sha([r['first_token_id'] for r in dm['passages']]),
            'literal_ids': value_sha(dm['literals'])},
        'rules': ['Remove source path fields and explicit token ID fields from shared metadata.',
                  'Reconstruct current IDs privately from the pinned tokenizer and input bytes.',
                  'Omit historical T16/T17 first-ID list; keep prior membership as labeled inherited attestation.',
                  'Preserve original hash edges as historical commitments; record new hashes separately.'],
        'history_scope': 'Frozen original attestations, not a rescan of private history by the recipient.'}
    for n in COPIED_EXACT:
        objects[n] = read(inputs / n)
    objects['historical_preflight.json'] = read(inputs / 'preflight.json')
    assert set(objects) == set(FILES)
    for n, obj in objects.items(): safe(obj, n)
    for n, obj in objects.items():
        if n in COPIED_EXACT:
            shutil.copyfile(inputs / n, out / n)
        elif n == 'historical_preflight.json':
            shutil.copyfile(inputs / 'preflight.json', out / n)
        else:
            dump(out / n, obj)
    identity = {'kind': 't18_wiki_safe_metadata_v1', 'model_id': MODEL_ID, 'revision': REVISION,
        'canonical_input_sha256': EXPECTED, 'canonical_calibration_maps_sha256': MAP_SHA,
        'canonical_calibration_tokens_sha256': CAL_TOKENS_SHA, 'dependencies': DEPENDENCIES,
        'metadata_sha256': {n: sha(out / n) for n in FILES},
        'changed_metadata_files': list(CHANGED), 'exact_canonical_files': sorted(set(EXPECTED) - set(CHANGED)),
        'calibration_passages': 4, 'evaluation_passages': 16, 'evaluation_rejections': 58,
        'history_union_count': 561, 'history_contract_sha256': CONTRACT_SHA,
        'scope': 'No paths, source text, token IDs/arrays or model weights. Historical evidence is inherited; 20 current windows are verified locally.'}
    safe(identity, 'identity.json'); dump(out / 'identity.json', identity)
    assert sum(p.stat().st_size for p in out.iterdir()) < ALLOWANCE
    return identity


def validate_metadata(metadata):
    assert re.fullmatch(r'[a-f0-9]{64}', METADATA_IDENTITY_SHA256), 'Draft metadata identity not frozen'
    assert {p.name for p in metadata.iterdir()} == set(FILES) | {'identity.json'}, 'Exact metadata member universe required'
    assert all(p.is_file() and not p.is_symlink() for p in metadata.iterdir())
    assert sha(metadata / 'identity.json') == METADATA_IDENTITY_SHA256
    identity = read(metadata / 'identity.json'); safe(identity, 'identity.json')
    assert identity['kind'] == 't18_wiki_safe_metadata_v1'
    assert identity['model_id'] == MODEL_ID and identity['revision'] == REVISION
    assert identity['canonical_input_sha256'] == EXPECTED and identity['dependencies'] == DEPENDENCIES
    assert identity['canonical_calibration_maps_sha256'] == MAP_SHA and identity['canonical_calibration_tokens_sha256'] == CAL_TOKENS_SHA
    assert identity['changed_metadata_files'] == list(CHANGED)
    assert identity['exact_canonical_files'] == sorted(set(EXPECTED) - set(CHANGED))
    assert set(identity['metadata_sha256']) == set(FILES)
    dependencies()
    for n, h in identity['metadata_sha256'].items():
        assert sha(metadata / n) == h, ('Safe metadata changed', n)
        safe(read(metadata / n), n)
    for n in COPIED_EXACT:
        assert sha(metadata / n) == EXPECTED[n]
    assert sha(metadata / 'historical_preflight.json') == EXPECTED['preflight.json']
    return identity


def verify_capacity(inv, proof):
    spans = sorted((r['char_start'], r['char_end']) for r in inv['exclusions']['hume'])
    merged = []
    for lo, hi in spans:
        assert 0 <= lo < hi <= inv['sources']['hume']['characters']
        if merged and lo <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], hi)
        else: merged.append([lo, hi])
    assert merged == proof['hume']['union']
    edges = [[0, 0]] + merged + [[inv['sources']['hume']['characters']] * 2]
    gaps = [b[0] - a[1] for a, b in zip(edges, edges[1:])]
    assert max(gaps) == proof['hume']['maximum_gap'] == 5492 and max(gaps) < 8192


def reconstruct(config, metadata, calibration, out, tokenizer_factory=None):
    empty_out(out); started = time.time()
    state = {'status': 'checking', 'phase': 'metadata', 'model_forwards': 0, 'model_loaded': False,
             'gpu_used': False, 'calibration_fitted': False, 'script_sha256': sha(__file__)}
    dump(out / 'reconstruction.json', state)
    try:
        state['free_bytes_before'] = space_guard(out)
        identity = validate_metadata(metadata)
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer
        selection = importlib.import_module('gate_calibration_factorial_inputs')
        geometry = importlib.import_module('gate_direction_geometry')
        torch.set_num_threads(1)
        cfg = read(config); gc = cfg['models']['gpt2']
        assert gc['model_id'] == MODEL_ID and gc['revision'] == REVISION
        dm = read(metadata / 'data_manifest.template.json'); cm = read(metadata / 'calibration_inputs.template.json')
        inv = read(metadata / 'exclusion_inventory.json'); pinv = read(metadata / 'prior_exclusion_inventory.json')
        history = read(metadata / 'history_token_hashes.json'); start_template = read(metadata / 'start_identity.template.json')
        assert inv['exclusions'] == pinv['exclusions'] and inv['provenance_hashes'] == pinv['provenance_hashes']
        assert len(inv['provenance_hashes']) == 185
        assert inv['sources'] == dm['sources'] == pinv['sources']
        verify_capacity(inv, read(metadata / 'capacity_attestation.json'))
        assert sha(calibration / 'calibration_maps.pt') == MAP_SHA and sha(calibration / 'calibration_tokens.pt') == CAL_TOKENS_SHA
        tokpath = Path(gc['path'])
        for n, h in dm['tokenizer_sha256'].items():
            assert Path(n).name == n and sha(tokpath / n) == h, ('Tokenizer mismatch', n)
        tokenizer_factory = tokenizer_factory or AutoTokenizer.from_pretrained
        tokenizer = tokenizer_factory(tokpath, local_files_only=True); assert tokenizer.is_fast
        state['phase'] = 'source_verification'; texts = {}
        for c in PHYSICAL_SOURCES:
            source = dm['sources'][c]; p = Path(cfg['sources'][c])
            assert p.name == source['filename'] and sha(p) == source['file_sha256'], ('Source mismatch', c)
            text = '\n\n'.join(Dataset.from_file(str(p))['text']) if c.startswith('wiki_') else p.read_text()
            if c in ('utilitarianism', 'ethics'): text = selection.strip_book(text)
            assert len(text) == source['characters'] and selection.textsha(text) == source['processed_text_sha256']
            texts[c] = text
        texts['optimizers_new'] = '\n\n'.join(texts[c] for c in ('adadelta', 'adamax', 'asgd'))
        assert selection.textsha(texts['optimizers_new']) == dm['sources']['optimizers_new']['processed_text_sha256']
        assert len(texts['optimizers_new']) == dm['sources']['optimizers_new']['characters']
        components = dm['sources']['optimizers_new']['components']; cursor = 0
        for component, c in zip(components, ('adadelta', 'adamax', 'asgd')):
            assert component == {'source': c, 'char_start': cursor, 'char_end': cursor + len(texts[c]), 'file_sha256': dm['sources'][c]['file_sha256']}
            cursor += len(texts[c]) + 2
        assert len(components) == 3
        for c, rows in inv['exclusions'].items():
            for row in rows: assert 0 <= row['char_start'] < row['char_end'] <= dm['sources'][c]['characters']
        union = sorted(set().union(*(set(v) for v in history['families'].values())))
        assert union == history['evaluation_union'] and len(union) == 561
        assert history['calibration_union'] == sorted(set(union) - set(history['families']['retained_fresh_calibration']))
        state['phase'] = 'retained_calibration_verification'; cal = {}
        selection.select_panel(('wiki_test',), texts, tokenizer, inv['exclusions'], set(history['calibration_union']), components, True, cal)
        assert clean(cal['selected']) == cm['passages'] and cal['rejections']['wiki_test'] == cm['rejections']
        assert len(cal['items']) == 4 and len(cm['rejections']) == 7
        torch.save(cal['items'], out / 'partial_calibration_tokens.pt')
        (out / 'partial_calibration_tokens.pt').rename(out / 'calibration_tokens.pt')
        assert sha(out / 'calibration_tokens.pt') == EXPECTED['calibration_tokens.pt'], 'Canonical retained tensor archive differs'
        state['phase'] = 'evaluation_policy_replay'; ev = {}
        selection.select_panel(EVALUATION, texts, tokenizer, inv['exclusions'], set(union), components, False, ev)
        assert clean(ev['selected']) == dm['passages'] and ev['rejections'] == dm['rejections']
        assert len(ev['items']) == 16 and sum(len(v) for v in ev['rejections'].values()) == 58
        torch.save(ev['items'], out / 'tokens.pt'); assert sha(out / 'tokens.pt') == EXPECTED['tokens.pt']
        cm['passages'] = cal['selected']; dm['passages'] = ev['selected']
        for name, literal in dm['literals'].items():
            if 'literal' in literal: literal['ids'] = tokenizer(literal['literal'], add_special_tokens=False)['input_ids']
            else:
                assert name == 'start_special' and literal['label'] == 'bos'
                literal['id'] = tokenizer.bos_token_id
        for n in ('exclusion_inventory.json', 'prior_exclusion_inventory.json', *COPIED_EXACT):
            shutil.copyfile(metadata / n, out / n)
        dm['exclusion_inventory_sha256'] = sha(out / 'exclusion_inventory.json')
        dump(out / 'calibration_inputs.json', cm); dump(out / 'data_manifest.json', dm)
        state['phase'] = 'direction_and_start_identity_reconstruction'
        maps = torch.load(calibration / 'calibration_maps.pt', weights_only=True, map_location='cpu')
        directions = geometry.directions(maps)
        torch.save(directions, out / 'directions.pt'); (out / 'directions.pt').rename(out / 'mixture_directions.pt')
        dump(out / 'mixture_directions.json', {l: {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in row.items()} for l, row in directions.items()})
        old = torch.load(calibration / 'calibration_tokens.pt', weights_only=True, map_location='cpu')
        assert len(old) == 8 and [(x['corpus'], x['window']) for x in old] == [(c, i) for c in ('wiki_test', 'hume') for i in range(4)]
        oldstarts = {int(x['tokens'][0]) for x in old if x['corpus'] == 'wiki_test'}
        mixstarts = {int(x['tokens'][0]) for x in old}; freshstarts = {int(x['tokens'][0]) for x in cal['items']}
        bos = dm['literals']['start_special']['id']
        starts = {'W_old4_first_token_ids': sorted(oldstarts), 'W_fresh4_first_token_ids': sorted(freshstarts), 'M8_first_token_ids': sorted(mixstarts),
            'fresh_seen_in_old_wiki': [int(x['tokens'][0]) in oldstarts for x in cal['items']],
            'fresh_seen_in_M8': [int(x['tokens'][0]) in mixstarts for x in cal['items']],
            'fresh_seen_in_T16_T17_evaluation': start_template['fresh_seen_in_T16_T17_evaluation'],
            'evaluation_first_token_ids': [int(x['tokens'][0]) for x in ev['items']],
            'evaluation_seen_in_old_wiki': [int(x['tokens'][0]) in oldstarts for x in ev['items']],
            'evaluation_seen_in_fresh': [int(x['tokens'][0]) in freshstarts for x in ev['items']],
            'evaluation_seen_in_M8': [int(x['tokens'][0]) in mixstarts for x in ev['items']],
            'bos_id': bos, 'bos_seen_in_old_wiki': bos in oldstarts, 'bos_seen_in_fresh': bos in freshstarts, 'bos_seen_in_M8': bos in mixstarts}
        assert clean(starts) == start_template
        starts['prior_membership_scope'] = 'fresh_seen_in_T16_T17_evaluation is inherited attestation; old first-ID list omitted and not independently reconstructed.'
        starts['canonical_summary_sha256'] = EXPECTED['start_identity_summary.json']; dump(out / 'start_identity_summary.json', starts)
        for n in identity['exact_canonical_files']: assert sha(out / n) == EXPECTED[n], ('Exact canonical file differs', n)
        pf = {'status': 'passed', 'phase': 'completed', 'kind': 'metadata_only_T18_reconstruction',
            'model_forwards': 0, 'model_loaded': False, 'gpu_used': False, 'calibration_fitted': False,
            'calibration_passages': 4, 'evaluation_passages': 16, 'evaluation_inputs': 32,
            'evaluation_passages_per_corpus': {c: 4 for c in EVALUATION}, 'calibration_passages_per_corpus': {'wiki_test': 4},
            'canonical_preflight_sha256': EXPECTED['preflight.json'], 'metadata_identity_sha256': METADATA_IDENTITY_SHA256,
            'history_attestation_only': True, 'calibration_rejections_verified': 7, 'evaluation_rejections_verified': 58,
            'selector_history_hashes': 561, 'output_sha256': {n: sha(out / n) for n in EXPECTED if n != 'preflight.json'}}
        dump(out / 'preflight.json', pf)
        bindings = {'kind': 'explicit_T18_metadata_binding_only', 'original': EXPECTED,
            'regenerated': {n: sha(out / n) for n in EXPECTED}, 'changed_metadata_files': list(CHANGED),
            'exact_canonical_files': identity['exact_canonical_files'], 'metadata_identity_sha256': METADATA_IDENTITY_SHA256}
        dump(out / 'input_bindings.json', bindings)
        validate_reconstructed_semantics(out, metadata)
        assert sum(p.stat().st_size for p in out.iterdir()) + 8192 < ALLOWANCE
        assert shutil.disk_usage(out).free >= RESERVE + 8192
        state.update(status='passed', phase='completed', calibration_windows=4, evaluation_windows=16,
            calibration_rejections_verified=7, evaluation_rejections_verified=58, selector_history_hashes=561,
            source_files_verified=7, tokenizer_sha256=dm['tokenizer_sha256'], input_bindings_sha256=sha(out / 'input_bindings.json'),
            canonical_exact={n: sha(out / n) == EXPECTED[n] for n in identity['exact_canonical_files']},
            history_scope='Inherited attested history; no old raw experiment directories, Llama/OLMo tokenizer or model weights read.')
    except BaseException as error:
        state.update(status='failed', failed_phase=state['phase'], error_type=type(error).__name__, error=str(error)); raise
    finally:
        state.update(wall_seconds=time.time()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            output_bytes_before_final_manifest=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        dump(out / 'reconstruction.json', state)
    return state


def validate_reconstructed_semantics(out, metadata):
    """Independent allowlisted metadata transformations; no model/tokenizer needed."""
    identity = validate_metadata(metadata); binding = read(out / 'input_bindings.json')
    assert set(binding) == {'kind', 'original', 'regenerated', 'changed_metadata_files', 'exact_canonical_files', 'metadata_identity_sha256'}
    assert binding['kind'] == 'explicit_T18_metadata_binding_only'
    assert binding['original'] == EXPECTED and binding['changed_metadata_files'] == list(CHANGED)
    assert binding['exact_canonical_files'] == identity['exact_canonical_files'] and binding['metadata_identity_sha256'] == METADATA_IDENTITY_SHA256
    assert set(binding['regenerated']) == set(EXPECTED)
    for n, h in binding['regenerated'].items(): assert sha(out / n) == h
    for n in identity['exact_canonical_files']: assert binding['regenerated'][n] == EXPECTED[n]
    for n in ('exclusion_inventory.json', 'prior_exclusion_inventory.json'):
        assert sha(out / n) == sha(metadata / n)
    private = read(metadata / 'sanitization.json')['private_value_commitments']
    raw_cm = read(out / 'calibration_inputs.json'); raw_dm = read(out / 'data_manifest.json')
    assert value_sha([r['first_token_id'] for r in raw_cm['passages']]) == private['calibration_first_ids']
    assert value_sha([r['first_token_id'] for r in raw_dm['passages']]) == private['evaluation_first_ids']
    assert value_sha(raw_dm['literals']) == private['literal_ids']
    cm = clean(raw_cm); assert cm == read(metadata / 'calibration_inputs.template.json')
    dm = clean(raw_dm); template = read(metadata / 'data_manifest.template.json')
    assert dm['exclusion_inventory_sha256'] == sha(out / 'exclusion_inventory.json')
    dm['exclusion_inventory_sha256'] = template['exclusion_inventory_sha256']; assert dm == template
    allowed_cm = read(metadata / 'calibration_inputs.template.json')
    allowed_dm = copy.deepcopy(template)
    for expected_rows, actual_rows in ((allowed_cm['passages'], raw_cm['passages']), (allowed_dm['passages'], raw_dm['passages'])):
        assert len(expected_rows) == len(actual_rows)
        for expected_row, actual_row in zip(expected_rows, actual_rows):
            expected_row['first_token_id'] = actual_row['first_token_id']
    allowed_dm['literals'] = raw_dm['literals']
    allowed_dm['exclusion_inventory_sha256'] = sha(out / 'exclusion_inventory.json')
    assert raw_cm == allowed_cm and raw_dm == allowed_dm, 'Only declared ID restoration and one hash edge may change'
    starts = read(out / 'start_identity_summary.json')
    assert starts.pop('canonical_summary_sha256') == EXPECTED['start_identity_summary.json']
    assert starts.pop('prior_membership_scope') == 'fresh_seen_in_T16_T17_evaluation is inherited attestation; old first-ID list omitted and not independently reconstructed.'
    assert 'prior_T16_T17_evaluation_first_token_ids' not in starts
    assert clean(starts) == read(metadata / 'start_identity.template.json')
    assert value_sha(starts) == private['current_start_summary']
    pf = read(out / 'preflight.json')
    assert set(pf) == {'status', 'phase', 'kind', 'model_forwards', 'model_loaded', 'gpu_used', 'calibration_fitted',
        'calibration_passages', 'evaluation_passages', 'evaluation_inputs', 'evaluation_passages_per_corpus',
        'calibration_passages_per_corpus', 'canonical_preflight_sha256', 'metadata_identity_sha256',
        'history_attestation_only', 'calibration_rejections_verified', 'evaluation_rejections_verified', 'selector_history_hashes', 'output_sha256'}
    assert pf['status'] == 'passed' and pf['phase'] == 'completed' and pf['kind'] == 'metadata_only_T18_reconstruction'
    assert pf['model_forwards'] == 0 and pf['model_loaded'] is False and pf['gpu_used'] is False and pf['calibration_fitted'] is False
    assert pf['calibration_passages'] == 4 and pf['evaluation_passages'] == 16 and pf['evaluation_inputs'] == 32
    assert pf['evaluation_passages_per_corpus'] == {c: 4 for c in EVALUATION} and pf['calibration_passages_per_corpus'] == {'wiki_test': 4}
    assert pf['canonical_preflight_sha256'] == EXPECTED['preflight.json'] and pf['metadata_identity_sha256'] == METADATA_IDENTITY_SHA256
    assert pf['history_attestation_only'] is True and pf['calibration_rejections_verified'] == 7 and pf['evaluation_rejections_verified'] == 58 and pf['selector_history_hashes'] == 561
    assert pf['output_sha256'] == {n: sha(out / n) for n in EXPECTED if n != 'preflight.json'}
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    exp = sub.add_parser('export-metadata')
    for n in ('inputs', 'contract', 'audit', 'out'): exp.add_argument('--'+n, type=Path, required=True)
    rebuild = sub.add_parser('rebuild')
    for n in ('config', 'metadata', 'calibration', 'out'): rebuild.add_argument('--'+n, type=Path, required=True)
    verify = sub.add_parser('verify-reconstructed')
    for n in ('metadata', 'out'): verify.add_argument('--'+n, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'export-metadata':
        may_record_failure = not args.out.exists() or not any(args.out.iterdir())
        try:
            export_metadata(args.inputs, args.contract, args.audit, args.out)
        except BaseException as error:
            if may_record_failure and args.out.is_dir() and not (args.out / 'identity.json').exists():
                dump(args.out / 'export_failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                    'error': str(error), 'model_forwards': 0, 'model_loaded': False, 'gpu_used': False,
                    'script_sha256': sha(__file__)})
            raise
        print('EXPORTED_T18_METADATA', sha(args.out / 'identity.json'))
    elif args.mode == 'rebuild':
        result = reconstruct(args.config, args.metadata, args.calibration, args.out)
        print('REBUILT_T18_INPUTS', result['status'], 'zero model forwards')
    else:
        validate_reconstructed_semantics(args.out, args.metadata); print('VERIFIED_T18_METADATA_TRANSFORMS')


if __name__ == '__main__': main()
