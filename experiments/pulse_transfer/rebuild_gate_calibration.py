"""Explicit eight-forward calibration rebuild; never selects or evaluates maps.

This is a separate replication artifact. It never substitutes its output into
the reference-map experiment automatically. See GATE_CALIBRATION_REBUILD.md.
"""
import argparse
import json
import platform
import resource
import time
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from gate_inputs import prepare_gate_inputs
from gpt2_gate_subspace import GateSubspace, GateRunner, fit_maps, maps_digest, tensor_hash
from gpt2_layer_writes import LayerWrites
from run_transfer import ROOT, dump, modelhash, sha, th

SOURCES = ('rebuild_gate_calibration.py', 'GATE_CALIBRATION_REBUILD.md', 'gate_inputs.py', 'gpt2_gate_subspace.py',
           'gpt2_query_source.py', 'gpt2_regrowth_kv.py', 'gpt2_layer_writes.py',
           'gpt2_writes.py', 'llama_source_kv.py', 'run_transfer.py',
           'fresh_inputs_v2.py', 'GPT2_GATE_SUBSPACE.md')


def tensor_comparison(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'exact': torch.equal(a, b),
            'max_absolute_error': float((a.double()-b.double()).abs().max())}


def compare_reference(out, reference):
    """Report differences; a mismatch is never a fallback or a tuning signal."""
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    maps, refmaps = load(out/'calibration_maps.pt'), load(reference/'calibration_maps.pt')
    raw, refraw = load(out/'calibration_raw.pt'), load(reference/'calibration_raw.pt')
    assert set(maps) == set(refmaps) and set(raw) == set(refraw)
    comparison = {'reference_files': {n: sha(reference/n) for n in
                  ('calibration_maps.pt', 'calibration_raw.pt')}, 'layers': {}, 'raw': {}}
    for layer, mapping in maps.items():
        assert set(mapping) == set(refmaps[layer])
        row = {'tensors': {k: tensor_comparison(v, refmaps[layer][k]) for k,v in mapping.items()},
               'subspaces': {}}
        for name in ('basis_r', 'basis_b', 'basis_qr'):
            for rank in (1,3,6):
                u, v = mapping[name][:,:rank].double(), refmaps[layer][name][:,:rank].double()
                row['subspaces'][name+'_'+str(rank)] = {
                    'principal_cosines': torch.linalg.svdvals(u.T@v).tolist(),
                    'projector_max_absolute_error': float((u@u.T-v@v.T).abs().max())}
        comparison['layers'][layer] = row
    for key, layers in raw.items():
        assert set(layers) == set(refraw[key])
        comparison['raw'][key] = {}
        for layer, fields in layers.items():
            assert set(fields) == set(refraw[key][layer])
            comparison['raw'][key][layer] = {k: tensor_comparison(v, refraw[key][layer][k]) for k,v in fields.items()}
    comparison['all_map_tensors_exact'] = all(t['exact'] for r in comparison['layers'].values() for t in r['tensors'].values())
    comparison['all_raw_tensors_exact'] = all(t['exact'] for ls in comparison['raw'].values() for fs in ls.values() for t in fs.values())
    comparison['byte_exact'] = {n: sha(out/n)==sha(reference/n) for n in
                                ('calibration_maps.pt', 'calibration_raw.pt')}
    comparison['note'] = 'Rebuilt maps remain separate; no automatic substitution, rank selection, or evaluation.'
    return comparison


def precision_settings():
    return {'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'flash_sdp_enabled': torch.backends.cuda.flash_sdp_enabled(),
            'memory_efficient_sdp_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp_enabled': torch.backends.cuda.math_sdp_enabled(),
            'cudnn_sdp_enabled': torch.backends.cuda.cudnn_sdp_enabled()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args(); out = args.out
    assert not out.exists() or not any(out.iterdir()), 'Output must be new or empty'
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    old = json.loads((ROOT/'gpt2_run/manifest.json').read_text())
    assert sha(ROOT/'gpt2_run/selection.json') == old['selection_sha256']
    selected = json.loads((ROOT/'gpt2_run/selection.json').read_text())['selected']
    assert selected['layer'] == 2 and selected['row'] == 447
    assert cfg['models']['gpt2']['revision'] == old['revision']
    reference_hashes = None if args.reference is None else {
        n: sha(args.reference/n) for n in ('calibration_maps.pt','calibration_raw.pt')}
    torch.set_num_threads(4); torch.manual_seed(20261021)
    started = time.time(); path = cfg['models']['gpt2']['path']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    cal, items, dm = prepare_gate_inputs(cfg, tokenizer, out)
    assert len(cal) == 8 and len(items) == 16
    calstarts = [int(v['tokens'][0]) for v in cal]
    evalstarts = [int(v['tokens'][0]) for v in items]
    bos = dm['literals']['start_special']['id']
    dump(out/'start_identity_summary.json', {
        'calibration_first_token_ids':calstarts, 'evaluation_natural_first_token_ids':evalstarts,
        'bos_id':bos, 'evaluation_natural_matches_calibration':[x in set(calstarts) for x in evalstarts],
        'bos_seen_in_calibration':bos in set(calstarts),
        'note':'Natural/BOS are not assumed to be unseen token identities.'})
    meta = {'status': 'running', 'kind': 'calibration_only_rebuild',
            'started_utc': time.strftime('%FT%TZ',time.gmtime()),
            'model_id': old['model_id'], 'revision': old['revision'], 'dtype': 'float32',
            'attention': 'sdpa_default', 'seed': 20261021, 'qr_seed': 20261025,
            'expected_forwards': 8, 'evaluation_forwards': 0,
            'source_hashes': {n: sha(ROOT/n) for n in SOURCES},
            'calibration_inputs_sha256': sha(out/'calibration_inputs.json'),
            'evaluation_inputs_sha256': sha(out/'data_manifest.json'),
            'calibration_tokens_sha256': sha(out/'calibration_tokens.pt'),
            'reference_hashes': reference_hashes,
            'precision_settings_before': precision_settings(),
            'runtime_versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                'transformers': transformers.__version__, 'numpy': version('numpy'),
                'cuda_runtime': torch.version.cuda},
            'note': 'Original helper also prepares evaluation token files; no evaluation input is forwarded. Saved calibration KL is self-reference, not native-C fidelity.'}
    dump(out/'manifest.json', meta)
    raw = {}; metrics = {}; operations = []; hidden = {}; model = probe = gate = runner = None
    active = None
    try:
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                    device_map='cuda', attn_implementation='sdpa', local_files_only=True)
        model.eval(); model.requires_grad_(False)
        before = modelhash(model); assert before == old['model_hash_before']
        versions = {n:p._version for n,p in model.named_parameters()}
        meta['model_hash_before'] = before; dump(out/'manifest.json',meta)
        probe = LayerWrites(model,selected,[447,138]); gate = GateSubspace(model)
        runner = GateRunner(model,probe,out)
        def capture_hidden(mod, a, output): hidden['value'] = output.detach().cpu().clone()
        gate.handles.append(model.transformer.ln_f.register_forward_hook(capture_hidden))
        for item in cal:
            active = {'corpus':item['corpus'],'window':item['window']}
            key = f"{item['corpus']}__{item['window']}"
            probe.reset_layers(E=False,blocked=[],special=None)
            gate.reset_gate(mode='calibration',zero=True,mask='causal'); hidden.clear()
            z, logits, _, _ = runner.ev(item['tokens'],baseline=None,keep=True)
            z['final_hidden_sha256'] = tensor_hash(hidden['value'])
            assert probe.locality == 30 and probe.writer_zero_checks == 0
            assert gate.query_checks == gate.context_checks == gate.projection_checks == 8
            assert len(gate.gate_checks) == 8
            raw[key] = {l:{k:d[k] for k in ('b','mass')} for l,d in gate.gate_cache.items()}
            metrics[key] = {'token_sha256':th(item['tokens']),'kl_reference':'self/no external C reference',**z}
            operations.append({**active,'gate_checks':gate.gate_checks,
                'max_context_error':max(d['context_max_abs_error'] for d in gate.diag.values())})
            torch.save(raw,out/'calibration_raw.pt'); torch.save(metrics,out/'calibration_metrics.pt')
            dump(out/'calibration_operations.json',operations); del logits
        maps, spectra = fit_maps(raw)
        torch.save(maps,out/'calibration_maps.pt'); dump(out/'calibration_spectra.json',spectra)
        digest = maps_digest(maps)
        meta['calibration_frozen_utc'] = time.strftime('%FT%TZ',time.gmtime())
        if args.reference is not None:
            assert reference_hashes == {n:sha(args.reference/n) for n in reference_hashes}
            dump(out/'reference_comparison.json',compare_reference(out,args.reference))
        assert runner.count == 8 and len(raw) == len(metrics) == 8
        after = modelhash(model); assert before == after
        assert versions == {n:p._version for n,p in model.named_parameters()}
        assert precision_settings() == meta['precision_settings_before']
        assert meta['source_hashes'] == {n:sha(ROOT/n) for n in SOURCES}
        meta.update(status='completed', completed_utc=time.strftime('%FT%TZ',time.gmtime()),
            forward_passes=8, calibration_forwards=8, evaluation_forwards=0,
            calibration_raw_sha256=sha(out/'calibration_raw.pt'),
            calibration_maps_sha256=sha(out/'calibration_maps.pt'),
            calibration_maps_tensor_digest=digest, model_hash_after=after,
            parameter_versions_unchanged=True, wall_seconds=time.time()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(out/'manifest.json',meta); print('CALIBRATION_REBUILD_COMPLETE',8,flush=True)
    except BaseException as error:
        torch.save(raw,out/'calibration_raw.pt'); torch.save(metrics,out/'calibration_metrics.pt')
        dump(out/'calibration_operations.json',operations)
        torch.save({'active_input':active,'current_hidden':hidden.get('value'),
                    'gate_failure':None if gate is None else gate.failure_gate,
                    'reconstruction_failure':None if gate is None else gate.failure_reconstruction},out/'failure_state.pt')
        meta.pop('completed_utc',None)
        meta.update(status='failed',failed_utc=time.strftime('%FT%TZ',time.gmtime()),
                    forward_passes=0 if runner is None else runner.count,
                    failure_type=type(error).__name__,failure_message=str(error))
        dump(out/'manifest.json',meta); raise
    finally:
        if gate is not None: gate.close()
        if probe is not None: probe.close()


if __name__ == '__main__': main()
