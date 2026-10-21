"""Offline fixed-source replacement experiment; see LLAMA_REPLACEMENTS.md."""
import argparse, hashlib, json, math, resource, time
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_transfer import ROOT, sha, th, dump, strip_book, modelhash, Probe, Runner
from fresh_inputs_v2 import prepare_fresh
from llama_source_kv import KV


def calibration_inputs(cfg, tok, parent, out):
    old = json.loads((parent / 'data_manifest.json').read_text())
    items, records = [], []
    for corpus in ('wiki_test', 'hume'):
        path = Path(cfg['sources'][corpus])
        assert sha(path) == old['sources'][corpus]['sha256']
        s = '\n\n'.join(Dataset.from_file(str(path))['text']) if corpus == 'wiki_test' else strip_book(path.read_text())
        assert hashlib.sha256(s.encode()).hexdigest() == old['sources'][corpus]['processed_text_sha256']
        spans = []
        for r in old['passages']:
            if r['split'] != 'development' or r['corpus'] != corpus:
                continue
            a = r['char_start']
            enc = tok(s[a:a+8192], add_special_tokens=False, return_offsets_mapping=True)
            assert th(torch.tensor(enc['input_ids'][:512])) == r['window_token_sha256']
            spans.append((a, a + enc['offset_mapping'][511][1]))
        for i, fraction in enumerate((.2, .4, .6, .8)):
            a = math.floor(fraction * (len(s) - 8192))
            enc = tok(s[a:a+8192], add_special_tokens=False, return_offsets_mapping=True)
            assert len(enc['input_ids']) >= 512
            b = a + enc['offset_mapping'][511][1]
            assert all(b <= aa or a >= bb for aa, bb in spans), (corpus, a, b, spans)
            spans.append((a, b))
            x = torch.tensor(enc['input_ids'][:512])
            items.append({'corpus': corpus, 'window': i, 'tokens': x})
            records.append({'corpus': corpus, 'window': i, 'char_start': a, 'used_char_end': b,
                            'first_token_id': int(x[0]), 'window_token_sha256': th(x),
                            'passage_sha256': hashlib.sha256(s[a:a+8192].encode()).hexdigest()})
    torch.save(items, out / 'calibration_tokens.pt')
    dump(out / 'calibration_inputs.json', {'passages': records,
         'distinct_first_tokens': len({r['first_token_id'] for r in records})})
    return items


class MeanPulse:
    def __init__(self, probe, mean):
        self.p, self.mean, self.mode, self.checks = probe, mean, None, []
        self.handle = probe.downs[probe.selected['layer']].register_forward_hook(self.hook)

    def hook(self, module, args, output):
        if self.mode is None:
            return output
        assert self.p.patch is None
        r, c = self.p.selected['row'], self.p.selected['column']
        a = args[0].clone()
        a[0, 0, c] = self.mean
        assert torch.equal(a[:, 1:], args[0][:, 1:])
        keep = torch.ones(a.shape[-1], dtype=torch.bool, device=a.device)
        keep[c] = False
        assert torch.equal(a[0, 0, keep], args[0][0, 0, keep])
        z = F.linear(a, module.weight, module.bias)
        result = output.clone()
        if self.mode == 'row':
            result[0, 0, r] = z[0, 0, r]
            keepout = torch.ones(output.shape[-1], dtype=torch.bool, device=output.device)
            keepout[r] = False
            assert torch.equal(result[0, 0, keepout], output[0, 0, keepout])
        else:
            assert self.mode == 'column'
            result[0, 0] = z[0, 0]
        assert torch.equal(result[:, 1:], output[:, 1:])
        self.p.cache['selected']['output_source_after'] = result[0, 0].detach().cpu().clone()
        self.checks.append({'mode': self.mode, 'mean_activation': float(a[0, 0, c]), 'locality': True})
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists()
    parent = ROOT / 'llama_run'
    old = json.loads((parent / 'manifest.json').read_text())
    assert sha(parent / 'selection.json') == old['selection_sha256']
    selected = json.loads((parent / 'selection.json').read_text())['selected']
    dump(out / 'selection.json', selected)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    path = cfg['models']['llama']['path']
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    cal = calibration_inputs(cfg, tok, parent, out)
    items, dm = prepare_fresh(cfg, tok, parent, out, additional_parents=(ROOT / 'llama_source_kv_run',))
    calibration_ids = {int(item['tokens'][0]) for item in cal}
    dump(out / 'first_token_coverage.json', [{'corpus': item['corpus'], 'window': item['window'],
         'identity': identity, 'first_token_id': first_id, 'seen_in_calibration': first_id in calibration_ids}
         for item in items for identity, first_id in (('natural', int(item['tokens'][0])),
                                                      ('bos', dm['literals']['start_special']['id']))])
    meta = {'started_utc': time.strftime('%FT%TZ', time.gmtime()), 'model_id': old['model_id'],
            'revision': old['revision'], 'dtype': 'bfloat16', 'attention': 'sdpa', 'seed': 20261021,
            'parent_manifest_sha256': sha(parent / 'manifest.json'), 'selection_sha256': sha(out / 'selection.json'),
            'source_hashes': {p: sha(ROOT / p) for p in ('llama_replacements.py', 'run_transfer.py',
                             'fresh_inputs_v2.py', 'llama_source_kv.py', 'LLAMA_REPLACEMENTS.md')}}
    dump(out / 'manifest.json', meta)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map='cuda',
                                                attn_implementation='sdpa', local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    before = modelhash(model)
    assert before == old['model_hash_before']
    versions = {n: p._version for n, p in model.named_parameters()}
    p = Probe(model, 'llama')
    p.selected, p.capture_all = selected, False
    kv = KV(model, selected['layer'] + 1)
    run = Runner(model, p, out)
    calibration, clean_donors, patch_checks = [], {}, []
    pulse = None
    try:
        for item in cal:
            p.setpatch(None)
            kv.reset()
            z, _, cap, _ = run.ev(item['tokens'])
            calibration.append({'corpus': item['corpus'], 'window': item['window'],
                                'first_token_id': int(item['tokens'][0]), 'token_sha256': th(item['tokens']),
                                'activation': cap['selected']['a0'], 'loss': z['loss'],
                                'kv': {(layer, kind): vec.clone() for layer, dd in kv.cache.items() for kind, vec in dd.items()}})
        means = {}
        for label, corpus in (('pooled', None), ('wiki', 'wiki_test'), ('hume', 'hume')):
            rows = [r for r in calibration if corpus is None or r['corpus'] == corpus]
            means[label] = {key: torch.stack([r['kv'][key].float() for r in rows]).mean(0).bfloat16()
                            for key in calibration[0]['kv']}
        rng = torch.Generator().manual_seed(20261021)
        random_mean = {}
        for key, value in means['pooled'].items():
            random_mean[key] = value[torch.randperm(value.numel(), generator=rng)] * (2 * torch.randint(2, value.shape, generator=rng) - 1)
            assert torch.equal(random_mean[key].abs().sort().values, value.abs().sort().values)
        activation_mean = torch.tensor([r['activation'] for r in calibration]).mean().bfloat16()
        frozen = {'calibration': calibration, 'means': means, 'random_mean': random_mean, 'activation_mean': activation_mean}
        torch.save(frozen, out / 'calibration.pt')
        meta['calibration_sha256'] = sha(out / 'calibration.pt')
        meta['calibration_inputs_sha256'] = sha(out / 'calibration_inputs.json')
        dump(out / 'manifest.json', meta)
        pulse = MeanPulse(p, activation_mean)

        def ev(x, world='C', patch=None, mean_pulse=None, baseline=None, keep=False):
            pulse.mode = mean_pulse
            kv.reset(patch)
            if world == 'D':
                p.setpatch('column')
                r, c = selected['row'], selected['column']
                p.patch['weight'][r, c] = p.downs[selected['layer']].weight[r, c]
            else:
                p.setpatch({'C': None, 'R': 'row', 'N': 'column', 'oracle': 'oracle'}[world])
            z, logits, cap, norms = run.ev(x, baseline=baseline, keep=keep)
            if world == 'D':
                assert torch.equal(cap['selected']['output_source_before'][selected['row']],
                                   cap['selected']['output_source_after'][selected['row']])
            return z, logits, cap, norms

        for item in items:
            for identity in ('natural', 'bos'):
                x = item['tokens'].clone()
                if identity == 'bos':
                    x[0] = dm['literals']['start_special']['id']
                base = {'panel': 'replacement', 'corpus': item['corpus'], 'window': item['window'], 'length': 512, 'identity': identity}
                key = f"replacement__{item['corpus']}__{item['window']}__512__{identity}"
                b, bl, cap, norms = ev(x, keep=True)
                clean = {(layer, kind): v.clone() for layer, dd in kv.cache.items() for kind, v in dd.items()}
                clean_donors[key] = clean
                run.rec(key + '__C', {**base, 'trial': 'C'}, x, b, b, cap, norms)
                trials = [('R', 'R', None, None), ('D', 'D', None, None), ('N', 'N', None, None),
                          ('N_cleanKV', 'N', clean, None)]
                for world in ('C', 'N'):
                    for kinds in ('k', 'v', 'kv'):
                        trials.append((world + '_mean_' + kinds, world,
                                       {k: v for k, v in means['pooled'].items() if k[1] in kinds}, None))
                for label in ('wiki', 'hume'):
                    trials.append(('C_mean_' + label + '_kv', 'C', means[label], None))
                trials.append(('C_random_mean_kv', 'C', random_mean, None))
                for mode in ('row', 'column'):
                    trials.append(('mean_pulse_' + mode, 'C', None, mode))
                if identity == 'natural' and item['window'] == 0:
                    trials.extend([('repeat', 'C', None, None), ('oracle', 'oracle', None, None)])
                refs = {}
                for trial, world, patch, mode in trials:
                    z, _, cap, norms = ev(x, world, patch, mode, baseline=bl)
                    run.rec(key + '__' + trial, {**base, 'trial': trial}, x, z, b, cap, norms)
                    if patch is not None:
                        patch_checks.append({**base, 'trial': trial, 'patches': len(kv.errors), 'max_error': max(kv.errors)})
                    target = b if trial in ('N_cleanKV', 'repeat', 'oracle') else refs.get('C_mean_kv') if trial == 'N_mean_kv' else None
                    if target is not None:
                        check = {**base, 'trial': trial, 'loss_exact': torch.equal(z['loss'][1:], target['loss'][1:]),
                                 'pred_exact': torch.equal(z['pred'][1:], target['pred'][1:])}
                        run.checks.append(check)
                        assert check['loss_exact'] and check['pred_exact'], check
                    refs[trial] = z
                del bl
            run.checkpoint()
            torch.save(clean_donors, out / 'clean_donors.pt')
            dump(out / 'patch_checks.json', patch_checks)
            dump(out / 'pulse_checks.json', pulse.checks)
            print('REPLACEMENTS_DONE', item['corpus'], item['window'], run.count, flush=True)
        meta.update(completed_utc=time.strftime('%FT%TZ', time.gmtime()), forward_passes=run.count,
                    calibration_forwards=len(cal), records=len(run.raw), wall_seconds=time.time()-started,
                    model_hash_before=before, model_hash_after=modelhash(model),
                    parameter_versions_unchanged=versions == {n: p._version for n, p in model.named_parameters()},
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert meta['model_hash_before'] == meta['model_hash_after'] and meta['parameter_versions_unchanged']
        dump(out / 'manifest.json', meta)
        print('COMPLETE', meta['forward_passes'], meta['wall_seconds'], flush=True)
    finally:
        if pulse is not None:
            pulse.handle.remove()
        kv.close()
        p.close()


if __name__ == '__main__':
    main()
