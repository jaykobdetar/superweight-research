"""T19 tokenizer-only fresh panel, extending the frozen T18 exposure inventory.

No checkpoint weights, calibration fit or model forward. Historical evidence is
pinned and inherited explicitly; current source bytes/spans are rechecked.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse, copy, hashlib, json, resource, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FROZEN_SELECTOR = '810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a'
HISTORY_SCAN = '351dbebe91387f1c228a2a35938cc621ec88eb56c49912e15683cc030897d20b'
T18_MANIFEST = 'f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060'
RESERVE = 2 * 1024**3
ALLOWANCE = 16 * 1024**2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

assert sha(ROOT/'gate_calibration_factorial_inputs.py') == FROZEN_SELECTOR
import gate_calibration_factorial_inputs as old
import torch
from datasets import Dataset
from transformers import AutoTokenizer

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a=ap.parse_args(); out=a.out
    assert not out.exists() or not any(out.iterdir()), 'Use a new directory; preserve prior attempts'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); started=time.time(); progress={}
    meta={'status':'checking', 'phase':'pins', 'model_forwards':0, 'model_loaded':False,
          'gpu_used':False, 'calibration_fitted':False, 'physical_length':512,
          'script_sha256':sha(__file__), 'protocol_sha256':sha(ROOT/'GPT2_NATIVE_VALUE_BOUNDARY_PROPOSAL.md'),
          'selector_sha256':FROZEN_SELECTOR, 'history_scan_sha256':HISTORY_SCAN,
          'history_pin_file_sha256':sha(ROOT/'gate_native_value_history_pins.json'),
          'started_utc':time.strftime('%FT%TZ',time.gmtime())}
    dump(out/'preflight.json', meta)
    try:
        assert shutil.disk_usage(out).free >= RESERVE+ALLOWANCE
        pins=read(ROOT/'gate_native_value_history_pins.json')
        assert len(pins['files'])==144
        for name,h in pins['files'].items():
            p=WORK/name
            assert p.resolve().is_relative_to(WORK.resolve()) and sha(p)==h, ('Changed history',name)
        assert sha(ROOT/'review/gate_native_value_history_scan.json')==HISTORY_SCAN
        prior_scan=read(ROOT/'review/gate_native_value_history_scan.json')
        scan=old.scan_new_sources(out, read(ROOT/'calibration_wiki_inputs/data_manifest.json')['sources'])
        assert scan==prior_scan, 'Historical source coverage changed; resolve before selecting'
        assert sorted({r['manifest'] for r in scan['new_source_matches']})==pins['expected_new_source_manifests']
        assert len(scan['files'])==123
        dump(out/'historical_manifest_scan.json',scan)
        t18=ROOT/'gpt2_gate_wiki_calibration_run'
        assert sha(t18/'manifest.json')==T18_MANIFEST
        run=read(t18/'manifest.json')
        assert run['status']=='completed' and run['forward_passes']==1548
        assert run['calibration_map_digests']['F']=='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'
        refs={n:sha(t18/n) for n in ('calibration_maps.npz','calibration_directions.npz','calibration_geometry.npz')}
        assert refs['calibration_maps.npz']==run['calibration_maps_sha256']
        assert refs['calibration_directions.npz']==run['calibration_directions_sha256']
        previous=read(ROOT/'calibration_wiki_inputs/data_manifest.json')
        sources=previous['sources']; texts={}
        meta['phase']='source_bytes_and_tokenizer'
        for c in ('wiki_validation','utilitarianism','ethics','adadelta','adamax','asgd'):
            spec=sources[c]; p=Path(spec['path'])
            assert sha(p)==spec['file_sha256'], ('Changed source',c)
            text='\n\n'.join(Dataset.from_file(str(p))['text']) if c=='wiki_validation' else old.strip_book(p.read_text()) if c in ('utilitarianism','ethics') else p.read_text()
            assert len(text)==spec['characters'] and old.textsha(text)==spec['processed_text_sha256']
            texts[c]=text
        texts['optimizers_new']='\n\n'.join(texts[c] for c in ('adadelta','adamax','asgd'))
        assert old.textsha(texts['optimizers_new'])==sources['optimizers_new']['processed_text_sha256']
        cursor=0; components=[]
        for name in ('adadelta','adamax','asgd'):
            components.append({'source':name,'char_start':cursor,'char_end':cursor+len(texts[name]),'file_sha256':sources[name]['file_sha256']})
            cursor+=len(texts[name])+2
        assert components==sources['optimizers_new']['components']
        cfg=read(a.config); model=read(WORK/'superweight-causal-circuits/manifests/transfer_models.json')['gpt2']
        configured=cfg['models']['gpt2']; assert configured['model_id']==model['model_id'] and configured['revision']==model['revision']
        tokenizer_path=Path(configured['path']); tokenizer_hashes={}
        for name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
            h=sha(tokenizer_path/name); assert h==model['checkpoint_files'][name]['sha256']; tokenizer_hashes[name]=h
        tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True); assert tok.is_fast
        # Extend, never replace, the inherited historical exclusion inventory.
        inherited=read(ROOT/'calibration_wiki_inputs/exclusion_inventory.json')
        exclusions=copy.deepcopy(inherited['exclusions'])
        retained_buffers=copy.deepcopy(inherited['retained_calibration_buffers'])
        assert len(retained_buffers)==4
        for rec in retained_buffers:
            assert rec['corpus']=='wiki_test' and rec['char_end']-rec['char_start']==8192
            exclusions[rec['corpus']].append({'char_start':rec['char_start'],'char_end':rec['char_end'],
                'origin':'pulse_transfer/calibration_wiki_inputs/calibration_inputs.json',
                'family':'T18_fresh_F_full8192_calibration_buffer'})
        history=read(ROOT/'wiki_replay_metadata/history_token_hashes.json')
        old_hashes=set(history['evaluation_union']); assert len(old_hashes)==561
        old_hashes.update(scan['historical_window_token_hashes'])
        previous_items=old.load(ROOT/'calibration_wiki_inputs/tokens.pt')
        assert len(previous_items)==len(previous['passages'])==16
        for rec,item in zip(previous['passages'],previous_items):
            assert (item['corpus'],item['window'])==(rec['corpus'],rec['window'])
            x=item['tokens']; assert old.th(x)==rec['window_token_sha256']
            c=rec['corpus']; start=rec['char_start']; buffer=texts[c][start:start+8192]
            assert old.textsha(buffer)==rec['passage_sha256']
            encoded=tok(buffer,add_special_tokens=False,return_offsets_mapping=True)
            assert torch.equal(torch.tensor(encoded['input_ids'][:512],dtype=torch.long),x)
            end=start+max(hi for lo,hi in encoded['offset_mapping'][:512]); assert end==rec['used_char_end']
            exclusions[c].append({'char_start':start,'char_end':end,'origin':'pulse_transfer/calibration_wiki_inputs/data_manifest.json',
                'family':'T18_used512_token_span','window':rec['window'],'token_sha256':old.th(x)})
            old_hashes.add(old.th(x))
        # Fresh F calibration already appears in the inherited561 union.
        fresh_cal=old.load(ROOT/'calibration_wiki_inputs/calibration_tokens.pt')
        assert len(fresh_cal)==4 and all(old.th(v['tokens']) in old_hashes for v in fresh_cal)
        for c,records in exclusions.items():
            for rec in records: assert 0<=rec['char_start']<rec['char_end']<=sources[c]['characters']
        frozen_history=sorted(old_hashes)
        dump(out/'exclusion_inventory.json',{'sources':sources,'exclusions':exclusions,
            'retained_calibration_buffers':retained_buffers,
            'inherited_exclusion_sha256':sha(ROOT/'calibration_wiki_inputs/exclusion_inventory.json'),
            'T18_input_manifest_sha256':sha(ROOT/'calibration_wiki_inputs/data_manifest.json'),
            'historical_hash_families_sha256':sha(ROOT/'wiki_replay_metadata/history_token_hashes.json'),
            'history_token_hashes':frozen_history,'freshness_scope':'Recorded research exposure, not pretraining or universal historical exposure.'})
        meta['phase']='selection'
        old.select_panel(old.EVALUATION,texts,tok,exclusions,old_hashes,components,False,progress)
        assert len(progress['items'])==len(progress['selected'])==16
        torch.save(progress['items'],out/'tokens.pt')
        dump(out/'data_manifest.json',{'sources':sources,'passages':progress['selected'],'rejections':progress['rejections'],
            'literals':previous['literals'],'tokenizer_sha256':tokenizer_hashes,'physical_length':512,
            'passages_per_corpus':{c:4 for c in old.EVALUATION},'exclusion_inventory_sha256':sha(out/'exclusion_inventory.json'),
            'calibration_reference_manifest_sha256':T18_MANIFEST,'calibration_reference_files':refs,
            'calibration_fit':'Frozen T18 F maps; no new calibration',
            'selection_rule':{'grid_denominator':1000,'preferred_numerators':list(old.PREFERRED),
                'order':'distance then numerator; four passages required per corpus',
                'evaluation':'used512-token span','historical_calibration':'inherited full8192 buffers unchanged',
                'code_used_span_inside_single_component':True}})
        calstarts={int(v['tokens'][0]) for v in fresh_cal}; oldstarts={int(v['tokens'][0]) for v in previous_items}; bos=previous['literals']['start_special']['id']
        dump(out/'start_identity_summary.json',{'evaluation_first_token_ids':[int(v['tokens'][0]) for v in progress['items']],
            'evaluation_seen_in_F':[int(v['tokens'][0]) in calstarts for v in progress['items']],
            'evaluation_seen_in_T18':[int(v['tokens'][0]) in oldstarts for v in progress['items']],
            'bos_id':bos,'bos_seen_in_F':bos in calstarts,'bos_seen_in_T18':bos in oldstarts,
            'start_intervention':'Replace token0 only; preserve all subsequent token IDs and physical length'})
        assert sha(t18/'manifest.json')==T18_MANIFEST
        assert all(sha(WORK/n)==h for n,h in pins['files'].items())
        size=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        assert size<ALLOWANCE and shutil.disk_usage(out).free>=RESERVE
        meta.update(status='passed',phase='completed',evaluation_passages=16,evaluation_inputs=32,
            inherited_history_count=561,exclusion_hash_count=len(frozen_history),preparation_bytes=size,
            retained_calibration_map_digest=run['calibration_map_digests']['F'])
    except BaseException as error:
        meta.update(status='failed',failure_phase=meta['phase'],error_type=type(error).__name__,error=str(error))
        if progress.get('items'): torch.save(progress['items'],out/'partial_tokens.pt')
        dump(out/'failure_selection.json',{k:v for k,v in progress.items() if k!='items'})
        raise
    finally:
        meta.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=time.strftime('%FT%TZ',time.gmtime()))
        meta['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='preflight.json'}
        dump(out/'preflight.json',meta)
    print('NATIVE_VALUE_INPUTS',meta['status'],'0 model forwards',flush=True)

if __name__=='__main__': main()
