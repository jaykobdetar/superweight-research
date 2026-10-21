"""Tokenizer-only localization eligibility and fixed orthogonal directions.

No model class, model weights, logits, loss or activations are loaded. Existing
calibration maps and saved input-token files are read as provenance data.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
CORPORA=('wiki_validation','dubois','hobbes','optimizers')
TRANSFER=('gpt2_run','gpt2_groups_run','gpt2_writes_run','gpt2_layer_writes_run',
    'gpt2_regrowth_kv_run','gpt2_value_geometry_run','gpt2_query_source_run',
    'gpt2_query_mixture_run','gpt2_gate_subspace_run','gpt2_gate_subspace_v2_run',
    'gpt2_gate_fixed512_run','llama_run','llama_source_kv_run','llama_replacements_run')
MAP_SHA='ed62f089168abeec63434e5f6dfa83ff8e7e139bc362e952456ea34859b5a098'
PREFERRED_NUMERATORS=(125,375,625,875)
FROZEN_NUMERATORS={'wiki_validation':(122,377,623,875),'dubois':(69,368,619,875),
    'hobbes':(123,373,623,875),'optimizers':(881,967,991)}
FAILED_PREFLIGHT_SHA='2fd5f040e872949287fbe566a9f348e6684d5984c3d6ada888be163189b351a4'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def th(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def strip_book(s):
    a=s.find('*** START OF');b=s.rfind('*** END OF')
    return s[s.find('\n',a)+1:b if b>=0 else None]if a>=0 else s

def overlap(a,b,lo,hi):return a<hi and b>lo

def directions(maps):
    result={}
    for l in range(4,12):
        u0=maps[l]['basis_r'][:,0].double();norm=float(u0.norm())
        assert torch.isfinite(u0).all()and abs(norm-1)<=1e-10
        u=u0/u0.norm();chosen=None
        for j in range(12):
            q=maps[l]['basis_qr'][:,j].double();assert torch.isfinite(q).all()
            w=q-u*torch.dot(u,q);wn=float(w.norm())
            assert torch.isfinite(w).all()
            if wn>=1e-6:chosen=(j,w,wn);break
        assert chosen is not None
        j,w,wn=chosen;v=w/w.norm();pivot=int(v.abs().argmax());sign=-1 if float(v[pivot])<0 else 1;v=v*sign
        assert abs(float(torch.dot(u,v)))<=1e-12 and abs(float(v.norm())-1)<=1e-12
        uf,vf=u.float(),v.float()
        assert abs(float(uf.norm()-vf.norm()))<=1e-6 and abs(float(torch.dot(uf,vf)))<=1e-6
        result[l]={'u':u,'v':v,'qr_column':j,'residual_norm':wn,'sign_pivot':pivot,'sign_multiplier':sign,
            'u_shape':list(u.shape),'v_shape':list(v.shape),'dtype':str(u.dtype),'u_sha256':th(u),'v_sha256':th(v),
            'u_norm_error':abs(float(u.norm())-1),'v_norm_error':abs(float(v.norm())-1),
            'orthogonality_error':abs(float(torch.dot(u,v))),
            'fp32_norm_difference':abs(float(uf.norm()-vf.norm())),
            'fp32_orthogonality_error':abs(float(torch.dot(uf,vf)))}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out
    assert not out.exists()or not any(out.iterdir()),'Use a new output directory'
    out.mkdir(parents=True,exist_ok=True);started=time.time();torch.set_num_threads(1)
    protocol=ROOT/'GPT2_GATE_LOCALIZATION_PROPOSAL.md'
    meta={'status':'checking','model_forwards':0,'model_loaded':False,'gpu_used':False,
        'input_policy_version':2,'passages_per_corpus':{c:len(v)for c,v in FROZEN_NUMERATORS.items()},
        'script_sha256':sha(__file__),'proposal_sha256':sha(protocol),
        'candidate_fraction_denominator':1000,'candidate_numerators':list(range(1001)),
        'preferred_numerators':list(PREFERRED_NUMERATORS),'frozen_numerators':FROZEN_NUMERATORS,'eligibility_before_outputs':True}
    dump(out/'preflight.json',meta)
    input_hashes={};exclusions={c:[] for c in CORPORA};history=[];records=[];rejections={};items=[]
    def remember(p):
        p=Path(p);input_hashes[str(p.relative_to(ROOT.parent))]=sha(p);return read(p)
    def exclude(c,a,b,family,origin,**extra):
        assert 0<=a<b<=len(texts[c]),(family,c,a,b)
        exclusions[c].append({'char_start':a,'char_end':b,'family':family,'origin':origin,**extra})
    try:
        failed=ROOT/'gpt2_gate_localization_inputs'
        assert sha(failed/'preflight.json')==FAILED_PREFLIGHT_SHA
        fm=remember(failed/'preflight.json');assert fm['status']=='failed'and fm['model_forwards']==0
        for name in('failure_selection.json','exclusion_inventory.json'):
            assert sha(failed/name)==fm['output_sha256'][name]
        previous=remember(failed/'failure_selection.json')['selected'];assert len(previous)==15
        capacity=remember(ROOT/'review/gate_localization_eligibility_audit.json')
        assert capacity['failed_preflight_sha256']==FAILED_PREFLIGHT_SHA
        assert capacity['maximum_nonoverlapping_grid_windows']==3 and capacity['best_four_by_quartile_distance']is None
        assert tuple(r['numerator']for r in capacity['maximum_packing'])==FROZEN_NUMERATORS['optimizers']
        cfg=read(args.config);old=remember(ROOT/'gpt2_run/data_manifest.json')
        model_manifest=remember(ROOT.parent/'superweight-causal-circuits/manifests/transfer_models.json')['gpt2']
        gcfg=cfg['models']['gpt2'];assert gcfg['revision']==model_manifest['revision']and gcfg['model_id']==model_manifest['model_id']
        gtpath=Path(gcfg['path']);tokenizer_hashes={}
        for name in('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
            expected=model_manifest['checkpoint_files'][name];p=gtpath/name
            assert p.stat().st_size==expected['bytes']and sha(p)==expected['sha256'];tokenizer_hashes[name]=sha(p)
        tok=AutoTokenizer.from_pretrained(gtpath,local_files_only=True);assert tok.is_fast
        source_texts={};raw_texts={};source_ids={}
        for c,sp in cfg['sources'].items():
            p=Path(sp);d=old['sources'][c];assert sha(p)==d['sha256']
            s='\n\n'.join(Dataset.from_file(str(p))['text'])if c.startswith('wiki_')else p.read_text()
            raw_texts[c]=s;ss=strip_book(s)if c in('hume','dubois','hobbes')else s
            assert textsha(ss)==d['processed_text_sha256'];source_texts[c]=ss
            source_ids[c]={'file_sha256':sha(p),'processed_text_sha256':textsha(ss),'characters':len(ss)}
        source_texts['optimizers']='\n\n'.join(source_texts[k]for k in('sgd','adam','adamw'))
        raw_texts['optimizers']=source_texts['optimizers'];texts={c:source_texts[c]for c in CORPORA}
        # Each tokenizer call is deterministic and contains no model inference.
        enc_cache={}
        def window(c,a):
            key=(c,a)
            if key not in enc_cache:
                ss=texts[c][a:a+8192];assert len(ss)==8192
                e=tok(ss,add_special_tokens=False,return_offsets_mapping=True)
                assert len(e['input_ids'])>=512
                x=torch.tensor(e['input_ids'][:512],dtype=torch.long)
                lo=min(t[0]for t in e['offset_mapping'][:512]);hi=max(t[1]for t in e['offset_mapping'][:512])
                # Conservatively include any leading ignored characters.
                enc_cache[key]=(x,a+hi,textsha(ss),lo)
            return enc_cache[key]
        llamabase=remember(ROOT/'llama_run/data_manifest.json')
        llprior=remember(ROOT/'llama_source_kv_run/data_manifest.json')
        assert llprior['old_manifest_sha256']==sha(ROOT/'llama_run/data_manifest.json')
        for run in TRANSFER:
            p=ROOT/run/'data_manifest.json';d=remember(p)
            for rec in d['passages']:
                c=rec['corpus']
                if c not in CORPORA or rec.get('split','evaluation')!='evaluation':continue
                a=rec['char_start'];assert textsha(texts[c][a:a+8192])==rec['passage_sha256']
                if run.startswith('gpt2'):
                    x,b,_,_=window(c,a);assert th(x)==rec['window_token_sha256']
                    if 'used_char_end'in rec:assert rec['used_char_end']==b
                    verification='matching GPT2 tokenizer offsets and saved window hash'
                elif 'used_char_end'in rec:
                    b=rec['used_char_end'];assert a<b<=a+8192
                    verification='saved Llama actual used_char_end; hashed manifest and extraction verified'
                else:
                    matching=[b for aa,b in llprior['old_used_spans'][c]if aa==a];assert len(matching)==1
                    b=matching[0];assert rec in llamabase['passages']
                    verification='parent-linked saved Llama source-KV old_used_spans; not approximated with GPT2'
                exclude(c,a,b,'transfer_evaluation',str(p.relative_to(ROOT.parent)),verification=verification)
        # Full extraction buffers, including historical Llama and original development.
        buffers=[]
        for run in('gpt2_run','llama_run'):
            p=ROOT/run/'data_manifest.json';d=remember(p)
            for rec in d['passages']:
                if rec['split']=='development':buffers.append((p,rec))
        for run in('llama_replacements_run','gpt2_gate_subspace_run','gpt2_gate_subspace_v2_run','gpt2_gate_fixed512_run'):
            p=ROOT/run/'calibration_inputs.json';d=remember(p)
            buffers.extend((p,r)for r in d['passages'])
        for p,rec in buffers:
            c=rec['corpus'];a=rec['char_start'];b=a+8192
            assert textsha(source_texts[c][a:b])==rec['passage_sha256']
            history.append({'corpus':c,'char_start':a,'char_end':b,'origin':str(p.relative_to(ROOT.parent)),
                'source_identity':source_ids[c],'excluded_from_candidates':c in CORPORA})
            if c in CORPORA:exclude(c,a,b,'full_calibration_development_buffer',str(p.relative_to(ROOT.parent)))
        # Exact OLMo window validation and coordinate mapping on shared sources.
        om=remember(ROOT.parent/'pulse_causal_tranche/run/manifest.json');op=Path(om['model'])
        assert op.name==om['revision'];ot=AutoTokenizer.from_pretrained(op,local_files_only=True);assert ot.is_fast
        olmo_files={n:sha(op/n)for n in('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json')}
        cont=ROOT.parent/'pulse_continuation'
        manifests={n:remember(cont/n/'data_manifest.json')for n in('confirmation_run','later_uses_run','newline_heads_run')}
        saved={n:load(cont/n/'tokens.pt')for n in manifests}
        for n in saved:input_hashes[str((cont/n/'tokens.pt').relative_to(ROOT.parent))]=sha(cont/n/'tokens.pt')
        mapping=[]
        for c in CORPORA:
            d=manifests['confirmation_run'][c]
            for source in d['sources']:assert sha(Path(source['path']))==source['sha256']
            ps=[Path(source['path'])for source in d['sources']]
            original_text='\n\n'.join(Dataset.from_file(str(ps[0]))['text'])if c=='wiki_validation'else'\n\n'.join(p.read_text()for p in ps)
            raw=raw_texts[c];assert raw==original_text
            shift=raw.find(texts[c]);assert shift>=0 and raw.find(texts[c],shift+1)<0
            assert raw[shift:shift+len(texts[c])]==texts[c]
            e=ot(raw,add_special_tokens=False,return_offsets_mapping=True)
            ids=torch.tensor(e['input_ids'],dtype=torch.long);offsets=e['offset_mapping'];assert len(ids)==d['token_count']
            mapping.append({'corpus':c,'unstripped_text_sha256':textsha(raw),'processed_text_sha256':textsha(texts[c]),
                            'character_shift':shift,'token_count':len(ids)})
            for n,dm in manifests.items():
                assert dm[c]['token_count']==len(ids)
                for split in(('calibration','evaluation')if n=='confirmation_run'else('evaluation',)):
                    starts=dm[c][split+'_starts']
                    xs=saved[n][c][split]if n=='confirmation_run'else saved[n]['evaluation'][c]if n=='newline_heads_run'else saved[n][c]
                    assert len(xs)==len(starts)
                    for i,(s,x)in enumerate(zip(starts,xs)):
                        assert x.shape==(512,)and torch.equal(ids[s:s+512],x),(n,c,split,i)
                        lo=min(t[0]for t in offsets[s:s+512]);hi=max(t[1]for t in offsets[s:s+512])
                        a=max(0,lo-shift);b=min(len(texts[c]),hi-shift)
                        if a<b:exclude(c,a,b,'olmo_'+split,n,token_start=s,token_length=512,
                            original_char_start=lo,original_char_end=hi,token_sha256=th(x),verification='exact original tokenizer IDs and offsets')
            del e,ids,offsets
        del saved,ot
        # Publish complete exclusion evidence before candidate selection.
        inventory={'input_hashes':input_hashes,'evaluation_and_olmo_calibration':exclusions,
            'full_transfer_calibration_development_buffers':history,'olmo_coordinate_mapping':mapping,
            'other_olmo_sources':'Wiki-test/Hume, Plato and modeling_olmo/cache_utils Python texts are distinct from these four candidate sources; derivative runs reuse confirmation/later/newline inputs.',
            'coverage':'All verified source windows on the four candidate corpora; prepared failed-GPT2 panels are excluded conservatively as well.'}
        dump(out/'exclusion_inventory.json',inventory)
        for c in CORPORA:
            fresh=[];rejections[c]=[]
            for w,j in enumerate(FROZEN_NUMERATORS[c]):
                target=PREFERRED_NUMERATORS[w];a=j*(len(texts[c])-8192)//1000
                x,b,ph,_=window(c,a)
                assert not any(overlap(a,b,r['char_start'],r['char_end'])for r in exclusions[c])
                assert not any(overlap(a,b,*r)for r in fresh)
                fresh.append((a,b));items.append({'corpus':c,'window':w,'tokens':x})
                record={'corpus':c,'window':w,'fraction_numerator':j,'fraction_denominator':1000,
                    'preferred_numerator':target,'char_start':a,'used_char_end':b,'extraction_char_end':a+8192,
                    'passage_sha256':ph,'window_token_sha256':th(x),'first_token_id':int(x[0])}
                assert record==next(r for r in previous if r['corpus']==c and r['window']==w)
                records.append(record)
        assert len(items)==len(records)==15
        mapfile=ROOT/'gpt2_gate_fixed512_run/calibration_maps.pt';assert sha(mapfile)==MAP_SHA
        ds=directions(load(mapfile));torch.save(ds,out/'directions.pt')
        dump(out/'directions.json',{l:{k:(v.tolist()if isinstance(v,torch.Tensor)else v)for k,v in d.items()}for l,d in ds.items()})
        torch.save(items,out/'tokens.pt')
        dm={'sources':source_ids,'passages':records,'literals':old['literals'],
            'selection_rule':{k:meta[k]for k in('input_policy_version','candidate_fraction_denominator','candidate_numerators','preferred_numerators','frozen_numerators')},
            'excluded_inventory_sha256':sha(out/'exclusion_inventory.json'),'rejections':rejections,
            'tokenizer_sha256':tokenizer_hashes,'olmo_tokenizer_sha256':olmo_files,'original_maps_sha256':MAP_SHA}
        dump(out/'data_manifest.json',dm)
        cal=load(ROOT/'gpt2_gate_fixed512_run/calibration_tokens.pt');starts={int(i['tokens'][0])for i in cal}
        dump(out/'start_identity_summary.json',{'calibration_first_token_ids':sorted(starts),
            'evaluation_natural_first_token_ids':[int(i['tokens'][0])for i in items],
            'natural_seen_in_calibration':[int(i['tokens'][0])in starts for i in items],
            'bos_id':old['literals']['start_special']['id'],'bos_seen_in_calibration':old['literals']['start_special']['id']in starts})
        meta.update(status='passed',passages=15,identities=2,model_inputs=30,
            all_first12_prose_records_match_failed_preflight=True,all15_selected_records_match_failed_preflight=True,
            capacity_claim='Maximum3 optimizer windows on the frozen1001-fraction grid, not every possible character start.',
            revision_reason='Input availability before any new model output; same exclusions, no efficacy selection.')
    except BaseException as error:
        meta.update(status='failed',error_type=type(error).__name__,error=str(error))
        dump(out/'failure_selection.json',{'selected':records,'rejections':rejections,'exclusions':exclusions,'input_hashes':input_hashes})
        raise
    finally:
        meta.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            completed_utc=time.strftime('%FT%TZ',time.gmtime())if meta['status']=='passed'else None)
        meta['output_sha256']={p.name:sha(p)for p in sorted(out.iterdir())if p.is_file()and p.name!='preflight.json'}
        dump(out/'preflight.json',meta)
    print('LOCALIZATION_INPUT_PREFLIGHT',meta['status'],len(records),'zero model forwards')

if __name__=='__main__':main()
