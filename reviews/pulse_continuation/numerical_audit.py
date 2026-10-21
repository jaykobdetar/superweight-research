"""Audit saved factorial outputs and reconstruct attention on CPU only."""
import hashlib,json,os,resource
from pathlib import Path
from collections import defaultdict
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'factorial_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'factorial.py').read_bytes()).hexdigest()
raw=torch.load(OUT/'per_token.pt',weights_only=True,map_location='cpu')
rows=[json.loads(s) for s in (OUT/'windows.jsonl').read_text().splitlines()]
groups=defaultdict(dict)
for row in rows:
    key=row['group']+'__'+row['trial'];assert row['window'] not in groups[key];groups[key][row['window']]=row
assert len(rows)==1488 and len(groups)==186 and meta['forward_passes']==1424
assert set(raw)==set(groups)
smallest_kl=0
for key,items in raw.items():
    assert len(items)==8
    for z in items:
        r=groups[key][z['window']]
        assert z['token_sha256']==r['token_sha256']
        assert torch.isfinite(z['loss']).all() and torch.isfinite(z['kl']).all()
        assert abs(float(z['loss'].double().mean())-r['nll'])<1e-12
        assert abs(float(z['kl'].double().mean())-r['kl_clean_to_condition'])<1e-12
        assert len(z['loss'])==511
        smallest_kl=min(smallest_kl,float(z['kl'].min()))
raw={k:{r['window']:r for r in v} for k,v in raw.items()}
for key,items in raw.items():
    clean=raw[key.split('__')[0]+'__clean']
    for i,z in items.items():
        row=groups[key][i]
        assert abs(float(clean[i]['loss'].double().mean())-row['baseline_nll'])<1e-12
        assert abs(float(z['pred'].eq(clean[i]['pred']).double().mean())-row['agreement'])<1e-12
result={'rows':len(rows),'conditions':len(groups),'forward_passes':meta['forward_passes'],'raw_window_means_match':True,'smallest_token_kl':smallest_kl}
for corpus in ['wiki','hume']:
    for group in ['natural_'+corpus]+[f'prefix_{corpus}_{p}' for p in [0,16,128,512]]:
        heads=[15,0] if group.startswith('natural') else [15]
        for direction,state in [('damage','clean'),('rescue','deleted')]:
            for head in heads:
                pairs=[('q',None),('qkv','kv')]
                if group.startswith('natural'):pairs += [('qk','k'),('qv','v'),('none',None)]
                for a,b in pairs:
                    left=raw[group+f'__H{head}_{a}_{direction}']
                    right=raw[group+(f'__H{head}_{b}_{direction}' if b else '__'+state)]
                    for i in range(8):
                        for kind in ['loss','pred','kl']:assert torch.equal(left[i][kind],right[i][kind]),(group,a,b,kind)
result['q_has_no_effect_loss_prediction_kl']=True
tokens=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu')
for corpus,windows in tokens.items():
    for i,w in enumerate(windows):
        suffix=w['natural'].clone();suffix[0]=510
        for p in [0,16,128,512]:
            x=torch.cat([w['prefix'][-p:],suffix]) if p else suffix.clone();x[0]=510
            assert torch.equal(x[p+1:],suffix[1:])
            h=hashlib.sha256(x.numpy().tobytes()).hexdigest()
            assert raw[f'prefix_{corpus}_{p}__clean'][i]['token_sha256']==h
result['prefix_same_511_targets_and_input_hashes']=True

def interval(vals):
    x=np.asarray(vals);ix=np.random.default_rng(20261021).integers(0,len(x),(2000,len(x)))
    a,b=np.quantile(x[ix].mean(1),[.025,.975]);return {'mean':float(x.mean()),'low':float(a),'high':float(b)}
def effects(group,trial,reference):
    a=groups[group+'__'+trial];b=groups[group+'__'+reference]
    assert set(a)==set(b)
    assert all(a[i]['token_sha256']==b[i]['token_sha256'] for i in a)
    return interval([a[i]['nll']-b[i]['nll'] for i in sorted(a)])
result['natural']={};result['normalization']={};result['prefix']={}
for corpus in ['wiki','hume']:
    g='natural_'+corpus;result['natural'][corpus]={'deletion':effects(g,'deleted','clean')}
    for direction,state in [('damage','clean'),('rescue','deleted')]:
        for kind in ['k','v','kv']:
            result['natural'][corpus][kind+'_'+direction]=effects(g,'H15_'+kind+'_'+direction,state)
        result['normalization'][corpus+'_'+direction]={m:effects(g,'norm_'+m+'_'+direction,state) for m in ['coordinate','complement','full','donor_scale']}
    for p in [0,16,128,512]:
        g=f'prefix_{corpus}_{p}'
        r={'deletion':effects(g,'deleted','clean'),'H15_kv_damage':effects(g,'H15_kv_damage','clean'),'H15_kv_rescue':effects(g,'H15_kv_rescue','deleted')}
        if p:r['later_same_token_deletion']=effects(g,'delete_later_same_token','clean')
        result['prefix'][corpus+'_'+str(p)]=r

def rotate(x):
    a,b=x.chunk(2,dim=-1);return torch.cat([-b,a],dim=-1)
recons=[];norm_errors=[]
for corpus in ['wiki','hume']:
    for i in range(8):
        saved=torch.load(OUT/f'local_{corpus}_{i}.pt',weights_only=True,map_location='cpu')
        for kind in ['q','k','v']:assert torch.equal(saved['clean'][kind][:,1:],saved['deleted'][kind][:,1:])
        a,b=saved['clean']['input'].clone(),saved['deleted']['input'].clone();a[:,:,1764]=b[:,:,1764]
        assert torch.equal(a,b)
        for state in ['clean','deleted']:
            c=saved[state];co=c['cos'][0];si=c['sin'][0]
            norm=torch.nn.functional.layer_norm(c['input'].float(),(2048,),eps=1e-5).to(c['norm'].dtype)
            norm_errors.append(float((norm.float()-c['norm'].float()).abs().max()))
            assert torch.allclose(norm,c['norm'],rtol=.002,atol=2e-5)
            for head in [0,15]:
                sl=slice(head*128,(head+1)*128)
                q=c['q'][0,:,sl].clamp(-8,8);k=c['k'][0,:,sl].clamp(-8,8);v=c['v'][0,:,sl].clamp(-8,8).float()
                qr=q*co+rotate(q)*si;kr=k*co+rotate(k)*si;actual=c['context'][0,:,sl].float()
                rs={}
                for rounding in ['original_diagnostic','native_fp16_rope']:
                    qq=qr.to(q.dtype).float() if rounding=='native_fp16_rope' else qr.float()
                    kk=kr.to(k.dtype).float() if rounding=='native_fp16_rope' else kr.float()
                    score=qq@kk.T/(128**.5);score.masked_fill_(torch.ones_like(score,dtype=torch.bool).triu(1),-torch.inf)
                    prob=score.softmax(-1);context=prob@v;conditional=score[1:,1:].softmax(-1)@v[1:]
                    decomposed=prob[1:,:1]*v[0]+(1-prob[1:,:1])*conditional
                    assert torch.allclose(decomposed,context[1:],rtol=2e-4,atol=3e-5)
                    rs[rounding]={'relative_error':float((context-actual).norm()/actual.norm()),'mean_key0_mass':float(prob[1:,0].mean()),'value0_norm':float(v[0].norm())}
                    if rounding=='original_diagnostic':
                        d=saved['decomposition'][state+'_H'+str(head)]
                        assert torch.equal(prob[:,0],d['attention0']) and torch.equal(conditional,d['conditional_mixture'])
                recons.append({'corpus':corpus,'window':i,'state':state,'head':head,**rs})
result['diagnostic_reconstruction']=recons
result['max_original_reconstruction_error']=max(r['original_diagnostic']['relative_error'] for r in recons)
result['max_native_rope_reconstruction_error']=max(r['native_fp16_rope']['relative_error'] for r in recons)
result['max_key0_mass_rounding_change']=max(abs(r['native_fp16_rope']['mean_key0_mass']-r['original_diagnostic']['mean_key0_mass']) for r in recons)
result['source_only_coordinate1764_and_later_qkv_exact']=True
result['max_cpu_vs_gpu_normalization_abs_error']=max(norm_errors)
result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='diagnostic_reconstruction'},indent=2))
