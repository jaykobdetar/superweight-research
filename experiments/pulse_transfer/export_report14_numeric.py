"""Export only allowlisted T18 numerical values, definitions and audit records.

No token IDs, predictions, text, q/k/v, hidden states, logits or local config.
The complete original raw inventory is retained by hash, not copied payload.
"""
from pathlib import Path
import json,shutil,math
import numpy as np
import torch
from gate_wiki_storage import read_npz
from run_transfer import sha,dump
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'gpt2_gate_wiki_calibration_run'
REPO=ROOT.parent/'superweight-causal-circuits';OUT=REPO/'results/gate_wiki_calibration'
def j(p):return json.loads(p.read_text())
def fingerprint_only(x):
    if not isinstance(x,dict):return False
    if set(x)=={'sha256','shape','dtype'}:
        return isinstance(x['sha256'],str)and len(x['sha256'])==64 and all(c in '0123456789abcdef'for c in x['sha256'])and isinstance(x['shape'],list)and all(type(v)is int and v>0 for v in x['shape'])and x['dtype']in('torch.float32','torch.float64','torch.int64','torch.bool')
    return bool(x)and all(isinstance(k,int)or str(k).isdigit()for k in x)and all(fingerprint_only(v)for v in x.values())
def safe_json(x):
    if isinstance(x,dict):
        for k,v in x.items():
            assert k not in ('path','id','ids','tokens','pred','targets','text'),k
            if k in ('hidden','logits','q','k','v'):assert fingerprint_only(v),k
            assert 'first_token'not in str(k) and 'token_ids'not in str(k),k
            safe_json(v)
    elif isinstance(x,list):
        for v in x:safe_json(v)
    elif isinstance(x,str):assert '/home/'not in x and '/tmp/'not in x,x
    else:assert x is None or type(x)in(bool,int,float)and (type(x)is not float or math.isfinite(x))
def put_json(p,x):safe_json(x);dump(p,x)
def main():
    m=j(RUN/'manifest.json');assert m['status']=='completed'
    assert j(RUN/'saved_state_audit.json')['status']==j(ROOT/'review/gate_wiki_numerical_audit.json')['status']=='passed'
    OUT.mkdir(parents=True,exist_ok=True);shards=OUT/'inputs';shards.mkdir(exist_ok=True)
    for n in ('manifest.json','numeric_inventory.json','saved_state_audit.json','wiki_factorial_analysis.json'):
        put_json(OUT/n,j(RUN/n))
    index=[];arrays_count=0
    for p in sorted((RUN/'inputs').glob('*.npz')):
        d=read_npz(p);raw={k.rsplit('__',1)[-1]:v for k,v in d['per_token'].items()};a={}
        for mode,z in raw.items():
            for field in ('loss','kl','kl_p','agreement_p'):
                assert z[field].shape==(511,) and torch.isfinite(z[field]).all()
                a[mode+'__'+field]=z[field].numpy()
            a[mode+'__agreement_c']=z['pred'].eq(raw['C']['pred']).numpy()
        for mode,z in d['suffix_metrics'].items():
            for field in ('loss','kl','kl_p','agreement_p'):
                assert z[field].shape==((127,)if field=='loss'else(128,))
                a['suffix__'+mode+'__'+field]=z[field].numpy()
            a['suffix__'+mode+'__agreement_c']=z['pred'].eq(raw['C']['pred'][:128]).numpy()
        for mode,layers in d['norms'].items():
            for layer,vals in layers.items():
                for field,t in vals.items():
                    assert t.shape==(511,)and t.dtype==(torch.float32 if field=='s'else torch.float64) and torch.isfinite(t).all()
                    a['norm__'+mode+'__'+str(layer)+'__'+field]=t.numpy()
        assert all(v.dtype in(np.dtype('float32'),np.dtype('float64'),np.dtype('bool'))for v in a.values())
        np.savez_compressed(shards/p.name,**a);arrays_count+=len(a)
        # Runtime evidence contains identities/hashes, never raw donor arrays.
        evidence={k:d[k]for k in ('metadata','checks','operations','suffix_checks','suffix_operations')}
        put_json(shards/(p.stem+'.json'),evidence)
        index.append({**d['metadata'],'file':p.name,'original_sha256':sha(p),
            'main_modes':list(raw),'suffix_modes':list(d['suffix_metrics']),
            'token_sha256':raw['P']['token_sha256'],'score_index':1,
            'raw_array_fields':sorted(a),'norm_layers':{k:list(v)for k,v in d['norms'].items()}})
    assert len(index)==32;put_json(OUT/'index.json',index)
    # Fit inputs and fitted maps contain no tokens, text or hidden/readout states.
    fit={}
    for p in sorted((RUN/'calibration').glob('*.npz')):
        for layer,vals in read_npz(p)['raw'].items():
            assert set(vals)=={'b','mass'}
            for field,t in vals.items():fit[p.stem+'__'+str(layer)+'__'+field]=t.numpy()
    for label,layers in read_npz(RUN/'calibration_maps.npz').items():
        for layer,vals in layers.items():
            assert set(vals)=={'alpha','mean_b','mean_mass','basis_r','basis_b','basis_qr'}
            for field,t in vals.items():fit['map__'+label+'__'+str(layer)+'__'+field]=t.numpy()
    dd=read_npz(RUN/'calibration_directions.npz');meta={}
    for label,layers in dd.items():
        meta[label]={}
        for layer,vals in layers.items():
            meta[label][layer]={k:v for k,v in vals.items()if not isinstance(v,torch.Tensor)}
            for field,t in vals.items():
                if isinstance(t,torch.Tensor):
                    assert field in('u','v');fit['direction__'+label+'__'+str(layer)+'__'+field]=t.numpy()
    assert all(np.isfinite(v).all()for v in fit.values())
    np.savez_compressed(OUT/'calibration_numeric.npz',**fit)
    put_json(OUT/'direction_metadata.json',meta)
    for n in ('calibration_spectra','calibration_geometry'):put_json(OUT/(n+'.json'),read_npz(RUN/(n+'.npz')))
    inventory={p.relative_to(RUN).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(RUN.rglob('*'))if p.is_file()}
    dump(ROOT/'wiki_calibration_raw_inventory.json',{'files':inventory,'total_bytes':sum(v['bytes']for v in inventory.values())})
    files={p.relative_to(OUT).as_posix():sha(p)for p in sorted(OUT.rglob('*'))if p.is_file()}
    put_json(REPO/'manifests/wiki_calibration_numeric_export.json',{'schema':1,'input_shards':32,'main_records':776,'suffix_records':768,'exported_array_count':arrays_count,
        'exported_files':files,'raw_inventory':inventory,
        'scope':'Per-target losses/KLs/boolean agreements; all gate-norm arrays; fresh calibration b/mass and fitted maps/directions; runtime identity/check ledgers. No token IDs, predictions, source text, local config, checkpoint weights, full logits, hidden states or q/k/v donor arrays.'})
    print('EXPORTED_WIKI_NUMERIC',len(files),arrays_count,sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file()))
if __name__=='__main__':main()
