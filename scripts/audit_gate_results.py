"""NumPy-only audit of exported gate metrics, paired intervals and failures."""
import argparse
from collections import defaultdict
import hashlib,json,math,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(x):return float(np.asarray(x,dtype=np.float64).mean())
def close(a,b):
    assert np.allclose(a,b,rtol=0,atol=1e-12),(a,b)
def bootstrap(x):
    x=np.array(x,dtype=np.float64);assert len(x)in(3,4)
    rng=np.random.default_rng(20261021)
    return np.quantile(x[rng.integers(0,len(x),(2000,len(x)))].mean(1),[.025,.975])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.output:a.output.write_text(json.dumps({'status':'checking','note':'Only a final passed state confirms this audit.'},indent=2)+'\n')
    manifest=read(ROOT/'manifests/gate_numeric_export.json')
    expected_runs={'gpt2_gate_subspace_run','native_prefix_diagnostic_run',
        'gpt2_gate_calibration_rebuild','gpt2_gate_subspace_v2_run','gpt2_gate_fixed512_run'}
    assert len(manifest)==5 and {m['run']for m in manifest}==expected_runs
    result=[];full=None
    for m in manifest:
        p=ROOT/'results/pulse_transfer'/m['run']
        for f,h in m['exported_files'].items():assert sha(p/f)==h,(m['run'],f)
        assert read(p/'manifest.json')['status']==m['status']
        if not m['records']:continue
        rows=[json.loads(x)for x in(p/'windows.jsonl').read_text().splitlines()]
        idx=read(p/'per_token_index.json');di={x['key']:x for x in idx};rr={x['key']:x for x in rows}
        assert len(rows)==len(rr)==len(di)==len(idx)==m['records']
        assert set(di)==set(rr) and sorted(x['index']for x in idx)==list(range(len(idx)))
        values={};n=0;expected=set()
        with np.load(p/'per_token_numeric.npz',allow_pickle=False)as z:
            for e in idx:
                r=rr[e['key']];i=e['index'];assert e['score_index']==r['score_index']
                assert e['token_sha256']==r['token_sha256']
                fields={}
                for suffix,field in(('loss','nll'),('kl','kl'),('kl_p','kl_p'),('matches_native','agreement'),('matches_p','agreement_p')):
                    name=f'{i}_{suffix}'
                    assert name in z,('Missing required metric',m['run'],name)
                    expected.add(name);v=z[name];assert v.shape==(r['scored_tokens'],)and np.isfinite(v).all()
                    if suffix.startswith('matches_'):assert v.dtype==np.bool_
                    else:assert np.issubdtype(v.dtype,np.floating)
                    close(mean(v),r[field]);fields[field]=v.astype(np.float64);n+=1
                values[e['key']]=fields
            assert set(z.files)==expected
        for e in idx:
            r=rr[e['key']]
            for ref,field in((e['baseline_key'],'baseline_nll'),(e['p_baseline_key'],'p_baseline_nll')):
                if ref in values:close(mean(values[ref]['nll']),r[field]);n+=1
        result.append({'run':m['run'],'status':m['status'],'records':len(rows),'scalar_reductions':n})
        if m['run']=='gpt2_gate_fixed512_run':full=(p,rows,values)
    assert full is not None
    p,rows,values=full;assert read(p/'audit.json')['status']=='passed'
    conditions=read(p/'conditions.json');assert len(conditions)==21
    group=defaultdict(list)
    for r in rows:
        if r['trial']in conditions:group[r['corpus'],r['identity'],r['trial']].append(r)
    checked=0
    for excluded,prefix in((False,''),(True,'unexposed_')):
        for segment,file in((None,'summary.json'),('segments','segment_summary.json')):
            summaries=read(p/(prefix+file))
            assert len(summaries)==(168 if segment is None else 336)
            for s in summaries:
                gs=[r for r in group[s['corpus'],s['identity'],s['trial']]if not(excluded and r['corpus']=='wiki_validation'and r['window']==0)]
                sl=slice(None)if segment is None else(slice(0,126)if s['segment']=='early_2_127'else slice(126,None))
                for ref,keypref in(('C',''),('P','p_')):
                    deltas=[];kls=[];agreements=[]
                    for r in gs:
                        key=r['key'];b=key.rsplit('__',1)[0]+'__'+ref
                        v=values[key];deltas.append(mean(v['nll'][sl])-mean(values[b]['nll'][sl]))
                        kls.append(mean(v['kl'if ref=='C'else'kl_p'][sl]));agreements.append(mean(v['agreement'if ref=='C'else'agreement_p'][sl]))
                    assert s[keypref+'windows']==len(gs)
                    close(s[keypref+'mean_delta_nll'],mean(deltas));close(s[keypref+'ppl_ratio'],math.exp(mean(deltas)))
                    close(s[keypref+'ppl_ratio_ci95'],np.exp(bootstrap(deltas)))
                    close(s[keypref+'mean_kl'],mean(kls));close(s[keypref+'mean_agreement'],mean(agreements))
                    close(s[keypref+'per_window_delta_nll'],deltas);checked+=1
        contrasts=read(p/(prefix+'paired_contrasts.json'));paired=read(p/(prefix+'paired_summary.json'));g=defaultdict(list)
        assert len(contrasts)==(960 if excluded else 1024)and len(paired)==256
        for c in contrasts:
            left,right=c['trial'].split('_minus_');stem=f"gate__{c['corpus']}__{c['window']}__512__{c['identity']}__"
            delta=mean(values[stem+left]['nll'])-mean(values[stem+right]['nll'])
            close(delta,c['delta_nll']);g[c['corpus'],c['identity'],c['trial']].append(delta)
        for s in paired:
            ds=g[s['corpus'],s['identity'],s['trial']];assert s['n']==len(ds)
            close(s['mean_delta_nll'],mean(ds));close(s['ci95_delta_nll'],bootstrap(ds));close(s['per_window_delta_nll'],ds)
    output={'status':'passed','runs':result,'dual_reference_summary_checks':checked,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'Reaggregate exported loss/KL and agreement masks, full/early/later views and paired bootstrap intervals. Does not reconstruct KL from logits or verify hook arithmetic, private tokens or absent activation arrays. Failed runs remain failed.'}
    if a.output:a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
