"""Recompute saved attention diagnostics with native post-RoPE FP16 rounding.

CPU only; no experiment code imports, model loading, GPU, or source-file changes.
Original run artifacts remain untouched.
"""
import hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'factorial_run'

def rotate_half(x):
    a,b=x.chunk(2,dim=-1)
    return torch.cat((-b,a),dim=-1)

def corrected(cache,head):
    sl=slice(head*128,(head+1)*128)
    cos,sin=cache['cos'][0],cache['sin'][0]
    q=cache['q'][0,:,sl].clamp(-8,8)
    k=cache['k'][0,:,sl].clamp(-8,8)
    v=cache['v'][0,:,sl].clamp(-8,8).float()
    # Matches apply_rotary_pos_emb: FP32 rotations, then original projection dtype.
    q=(q*cos+rotate_half(q)*sin).to(q.dtype).float()
    k=(k*cos+rotate_half(k)*sin).to(k.dtype).float()
    score=(q@k.T)/(128**.5)
    score.masked_fill_(torch.ones_like(score,dtype=torch.bool).triu(1),-torch.inf)
    p=score.softmax(-1)
    expected=p@v
    actual=cache['context'][0,:,sl].float()
    u=score[1:,1:].softmax(-1)@v[1:]
    error=float((expected-actual).norm()/actual.norm().clamp_min(1e-12))
    return {
        'head':head,'reconstruction_relative_error':error,
        'key0_mass_mean':float(p[1:,0].mean()),
        'key0_mass_median':float(p[1:,0].median()),
        'key0_mass_late':float(p[min(128,len(p)-1):,0].mean()),
        'value0_norm':float(v[0].norm()),'key0_norm':float(k[0].norm()),
        'other_values_rms_norm':float(v[1:].square().sum(-1).mean().sqrt()),
        'conditional_mixture_rms_norm':float(u.square().sum(-1).mean().sqrt()),
        'source_residual_std':float(cache['input'][0,0].float().std(unbiased=False)),
        'normalized_source_coordinate':float(cache['norm'][0,0,1764]),
    }

original={(r['corpus'],r['window']):r for r in map(json.loads,(OUT/'diagnostics.jsonl').read_text().splitlines())}
rows=[];hashes={}
for corpus in ['wiki','hume']:
    for i in range(8):
        path=OUT/f'local_{corpus}_{i}.pt'
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        cache=torch.load(path,weights_only=True,map_location='cpu')
        metrics={};changes={}
        for state in ['clean','deleted']:
            for head in [15,0]:
                key=f'{state}_H{head}';c=corrected(cache[state],head);old=original[corpus,i]['metrics'][key]
                metrics[key]=c
                changes[key]={k:c[k]-old[k] for k in c if k!='head'}
        rows.append({'corpus':corpus,'window':i,'metrics':metrics,'corrected_minus_original':changes})
max_err=max(m['reconstruction_relative_error'] for r in rows for m in r['metrics'].values())
max_mass_delta=max(abs(m['key0_mass_mean']) for r in rows for m in r['corrected_minus_original'].values())
result={
    'method':'Preclip cached Q/K/V -> native clamp -> RoPE -> original projection dtype -> FP32 causal softmax and value product',
    'scope':'Natural-window clean/deleted caches only; normalization-intervention full caches were not saved',
    'model_loaded':False,'device':'cpu','original_artifacts_modified':False,
    'remaining_difference':'Explicit FP32 attention differs slightly from native fused SDPA; retain reconstructed-identity controls for causal decomposition patches',
    'maximum_reconstruction_relative_error':max_err,'maximum_mean_key0_mass_change':max_mass_delta,
    'source_cache_sha256':hashes,'windows':rows,
    'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}
destination=Path(__file__).with_name('corrected_attention_diagnostics.json')
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'saved':str(destination),'windows':len(rows),'maximum_reconstruction_relative_error':max_err,'maximum_mean_key0_mass_change':max_mass_delta,'peak_rss_kib':result['peak_rss_kib']},indent=2))
