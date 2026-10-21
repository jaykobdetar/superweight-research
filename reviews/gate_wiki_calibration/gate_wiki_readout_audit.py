"""T18 fixed44-row CPU-FP64 unembedding check; only one checkpoint tensor."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import sys,json,hashlib,time,resource,math
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gate_wiki_storage import read_npz
from gate_wiki_calibration import SNAPSHOT_MODES,READOUT_ROWS
RUN=ROOT/'gpt2_gate_wiki_calibration_run';EXPECTED_WEIGHT='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
def j(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def fp(t):return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest())
def loss(z,t):return z.logsumexp(-1)-z.gather(-1,t[:,None]).flatten()
def kl(a,b):
 lp=a.log_softmax(-1);lq=b.log_softmax(-1);return (lp.exp()*(lp-lq)).sum(-1)
def main():
 started=time.time();torch.set_num_threads(2);m=j(RUN/'manifest.json');assert m['status']=='completed'and j(ROOT/'review/gate_wiki_numerical_audit.json')['status']=='passed_core_pending_readout'
 config=j(ROOT/'inputs.local.json');checkpoint=Path(config['models']['gpt2']['path'])/'model.safetensors';assert sha(checkpoint)==EXPECTED_WEIGHT
 with safe_open(checkpoint,framework='pt',device='cpu')as f:
  names=[k for k in ('wte.weight','transformer.wte.weight')if k in f.keys()];assert len(names)==1;w32=f.get_tensor(names[0])
 assert fp(w32)==m['readout']['weight']and m['readout']['bias']is None and m['readout']['tied_embedding_storage'];w=w32.double();wa=w.abs();del w32
 d=read_npz(RUN/'inputs/gate__wiki_validation__0__512__natural.npz');raw={k.rsplit('__',1)[-1]:v for k,v in d['per_token'].items()};samples=d['sample_readouts'];assert set(samples)==set(SNAPSHOT_MODES)
 actual_p=samples['P']['logits'].double();hp=samples['P']['hidden'][0,list(READOUT_ROWS)].double();cpu_p=hp@w.T;stats=[];zero=0;badzero=0;argmax=0
 unit=2**-24;operations=2*768+2;gamma=operations*unit/(1-operations*unit)
 for mode in SNAPSHOT_MODES:
  s=samples[mode];h=s['hidden'][0,list(READOUT_ROWS)].double();actual=s['logits'].double();cpu=h@w.T;err=cpu-actual;bound=gamma*(h.abs()@wa.T)+operations*2**-126+1e-12
  assert torch.isfinite(cpu).all()and torch.isfinite(actual).all()and (err.abs()<=bound).all(),('prospective dot-product bound failed',mode,float((err.abs()-bound).max()))
  norms=actual.norm(dim=-1);en=err.norm(dim=-1);nonzero=norms>0;zero+=int((~nonzero).sum());badzero+=int(((~nonzero)&(en!=0)).sum());relative=en[nonzero]/norms[nonzero]
  target=s['targets'];cl=loss(cpu,target);al=loss(actual,target);saved=raw[mode]['loss'][list(READOUT_ROWS)].double();pk=kl(cpu_p,cpu);ak=kl(actual_p,actual);savedkl=raw[mode]['kl_p'][list(READOUT_ROWS)].double()
  assert torch.equal(actual.argmax(-1),raw[mode]['pred'][list(READOUT_ROWS)]);mismatches=int((cpu.argmax(-1)!=actual.argmax(-1)).sum());argmax+=mismatches
  delta=err.abs().max(-1).values;assert (cl-al).abs().max()<=2*delta.max()+1e-12
  stats.append({'mode':mode,'rows':4,'max_absolute_logit_error':float(err.abs().max()),'max_RMS_logit_error':float(err.square().mean(-1).sqrt().max()),'max_relative_L2_error':float(relative.max())if len(relative)else None,'max_gamma_bound_ratio':float((err.abs()/bound).max()),'max_NLL_CPU_vs_actual_logit_error':float((cl-al).abs().max()),'max_NLL_CPU_vs_runtime_error':float((cl-saved).abs().max()),'max_NLL_actual_FP64_vs_runtime_error':float((al-saved).abs().max()),'max_KL_either_direction_CPU_actual':float(torch.maximum(kl(actual,cpu),kl(cpu,actual)).max()),'max_KL_P_CPU_vs_runtime_error':float((pk-savedkl).abs().max()),'max_KL_P_actual_FP64_vs_runtime_error':float((ak-savedkl).abs().max()),'CPU_actual_argmax_mismatches':mismatches})
 keys=[k for k in stats[0]if k.startswith('max_')];maxima={k:max(v[k]for v in stats if v[k]is not None)for k in keys}
 result={'status':'passed','script_sha256':sha(Path(__file__)),'manifest_sha256':sha(RUN/'manifest.json'),'weight_file_sha256':EXPECTED_WEIGHT,'loaded_tensor_name':names[0],'sampled_rows':44,'sampled_modes':11,'query_rows':list(READOUT_ROWS),'gamma_bound':{'unit_roundoff':unit,'operations':operations,'gamma':gamma,'formula':'gamma * abs(hidden) @ abs(weight).T + operations*2^-126 +1e-12','scope':'Inherited prospectively used conservative FP32 dot-product bound on only the44 saved actual GPU rows.'},'maxima':maxima,'per_mode':stats,'zero_actual_norm_rows':zero,'zero_actual_norm_nonzero_error_rows':badzero,'relative_L2_definition':'L2(CPU_FP64-saved_GPU_FP32)/L2(saved_GPU_FP32) only on nonzero reference norms; zero norms counted separately','CPU_actual_argmax_mismatches':argmax,'pretrained_model_loaded':False,'model_forwards':0,'single_weight_tensor_loaded':True,'gpu_used':False,'wall_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Only four fixed queries of11 singleton/P states on one natural Wiki window have raw GPU logit rows; no full-query/all-input reconstruction or native-C KL validation is claimed. Runtime saved metrics remain primary.'}
 assert result['peak_rss_kib']<3*1024**2
 return result
if __name__=='__main__':
 result=main();out=ROOT/'review/gate_wiki_readout_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
