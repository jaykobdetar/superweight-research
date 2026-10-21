"""CPU gate calibration, causal reconstruction-control and paired-result audit."""
import ast,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'global_gates_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta
for field,p in [('script_sha256',ROOT/'global_gates.py'),('frozen_mean_cache_sha256',ROOT/'cache_run/calibration.pt'),('evaluation_tokens_sha256',ROOT/'residual_paths_run/tokens.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']and meta['forward_passes']==656
raw,groups,result=load_audit(OUT,require_tables=False);assert result['rows']==640 and result['groups']==80
result['summary_tables_available']=(OUT/'summary.csv').exists()and(OUT/'contrasts.csv').exists()
cal=torch.load(OUT/'calibration.pt',map_location='cpu',weights_only=True);masses=torch.load(OUT/'calibration_source_mass.pt',map_location='cpu',weights_only=True)
old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',map_location='cpu',weights_only=True)
ev=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);dm=json.loads((ROOT/'residual_paths_run/data_manifest.json').read_text())
mass_errors=[];cal_errors=[]
def rotate_half(x):return torch.cat((-x[...,64:],x[...,:64]),dim=-1)
for c in ['wiki','hume']:
 xs=torch.load(OUT/f'calibration_tokens_{c}.pt',map_location='cpu',weights_only=True);assert len(xs)==len(masses[c])==8
 starts=old['starts'][c]['development']
 for x,s in zip(xs,starts):
  assert torch.equal(x,old['corpora'][c][s:s+512])
  for es in dm[c]['evaluation_starts']:assert s+512<=es or s>=es+512
 snap=torch.load(OUT/f'calibration_qk_snapshot_{c}.pt',map_location='cpu',weights_only=True)
 for l in range(2,16):
  a=torch.stack([z[l]for z in masses[c]]);assert a.shape==(8,16,512)and torch.isfinite(a).all()and (a>=0).all()and(a<=1).all()and torch.equal(a[:,:,0],torch.ones(8,16))
  gc=a[:,:,1:].mean((0,2));bc=torch.stack([a[:,:,s:e].mean((0,2))for s,e in zip([1,16,64,128],[16,64,128,512])],1)
  cal_errors.extend([float((gc-cal[c][l]['global']).abs().max()),float((bc-cal[c][l]['binned']).abs().max())])
  z=snap[l];co=z['cos'][:,None];si=z['sin'][:,None]
  q=z['q'].clamp(-8,8).reshape(1,512,16,128).transpose(1,2);k=z['k'].clamp(-8,8).reshape(1,512,16,128).transpose(1,2)
  q=(q*co+rotate_half(q)*si).to(q.dtype).float()[0];k=(k*co+rotate_half(k)*si).to(k.dtype).float()[0]
  logits=(q@k.transpose(1,2))/(128**.5);logits.masked_fill_(torch.ones(512,512,dtype=torch.bool).triu(1)[None],-torch.inf)
  rebuilt=logits.softmax(-1)[:,:,0];mass_errors.append(float((rebuilt-a[0]).abs().max()))
 del snap
assert max(cal_errors)<1e-7 and max(mass_errors)<3e-5
# Isolate the context-recombination hook without importing experiment/model modules.
node=next(n for n in ast.parse((ROOT/'global_gates.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='Gates')
env={'torch':torch,'math':math,'rotate_half':rotate_half};exec(compile(ast.Module(body=[node],type_ignores=[]),'<Gates AST>','exec'),env)
h=env['Gates'].__new__(env['Gates']);v=torch.randn(1,1,2048).half();h.mean={2:{'v':v}};o=torch.randn(1,512,2048).half()
for mode in ['global','binned','bias','zero','one']:
 h.reset(mode,cal['wiki']);actual=h.context(2,(o,))[0]
 if mode in ['global','bias']:a=cal['wiki'][2]['global'][:,None].expand(-1,512)
 elif mode=='binned':
  a=torch.ones(16,512)
  for j,(s,e)in enumerate(zip([1,16,64,128],[16,64,128,512])):a[:,s:e]=cal['wiki'][2]['binned'][:,j,None]
 else:a=torch.full((16,512),float(mode=='one'))
 u=o.reshape(1,512,16,128).float();vv=v.reshape(1,1,16,128).float();aa=a.T[None,:,:,None]
 expected=(u+aa*vv if mode=='bias'else(1-aa)*u+aa*vv).reshape(1,512,2048).half();expected[:,0]=o[:,0]
 assert torch.equal(actual,expected)and torch.equal(actual[:,0],o[:,0])
 if mode=='zero':assert torch.equal(actual,o)
quiet=torch.load(ROOT/'quiet_anchor_run/per_token.pt',map_location='cpu',weights_only=True);quiet={k:{z['window']:z for z in v}for k,v in quiet.items()}
variants=['oracle','remove_source','constant_value_only','wiki_global','wiki_binned','wiki_bias','hume_global','hume_binned','hume_bias'];comparisons={};vs_mean={}
for c,ws in ev.items():
 g='gates_'+c
 for i,x in enumerate(ws):
  for t,qt in [('clean','clean'),('mean_KV','K_mean_V_mean_clean'),('remove_source_clean','remove_source_all_clean')]:
   for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t][i][k],quiet['quiet_'+c+'__'+qt][i][k])
  for name in variants:
   for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+name+'_clean'][i][k],raw[g+'__'+name+'_deleted'][i][k])
 for t in ['mean_KV']+[v+'_clean'for v in variants]:
  comparisons[c+'__'+t]=compare(raw,groups,g,t,'clean');rows=groups[g+'__'+t].values()
  comparisons[c+'__'+t].update(mean_kl=float(np.mean([z['kl_clean_to_condition']for z in rows])),agreement=float(np.mean([z['agreement']for z in rows])))
  vs_mean[c+'__'+t]=compare(raw,groups,g,t,'mean_KV')
 for key,zs in groups.items():
  if key.startswith(g+'__'):assert all(z['score_start']==1 and z['scored_tokens']==510 for z in zs.values())
checks=json.loads((OUT/'reconstruction_checks.json').read_text());assert len(checks)==32
for z in checks:
 g='gates_'+z['corpus'];i=z['window'];a=raw[g+'__oracle_clean'][i];b=raw[g+'__mean_KV'][i]
 assert z['oracle_mean_delta_nll']==float((a['loss']-b['loss']).double().mean())
 assert z['oracle_max_abs_token_loss_delta']==float((a['loss']-b['loss']).abs().max())
 assert z['oracle_argmax_agreement_to_mean']==float(a['pred'].eq(b['pred']).double().mean())
result.update(calibration_tokens_and_separation_passed=True,max_calibration_mean_replay_error=max(cal_errors),max_snapshot_source_mass_cpu_gpu_error=max(mass_errors),synthetic_recombination_modes_and_q0_localization_passed=True,all_state_invariance_exact=True,native_mean_and_removal_anchors_exact=True,reconstruction_checks_replayed=True,comparisons=comparisons,vs_mean_KV=vs_mean,oracle_checks=checks,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['comparisons','vs_mean_KV','oracle_checks']},indent=2))
