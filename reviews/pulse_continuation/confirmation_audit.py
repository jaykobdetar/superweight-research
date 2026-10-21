"""CPU-only fresh-panel calibration, identity and metric audit."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,compare,interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'confirmation_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']and meta['forward_passes']==736
for field,p in [('script_sha256',ROOT/'confirmation.py'),('design_sha256',ROOT/'CONFIRMATION.md'),('mean_cache_sha256',ROOT/'cache_run/calibration.pt'),('old_gate_calibration_sha256',ROOT/'global_gates_run/calibration.pt'),('subspace_fit_sha256',ROOT/'gate_subspaces_run/fitted_subspaces.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,require_tables=False);assert result['rows']==704 and result['groups']==88
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);dm=json.loads((OUT/'data_manifest.json').read_text())
cal=torch.load(OUT/'fresh_gate_calibration.pt',map_location='cpu',weights_only=True);mass=torch.load(OUT/'calibration_source_mass.pt',map_location='cpu',weights_only=True)
checks=json.loads((OUT/'reconstruction_checks.json').read_text());assert len(checks)==64
ratios={};controls={};effects={};id_effects={};fiterr=[];seen={};duplicates=[];sensitivity={}
for c,data in sets.items():
 for part,ws in data.items():
  for i,x in enumerate(ws):
   h=hashlib.sha256(x.numpy().tobytes()).hexdigest()
   if h in seen:duplicates.append({'first':seen[h],'second':[c,part,i],'sha256':h})
   seen[h]=[c,part,i]
for c,data in sets.items():
 d=dm[c]
 for source in d['sources']:assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
 code=c=='optimizers';expectedcal=list(range(0,4096,512))if code else list(range(8192,16384,1024));expectedeval=np.linspace(8192 if code else 32768,d['token_count']-512,8,dtype=int).tolist()
 assert d['calibration_starts']==expectedcal and d['evaluation_starts']==expectedeval and not d['special_tokens']
 assert all(abs(a-b)>=512 for i,a in enumerate(expectedcal+expectedeval)for b in (expectedcal+expectedeval)[i+1:])
 assert len(data['calibration'])==len(data['evaluation'])==len(mass[c])==8
 for l in range(2,16):
  z=torch.stack([m[l]for m in mass[c]]);assert z.shape==(8,16,512)and torch.isfinite(z).all()and(z>=0).all()and(z<=1).all()
  m=z[:,:,1:].mean(2).mean(0);fiterr.append(float((m-cal[c][l]['global']).abs().max()));assert torch.equal(m,cal[c][l]['global'])
 for identity in ['natural','The']:
  g='confirm_'+c+'_'+identity
  for i,xx in enumerate(data['evaluation']):
   assert xx.shape==(512,);x=xx.clone()
   if identity=='The':x[0]=510
   digest=hashlib.sha256(x.numpy().tobytes()).hexdigest()
   assert all(rows[i]['token_sha256']==digest and rows[i]['score_start']==1 and rows[i]['scored_tokens']==510 for key,rows in groups.items()if key.startswith(g+'__'))
  controls[c+'__'+identity]=compare(raw,groups,g,'oracle','mean_KV')
  for t in ['clean','deleted','mean_KV','meanK_zeroV','remove_source','oracle','global_wiki','global_hume','global_fresh','pca4','pca12']:
   delta=effect(groups,g,t,'clean');ratios[c+'__'+identity+'__'+t]={'ratio':math.exp(delta['mean']),'ratio_low':math.exp(delta['low']),'ratio_high':math.exp(delta['high']),'delta_nll':delta,'agreement':float(np.mean([z['agreement']for z in groups[g+'__'+t].values()])),'kl':float(np.mean([z['kl_clean_to_condition']for z in groups[g+'__'+t].values()]))}
  for a,b in [('meanK_zeroV','remove_source'),('pca12','pca4'),('global_fresh','mean_KV'),('global_fresh','global_hume'),('global_fresh','global_wiki')]:effects[c+'__'+identity+'__'+a+'_minus_'+b]=effect(groups,g,a,b)
 for t in ['clean','deleted','mean_KV','meanK_zeroV','remove_source','oracle','global_wiki','global_hume','global_fresh','pca4','pca12']:
  # Source identity can still propagate through layers 0/1 before these L2+ cuts.
  a=raw['confirm_'+c+'_The__'+t];b=raw['confirm_'+c+'_natural__'+t]
  id_effects[c+'__'+t]={'The_minus_natural_nll':interval([float(a[i]['loss'].double().mean()-b[i]['loss'].double().mean())for i in range(8)]),'max_loss_difference':max(float((a[i]['loss']-b[i]['loss']).abs().max())for i in range(8))}
for z in checks:
 c,i,identity=z['corpus'],z['window'],z['identity'];g='confirm_'+c+'_'+identity;a=raw[g+'__oracle'][i];b=raw[g+'__mean_KV'][i]
 assert abs(float((a['loss']-b['loss']).double().mean())-z['oracle_delta_nll_to_mean'])<1e-12
 assert float(a['pred'].eq(b['pred']).double().mean())==z['oracle_agreement_to_mean']
for c in ['dubois','hobbes']:
 for identity in ['natural','The']:
  g='confirm_'+c+'_'+identity
  for t in ['mean_KV','meanK_zeroV','remove_source','global_fresh','pca4','pca12']:
   delta=interval([groups[g+'__'+t][i]['nll']-groups[g+'__clean'][i]['nll']for i in range(7)])
   sensitivity[c+'__'+identity+'__'+t]={'n':7,'ratio':math.exp(delta['mean']),'delta_nll':delta}
  for a,b in [('meanK_zeroV','remove_source'),('global_fresh','mean_KV'),('pca12','pca4')]:
   sensitivity[c+'__'+identity+'__'+a+'_minus_'+b]=interval([groups[g+'__'+a][i]['nll']-groups[g+'__'+b][i]['nll']for i in range(7)])
result.update(source_hashes_verified=True,span_disjointness_and_schedule_verified=True,fresh_gate_fit_max_error=max(fiterr),matched_suffix_token_hashes_verified=True,reconstruction_checks_replayed=64,duplicate_token_windows=duplicates,book_sensitivity_excluding_final_window=sensitivity,ratios=ratios,paired_effects=effects,oracle_controls=controls,identity_baseline_changes=id_effects,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
summary=[]
for key,items in groups.items():
 g,t=key.split('__');book=g.startswith('confirm_dubois_')or g.startswith('confirm_hobbes_')
 rs=[z for i,z in sorted(items.items())if not(book and i==7)]
 d=interval([z['nll']-z['baseline_nll']for z in rs]);a=interval([z['agreement']for z in rs]);k=interval([z['kl_clean_to_condition']for z in rs])
 summary.append({'group':g,'trial':t,'windows':len(rs),'nll':float(np.mean([z['nll']for z in rs])),'ppl_ratio':math.exp(d['mean']),'ratio_low':math.exp(d['low']),'ratio_high':math.exp(d['high']),'delta_nll':d['mean'],'delta_low':d['low'],'delta_high':d['high'],'agreement':a['mean'],'agreement_low':a['low'],'agreement_high':a['high'],'kl':k['mean'],'kl_low':k['low'],'kl_high':k['high']})
with (Path(__file__).parent/'confirmation_excluding_shared_footer_summary.csv').open('w')as f:
 writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
result['sensitivity_summary_rows']=len(summary)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['ratios','paired_effects','oracle_controls','identity_baseline_changes','book_sensitivity_excluding_final_window']},indent=2))
