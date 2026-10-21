"""Independent CPU-only geometric reconstruction, controls and metrics replay."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'source_geometry_run'
m=json.loads((OUT/'manifest.json').read_text());assert m['completed_utc'] and m['forward_passes']==616 and m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged']
for f,p in [('script_sha256',ROOT/'source_geometry.py'),('design_sha256',ROOT/'SOURCE_GEOMETRY.md')]:assert m[f]==hashlib.sha256(p.read_bytes()).hexdigest()
for f,h in m['input_hashes'].items():assert h==hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,4);assert result['rows']==616 and result['groups']==154
tr=torch.load(OUT/'donors.pt',weights_only=True,map_location='cpu');random=torch.load(OUT/'random_controls.pt',weights_only=True,map_location='cpu');checks=json.loads((OUT/'checks.json').read_text());obs=json.loads((OUT/'geometry_observations.json').read_text());assert len(tr)==28 and len(checks)==168 and len(obs)==1176
u=torch.full((2048,),-1/2048,dtype=torch.float32);u[1764]+=1;u/=u.norm()
max_rand_mean=max_rand_dot=max_rand_radius=0.
for seed,vs in random.items():
 gen=torch.Generator().manual_seed(seed)
 for l,v in vs.items():
  z=torch.randn(2048,generator=gen);z-=z.mean();z-=u*(z@u);z/=z.norm();assert torch.equal(v,z)
  max_rand_mean=max(max_rand_mean,abs(float(v.mean())));max_rand_dot=max(max_rand_dot,abs(float(v@u)));max_rand_radius=max(max_rand_radius,abs(float(v.norm())-1))
geometry={};exact_vectors=0
for key,z in tr.items():
 g,i=key.split('__');i=int(i);geometry[key]={}
 for world,ds in z['donors'].items():
  geometry[key][world]={}
  for l,d in ds.items():
   x=d['norm'].float();mean=x.mean();y=x-mean;r=y.norm();a=(y@u)/r;b=y/r-a*u;s=b.norm();v=b/s;assert torch.isfinite(v).all()and r>0 and s>0
   expected={'mean':float(mean),'radius':float(r),'a':float(a),'s':float(s)};o=next(q for q in obs if q['group']==g and q['window']==i and q['world']==world and q['layer']==l)
   for k,val in expected.items():assert o[k]==val
   reconstructed=(mean+r*(a*u+s*v)).half();assert torch.equal(reconstructed,d['norm']);exact_vectors+=1;geometry[key][world][l]=expected
  ss=z['source']+1;ref={'C':'clean','R':'row','N':'column'}[world]
  for t in ['full_'+world,'angle_'+world+'_direction_'+world]:
   aa,bb=raw[g+'__'+t][i],raw[g+'__'+ref][i];assert torch.equal(aa['loss'],bb['loss']) and torch.equal(aa['pred'],bb['pred']) and torch.equal(aa['kl'],bb['kl']);assert groups[g+'__'+t][i]['score_start']==ss
 assert torch.equal(z['donors']['R'][2]['raw'][1764],z['donors']['N'][2]['raw'][1764])
for z in checks:
 if z['kind']=='closure':assert z['loss_exact']and z['pred_exact']
 else:assert z['max_norm_error']==z['nll_drift']==0 and z['agreement']==1
cs=list(csv.DictReader((OUT/'geometry_contrasts.csv').open()));assert len(cs)==84
for z in cs:
 g=z['group'];ts=json.loads(z['terms']);st=interval([sum(c*groups[g+'__'+t][i]['nll']for t,c in ts)for i in range(4)])
 for k,v in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(z[k])-st[v])<1e-12
ratios={};windows=[]
for key,rs in groups.items():
 g,t=key.split('__');e=effect(groups,g,t,'clean');ratios[key]={'ppl_ratio':math.exp(e['mean']),'delta_nll':e,'agreement':interval([r['agreement']for r in rs.values()]),'kl':interval([r['kl_clean_to_condition']for r in rs.values()])}
 for i,r in rs.items():windows.append({'group':g,'trial':t,'window':i,'ppl_ratio':math.exp(r['nll']-r['baseline_nll']),'delta_nll':r['nll']-r['baseline_nll'],'agreement':r['agreement']})
anchors=0
for kind,file in [('first','column_factorial_run'),('newline','newline_natural_run')]:
 old=torch.load(ROOT/file/'per_token.pt',weights_only=True,map_location='cpu');old={k:{a['window']:a for a in v}for k,v in old.items()}
 for g in sorted({k.split('__')[0]for k in raw if k.startswith(kind+'_')}):
  c=g[len(kind)+1:]
  for t in ['clean','row','column']:
   oldkey=('first_'+c+'_natural__'+{'clean':'clean','row':'delete_row','column':'delete_column'}[t])if kind=='first'else('natural_'+c+'__'+{'clean':'clean','row':'clean_row_p','column':'clean_column_p'}[t])
   for i,a in raw[g+'__'+t].items():
    b=old[oldkey][i];n=len(b['loss']);assert torch.equal(a['loss'][-n:],b['loss'])and torch.equal(a['pred'][-n:],b['pred'])and a['token_sha256']==b['token_sha256'];anchors+=1
result.update(full_closures_replayed=84,same_donor_output_closures_replayed=84,exact_half_vector_reconstructions=exact_vectors,geometry_observations_replayed=len(obs),random_controls_replayed=42,max_random_mean=max_rand_mean,max_random_axis_dot=max_rand_dot,max_random_norm_error=max_rand_radius,prior_anchors_or_overlapping_suffixes=anchors,geometry_contrasts_replayed=len(cs),raw_L2_target_coordinate_equal_row_column_all_28=True,ratios=ratios,geometry=geometry,per_window=windows,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['ratios','geometry','per_window']},indent=2))
for g in sorted({k.split('__')[0]for k in raw}):
 print(g)
 for t in ['row','column','angle_C_direction_R','angle_N_direction_R','angle_R_direction_N','negative_axis_only']:
  z=ratios[g+'__'+t];print(t,round(z['ppl_ratio'],5),round(z['agreement']['mean'],5))
