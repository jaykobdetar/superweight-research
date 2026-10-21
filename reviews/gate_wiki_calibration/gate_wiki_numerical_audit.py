"""Independent T18 raw/statistics/geometry audit. CPU only; require completed run."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import hashlib,json,sys,time,resource,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gate_wiki_storage import read_npz,restore_snapshot
from gpt2_gate_subspace import maps_digest
from gate_wiki_calibration import CONDITIONS,mode_spec,SNAPSHOT_MODES
RUN=ROOT/'gpt2_gate_wiki_calibration_run'
def j(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def th(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def fp(x):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=th(x))
def same(a,b):
 if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b)and len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b

def main():
 started=time.time();torch.set_num_threads(2);m=j(RUN/'manifest.json');assert m['status']=='completed'and m['phase']=='completed';assert j(RUN/'saved_state_audit.json')['status']=='passed'
 assert m['forward_passes']==m['forward_attempts']==1548 and m['records']==776
 assert (m['calibration_forwards'],m['evaluation_forwards'],m['suffix_forwards'])==(4,776,768)
 assert m['model_hash_before']==m['model_hash_after']and m['parameter_versions_unchanged']and m['calibration_maps_unchanged']and m['precision_settings_before']==m['precision_settings_after']
 for n,h in m['source_hashes'].items():assert sha(RUN/'frozen_sources'/n)==h
 for n,h in m['input_dependencies'].items():assert sha(RUN/n)==h
 inv=j(RUN/'numeric_inventory.json')
 for n,r in inv.items():assert sha(RUN/n)==r['sha256']and (RUN/n).stat().st_size==r['bytes']
 maps=read_npz(RUN/'calibration_maps.npz');dirs=read_npz(RUN/'calibration_directions.npz');spec=read_npz(RUN/'calibration_spectra.npz');geo=read_npz(RUN/'calibration_geometry.npz')
 assert {k:maps_digest(v)for k,v in maps.items()}==m['calibration_map_digests']
 same(maps['O'],torch.load(ROOT/'wiki_calibration_preparation/old_wiki_maps.pt',weights_only=True,map_location='cpu'));same(maps['M'],torch.load(ROOT/'gpt2_gate_fixed512_run/calibration_maps.pt',weights_only=True,map_location='cpu'))
 cal={};fiterr={'mean':0.,'covariance':0.,'eigenvalues':0.,'LOO_cosines':0.};fitcounts={'full':0,'LOO':0,'cosines':0};geom=[]
 def space(x,report,kind):
  y=x.reshape(-1,12).double();mu=y.mean(0);c=y-mu;cov=c.T@c/(len(y)-1);_,s,vh=torch.linalg.svd(c,full_matrices=False);ev=s.square()/(len(y)-1)
  assert report['vectors']==len(y)and report['covariance_denominator']==len(y)-1
  for label,actual,saved,tol in [('mean',mu,report['mean'],1e-12),('covariance',cov,report['covariance'],1e-12),('eigenvalues',ev,report['eigenvalues'],1e-10)]:
   err=float((actual-torch.tensor(saved,dtype=torch.float64)).abs().max());fiterr[label]=max(fiterr[label],err);assert err<=tol,(kind,label,err)
  assert float((torch.linalg.eigvalsh(cov).flip(0)-ev).abs().max())<=1e-9;fitcounts[kind]+=1
  return mu,vh.T,ev
 for p in sorted((RUN/'calibration').glob('*.npz')):
  z=read_npz(p);cal[p.stem]=z['raw'];assert z['metrics']['kl_reference']=='self/no-native-C'and torch.count_nonzero(z['metrics']['kl'])==0
 assert list(cal)==['wiki_test__'+str(i)for i in range(4)]
 for layer in range(4,12):
  b=torch.stack([z[layer]['b'][:,:510].T for z in cal.values()]).double();a=torch.stack([z[layer]['mass'][:,:510].T for z in cal.values()]).double();logq=torch.arange(1,511,dtype=torch.float64).log()[None,:,None]
  spaces={'mass':a,'b':b,'r':b-logq,'position_centered_b':b-b.mean(0,keepdim=True)};bases={}
  for name,x in spaces.items():
   mu,u,e=space(x,spec['F'][layer][name],'full');bases[name]=u
   if name in ['r','b','mass']:
    actual=maps['F'][layer][{'r':'alpha','b':'mean_b','mass':'mean_mass'}[name]];assert torch.allclose(mu,actual,rtol=0,atol=1e-12)
   if name in ['r','b']:
    got=maps['F'][layer]['basis_'+name];assert torch.allclose(got@got.T,u@u.T,rtol=0,atol=1e-12);assert abs(float(got[:,0]@u[:,0]))>=1-1e-12
  for omit,lo in enumerate(spec['F'][layer]['leave_one_passage_out']):
   assert lo['omitted_passage']==list(cal)[omit];keep=[i for i in range(4)if i!=omit]
   for name,x in spaces.items():
    v=b[keep]-b[keep].mean(0,keepdim=True)if name=='position_centered_b'else x[keep];mu,u,e=space(v,lo['spaces'][name],'LOO')
    for rank,values in lo['spaces'][name]['principal_cosines_to_full'].items():
     k=int(rank);got=torch.linalg.svdvals(bases[name][:,:k].T@u[:,:k]);err=float((got-torch.tensor(values,dtype=torch.float64)).abs().max());assert err<=1e-10;fiterr['LOO_cosines']=max(fiterr['LOO_cosines'],err);fitcounts['cosines']+=1
  for label in ['O','F','M']:
   u0=maps[label][layer]['basis_r'][:,0].double();u=u0/u0.norm();selected=None
   for col in range(12):
    q=maps[label][layer]['basis_qr'][:,col];v=q-u*(u@q)
    if v.norm()>=1e-6:selected=(col,v/v.norm());break
   col,v=selected;pivot=int(v.abs().argmax());v*=1 if v[pivot]>=0 else-1
   assert torch.allclose(dirs[label][layer]['u'],u,rtol=0,atol=1e-12)and torch.allclose(dirs[label][layer]['v'],v,rtol=0,atol=1e-12)
  delta=maps['F'][layer]['alpha']-maps['O'][layer]['alpha'];cos=float(dirs['O'][layer]['u']@dirs['F'][layer]['u']);assert abs(cos-geo[layer]['signed_pc1_cosine'])<=1e-12
  assert abs(float(delta.norm())-geo[layer]['mean_displacement_l2'])<=1e-12
  vals=spec['F'][layer]['r']['eigenvalues'];gap=(vals[0]-vals[1])/vals[0];assert gap>1e-12 and abs(gap-geo[layer]['F_relative_leading_gap'])<=1e-12
  geom.append({'layer':layer,'signed_cosine':cos,'mean_displacement_l2':float(delta.norm()),'fresh_relative_gap':gap,'fresh_PC1_variance_fraction':vals[0]/sum(vals),'fresh_LOO_PC1_absolute_cosines':[v['spaces']['r']['principal_cosines_to_full']['1'][0]for v in spec['F'][layer]['leave_one_passage_out']]})
 analysis=j(RUN/'wiki_factorial_analysis.json');savedrows={(r['corpus'],r['identity'],r['window'],r['view'],r['mode']):r for r in analysis['window_metrics']};assert len(savedrows)==2328
 savednorms={(r['corpus'],r['identity'],r['window'],r['mode'],r['layer'],r['view']):r for r in analysis['norm_summary']};assert len(savednorms)==9408;norm_stat_checks=0;norm_stat_max=0.
 rows={};counts={'shards':0,'records':0,'suffix':0,'main_checks':0,'gate_checks':32,'norms':0,'snapshots':0};scalar_max=0.;norm_max=0.;extrema={'max_gain':0.,'min_mass':float('inf'),'max_conditional_error':0.};snapshot_errors={}
 tokens=torch.load(RUN/'tokens.pt',weights_only=True,map_location='cpu');lookup={(v['corpus'],v['window']):v['tokens']for v in tokens};bos=j(RUN/'data_manifest.json')['literals']['start_special']['id'];views={'full':slice(1,511),'early':slice(1,127),'later':slice(127,511)}
 for p in sorted((RUN/'inputs').glob('*.npz')):
  d=read_npz(p);base=d['metadata'];ci=base['corpus'],base['identity'];w=base['window'];x=lookup[ci[0],w].clone()
  if ci[1]=='bos':x[0]=bos
  raw={k.rsplit('__',1)[-1]:v for k,v in d['per_token'].items()};assert len(raw)==len(d['per_token']);counts['shards']+=1;counts['records']+=len(raw);counts['main_checks']+=len(d['checks'])
  c,pp=raw['C'],raw['P'];expected=set(CONDITIONS)|({'repeat','identity'}if ci[1]=='natural'and w==0 else set());assert set(raw)==expected
  for mode,z in raw.items():
   assert z['token_sha256']==th(x)
   for trial,reference in [('current','P'),('fullrank','P')]+([('repeat','C'),('identity','C')]if 'repeat'in raw else []):
    if mode==trial:
     for k in ['loss','kl','kl_p','pred','agreement_p']:assert torch.equal(z[k],raw[reference][k])
     assert z['final_hidden_sha256']==raw[reference]['final_hidden_sha256']
   for view,sl in views.items():
    a=z['loss'][sl].numpy().astype('float64');bv=pp['loss'][sl].numpy().astype('float64');cv=c['loss'][sl].numpy().astype('float64')
    got=dict(nll=float(a.mean()),delta_nll_p=float((a-bv).mean()),delta_nll_c=float((a-cv).mean()),kl_c=float(z['kl'][sl].double().mean()),kl_p=float(z['kl_p'][sl].double().mean()),agreement_c=float(z['pred'][sl].eq(c['pred'][sl]).double().mean()),agreement_p=float(z['pred'][sl].eq(pp['pred'][sl]).double().mean()))
    key=(*ci,w,view,mode);assert key not in rows;rows[key]=got
    for k,v in got.items():err=abs(v-savedrows[key][k]);scalar_max=max(scalar_max,err);assert err<=2e-12
  ops={v['trial']:v for v in d['operations']};assert set(ops)==set(raw)
  for mode in CONDITIONS:
   sm=d['suffix_metrics'][mode];counts['suffix']+=1
   for k in ['loss','kl','kl_p','pred','agreement_p']:assert torch.equal(sm[k],raw[mode][k][:127 if k=='loss'else 128])
   if mode!='C':assert ops[mode]['source_fingerprints']==ops['P']['source_fingerprints']
   if len(mode_spec(mode)[3])==1:
    assert ops[mode]['gate_fingerprints'][10]['b']==ops['P']['gate_fingerprints'][10]['b']and ops[mode]['gate_fingerprints'][10]['mass']==ops['P']['gate_fingerprints'][10]['mass']and ops[mode]['current_context_hashes'][10]==ops['P']['current_context_hashes'][10]
  for op in d['operations']+d['suffix_operations']:
   for z in op['gate_checks'].values():
    counts['gate_checks']+=1;extrema['max_gain']=max(extrema['max_gain'],z['max_gain']);extrema['min_mass']=min(extrema['min_mass'],z['min_mass']);extrema['max_conditional_error']=max(extrema['max_conditional_error'],z['conditional_error'])
  for mode,ls in d['norms'].items():
   for l,vs in ls.items():
    counts['norms']+=1
    for k,v in vs.items():
     assert fp(v)==ops[mode]['norm_fingerprints'][l][k]
     for view,sl in {'full':slice(0,510),'early':slice(0,126),'later':slice(126,510)}.items():
      a=v[sl].double();got={'mean':float(a.mean()),'rms':float(a.square().mean().sqrt()),'max_abs':float(a.abs().max())};saved=savednorms[ci[0],ci[1],w,mode,l,view][k]
      for field,val in got.items():
       err=abs(val-saved[field]);norm_stat_max=max(norm_stat_max,err);assert err<=2e-12;norm_stat_checks+=1
    if mode!='P':
     u,v=vs['hypothetical_pc1_norm'],vs['hypothetical_orthogonal_norm'];err=float(((u-v).abs()/torch.maximum(torch.ones_like(u),torch.maximum(u,v))).max());norm_max=max(norm_max,err);assert err<=1e-6
  if d['snapshots']is not None:
   for mode in SNAPSHOT_MODES:
    s=restore_snapshot(d['snapshots'],mode);counts['snapshots']+=1;q,k,v=[s[n].double()for n in ['q','k','v']];mask=s['mask'];z=q@k.transpose(-1,-2)/8;z=z.masked_fill(torch.ones(512,512,dtype=torch.bool).triu(1),-float('inf'));z=z.masked_fill(~mask,-float('inf'))if mask.dtype==torch.bool else z+mask
    probs=z.softmax(-1);a=probs[0,:,1:,1:].sum(-1);b=(z[:,:,1:,1:].logsumexp(-1)-z[:,:,1:,0])[0];u=(z[:,:,1:,1:].softmax(-1)@v[:,:,1:])[0];context=(probs@v).transpose(1,2).reshape(1,512,768)
    errs={'b':float((b-s['gate_state']['b'].double()).abs().max()),'mass':float((a-s['gate_state']['mass'].double()).abs().max()),'stable_mixture':float((u-s['stable_mixture'].double()).abs().max()),'native_context':float((context-s['actual_context'].double()).abs().max())}
    fam,center,basis,affected=mode_spec(mode);sb=s['gate_state']['b'].double();sa=s['gate_state']['mass'].double();target=sa;tb=sb
    if mode!='P':
     # Mathematical FP64 recomputation of the actual FP32-cast maps/directions.
     alpha=maps[center][10]['alpha'].float().double();du=dirs[basis][10]['u'].float().double();dv=dirs[basis][10]['v'].float().double();logq=torch.arange(1,512,dtype=torch.float32).log().double();coef=(sb.T-logq[:,None]-alpha)@du;tb=sb-(coef[:,None]*(du if fam=='remove'else dv)).T;target=tb.sigmoid()
    actual=s['actual_context'].double();delivered=actual.clone();delivered[:,1:]=(actual[:,1:].reshape(511,12,64).transpose(0,1)*(target/sa)[...,None]).transpose(0,1).reshape(1,511,768)
    errs.update(target_b=float((tb-s['gate_state']['target_b'].double()).abs().max()),target_mass=float((target-s['gate_state']['target_mass'].double()).abs().max()),delivered_context=float((delivered-s['delivered_context'].double()).abs().max()))
    assert all(math.isfinite(t)for t in errs.values());snapshot_errors[mode]=errs
    assert torch.equal(s['actual_context'][:,0],s['delivered_context'][:,0]);assert torch.count_nonzero(s['v'][:,:,0])==0
  del d
 assert counts=={'shards':32,'records':776,'suffix':768,'main_checks':1128,'gate_checks':11808,'norms':3136,'snapshots':11}
 rng=np.random.default_rng(20261021);ids=rng.integers(0,4,(2000,4));staterr=0.;summarychecks=contrastchecks=0
 def stat(values,record):
  nonlocal staterr
  values=np.asarray(values,dtype='float64');assert record['n']==4 and np.allclose(record['per_window'],values,rtol=0,atol=2e-12)
  got=[float(values.mean()),*np.quantile(values[ids].mean(1),[.025,.975]).tolist()];saved=[record['mean'],*record['ci95']];err=max(abs(a-b)for a,b in zip(got,saved));staterr=max(staterr,err);assert err<=2e-12
 for r in analysis['summary']:
  for metric in ['nll','delta_nll_p','delta_nll_c','kl_c','kl_p','agreement_c','agreement_p']:
   values=[rows[r['corpus'],r['identity'],w,r['view'],r['mode']][metric]for w in range(4)];stat(values,r[metric]);summarychecks+=1
  assert abs(math.exp(r['delta_nll_p']['mean'])-r['ppl_ratio_p']['point'])<=2e-12
 for r in analysis['contrasts']:
  f=r['family'];ss=r['layer_set'];name=r['name']
  if f=='matched_control':cell=name.removeprefix('remove_minus_orthogonal_');terms={f'remove_{cell}_{ss}':1,f'orthogonal_{cell}_{ss}':-1}
  else:
   t={'center_at_O':{'FO':1,'OO':-1},'center_at_F':{'FF':1,'OF':-1},'direction_at_O':{'OF':1,'OO':-1},'direction_at_F':{'FF':1,'FO':-1},'interaction':{'FF':1,'FO':-1,'OF':-1,'OO':1},'M8_minus_OO':{'M8':1,'OO':-1},'M8_minus_FF':{'M8':1,'FF':-1}}[name];terms={f+'_'+c+'_'+ss:v for c,v in t.items()}
  assert terms==r['terms'];values=[sum(weight*rows[r['corpus'],r['identity'],w,r['view'],mode]['nll']for mode,weight in terms.items())for w in range(4)];stat(values,r['delta_nll']);contrastchecks+=1
 assert summarychecks==4032 and contrastchecks==912
 result={'status':'passed_core_pending_readout','source_sha256':sha(Path(__file__)),'manifest_sha256':sha(RUN/'manifest.json'),'counts':counts,'fresh_fit_counts':fitcounts,'fresh_fit_errors':fiterr,'calibration_geometry':geom,'per_token_window_metric_max_error':scalar_max,'summary_metric_groups':summarychecks,'paired_contrasts':contrastchecks,'bootstrap_max_error':staterr,'norm_match_max_error':norm_max,'norm_summary_scalar_checks':norm_stat_checks,'norm_summary_max_error':norm_stat_max,'runtime_extrema':extrema,'snapshot_FP64_errors':snapshot_errors,'snapshot_FP64_scope':'Descriptive independent floating-point comparison; runtime guards remain unchanged; no new acceptance tolerance inferred from these outcomes.','resources':{k:m[k]for k in ['wall_seconds','peak_process_rss_kib','peak_gpu_allocated_bytes']},'wall_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pretrained_model_loaded':False,'gpu_used':False,'limitations':['Only fixed11 attention/readout states are available as full arrays; other successful identities rely on saved fingerprints and runtime comparisons.','n4 passage bootstrap conditions on two fitted calibrations; no calibration-population uncertainty estimate.','Freshness is relative to recorded source exposures.']}
 assert result['peak_rss_kib']<3*1024**2
 return result
if __name__=='__main__':
 result=main();out=ROOT/'review/gate_wiki_numerical_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
