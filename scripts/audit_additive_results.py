"""NumPy-only independent-reader audit of every exported additive-study metric."""
import argparse,json,math,resource
from collections import defaultdict
from pathlib import Path
import numpy as np
from audit_gate_results import ROOT,read,sha,mean,close,bootstrap

CORPORA={'wiki_validation':4,'mill':4,'wollstonecraft':4,'optimizers_new':4}
IDENTITIES=('natural','bos')
FAMILIES=('remove_pc1','orthogonal_control','position')
LAYERS=tuple('L'+str(l)for l in range(4,12))+('all',)
CONDITIONS=['C','P','current','fullrank']+[f+'_'+l for f in FAMILIES for l in LAYERS]
CONTRASTS=[t for l in LAYERS for t in ('remove_pc1_'+l+'_minus_P','remove_pc1_'+l+'_minus_orthogonal_control','position_'+l+'_minus_P')]
INPUTS=[(c,w,i)for c,n in CORPORA.items()for w in range(n)for i in IDENTITIES]
def exact_keys(rows,fields,expected,label):
 keys=[tuple(r[f]for f in fields)for r in rows]
 assert len(keys)==len(set(keys)),('Duplicate',label)
 assert set(keys)==set(expected),('Incomplete or unexpected universe',label)

def exact_control_closures(rows,values):
 """All exported per-target fields must match their intended copy reference."""
 closures=0
 for r in rows:
  trial=r['trial']
  if trial not in ('current','fullrank','repeat','identity'):continue
  ref=r['key'].rsplit('__',1)[0]+'__'+('P'if trial in('current','fullrank')else'C')
  for field in('nll','kl','kl_p','agreement','agreement_p'):
   assert np.array_equal(values[r['key']][field],values[ref][field]),('Exact control closure',r['key'],ref,field)
  closures+=1
 assert closures==72,('Control closure count',closures)
 return closures

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
 if args.output:args.output.write_text(json.dumps({'status':'checking'})+'\n')
 m=read(ROOT/'manifests/additive_numeric_export.json');p=ROOT/'results/gate_additive'
 expected_files={'paired_contrasts.json', 'directions.json', 'paired_summary.json', 'per_token_index.json', 'disk_checks.json', 'original_metadata_hashes.json', 'manifest.json', 'suffix_checks.json', 'suffix_operations.json', 'calibration_operations.json', 'additive_index.json', 'per_token_numeric.npz', 'windows.jsonl', 'checks.json', 'additive_windows.json', 'additive_summary.json', 'calibration_inputs.json', 'audit.json', 'operation_checks.json', 'summary.json', 'fp32_nonadditivity_summary.json', 'data_manifest.json', 'additive_numeric.npz', 'conditions.json', 'fp32_nonadditivity.json', 'calibration_spectra.json', 'disk_preflight.json', 'calibration_dependency.json', 'segment_summary.json'}
 assert set(m['exported_files'])==expected_files and {f.name for f in p.iterdir()}==expected_files
 for f,h in m['exported_files'].items():assert sha(p/f)==h,('export hash',f)
 manifest=read(p/'manifest.json');audit=read(p/'audit.json')
 assert manifest['status']==manifest['phase']=='completed'and audit['status']=='passed'
 assert [manifest[k]for k in('forward_passes','forward_attempts','records','evaluation_forwards','suffix_forwards','calibration_forwards','inherited_calibration_forwards')]==[1992,1992,1000,1000,992,0,8]
 assert audit['forwards']==1992 and audit['records']==1000
 assert manifest['model_hash_before']==manifest['model_hash_after']=='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527'
 assert manifest['parameter_versions_unchanged']is True and manifest['calibration_maps_unchanged']is True
 assert manifest['precision_settings_before']==manifest['precision_settings_after']and manifest['physical_execution_lengths']==[512]
 assert manifest['calibration_maps_tensor_digest']=='fbf127a1a618e709b0d8e39625a156f11402c9703ac4b55ae3dcfd3711571c81'
 rows=[json.loads(x)for x in(p/'windows.jsonl').read_text().splitlines()];idx=read(p/'per_token_index.json');rr={r['key']:r for r in rows}
 assert len(rows)==len(idx)==len(rr)==m['records']==1000
 expected={(f'gate__{c}__{w}__512__{i}__{t}',)for c,w,i in INPUTS for t in CONDITIONS}
 expected|={(f'gate__{c}__0__512__natural__{t}',)for c in CORPORA for t in ('repeat','identity')}
 exact_keys(rows,('key',),expected,'records');exact_keys(idx,('key',),expected,'index')
 for r in rows:
  assert r['key']==f"gate__{r['corpus']}__{r['window']}__512__{r['identity']}__{r['trial']}"
  assert r['panel']=='gate'and r['length']==r['input_tokens']==512 and r['target_start']==2 and r['scored_tokens']==510
 assert {r['key']for r in idx}==set(rr)and sorted(r['index']for r in idx)==list(range(1000))
 values={};scalars=0
 with np.load(p/'per_token_numeric.npz',allow_pickle=False)as z:
  expected=set()
  for e in idx:
   r=rr[e['key']];assert e['score_index']==r['score_index']==1 and e['token_sha256']==r['token_sha256'];v={}
   for suffix,field in(('loss','nll'),('kl','kl'),('kl_p','kl_p'),('matches_native','agreement'),('matches_p','agreement_p')):
    name=f"{e['index']}_{suffix}";expected.add(name);x=z[name];assert x.shape==(510,)and np.isfinite(x).all()
    assert x.dtype==np.bool_ if suffix.startswith('matches_')else x.dtype==np.float32
    close(mean(x),r[field]);v[field]=x.astype(np.float64);scalars+=1
   values[e['key']]=v
  assert set(z.files)==expected
 closures=exact_control_closures(rows,values)
 for e in idx:
  stem=e['key'].rsplit('__',1)[0]+'__'
  assert e['baseline_key']==stem+'C'and e['p_baseline_key']==stem+'P'
  for k,f in(('baseline_key','baseline_nll'),('p_baseline_key','p_baseline_nll')):close(mean(values[e[k]]['nll']),rr[e['key']][f]);scalars+=1
 conditions=read(p/'conditions.json');assert conditions==CONDITIONS;groups=defaultdict(list)
 for r in rows:
  if r['trial']in conditions:groups[r['corpus'],r['identity'],r['trial']].append(r)
 summarychecks=0
 for filename,segments in(('summary.json',False),('segment_summary.json',True)):
  summaries=read(p/filename);assert len(summaries)==(496 if segments else 248)
  universe=[(c,i,t)for c in CORPORA for i in IDENTITIES for t in CONDITIONS]
  fields=('corpus','identity','trial')
  if segments:universe=[k+(s,)for k in universe for s in ('early_2_127','later_128_511')];fields+=('view',)
  exact_keys(summaries,fields,universe,filename)
  for s in summaries:
   gs=groups[s['corpus'],s['identity'],s['trial']];assert len(gs)==4
   sl=slice(None)if not segments else slice(0,126)if s['view']=='early_2_127'else slice(126,None)
   for ref,prefix in(('C',''),('P','p_')):
    delta=[];kl=[];agree=[]
    for r in gs:
     v=values[r['key']];b=values[r['key'].rsplit('__',1)[0]+'__'+ref]
     delta.append(mean(v['nll'][sl])-mean(b['nll'][sl]));kl.append(mean(v['kl'if ref=='C'else'kl_p'][sl]));agree.append(mean(v['agreement'if ref=='C'else'agreement_p'][sl]))
    assert s[prefix+'windows']==len(gs);close(s[prefix+'mean_delta_nll'],mean(delta));close(s[prefix+'ppl_ratio'],math.exp(mean(delta)));close(s[prefix+'ppl_ratio_ci95'],np.exp(bootstrap(delta)));close(s[prefix+'mean_kl'],mean(kl));close(s[prefix+'mean_agreement'],mean(agree));close(s[prefix+'per_window_delta_nll'],delta);summarychecks+=1
 contrasts=read(p/'paired_contrasts.json');paired=read(p/'paired_summary.json');assert len(contrasts)==864 and len(paired)==216;cg=defaultdict(list)
 exact_keys(contrasts,('corpus','window','identity','trial'),[(c,w,i,t)for c,w,i in INPUTS for t in CONTRASTS],'contrasts')
 exact_keys(paired,('corpus','identity','trial'),[(c,i,t)for c in CORPORA for i in IDENTITIES for t in CONTRASTS],'paired summaries')
 for c in contrasts:
  left,right=c['trial'].split('_minus_');right=right+'_'+left.rsplit('_',1)[1]if right=='orthogonal_control'else right
  stem=f"gate__{c['corpus']}__{c['window']}__512__{c['identity']}__";delta=mean(values[stem+left]['nll'])-mean(values[stem+right]['nll']);close(delta,c['delta_nll']);cg[c['corpus'],c['identity'],c['trial']].append(delta)
 def check_groups(summaries,g,key):
  for s in summaries:
   ds=g[s['corpus'],s['identity'],s[key]];assert s['n']==len(ds);close(s['mean_delta_nll'],mean(ds));close(s['ci95_delta_nll'],bootstrap(ds));close(s['per_window_delta_nll'],ds)
 check_groups(paired,cg,'trial')
 nonadd=read(p/'fp32_nonadditivity.json');ns=read(p/'fp32_nonadditivity_summary.json');assert len(nonadd)==96 and len(ns)==24;ng=defaultdict(list)
 exact_keys(nonadd,('corpus','window','identity','family'),[(c,w,i,f)for c,w,i in INPUTS for f in FAMILIES],'nonadditivity records')
 exact_keys(ns,('corpus','identity','family'),[(c,i,f)for c in CORPORA for i in IDENTITIES for f in FAMILIES],'nonadditivity summaries')
 for r in nonadd:
  assert r['length']==512
  stem=f"gate__{r['corpus']}__{r['window']}__512__{r['identity']}__";base=mean(values[stem+'P']['nll']);joint=mean(values[stem+r['family']+'_all']['nll'])-base;singles=[mean(values[stem+r['family']+'_L'+str(l)]['nll'])-base for l in range(4,12)]
  close(r['joint_delta_nll'],joint);close(r['singleton_delta_nll'],singles);close(r['sum_singleton_delta_nll'],sum(singles));close(r['joint_minus_sum_delta_nll'],joint-sum(singles));ng[r['corpus'],r['identity'],r['family']].append(joint-sum(singles))
 check_groups(ns,ng,'family')
 # Additive arrays contain scalar metrics only; no source/prediction token IDs or logits.
 ai=read(p/'additive_index.json');ar=read(p/'additive_windows.json');summ=read(p/'additive_summary.json')
 keys=[(c,w,i,f)for c,w,i in INPUTS for f in FAMILIES]
 exact_keys(ai,('corpus','window','identity','family'),keys,'additive index')
 assert len(ai)==96 and sorted(e['index']for e in ai)==list(range(96))
 fields={'loss_p','loss_joint','loss_additive','singleton_losses','sum_singleton_delta','J','R','joint_minus_sum','identity_error',
  'kl_joint_additive','kl_additive_joint','kl_p_additive','kl_native_additive','centered_logit_l2','centered_logit_rms','centered_logit_max_abs',
  'joint_delta_centered_l2','additive_delta_centered_l2','agreement_joint_additive','agreement_p_additive','agreement_native_additive'}
 views={'full':slice(1,511),'early_2_127':slice(1,127),'later_128_511':slice(127,511)}
 exact_keys(ar,('corpus','window','identity','family','view'),[k+(v,)for k in keys for v in views],'additive window views')
 ri={(r['corpus'],r['window'],r['identity'],r['family'],r['view']):r for r in ar}
 exact_keys(summ,('corpus','identity','family','view','metric'),[(c,i,f,v,k)for c in CORPORA for i in IDENTITIES for f in FAMILIES for v in views for k in fields-{'singleton_losses'}],'additive summary')
 ag=defaultdict(list);count=0;maxerr=0.
 with np.load(p/'additive_numeric.npz',allow_pickle=False)as z:
  assert set(z.files)=={f"{e['index']}_{k}"for e in ai for k in fields}
  for e in ai:
   name=f"gate__{e['corpus']}__{e['window']}__512__{e['identity']}.pt"
   assert e['source_file']==name and e['source_sha256']==m['raw_inventory']['additive/'+name]['sha256']
   stem=name[:-3]+'__'
   d={k:z[f"{e['index']}_{k}"]for k in fields}
   for k,v in d.items():
    assert v.shape==((8,511)if k=='singleton_losses'else(511,))and np.isfinite(v).all()
    assert v.dtype==(np.bool_ if k.startswith('agreement_')else np.float64)
   pairs=[('P',d['loss_p']),(e['family']+'_all',d['loss_joint'])]+[(e['family']+'_L'+str(l),d['singleton_losses'][l-4])for l in range(4,12)]
   for mode,losses in pairs:assert np.max(np.abs(losses[1:]-values[stem+mode]['nll']))<=1e-4
   assert min(d[k].min()for k in fields if k.startswith('kl_'))>=-1e-10
   ss=(d['singleton_losses']-d['loss_p']).sum(0);nn=d['loss_joint']-d['loss_p']-ss
   close(d['sum_singleton_delta'],ss);close(d['J'],d['loss_joint']-d['loss_additive'])
   close(d['R'],d['loss_additive']-d['loss_p']-ss);close(d['joint_minus_sum'],nn)
   residual=np.abs(nn-d['J']-d['R']);close(d['identity_error'],residual);assert residual.max()<=1e-12;maxerr=max(maxerr,float(residual.max()))
   for view,sl in views.items():
    row=ri[e['corpus'],e['window'],e['identity'],e['family'],view]
    for k in fields-{'singleton_losses'}:
     value=mean(d[k][sl]);close(value,row[k]);ag[e['corpus'],e['identity'],e['family'],view,k].append(value);count+=1
 for r in summ:
  v=ag[r['corpus'],r['identity'],r['family'],r['view'],r['metric']]
  assert r['n']==len(v)==4;close(r['mean'],mean(v));close(r['ci95'],bootstrap(v));close(r['per_window'],v)
 bootstrapmax=0.
 for c in CORPORA:
  for i in IDENTITIES:
   for f in FAMILIES:
    for view in views:
     jj,rr,nn=[np.array(ag[c,i,f,view,k])for k in('J','R','joint_minus_sum')]
     draw=np.random.default_rng(20261021).integers(0,4,(2000,4));err=np.max(np.abs(jj[draw].mean(1)+rr[draw].mean(1)-nn[draw].mean(1)))
     assert err<=1e-12;bootstrapmax=max(bootstrapmax,float(err))
 out={'status':'passed','records':1000,'exact_control_closures':closures,'scalar_reductions':scalars,'dual_reference_summary_checks':summarychecks,'paired_contrasts':864,'paired_groups':216,'nonadditivity_records':96,'nonadditivity_groups':24,'additive_inputs_families':96,'additive_window_views':288,'additive_metric_groups':1440,'additive_scalar_reductions':count,'max_identity_error':maxerr,'max_bootstrap_identity_error':bootstrapmax,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Audits exported loss/KL/boolean agreement arrays, additive scalar identities, exact per-target control closures and all aggregations, not original KL-from-logits, private tokens, hook arithmetic or absent complete contexts.'}
 if args.output:args.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
