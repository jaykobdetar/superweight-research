#!/usr/bin/env python3
"""Audit the allowlisted T18 numeric export using NumPy and the standard library.

No model, checkpoint, tokenizer, source corpus or private raw tensors are needed.
Checks reported runtime hashes; it cannot reconstruct omitted contexts/logits.
"""
import argparse, hashlib, json, math, re, time, resource
from pathlib import Path
import numpy as np
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
STARTS=('natural','bos');LAYERS=tuple(range(4,12));CELLS=('OO','OF','FO','FF');FAMILIES=('remove','orthogonal');SETS=('L10','all')
MODES=('C','P','current','fullrank')+tuple(f+'_'+c+'_'+s for f in FAMILIES for c in CELLS for s in SETS)+tuple(f+'_M8_'+s for f in FAMILIES for s in SETS)
FIELDS=('loss','kl','kl_p','agreement_p','agreement_c')
VIEWS={'full':slice(1,511),'early':slice(1,127),'later':slice(127,511)}
NORMVIEWS={'full':slice(0,510),'early':slice(0,126),'later':slice(126,510)}
METRICS=('nll','delta_nll_p','delta_nll_c','kl_c','kl_p','agreement_c','agreement_p')
TOL=2e-12
MAP_DIGESTS={'O':'e59fab6ed96e92d34a3286680b989f8ca12608edee7923baaf088f5d1b40573d','F':'8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8','M':'fbf127a1a618e709b0d8e39625a156f11402c9703ac4b55ae3dcfd3711571c81'}
MODEL_DIGEST='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527'
WEIGHT_DIGEST='e182e433b37dbdb47448e0413c840edf6965113c1fc8048c63b69795d8cf875a'
LAYOUT_DIGEST='c837ab07243683e08dcfd0288fcf76a7ca8372dfdad06d605a01a80fe23f3974'
FROZEN_AUDITOR='ebae29f3c04c937aa8abf7dcbb608af91ee0a957f3ffface128e914e6b5f1686'
STATISTICAL_SCOPE={'evaluation_passages_per_group':4,'bootstrap_repetitions':2000,'bootstrap_seed':20261021,'common_indices':True,'views_share_execution':True,'calibrations_fixed':True,'calibration_population_uncertainty_estimated':False,'multiple_comparison_adjustment':False,'estimand':'Finite center/direction interventions conditional on the P source-value-zero boundary; M8 comparisons confound sample count and mixture.'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def digest(s):assert isinstance(s,str)and re.fullmatch('[0-9a-f]{64}',s)
def close(a,b,tol=TOL):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all()
 error=float(np.max(np.abs(a-b),initial=0));assert error<=tol,(error,tol);return error

def keyed(rows,fields,expected):
 out={}
 for r in rows:
  k=tuple(r[f]for f in fields);assert k not in out,('duplicate',k);out[k]=r
 assert set(out)==set(expected),('coverage',len(out),len(expected));return out

def norm_layers(mode):
 if mode=='P':return LAYERS
 if mode in ('C','current','fullrank','repeat','identity'):return ()
 return (10,)if mode.endswith('_L10')else LAYERS

def norm_fields(mode):
 base=('logit_delta_norm','mass_delta_norm','context_delta_norm')
 return base if mode=='P'else base+('s','hypothetical_pc1_norm','hypothetical_orthogonal_norm')

def expected_arrays(corpus,identity,window):
 modes=MODES+(('repeat','identity')if identity=='natural'and window==0 else ())
 result={m+'__'+f:(511,np.dtype('bool'if f.startswith('agreement')else'float32'))for m in modes for f in FIELDS}
 result.update({'suffix__'+m+'__'+f:(127 if f=='loss'else 128,np.dtype('bool'if f.startswith('agreement')else'float32'))for m in MODES for f in FIELDS})
 result.update({'norm__'+m+'__'+str(l)+'__'+f:(511,np.dtype('float32'if f=='s'else'float64'))for m in MODES for l in norm_layers(m)for f in norm_fields(m)})
 return modes,result

def validate_arrays(a,c,i,w):
 modes,expected=expected_arrays(c,i,w);assert set(a)==set(expected),('array universe',c,i,w)
 for name,(n,dtype)in expected.items():assert a[name].shape==(n,)and a[name].dtype==dtype and np.isfinite(a[name]).all(),name
 for left,right in [('current','P'),('fullrank','P')]+([('repeat','C'),('identity','C')]if'repeat'in modes else[]):
  for f in FIELDS:assert np.array_equal(a[left+'__'+f],a[right+'__'+f]),('copy',left,f)
 for m in MODES:
  for f in FIELDS:assert np.array_equal(a['suffix__'+m+'__'+f],a[m+'__'+f][:127 if f=='loss'else 128]),('suffix',m,f)
 assert np.all(a['C__agreement_c'])and np.all(a['P__agreement_p'])
 assert np.count_nonzero(a['C__kl'])==np.count_nonzero(a['P__kl_p'])==0
 for m in MODES:
  for l in norm_layers(m):
   prefix='norm__'+m+'__'+str(l)+'__'
   assert all(np.all(a[prefix+f]>=0)for f in norm_fields(m)if f!='s')
   if m=='P':
    assert all(np.count_nonzero(a[prefix+f])==0 for f in norm_fields(m))
   else:
    u,v=a[prefix+'hypothetical_pc1_norm'],a[prefix+'hypothetical_orthogonal_norm']
    assert np.max(np.abs(u-v)/np.maximum(1,np.maximum(u,v)))<=1e-6
 return modes

def contrast_terms(family,name,which):
 if family=='matched_control':
  assert name.startswith('remove_minus_orthogonal_');cell=name.removeprefix('remove_minus_orthogonal_');assert cell in (*CELLS,'M8')
  return {'remove_'+cell+'_'+which:1,'orthogonal_'+cell+'_'+which:-1}
 assert family in FAMILIES
 t={'center_at_O':{'FO':1,'OO':-1},'center_at_F':{'FF':1,'OF':-1},'direction_at_O':{'OF':1,'OO':-1},'direction_at_F':{'FF':1,'FO':-1},'interaction':{'FF':1,'FO':-1,'OF':-1,'OO':1},'M8_minus_OO':{'M8':1,'OO':-1},'M8_minus_FF':{'M8':1,'FF':-1}}[name]
 return {family+'_'+c+'_'+which:v for c,v in t.items()}

def calibration_audit(run):
 with np.load(run/'calibration_numeric.npz',allow_pickle=False)as z:
  a={k:z[k]for k in z.files};assert len(a)==len(z.files)
 expected={f'wiki_test__{w}__{l}__{f}':((12,511),np.dtype('float32'))for w in range(4)for l in LAYERS for f in ('b','mass')}
 expected.update({f'map__{label}__{l}__{f}':((12,12)if f.startswith('basis')else(12,),np.dtype('float64'))for label in ('O','F','M')for l in LAYERS for f in ('alpha','mean_b','mean_mass','basis_r','basis_b','basis_qr')})
 expected.update({f'direction__{label}__{l}__{f}':((12,),np.dtype('float64'))for label in ('O','F','M')for l in LAYERS for f in ('u','v')})
 assert set(a)==set(expected)
 for k,(shape,dtype)in expected.items():assert a[k].shape==shape and a[k].dtype==dtype and np.isfinite(a[k]).all(),k
 specs=read(run/'calibration_spectra.json');geometry=read(run/'calibration_geometry.json');directions=read(run/'direction_metadata.json');maxerr=0.;fits=0
 assert set(specs)==set(directions)=={'O','F','M'}and set(geometry)=={str(l)for l in LAYERS}
 for l in LAYERS:
  b=np.stack([a[f'wiki_test__{w}__{l}__b'][:,:510].T for w in range(4)]).astype('float64');mass=np.stack([a[f'wiki_test__{w}__{l}__mass'][:,:510].T for w in range(4)]).astype('float64');assert np.all(mass>0)
  spaces={'mass':mass,'b':b,'r':b-np.log(np.arange(1,511))[None,:,None],'position_centered_b':b-b.mean(0,keepdims=True)}
  for name,x in spaces.items():
   y=x.reshape(-1,12);mu=y.mean(0);centered=y-mu;cov=centered.T@centered/(len(y)-1);ev=np.linalg.eigvalsh(cov)[::-1];sp=specs['F'][str(l)][name]
   assert sp['vectors']==2040 and sp['covariance_denominator']==2039
   for actual,saved,tol in [(mu,sp['mean'],1e-12),(cov,sp['covariance'],1e-12),(ev,sp['eigenvalues'],1e-9)]:maxerr=max(maxerr,close(actual,saved,tol))
   if name in ('b','r','mass'):close(mu,a[f'map__F__{l}__'+{'b':'mean_b','r':'alpha','mass':'mean_mass'}[name]],1e-12)
   fits+=1
  for label in ('O','F','M'):
   u,v=a[f'direction__{label}__{l}__u'],a[f'direction__{label}__{l}__v'];close([u@u,v@v,u@v],[1,1,0],1e-12)
   for field,value in [('u',u),('v',v)]:assert hashlib.sha256(value.tobytes()).hexdigest()==directions[label][str(l)][field+'_sha256']
   for field in ('basis_r','basis_b','basis_qr'):
    basis=a[f'map__{label}__{l}__{field}'];close(basis.T@basis,np.eye(12),1e-10)
  u=a[f'direction__O__{l}__u'];v=a[f'direction__F__{l}__u'];close(u@v,geometry[str(l)]['signed_pc1_cosine'],1e-12)
 for label in MAP_DIGESTS:
  h=hashlib.sha256()
  for l in LAYERS:
   for field in sorted(('alpha','mean_b','mean_mass','basis_r','basis_b','basis_qr')):
    v=a[f'map__{label}__{l}__{field}'];h.update(f'{l}:{field}:torch.float64:{tuple(v.shape)}'.encode());h.update(v.tobytes())
  assert h.hexdigest()==MAP_DIGESTS[label]
 return {'fresh_full_fits':fits,'arrays':len(a),'max_fit_error':maxerr}

def audit(root):
 start=time.time();run=root/'results/gate_wiki_calibration';export=read(root/'manifests/wiki_calibration_numeric_export.json');manifest=read(run/'manifest.json')
 assert export['schema']==1 and export['input_shards']==32 and export['main_records']==776 and export['suffix_records']==768
 assert manifest['status']==manifest['phase']=='completed'and manifest['forward_passes']==manifest['forward_attempts']==1548 and manifest['records']==776
 assert (manifest['calibration_forwards'],manifest['evaluation_forwards'],manifest['suffix_forwards'])==(4,776,768)
 assert manifest['model_id']=='openai-community/gpt2'and manifest['revision']=='75e09b43581151bd1d9ef6700faa605df408979f'
 assert manifest['model_hash_before']==manifest['model_hash_after']==MODEL_DIGEST
 assert manifest['calibration_map_digests']==MAP_DIGESTS
 assert manifest['readout']['weight']==dict(sha256=WEIGHT_DIGEST,shape=[50257,768],dtype='torch.float32')and manifest['readout']['bias']is None and manifest['readout']['tied_embedding_storage']
 for k in ['model_hash_before','model_hash_after','calibration_maps_sha256','calibration_directions_sha256']:digest(manifest[k])
 assert manifest['model_hash_before']==manifest['model_hash_after']and manifest['parameter_versions_unchanged']and manifest['calibration_maps_unchanged']and manifest['precision_settings_before']==manifest['precision_settings_after']
 for name,h in manifest['source_hashes'].items():digest(h);assert export['raw_inventory']['frozen_sources/'+name]['sha256']==h
 for name,h in manifest['input_dependencies'].items():digest(h);assert export['raw_inventory'][name]['sha256']==h
 expectedfiles={'manifest.json','numeric_inventory.json','saved_state_audit.json','wiki_factorial_analysis.json','index.json','calibration_numeric.npz','direction_metadata.json','calibration_spectra.json','calibration_geometry.json'}
 universe={(c,i,w)for c in CORPORA for i in STARTS for w in range(4)}
 for c,i,w in universe:
  stem=f'inputs/gate__{c}__{w}__512__{i}';expectedfiles|={stem+'.json',stem+'.npz'}
 assert set(export['exported_files'])==expectedfiles and {p.relative_to(run).as_posix()for p in run.rglob('*')if p.is_file()}==expectedfiles
 for name,h in export['exported_files'].items():digest(h);assert sha(run/name)==h,('export hash',name)
 for name,r in export['raw_inventory'].items():assert not Path(name).is_absolute()and'..'not in Path(name).parts;digest(r['sha256']);assert type(r['bytes'])is int and r['bytes']>0
 for name,r in read(run/'numeric_inventory.json').items():assert export['raw_inventory'][name]==r
 savedstate=read(run/'saved_state_audit.json');assert savedstate['status']=='passed'and savedstate['totals']==dict(inputs=32,records=776,suffix_records=768,checks=1128,gate_checks=11808,snapshot_states=11,norm_layer_states=3136)
 layout=savedstate['layout_replay'];assert savedstate['source_sha256']==layout['frozen_auditor_sha256']==FROZEN_AUDITOR and layout['adapter_sha256']==LAYOUT_DIGEST
 assert layout['field']=='O/F/M L4-L11 basis_qr only'and layout['matrices']==24 and layout['serialized_stride']==[12,1]and layout['original_stride']==[1,12]
 assert layout['map_values_and_digests_unchanged']is True and layout['directions_and_geometry_exact']is True and layout['tolerances_changed']is False
 ix=keyed(read(run/'index.json'),('corpus','identity','window'),universe)
 analysis=read(run/'wiki_factorial_analysis.json');assert analysis['statistical_scope']==STATISTICAL_SCOPE;rows={};norms={};totalarrays=records=controls=copychecks=normchecks=0;reductionerror=0.
 for (c,i,w),item in sorted(ix.items()):
  name=f'gate__{c}__{w}__512__{i}.npz';assert item['file']==name and item['length']==512 and item['score_index']==1;digest(item['token_sha256'])
  assert item['original_sha256']==export['raw_inventory']['inputs/'+name]['sha256']
  with np.load(run/'inputs'/name,allow_pickle=False)as z:a={k:z[k]for k in z.files};assert len(z.files)==len(a)
  modes=validate_arrays(a,c,i,w);assert item['main_modes']==list(modes)and item['suffix_modes']==list(MODES);assert item['raw_array_fields']==sorted(a)
  assert {m:tuple(v)for m,v in item['norm_layers'].items()}=={m:norm_layers(m)for m in MODES}
  records+=len(modes);controls+=24;copychecks+=2+(2 if'repeat'in modes else 0);totalarrays+=len(a)
  for mode in modes:
   for view,sl in VIEWS.items():
    loss=a[mode+'__loss'][sl].astype('float64');p=a['P__loss'][sl].astype('float64');cc=a['C__loss'][sl].astype('float64')
    rows[c,i,w,view,mode]={'nll':float(loss.mean()),'delta_nll_p':float((loss-p).mean()),'delta_nll_c':float((loss-cc).mean()),'kl_c':float(a[mode+'__kl'][sl].mean(dtype=np.float64)),'kl_p':float(a[mode+'__kl_p'][sl].mean(dtype=np.float64)),'agreement_p':float(a[mode+'__agreement_p'][sl].mean()),'agreement_c':float(a[mode+'__agreement_c'][sl].mean())}
  for mode in MODES:
   for l in norm_layers(mode):
    for view,sl in NORMVIEWS.items():
     vals={}
     for f in norm_fields(mode):
      v=a[f'norm__{mode}__{l}__{f}'][sl].astype('float64');vals[f]={'mean':float(v.mean()),'rms':float(np.sqrt(np.square(v).mean())),'max_abs':float(np.max(np.abs(v)))};normchecks+=3
     norms[c,i,w,mode,l,view]=vals
  evidence=read(run/'inputs'/name.replace('.npz','.json'));assert evidence['metadata']=={k:item[k]for k in evidence['metadata']}
  # These are recorded comparisons/hashes, not raw context reconstruction.
  expectedchecks={m+'_source_identity'for m in MODES if m!='C'}|{m+'_singleton'for m in MODES if m.endswith('_L10')}|{'current_closure','fullrank_closure'}
  if'repeat'in modes:expectedchecks|={'repeat','identity'}
  ck=keyed(evidence['checks'],('trial',),{(k,)for k in expectedchecks});assert len(ck)==len(expectedchecks)
  for r in evidence['checks']+evidence['suffix_checks']:
   for field in ('corpus','identity','window','length'):assert r[field]==item[field]
   flags={k:v for k,v in r.items()if k.endswith('_exact')};assert flags and all(v is True for v in flags.values())
  for r in evidence['checks']:
   if r['trial'].endswith('_source_identity'):
    required={'source_exact'};assert r['reference_fingerprints']==r['trial_fingerprints']
   elif r['trial'].endswith('_singleton'):required={'b_exact','mass_exact','context_exact','coefficient_exact'}
   else:
    required={k+'_exact'for k in ('loss','kl','pred','kl_p','agreement_p','hidden','logits')};assert r['logits_reference']==r['logits_control']
   assert required<=set(r)and all(r[k]is True for k in required)
  for r in evidence['suffix_checks']:
   required={k+'_exact'for k in ('logits','hidden','gates','source','mask','contexts','loss','pred','kl','kl_p','agreement_p')}
   assert required<=set(r)and all(r[k]is True for k in required)
   assert r['logits_reference']==r['logits_control']and r['reference_context_hashes']==r['control_context_hashes']
  keyed(evidence['suffix_checks'],('trial',),{(m,)for m in MODES})
  for group,names in [(evidence['operations'],modes),(evidence['suffix_operations'],MODES)]:
   ops=keyed(group,('trial',),{(m,)for m in names})
   for op in group:
    assert set(op['gate_checks'])==({str(l)for l in LAYERS}if op['trial']not in ('C','repeat','identity')else set())
    assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30
    assert math.isfinite(op['max_context_error'])and op['max_context_error']<=1e-4
    for g in op['gate_checks'].values():
     assert {k+'_error'for k in ('conditional','native_reconstruction','mixed_reconstruction','sigmoid_context','fullrank_logit','fullrank_mass','fullrank_context','formula')}<=set(g)
     assert g['finite']and g['source_context_exact']and g.get('unaffected_context_exact',True)and g['min_mass']>0
     for k,v in g.items():
      if k.endswith('_error'):assert math.isfinite(v)and v<=(1e-6 if k=='norm_match_error'else 1e-4)
     assert all(math.isfinite(v)and v<=1e-4 for v in g.get('center_shift_errors',{}).values())
  del a
 assert records==776 and controls==768 and copychecks==72 and totalarrays==export['exported_array_count']
 sr=keyed(analysis['window_metrics'],('corpus','identity','window','view','mode'),rows)
 for key,vals in rows.items():
  for f,v in vals.items():reductionerror=max(reductionerror,close(v,sr[key][f]))
 nr=keyed(analysis['norm_summary'],('corpus','identity','window','mode','layer','view'),norms)
 for key,vals in norms.items():
  assert set(nr[key])==set(('corpus','identity','window','mode','layer','view'))|set(vals)
  for f,stats in vals.items():
   for k,v in stats.items():reductionerror=max(reductionerror,close(v,nr[key][f][k]))
 ids=np.random.default_rng(20261021).integers(0,4,(2000,4));booterror=0.
 def stat(values,saved):
  nonlocal booterror
  x=np.asarray(values);assert saved['n']==4;close(x,saved['per_window']);got=[float(x.mean()),*np.quantile(x[ids].mean(1),[.025,.975])];booterror=max(booterror,close(got,[saved['mean'],*saved['ci95']]))
 summaryuniverse={(c,i,v,m)for c in CORPORA for i in STARTS for v in VIEWS for m in MODES};ss=keyed(analysis['summary'],('corpus','identity','view','mode'),summaryuniverse);summarychecks=0
 for (c,i,v,m),s in ss.items():
  for metric in METRICS:stat([rows[c,i,w,v,m][metric]for w in range(4)],s[metric]);summarychecks+=1
  close([math.exp(s['delta_nll_p']['mean']),*np.exp(s['delta_nll_p']['ci95'])],[s['ppl_ratio_p']['point'],*s['ppl_ratio_p']['ci95']])
 names=('center_at_O','center_at_F','direction_at_O','direction_at_F','interaction','M8_minus_OO','M8_minus_FF');contrastuniverse={(c,i,v,f,s,n)for c in CORPORA for i in STARTS for v in VIEWS for f in FAMILIES for s in SETS for n in names}|{(c,i,v,'matched_control',s,'remove_minus_orthogonal_'+cell)for c in CORPORA for i in STARTS for v in VIEWS for s in SETS for cell in (*CELLS,'M8')}
 contrasts=keyed(analysis['contrasts'],('corpus','identity','view','family','layer_set','name'),contrastuniverse)
 for (c,i,v,f,s,n),r in contrasts.items():
  terms=contrast_terms(f,n,s);assert terms==r['terms']
  scope='secondary_M8'if f=='matched_control'and n.endswith('_M8')else('secondary_mixture_and_sample_count_confound'if n.startswith('M8_')else'primary_within_Wiki4');assert r['scope']==scope
  stat([sum(weight*rows[c,i,w,v,mode]['nll']for mode,weight in terms.items())for w in range(4)],r['delta_nll'])
 calibration=calibration_audit(run)
 assert summarychecks==4032 and len(contrasts)==912 and len(norms)==9408 and normchecks==162432
 return {'status':'passed','input_shards':32,'main_records':records,'suffix_records':controls,'exact_copy_closures':copychecks,'arrays':totalarrays,'calibration':calibration,'summary_metric_groups':summarychecks,'paired_contrasts':len(contrasts),'norm_summary_scalar_checks':normchecks,'max_reduction_error':reductionerror,'max_bootstrap_error':booterror,'auditor_sha256':sha(Path(__file__)),'export_manifest_sha256':sha(root/'manifests/wiki_calibration_numeric_export.json'),'wall_seconds':time.time()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'NumPy-only exported metrics/norms/recorded runtime evidence and provenance hashes. No raw attention/readout reconstruction, corpus/token freshness replay, model load or pretrained forward.'}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path);args=parser.parse_args();result=audit(args.repo.resolve());text=json.dumps(result,indent=2,allow_nan=False)+'\n'
 if args.output:assert not args.output.exists();args.output.write_text(text)
 print(text)
