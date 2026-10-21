"""CPU saved-evidence audit and fixed paired summaries; no model forward.

Actual GPU-logit FP64 J/R values are the primary diagnostic. Independent CPU
readout reconstruction is separate evidence, not an exact replay claim.
"""
import argparse,json,math,resource
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
from run_transfer import dump,sha,th
from gpt2_gate_subspace_v2 import fingerprint
from gpt2_gate_localization import localization_map
from gpt2_gate_additive import (CONDITIONS,FAMILIES,SETS,LAYERS,CORPUS_COUNTS,INPUT_HASHES,READOUT_ROWS,
    ERROR_TOLERANCE,NORM_TOLERANCE,LOSS64_FP32_SANITY_TOLERANCE,mode_spec,ESTIMATED_OUTPUT_BYTES,ANALYSIS_ALLOWANCE_BYTES)
from analyze_gate_localization import load,paired,dual_aggregate,scalars,CONTRASTS,nonadditivity
from gate_additive_readout import AdditiveReadout

VIEWS={'full':slice(1,511),'early_2_127':slice(1,127),'later_128_511':slice(127,511)}
AD_FIELDS={'loss_p','loss_joint','loss_additive','singleton_losses','sum_singleton_delta','J','R','joint_minus_sum','identity_error',
 'kl_joint_additive','kl_additive_joint','kl_p_additive','kl_native_additive','centered_logit_l2','centered_logit_rms','centered_logit_max_abs',
 'joint_delta_centered_l2','additive_delta_centered_l2','agreement_joint_additive','agreement_p_additive','agreement_native_additive'}

def read(p):return json.loads(p.read_text())
def exact(records,key,expected):
 out={key(r):r for r in records};assert len(out)==len(records)and set(out)==set(expected);return out

METRIC_DTYPES={'loss':torch.float32,'kl':torch.float32,'kl_p':torch.float32,'pred':torch.int64,'agreement_p':torch.bool}
EXACT_METRICS={k+'_exact'for k in METRIC_DTYPES}
def require_flags(record,names):
 assert {k for k in record if k.endswith('_exact')}==set(names)
 assert all(record[k] is True for k in names)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);p=ap.parse_args().run;torch.set_num_threads(4)
 analysis_names={'audit.json','summary.json','segment_summary.json','paired_contrasts.json','paired_summary.json','fp32_nonadditivity.json','fp32_nonadditivity_summary.json','additive_windows.json','additive_summary.json'}
 def analysis_dump(path,data):
  assert path.parent==p and path.name in analysis_names
  encoded=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
  others=sum(f.stat().st_size for f in p.rglob('*')if f.is_file()and f!=path)
  analysis_others=sum((p/n).stat().st_size for n in analysis_names if (p/n).is_file()and p/n!=path)
  assert analysis_others+len(encoded)<=ANALYSIS_ALLOWANCE_BYTES
  assert others+len(encoded)<=ESTIMATED_OUTPUT_BYTES
  path.write_bytes(encoded)
 analysis_dump(p/'audit.json',{'status':'checking'})
 m=read(p/'manifest.json');assert m['status']==m['phase']=='completed'and m['study']=='gate_additive'
 assert m['forward_passes']==m['forward_attempts']==1992 and m['records']==m['evaluation_forwards']==1000 and m['suffix_forwards']==992
 assert m['calibration_forwards']==0 and m['inherited_calibration_forwards']==8
 assert m['model_hash_before']==m['model_hash_after']and m['parameter_versions_unchanged']and m['calibration_maps_unchanged']
 assert m['precision_settings_before']==m['precision_settings_after']and m['physical_execution_lengths']==[512]
 assert m['context_tolerance']==ERROR_TOLERANCE and read(p/'conditions.json')==list(CONDITIONS)
 assert m['readout']['query_rows']==list(READOUT_ROWS)and m['readout']['bias']is None
 assert m['readout']['weight']['shape']==[50257,768]and m['readout']['weight']['dtype']=='torch.float32'
 for n,h in m['source_hashes'].items():assert sha(p/'frozen_sources'/n)==h and sha(Path(__file__).parent/n)==h
 for n,h in m['input_dependencies'].items():assert sha(p/n)==h
 assert all(m['input_dependencies'][n]==h for n,h in INPUT_HASHES.items())
 dep=read(p/'calibration_dependency.json');assert sha(p/'calibration_dependency.json')==m['calibration_dependency_sha256']
 for n,h in dep['copied_files'].items():assert sha(p/n)==h
 tokens=load(p/'tokens.pt');dm=read(p/'data_manifest.json');tokenmap={(d['corpus'],d['window']):d['tokens']for d in tokens}
 expected_inputs=[(c,w,i)for c in CORPUS_COUNTS for w in range(4)for i in ('natural','bos')]
 def stem(c,w,i):return f'gate__{c}__{w}__512__{i}'
 expected={stem(c,w,i)+'__'+t for c,w,i in expected_inputs for t in CONDITIONS}
 expected|={stem(c,0,'natural')+'__'+t for c in CORPUS_COUNTS for t in ('repeat','identity')}
 rows=[json.loads(x)for x in(p/'windows.jsonl').read_text().splitlines()];rr=exact(rows,lambda r:r['key'],expected)
 raw=load(p/'per_token.pt');assert set(raw)==expected
 groups=defaultdict(list);segments=defaultdict(list);inputrows=defaultdict(dict)
 for key,r in rr.items():
  assert key==stem(r['corpus'],r['window'],r['identity'])+'__'+r['trial']
  assert r['length']==r['input_tokens']==512 and r['score_index']==1 and r['target_start']==2 and r['scored_tokens']==510
  x=tokenmap[r['corpus'],r['window']].clone()
  if r['identity']=='bos':x[0]=dm['literals']['start_special']['id']
  z=raw[key];s=key.rsplit('__',1)[0];c,ref=raw[s+'__C'],raw[s+'__P']
  assert th(x)==r['token_sha256']==z['token_sha256']and z['score_index']==1 and r['exposed_passage']is False
  for k,dt in METRIC_DTYPES.items():assert z[k].shape==(511,)and z[k].dtype==dt and torch.isfinite(z[k]).all()
  for k,v in scalars(z,c,ref,VIEWS['full']).items():assert abs(v-r[k])<1e-12
  assert torch.equal(z['agreement_p'],z['pred'].eq(ref['pred']))
  if r['trial']in('current','fullrank','repeat','identity'):
   ref=ref if r['trial']in('current','fullrank')else c
   assert z['final_hidden_sha256']==ref['final_hidden_sha256']
   assert all(torch.equal(z[k],ref[k])for k in ('loss','kl','kl_p','pred','agreement_p'))
  inputrows[r['corpus'],r['identity'],512,r['window']][r['trial']]=r
  if r['trial']in CONDITIONS:
   groups[r['corpus'],r['identity'],r['trial']].append(r)
   for v,sl in VIEWS.items():
    if v!='full':segments[r['corpus'],r['identity'],r['trial'],v].append({**scalars(z,c,raw[s+'__P'],sl),'window':r['window'],'exposed_passage':False})
 assert len(groups)==248 and all(len(v)==4 for v in groups.values())
 assert len(segments)==496 and all(len(v)==4 for v in segments.values())
 analysis_dump(p/'summary.json',[{'corpus':c,'identity':i,'trial':t,**dual_aggregate(v)}for(c,i,t),v in groups.items()])
 analysis_dump(p/'segment_summary.json',[{'corpus':c,'identity':i,'trial':t,'view':s,**dual_aggregate(v)}for(c,i,t,s),v in segments.items()])
 contrasts=[];cg=defaultdict(list)
 for(c,i,n,w),rs in inputrows.items():
  for name,a,b in CONTRASTS:
   d=rs[a]['nll']-rs[b]['nll'];contrasts.append({'corpus':c,'identity':i,'window':w,'trial':name,'delta_nll':d});cg[c,i,name].append(d)
 analysis_dump(p/'paired_contrasts.json',contrasts);analysis_dump(p/'paired_summary.json',[{'corpus':c,'identity':i,'trial':t,**paired(v)}for(c,i,t),v in cg.items()])
 nr,ns=nonadditivity(inputrows);analysis_dump(p/'fp32_nonadditivity.json',nr);analysis_dump(p/'fp32_nonadditivity_summary.json',ns)
 expected_checks={stem(c,w,i)+'__'+t+'_source_identity'for c,w,i in expected_inputs for t in CONDITIONS if t!='C'}
 expected_checks|={stem(c,w,i)+'__'+t+'_singleton'for c,w,i in expected_inputs for t in CONDITIONS if len(mode_spec(t)[1])==1}
 expected_checks|={stem(c,w,i)+'__'+t+'_closure'for c,w,i in expected_inputs for t in ('current','fullrank')}
 expected_checks|={stem(c,0,'natural')+'__'+t for c in CORPUS_COUNTS for t in ('repeat','identity')}
 checks=read(p/'checks.json');exact(checks,lambda r:stem(r['corpus'],r['window'],r['identity'])+'__'+r['trial'],expected_checks)
 assert len(checks)==1800
 for r in checks:
  t=r['trial']
  if t.endswith('_source_identity'):flags={'source_kv_exact','source_q_exact'}
  elif t.endswith('_singleton'):
   flags={'b_exact','mass_exact','context_exact'}
   if t.startswith(('remove_pc1','orthogonal_control')):flags.add('s_exact')
  else:
   flags=EXACT_METRICS|{'final_hidden_exact','logits_exact'}
   assert r['logits_reference']==r['logits_control']
  require_flags(r,flags)
 ops=read(p/'operation_checks.json');oi=exact(ops,lambda r:stem(r['corpus'],r['window'],r['identity'])+'__'+r['trial'],expected)
 suffix_keys={stem(c,w,i)+'__'+t for c,w,i in expected_inputs for t in CONDITIONS}
 so=exact(read(p/'suffix_operations.json'),lambda r:stem(r['corpus'],r['window'],r['identity'])+'__'+r['trial'],suffix_keys)
 sc=exact(read(p/'suffix_checks.json'),lambda r:stem(r['corpus'],r['window'],r['identity'])+'__'+r['trial'],suffix_keys)
 guard_count=0;maxima=defaultdict(float);minmass=1.
 for op in list(oi.values())+list(so.values()):
  assert set(op['gate_checks'])==(set()if op['trial']in('C','repeat','identity')else {str(l)for l in LAYERS})
  assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30
  assert op['max_context_error']<=ERROR_TOLERANCE
  for ch in op['gate_checks'].values():
   assert {k for k in ch if k.endswith('_error')}=={'norm_match_error','conditional_error','native_reconstruction_error','mixed_reconstruction_error','sigmoid_context_error','fullrank_logit_error','fullrank_mass_error','fullrank_context_error','formula_error'}
   guard_count+=1;assert ch['finite']and ch['source_context_exact']and ch['unaffected_context_exact']and ch['min_mass']>0
   assert ch['norm_match_error']<=NORM_TOLERANCE;minmass=min(minmass,ch['min_mass'])
   for k,v in ch.items():
    if k.endswith('_error'):assert math.isfinite(v)and v<=ERROR_TOLERANCE;maxima[k]=max(maxima[k],v)
   maxima['max_gain']=max(maxima['max_gain'],ch['max_gain'])
 assert guard_count==15360
 maps=load(p/'calibration_maps.pt');directions=load(p/'directions.pt');sources=load(p/'source_states.pt')
 assert set(sources)=={stem(c,w,i)+'__P'for c,w,i in expected_inputs}
 maps_checked=hidden_count=sampled_rows=additive_count=0;maxmap=maxidentity=maxlossdiff=0.;adrows=[];adgroups=defaultdict(list)
 for folder,indexname,count in (('gates','gate_files.json',32),('suffix','suffix_files.json',32),('prefix','prefix_files.json',32),('hidden','hidden_files.json',32),('additive','additive_files.json',32),('readout_rows','readout_files.json',4)):
  index=read(p/indexname);assert len(index)==count
  for name,h in index.items():assert sha(p/folder/name)==h
 for c,w,i in expected_inputs:
  s=stem(c,w,i);pair=f'gate__{c}__{w}__{i}';gs=load(p/'gates'/(s+'.pt'));hs=load(p/'hidden'/(s+'.pt'));ss=load(p/'suffix'/(pair+'.pt'));pr=read(p/'prefix'/(pair+'.json'))
  ts=set(CONDITIONS)|({'repeat','identity'}if w==0 and i=='natural'else set())
  assert set(gs)==set(pr)==ts and set(ss)==set(CONDITIONS)
  assert set(hs)==set(CONDITIONS)-{'current','fullrank'}
  for t,h in hs.items():
   assert h.shape==(1,512,768)and h.dtype==torch.float32 and torch.isfinite(h).all()
   assert fingerprint(h)['sha256']==raw[s+'__'+t]['final_hidden_sha256'];hidden_count+=1
  source=sources[s+'__P'];sourcefp={'kv':{str(l):{k:fingerprint(v)for k,v in d.items()}for l,d in source['kv'].items()},'q':{str(l):fingerprint(v)for l,v in source['q'].items()}}
  for t in ts:
   key=s+'__'+t;alias='P'if t in('current','fullrank')else'C'if t in('repeat','identity')else t
   assert pr[t]['hidden']==fingerprint(hs[alias][:,:128])
   affected=mode_spec(t)[1]if t in CONDITIONS else();layers=set(LAYERS)if t=='P'else set(affected)
   assert set(gs[t])==layers
   if t not in('C','repeat','identity'):assert oi[key]['source_fingerprints']==sourcefp
   for l,d in gs[t].items():
    assert set(d)=={'b','mass','target_b','target_mass'}and all(v.shape==(12,511)and v.dtype==torch.float32 and torch.isfinite(v).all()for v in d.values())
    assert {k:fingerprint(v)for k,v in d.items()}==oi[key]['gate_fingerprints'][str(l)]
    if len(affected)==1:assert torch.equal(d['b'],gs['P'][l]['b'])and torch.equal(d['mass'],gs['P'][l]['mass'])
    b,a,_,_,_,ne=localization_map(t,l,d['b'],d['mass'],{k:v.float()for k,v in maps[l].items()},{k:directions[l][k].float()for k in('u','v')})
    err=max(float((b-d['target_b']).abs().max()),float((a-d['target_mass']).abs().max()));maxmap=max(maxmap,err)
    assert err<=ERROR_TOLERANCE and ne<=NORM_TOLERANCE and (d['mass']>0).all();maps_checked+=1
   if t not in CONDITIONS:continue
   ch=sc[key];require_flags(ch,EXACT_METRICS|{'logits_exact','hidden_exact','gates_exact','source_q_exact','source_kv_exact','mask_exact','contexts_exact'})
   assert ch['logits_reference']==ch['logits_control']and ch['reference_context_hashes']==ch['control_context_hashes']
   assert ch['reference_context_hashes']==oi[key]['prefix_context_hashes']and ch['control_context_hashes']==so[key]['prefix_context_hashes']
   d=ss[t];assert d['hidden']==pr[t]['hidden']and set(d['gates'])==layers
   for l,state in d['gates'].items():
    for k,v in state.items():assert v.shape==(12,127)and torch.equal(v,gs[t][l][k][:,:127])
   assert set(d['raw'])==set(METRIC_DTYPES)
   for k,v in d['raw'].items():assert v.dtype==METRIC_DTYPES[k]and torch.equal(v,raw[key][k][:len(v)])and v.shape==((127 if k=='loss'else 128),)
   assert json.loads(json.dumps(d['source_fingerprints']))==so[key]['source_fingerprints']
  ad=load(p/'additive'/(s+'.pt'));assert set(ad)==set(FAMILIES)
  for f,d in ad.items():
   additive_count+=1;assert d['singleton_losses'].shape==(8,511)
   assert set(d)==AD_FIELDS|{'baseline_logit_fingerprint','joint_logit_fingerprint','additive_logit_fingerprint','fp64_vs_fp32_loss_sanity'}
   for k,v in d.items():
    if isinstance(v,torch.Tensor):assert v.dtype==(torch.bool if k.startswith('agreement_')else torch.float64)and torch.isfinite(v).all()and(v.shape==(511,)or k=='singleton_losses')
   nonadd=d['loss_joint']-d['loss_p']-(d['singleton_losses']-d['loss_p']).sum(0)
   assert (nonadd-d['joint_minus_sum']).abs().max()<1e-12
   assert (d['J']-(d['loss_joint']-d['loss_additive'])).abs().max()<1e-12
   assert (d['R']-(d['loss_additive']-d['loss_p']-(d['singleton_losses']-d['loss_p']).sum(0))).abs().max()<1e-12
   err=float((d['J']+d['R']-nonadd).abs().max());maxidentity=max(maxidentity,err);assert err<1e-12
   pairs=[('P',d['loss_p']),(f+'_all',d['loss_joint'])]+[(f+'_L'+str(l),d['singleton_losses'][n])for n,l in enumerate(LAYERS)]
   diffs={t:float((a-raw[s+'__'+t]['loss'].double()).abs().max())for t,a in pairs}
   assert diffs==d['fp64_vs_fp32_loss_sanity']['max_absolute_errors']and max(diffs.values())<=LOSS64_FP32_SANITY_TOLERANCE
   maxlossdiff=max(maxlossdiff,max(diffs.values()))
   for view,sl in VIEWS.items():
    vals={k:float(v[sl].double().mean())for k,v in d.items()if isinstance(v,torch.Tensor)and v.shape==(511,)}
    record={'corpus':c,'window':w,'identity':i,'family':f,'view':view,**vals};adrows.append(record)
    for metric,value in vals.items():adgroups[c,i,f,view,metric].append(value)
  if w==0 and i=='natural':
   q=load(p/'readout_rows'/(s+'.pt'));assert q['query_rows'].tolist()==list(READOUT_ROWS)
   x=tokenmap[c,w];assert torch.equal(q['targets'],x[q['query_rows']+1])
   assert set(q)=={'query_rows','targets',*CONDITIONS,*[f+'_additive'for f in FAMILIES]}
   assert q['query_rows'].dtype==q['targets'].dtype==torch.int64
   for t in CONDITIONS:assert q[t].dtype==torch.float32 and q[t].shape==(len(READOUT_ROWS),50257)and torch.isfinite(q[t]).all()
   for f in FAMILIES:
    assert q[f+'_additive'].dtype==torch.float64 and q[f+'_additive'].shape==(len(READOUT_ROWS),50257)and torch.isfinite(q[f+'_additive']).all()
    a=AdditiveReadout(q['P'],q['targets'])
    for l in LAYERS:a.add_singleton(l,q[f+'_L'+str(l)])
    dd=a.finish(q[f+'_all'],q['C']);assert torch.equal(a.additive,q[f+'_additive'])
    for metric,v in dd.items():
     actual=ad[f][metric][:,list(READOUT_ROWS)]if metric=='singleton_losses'else ad[f][metric][list(READOUT_ROWS)]
     assert torch.allclose(v,actual,atol=1e-11,rtol=0),(c,f,metric)
   sampled_rows+=len(CONDITIONS)*len(READOUT_ROWS)
  del hs,gs,ss,ad
 assert hidden_count==928 and maps_checked==1792 and additive_count==96 and sampled_rows==len(CONDITIONS)*4*len(READOUT_ROWS)
 snaps=load(p/'attention_snapshots.pt');assert set(snaps)=={'gate__wiki_validation__0__512__natural__'+t for t in('P','remove_pc1_all')}
 assert all(set(v)=={4,11}for v in snaps.values())
 analysis_dump(p/'additive_windows.json',adrows)
 summary=[];bootstrap_identity_max=0.
 for(c,i,f,v,metric),values in adgroups.items():
  stat=paired(values);summary.append({'corpus':c,'identity':i,'family':f,'view':v,'metric':metric,'n':stat['n'],
   'mean':stat['mean_delta_nll'],'ci95':stat['ci95_delta_nll'],'per_window':stat['per_window_delta_nll']})
  if metric=='J':
   jj,rr,nn=[np.array(adgroups[c,i,f,v,k])for k in('J','R','joint_minus_sum')]
   choices=np.random.default_rng(20261021).integers(0,4,(2000,4))
   error=float(np.max(np.abs(jj[choices].mean(1)+rr[choices].mean(1)-nn[choices].mean(1))))
   bootstrap_identity_max=max(bootstrap_identity_max,error);assert error<1e-12
 assert len(adrows)==288 and len(summary)==1440
 analysis_dump(p/'additive_summary.json',summary)
 output_bytes=sum(f.stat().st_size for f in p.rglob('*')if f.is_file());assert output_bytes<=ESTIMATED_OUTPUT_BYTES
 audit={'status':'passed','records':1000,'forwards':1992,'gate_runtime_checks':guard_count,'raw_main_gate_states':maps_checked,'hidden_states':hidden_count,'raw_gpu_logit_query_rows':sampled_rows,'additive_inputs_families':additive_count,'additive_window_views':len(adrows),'max_cpu_gate_error':maxmap,'max_J_R_identity_error':maxidentity,'max_FP64_FP32_loss_difference':maxlossdiff,'minimum_mass':minmass,'gate_runtime_maxima':dict(maxima),'output_bytes_before_audit':output_bytes,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Production saved-evidence audit, not independent review. All-query hidden-to-logit CPU reconstruction and fixed attention snapshots require separate independent audit. Main compact gate fingerprints are bound to saved raw states; suffix full gate fingerprints and source/context closures outside saved tensors are runtime/hash evidence. Bootstrap conditional on fixed calibration; correlated starts/views, no multiplicity correction.'}
 audit['paired_bootstrap_identity_max_error']=bootstrap_identity_max
 analysis_dump(p/'audit.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
