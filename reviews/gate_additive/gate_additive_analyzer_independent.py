"""Full-cardinality fabricated analyzer fixture; alias-backed tensors, NOT a storage benchmark."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
import sys,json,math,tempfile,shutil,subprocess,resource,copy
import torch,numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P));torch.set_num_threads(1)
from run_transfer import dump,sha,th
from gpt2_gate_subspace_v2 import fingerprint
from gpt2_gate_additive import CONDITIONS,FAMILIES,LAYERS,CORPUS_COUNTS,INPUT_HASHES,READOUT_ROWS,compact_gates,mode_spec
from gpt2_gate_localization import localization_map
from analyze_gate_localization import scalars,paired
from gate_additive_readout import AdditiveReadout
V=50257
with tempfile.TemporaryDirectory(prefix='additive-analyzer-independent-')as td:
 p=Path(td)
 for f in ('gates','suffix','prefix','hidden','additive','readout_rows','frozen_sources'):(p/f).mkdir()
 for n in INPUT_HASHES:shutil.copyfile(P/'additive_inputs'/n,p/n)
 for n in ('directions.pt','data_manifest.json','tokens.pt'):
  if not(p/n).exists():shutil.copyfile(P/'additive_inputs'/n,p/n)
 shutil.copyfile(P/'gpt2_gate_fixed512_run/calibration_maps.pt',p/'calibration_maps.pt')
 dep={'copied_files':{'calibration_maps.pt':sha(p/'calibration_maps.pt')}};dump(p/'calibration_dependency.json',dep)
 maps=torch.load(p/'calibration_maps.pt',weights_only=True,map_location='cpu');dirs=torch.load(p/'directions.pt',weights_only=True,map_location='cpu');tokens=torch.load(p/'tokens.pt',weights_only=True,map_location='cpu');dm=json.loads((p/'data_manifest.json').read_text());tm={(i['corpus'],i['window']):i['tokens']for i in tokens}
 dump(p/'conditions.json',list(CONDITIONS));h=torch.zeros(1,512,768);hf=fingerprint(h);pf=fingerprint(h[:,:128]);context={str(l):{k:pf for k in ('current','delivered')}for l in LAYERS}
 source={'q':{l:torch.zeros(12,64)for l in LAYERS},'kv':{l:{k:torch.zeros(768)for k in ('k','v')}for l in LAYERS}}
 sourcefp={'kv':{str(l):{k:fingerprint(v)for k,v in d.items()}for l,d in source['kv'].items()},'q':{str(l):fingerprint(v)for l,v in source['q'].items()}}
 common={};opcommon={}
 for t in CONDITIONS:
  common[t]={};opcommon[t]={}
  if t=='C':continue
  for l in LAYERS:
   b=torch.log(torch.arange(1,512).float())[None,:].expand(12,-1).contiguous()+maps[l]['alpha'].float()[:,None]+.1;a=b.sigmoid()
   tb,ta,_,_,_,ne=localization_map(t,l,b,a,{k:v.float()for k,v in maps[l].items()},{k:dirs[l][k].float()for k in ('u','v')})
   common[t][l]={'b':b,'mass':a,'target_b':tb,'target_mass':ta}
   opcommon[t][str(l)]={'finite':True,'source_context_exact':True,'unaffected_context_exact':True,'norm_match_error':ne,'min_mass':float(a.min()),'max_gain':float((ta/a).max()),**{k:0. for k in ('conditional_error','native_reconstruction_error','mixed_reconstruction_error','sigmoid_context_error','fullrank_logit_error','fullrank_mass_error','fullrank_context_error','formula_error')}}
 zero=torch.zeros(511);loss=torch.full((511,),math.log(V));loss64=torch.full((511,),math.log(V),dtype=torch.float64);pred=torch.zeros(511,dtype=torch.long);agree=torch.ones(511,dtype=torch.bool)
 # Uniform full-vocabulary logits give exact null diagnostics with nontrivial real target IDs.
 def diagnostic(f):
  d={k:torch.zeros(511,dtype=torch.float64)for k in ('sum_singleton_delta','J','R','joint_minus_sum','identity_error','kl_joint_additive','kl_additive_joint','kl_p_additive','kl_native_additive','centered_logit_l2','centered_logit_rms','centered_logit_max_abs','joint_delta_centered_l2','additive_delta_centered_l2')}
  d.update(loss_p=loss64,loss_joint=loss64,loss_additive=loss64,singleton_losses=loss64.expand(8,-1),agreement_joint_additive=agree,agreement_p_additive=agree,agreement_native_additive=agree)
  names=['P',f+'_all']+[f+'_L'+str(l)for l in LAYERS];err=float((loss64-loss.double()).abs().max())
  d.update(fp64_vs_fp32_loss_sanity={'max_absolute_errors':{t:err for t in names}},baseline_logit_fingerprint={'synthetic':True},joint_logit_fingerprint={'synthetic':True},additive_logit_fingerprint={'synthetic':True});return d
 ad={f:diagnostic(f)for f in FAMILIES};logits=torch.zeros(len(READOUT_ROWS),V);addlogits=logits.double()
 rows=[];raw={};checks=[];ops=[];sops=[];schecks=[];sources={};indices={f:{}for f in ('gates','suffix','prefix','hidden','additive','readout_rows')}
 def stem(c,w,i):return f'gate__{c}__{w}__512__{i}'
 for c in CORPUS_COUNTS:
  for w in range(4):
   for ident in ('natural','bos'):
    s=stem(c,w,ident);pair=f'gate__{c}__{w}__{ident}';x=tm[c,w].clone()
    if ident=='bos':x[0]=dm['literals']['start_special']['id']
    y=x.clone();y[128:]=torch.roll(y[128:],1)
    ts=list(CONDITIONS)+(['repeat','identity']if w==0 and ident=='natural'else[])
    zs={t:{'loss':loss,'kl':zero,'kl_p':zero,'pred':pred,'agreement_p':agree,'score_index':1,'token_sha256':th(x),'final_hidden_sha256':hf['sha256']}for t in ts}
    gs={};hs={t:h for t in CONDITIONS if t not in ('current','fullrank')};ss={};pr={t:{'hidden':pf}for t in ts};sources[s+'__P']=source
    for t in ts:
     base={'panel':'gate','corpus':c,'window':w,'identity':ident,'length':512,'trial':t,'exposed_passage':False};z=zs[t];raw[s+'__'+t]=z
     rows.append({**base,'key':s+'__'+t,'input_tokens':512,'target_start':2,'score_index':1,'scored_tokens':510,'token_sha256':th(x),**scalars(z,zs['C'],zs['P'],slice(1,None))})
     gs[t]=compact_gates(t,common.get(t,{}));op={**base,'gate_checks':opcommon.get(t,{}),'projection_checks':8,'query_checks':8,'context_checks':8,'writer_locality_checks':30,'max_context_error':0.,'prefix_context_hashes':context,'source_fingerprints':sourcefp,'gate_fingerprints':{str(l):{k:fingerprint(v)for k,v in d.items()}for l,d in common.get(t,{}).items()}};ops.append(op)
     if t not in ('C','repeat','identity'):checks.append({**base,'trial':t+'_source_identity','source_q_exact':True,'source_kv_exact':True})
     if t in ('current','fullrank'):checks.append({**base,'trial':t+'_closure',**{k+'_exact':True for k in ('loss','kl','pred','kl_p','agreement_p')},'final_hidden_exact':True,'logits_exact':True,'logits_reference':{'synthetic':'same'},'logits_control':{'synthetic':'same'}})
     if t in ('repeat','identity'):checks.append({**base,**{k+'_exact':True for k in ('loss','kl','pred','kl_p','agreement_p')},'final_hidden_exact':True,'logits_exact':True,'logits_reference':{'synthetic':'same'},'logits_control':{'synthetic':'same'}})
     if t in CONDITIONS and len(mode_spec(t)[1])==1:checks.append({**base,'trial':t+'_singleton','b_exact':True,'mass_exact':True,'context_exact':True,**({'s_exact':True}if mode_spec(t)[0]in ('remove_pc1','orthogonal_control')else {})})
     if t not in CONDITIONS:continue
     sops.append(op);lf={'shape':[128,V],'dtype':'torch.float32','sha256':'f'*64}
     schecks.append({**base,'logits_reference':lf,'logits_control':lf,'reference_context_hashes':context,'control_context_hashes':context,'logits_exact':True,'hidden_exact':True,'gates_exact':True,'source_q_exact':True,'source_kv_exact':True,'mask_exact':True,'contexts_exact':True,'loss_exact':True,'pred_exact':True,'kl_exact':True,'kl_p_exact':True,'agreement_p_exact':True})
     ss[t]={'hidden':pf,'gates':compact_gates(t,common[t],prefix=True),'source_fingerprints':sourcefp,'raw':{k:z[k][:(127 if k=='loss'else 128)]for k in ('loss','pred','kl','kl_p','agreement_p')}}
    for folder,value,name in [('gates',gs,s+'.pt'),('hidden',hs,s+'.pt'),('suffix',ss,pair+'.pt'),('additive',ad,s+'.pt')]:
     f=p/folder/name;torch.save(value,f);indices[folder][name]=sha(f)
    f=p/'prefix'/(pair+'.json');dump(f,pr);indices['prefix'][f.name]=sha(f)
    if w==0 and ident=='natural':
     q={'query_rows':torch.tensor(READOUT_ROWS),'targets':x[torch.tensor(READOUT_ROWS)+1],**{t:logits for t in CONDITIONS},**{f+'_additive':addlogits for f in FAMILIES}}
     f=p/'readout_rows'/(s+'.pt');torch.save(q,f);indices['readout_rows'][f.name]=sha(f)
 torch.save(raw,p/'per_token.pt');torch.save(sources,p/'source_states.pt');torch.save({stem('wiki_validation',0,'natural')+'__'+t:{4:{},11:{}}for t in ('P','remove_pc1_all')},p/'attention_snapshots.pt')
 (p/'windows.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
 for f,v in [('checks.json',checks),('operation_checks.json',ops),('suffix_operations.json',sops),('suffix_checks.json',schecks)]:dump(p/f,v)
 for folder,indexname in [('gates','gate_files.json'),('hidden','hidden_files.json'),('suffix','suffix_files.json'),('prefix','prefix_files.json'),('additive','additive_files.json'),('readout_rows','readout_files.json')]:dump(p/indexname,indices[folder])
 sourcehash={n:sha(P/n)for n in ('gpt2_gate_additive.py','gate_additive_readout.py','gate_additive_failure.py','analyze_gate_additive.py')}
 for n in sourcehash:shutil.copyfile(P/n,p/'frozen_sources'/n)
 manifest={'status':'completed','phase':'completed','study':'gate_additive','records':1000,'evaluation_forwards':1000,'forward_passes':1992,'forward_attempts':1992,'suffix_forwards':992,'calibration_forwards':0,'inherited_calibration_forwards':8,'model_hash_before':'a'*64,'model_hash_after':'a'*64,'parameter_versions_unchanged':True,'calibration_maps_unchanged':True,'precision_settings_before':{},'precision_settings_after':{},'physical_execution_lengths':[512],'context_tolerance':1e-4,'readout':{'query_rows':list(READOUT_ROWS),'bias':None,'weight':{'shape':[V,768],'dtype':'torch.float32'}},'source_hashes':sourcehash,'input_dependencies':INPUT_HASHES,'calibration_dependency_sha256':sha(p/'calibration_dependency.json')};dump(p/'manifest.json',manifest)
 print('FIXTURE_READY',sum(q.stat().st_size for q in p.rglob('*')if q.is_file()),flush=True)
 def invoke():return subprocess.run([sys.executable,str(P/'analyze_gate_additive.py'),str(p)],capture_output=True,text=True)
 result=invoke()
 if result.returncode:print(result.stdout,result.stderr);raise AssertionError('valid fabricated fixture rejected')
 good=json.loads((p/'audit.json').read_text());assert good['status']=='passed'and good['hidden_states']==928
 assert len(json.loads((p/'summary.json').read_text()))==248 and len(json.loads((p/'segment_summary.json').read_text()))==496
 assert len(json.loads((p/'additive_summary.json').read_text()))==1440
 tests=[]
 def mutation(name,modify):
  f=p/name;old=f.read_bytes();v=json.loads(old);modify(v);dump(f,v);rr=invoke();f.write_bytes(old)
  assert rr.returncode!=0,(name,'corruption accepted');assert json.loads((p/'audit.json').read_text())['status']=='checking';tests.append(name)
 mutation('manifest.json',lambda v:v.update(status='failed'))
 mutation('checks.json',lambda v:v.__setitem__(0,v[1]))
 mutation('operation_checks.json',lambda v:next(x for x in v if x['trial']=='P')['gate_checks'].pop('4'))
 mutation('suffix_checks.json',lambda v:v.__setitem__(0,v[1]))
 mutation('checks.json',lambda v:v[0].pop('source_q_exact'))
 mutation('suffix_checks.json',lambda v:v[0].pop('contexts_exact'))
 mutation('operation_checks.json',lambda v:next(x for x in v if x['trial']=='P')['gate_fingerprints']['4']['b'].update(sha256='0'*64))
 old=(p/'windows.jsonl').read_bytes();rows[0]['nll']+=.1;(p/'windows.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows));bad=invoke();assert bad.returncode!=0;(p/'windows.jsonl').write_bytes(old);tests.append('window scalar')
 # Refresh the relevant shard digest so a raw algebra error cannot hide behind inventory checking.
 f=p/'additive'/(stem('wiki_validation',0,'natural')+'.pt');old=f.read_bytes();v=torch.load(f,weights_only=True,map_location='cpu');v['remove_pc1']['J']=v['remove_pc1']['J'].clone();v['remove_pc1']['J'][3]=.01;torch.save(v,f)
 indexfile=p/'additive_files.json';oldindex=indexfile.read_bytes();ix=json.loads(oldindex);ix[f.name]=sha(f);dump(indexfile,ix);bad=invoke();assert bad.returncode!=0;f.write_bytes(old);indexfile.write_bytes(oldindex);tests.append('J corruption with refreshed shard hash')
 # Evidence schemas must reject absent raw fields and wrong dtypes even with fresh hashes.
 def tensor_mutation(folder,name,indexname,modify,label):
  f=p/folder/name;before=f.read_bytes();v=torch.load(f,weights_only=True,map_location='cpu');modify(v);torch.save(v,f)
  ixfile=None;ixbefore=None
  if indexname:
   ixfile=p/indexname;ixbefore=ixfile.read_bytes();idx=json.loads(ixbefore);idx[name]=sha(f);dump(ixfile,idx)
  rr=invoke();f.write_bytes(before)
  if ixfile:ixfile.write_bytes(ixbefore)
  assert rr.returncode!=0,('corruption accepted',label)
  assert json.loads((p/'audit.json').read_text())['status']=='checking';tests.append(label)
 sn='gate__wiki_validation__0__natural.pt';an=stem('wiki_validation',0,'natural')+'.pt'
 tensor_mutation('suffix',sn,'suffix_files.json',lambda v:v['P']['raw'].pop('kl_p'),'missing suffix KL field')
 tensor_mutation('', 'per_token.pt',None,lambda v:v[stem('wiki_validation',0,'natural')+'__C'].update(pred=pred.float()),'main prediction dtype')
 tensor_mutation('additive',an,'additive_files.json',lambda v:v['remove_pc1'].update(J=v['remove_pc1']['J'].float()),'additive FP64 dtype')
 tensor_mutation('readout_rows',an,'readout_files.json',lambda v:v.update(C=v['C'].double()),'sampled actual-logit FP32 dtype')
 # Independent non-null paired bootstrap confirms fixed seed and n4; the fabricated null fixture alone cannot test that.
 a=np.array([-.13,.07,.41,.82]);choice=np.random.default_rng(20261021).integers(0,4,(2000,4));expected=np.quantile(a[choice].mean(1),[.025,.975]);assert paired(a)['ci95_delta_nll']==expected.tolist()
 out={'status':'passed','fabricated_only':True,'storage_aliases_used':True,'not_a_storage_or_model_fidelity_test':True,'valid_audit':good,'summary_counts':{'full':248,'segments':496,'additive_metric_groups':1440},'corruptions_rejected':tests,'non_null_n4_bootstrap_exact':True,'stale_pass_cleared':True,'source_sha256':sourcehash,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'model_loaded':False,'gpu_used':False}
 assert max(out['peak_rss_kib'],out['peak_child_rss_kib'])<3*1024**2
 Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
