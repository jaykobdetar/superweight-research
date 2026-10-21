"""Independent T17 CPU audit. No production runner/analyzer or pretrained model import.
Refuses scientific reads until run and production analysis are completed.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import argparse,json,math,resource,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from gpt2_gate_fixed512_numerical_audit import j,load,sha,th,close,metrics,dual
from gpt2_gate_localization_numerical_audit import formula,paired,spec
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'gpt2_gate_additive_run'
LAYERS=tuple(range(4,12));FAMILIES=('remove_pc1','orthogonal_control','position');CORPORA=('wiki_validation','mill','wollstonecraft','optimizers_new')
MODES=('C','P','current','fullrank')+tuple(f+'_'+s for f in FAMILIES for s in tuple(f'L{l}'for l in LAYERS)+('all',))
VIEWS={'full':slice(1,511),'early_2_127':slice(1,127),'later_128_511':slice(127,511)}
def fp(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':th(t)}
def main():
 start=time.time();m=j(P/'manifest.json');assert m.get('status')==m.get('phase')=='completed'and m.get('completed_utc')
 assert j(P/'audit.json')['status']=='passed';torch.set_num_threads(4)
 assert [m[k]for k in ('forward_passes','forward_attempts','records','evaluation_forwards','suffix_forwards','calibration_forwards')]==[1992,1992,1000,1000,992,0]
 frozen=j(ROOT/'review/gate_additive_implementation_readiness.json')['source_sha256']
 for n,h in m['source_hashes'].items():assert sha(ROOT/n)==sha(P/'frozen_sources'/n)==h
 for n,h in frozen.items():assert m['source_hashes'][n]==h
 for n,h in m['input_dependencies'].items():assert sha(ROOT/'additive_inputs'/n)==sha(P/n)==h
 assert j(ROOT/'review/gate_additive_actual_input_audit.json')['status']=='passed'
 assert m['model_hash_before']==m['model_hash_after']=='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527'and m['parameter_versions_unchanged']and m['calibration_maps_unchanged']
 assert m['precision_settings_before']==m['precision_settings_after']and m['physical_execution_lengths']==[512]
 dep=j(P/'calibration_dependency.json');assert sha(P/'calibration_dependency.json')==m['calibration_dependency_sha256']
 for n,h in dep['copied_files'].items():assert sha(P/n)==sha(ROOT/'gpt2_gate_fixed512_run'/n)==h
 assert j(ROOT/'review/gpt2_gate_fixed512_numerical_audit.json')['status']=='passed'
 assert tuple(j(P/'conditions.json'))==MODES
 maps=load(P/'calibration_maps.pt');directions=load(P/'directions.pt')
 for l in LAYERS:
  u=maps[l]['basis_r'][:,0];u=u/u.norm();v=maps[l]['basis_qr'][:,0]-u*(u@maps[l]['basis_qr'][:,0]);v=v/v.norm();v=v*(-1 if v[v.abs().argmax()]<0 else 1)
  assert torch.equal(u,directions[l]['u'])and torch.equal(v,directions[l]['v'])
 raw=load(P/'per_token.pt');rows=[json.loads(x)for x in(P/'windows.jsonl').read_text().splitlines()];assert len(raw)==len(rows)==len({r['key']for r in rows})==1000
 groups=defaultdict(list);segments=defaultdict(list);inputs=defaultdict(dict);closures=0
 for r in rows:
  k=r['key'];s=k.rsplit('__',1)[0];z=raw[k];c,p=raw[s+'__C'],raw[s+'__P']
  assert r['length']==r['input_tokens']==512 and r['score_index']==z['score_index']==1 and r['target_start']==2 and r['scored_tokens']==510 and not r['exposed_passage']
  assert z['token_sha256']==r['token_sha256']==c['token_sha256']==p['token_sha256']
  close(metrics(z,c,p,VIEWS['full']),r)
  assert torch.equal(z['agreement_p'],z['pred'].eq(p['pred']))
  if r['trial']in('current','fullrank','repeat','identity'):
   ref=p if r['trial']in('current','fullrank')else c
   for field in('loss','pred','kl','kl_p','agreement_p'):assert torch.equal(z[field],ref[field])
   assert z['final_hidden_sha256']==ref['final_hidden_sha256'];closures+=1
  inputs[r['corpus'],r['identity'],r['window']][r['trial']]=r
  if r['trial']in MODES:
   key=(r['corpus'],r['identity'],r['trial']);groups[key].append(r)
   for view in ('early_2_127','later_128_511'):segments[key+(view,)].append(metrics(z,c,p,VIEWS[view]))
 for c in CORPORA:
  for i in ('natural','bos'):
   for w in range(4):assert set(inputs[c,i,w])==set(MODES)|({'repeat','identity'}if w==0 and i=='natural'else set())
 assert len(groups)==248 and len(segments)==496 and closures==72
 for fn,gr in [('summary.json',groups),('segment_summary.json',segments)]:
  saved=j(P/fn);assert len(saved)==len(gr)
  for r in saved:
   key=(r['corpus'],r['identity'],r['trial'])+((r['view'],)if'view'in r else());close(dual(gr[key]),r)
 cg=defaultdict(list);cs=j(P/'paired_contrasts.json');assert len(cs)==864
 for r in cs:
  a,b=r['trial'].split('_minus_');b='orthogonal_control_'+a.rsplit('_',1)[1]if b=='orthogonal_control'else b;rs=inputs[r['corpus'],r['identity'],r['window']];delta=rs[a]['nll']-rs[b]['nll'];close(delta,r['delta_nll']);cg[r['corpus'],r['identity'],r['trial']].append(delta)
 ps=j(P/'paired_summary.json');assert len(ps)==216
 for r in ps:close(paired(cg[r['corpus'],r['identity'],r['trial']]),r)
 ng=defaultdict(list);nr=j(P/'fp32_nonadditivity.json');assert len(nr)==96
 for r in nr:
  rs=inputs[r['corpus'],r['identity'],r['window']];f=r['family'];joint=rs[f+'_all']['nll']-rs['P']['nll'];ss=[rs[f+'_L'+str(l)]['nll']-rs['P']['nll']for l in LAYERS];d=joint-sum(ss)
  close(dict(joint_delta_nll=joint,singleton_delta_nll=ss,sum_singleton_delta_nll=sum(ss),joint_minus_sum_delta_nll=d),r);ng[r['corpus'],r['identity'],f].append(d)
 for r in j(P/'fp32_nonadditivity_summary.json'):close(paired(ng[r['corpus'],r['identity'],r['family']]),r)
 result={'status':'checking','raw_records':1000,'closures':closures,'full_summary_groups':248,'segment_summary_groups':496,'paired_contrasts':864,'paired_groups':216,'calibration_audit_reused_exact_hashes':True,'source_hashes':frozen}
 evidence(m,raw,maps,directions,result)
 result.update(wall_seconds=time.time()-start,review_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 assert result['review_peak_rss_kib']<3*1024**2
 return result

def evidence(m,raw,maps,directions,result):
 ops=j(P/'operation_checks.json');sops=j(P/'suffix_operations.json');sc=j(P/'suffix_checks.json');checks=j(P/'checks.json')
 assert(len(ops),len(sops),len(sc),len(checks))==(1000,992,992,1800)
 assert all(v is True for c in checks+sc for k,v in c.items()if k.endswith('_exact'))
 opi={(v['corpus'],v['window'],v['identity'],v['trial']):v for v in ops};soi={(v['corpus'],v['window'],v['identity'],v['trial']):v for v in sops};ci={(v['corpus'],v['window'],v['identity'],v['trial']):v for v in sc}
 guard=0;maxima=defaultdict(float)
 for op in ops+sops:
  assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30 and op['max_context_error']<=1e-4
  for ch in op['gate_checks'].values():
   guard+=1;assert ch['finite']and ch['source_context_exact']and ch['unaffected_context_exact']and ch['min_mass']>0 and ch['norm_match_error']<=1e-6
   for k,v in ch.items():
    if k.endswith('_error'):assert math.isfinite(v)and v<=1e-4;maxima[k]=max(maxima[k],v)
   maxima['max_gain']=max(maxima['max_gain'],ch['max_gain'])
 assert guard==15360
 for folder,index in [('gates','gate_files.json'),('hidden','hidden_files.json'),('prefix','prefix_files.json'),('suffix','suffix_files.json'),('additive','additive_files.json'),('readout_rows','readout_files.json')]:
  ix=j(P/index);assert len(ix)==(4 if folder=='readout_rows'else 32)
  for n,h in ix.items():assert sha(P/folder/n)==h
 source=load(P/'source_states.pt');assert len(source)==32
 assert all(torch.count_nonzero(x['v'])==0 for state in source.values()for x in state['kv'].values())
 tokens={(v['corpus'],v['window']):v['tokens']for v in load(P/'tokens.pt')};bos=j(P/'data_manifest.json')['literals']['start_special']['id']
 hidden=states=suffixes=0;maperr=jrerr=0.;ag=defaultdict(list);aws={ (v['corpus'],v['window'],v['identity'],v['family'],v['view']):v for v in j(P/'additive_windows.json')};assert len(aws)==288
 for name in j(P/'gate_files.json'):
  s=Path(name).stem;_,c,w,n,i=s.split('__');w=int(w);pair=f'gate__{c}__{w}__{i}';gs=load(P/'gates'/name);hs=load(P/'hidden'/name);su=load(P/'suffix'/(pair+'.pt'));pr=j(P/'prefix'/(pair+'.json'));ad=load(P/'additive'/name)
  assert set(hs)==set(MODES)-{'current','fullrank'};assert set(su)==set(MODES)
  x=tokens[c,w].clone()
  if i=='bos':x[0]=bos
  y=x.clone();y[128:]=torch.roll(y[128:],1)
  src=source[s+'__P'];srcfp={'kv':{str(l):{k:fp(v)for k,v in d.items()}for l,d in src['kv'].items()},'q':{str(l):fp(v)for l,v in src['q'].items()}}
  for t,h in hs.items():assert h.shape==(1,512,768)and h.dtype==torch.float32 and torch.isfinite(h).all()and th(h)==raw[s+'__'+t]['final_hidden_sha256'];hidden+=1
  for t,g in gs.items():
   key=(c,w,i,t);full=s+'__'+t;alias='P'if t in('current','fullrank')else'C'if t in('repeat','identity')else t
   assert pr[t]['hidden']==fp(hs[alias][:,:128]);assert raw[full]['token_sha256']==th(x)
   if t not in ('C','repeat','identity'):assert opi[key]['source_fingerprints']==srcfp
   layers=set(LAYERS)if t=='P'else set(spec(t)[1]);assert set(g)==layers
   for l,d in g.items():
    states+=1;assert {k:fp(v)for k,v in d.items()}==opi[key]['gate_fingerprints'][str(l)]
    b,a,_,ne=formula(t,l,d['b'],d['mass'],{k:v.float()for k,v in maps[l].items()},{k:directions[l][k].float()for k in ('u','v')})
    e=max(float((b-d['target_b']).abs().max()),float((a-d['target_mass']).abs().max()));maperr=max(maperr,e);assert e<1e-4 and ne<=1e-6
    if len(spec(t)[1])==1:assert torch.equal(d['b'],gs['P'][l]['b'])and torch.equal(d['mass'],gs['P'][l]['mass'])
   if t not in MODES:continue
   suffixes+=1;ch=ci[key];z=su[t]
   assert ch['original_token_sha256']==th(x)and ch['control_token_sha256']==th(y)
   assert ch['logits_reference']==ch['logits_control']and ch['reference_context_hashes']==ch['control_context_hashes']==opi[key]['prefix_context_hashes']==soi[key]['prefix_context_hashes']
   assert z['hidden']==pr[t]['hidden']and set(z['gates'])==layers and set(z['raw'])=={'loss','pred','kl','kl_p','agreement_p'}
   for l,d in z['gates'].items():
    for k,v in d.items():assert torch.equal(v,g[l][k][:,:127])
   for k,v in z['raw'].items():assert torch.equal(v,raw[full][k][:len(v)])and len(v)==(127 if k=='loss'else 128)
  for f,d in ad.items():
   ss=(d['singleton_losses']-d['loss_p']).sum(0);expected={'sum_singleton_delta':ss,'J':d['loss_joint']-d['loss_additive'],'R':d['loss_additive']-d['loss_p']-ss,'joint_minus_sum':d['loss_joint']-d['loss_p']-ss}
   for k,v in expected.items():assert float((v-d[k]).abs().max())<=1e-12
   e=float((d['J']+d['R']-d['joint_minus_sum']).abs().max());jrerr=max(jrerr,e);assert e<=1e-12
   for view,sl in VIEWS.items():
    vals={k:float(v[sl].double().mean())for k,v in d.items()if isinstance(v,torch.Tensor)and v.shape==(511,)};close(vals,aws[c,w,i,f,view])
    for k,v in vals.items():ag[c,i,f,view,k].append(v)
  if w==0 and i=='natural':
   from gate_additive_readout_audit import family_metrics
   gpu=load(P/'readout_rows'/name);qr=gpu['query_rows'];assert qr.tolist()==[0,1,126,127,254,510]and torch.equal(gpu['targets'],x[qr+1])
   for f in FAMILIES:
    dd,add,_=family_metrics(gpu['P'].double(),gpu['C'].double(),[gpu[f+'_L'+str(l)].double()for l in LAYERS],gpu[f+'_all'].double(),gpu['targets'])
    assert torch.equal(add,gpu[f+'_additive'])
    for metric,v in dd.items():
     saved=ad[f][metric][:,qr]if metric=='singleton_losses'else ad[f][metric][qr]
     assert float((v.double()-saved.double()).abs().max())<1e-11,(c,f,metric)
   del gpu
  del gs,hs,su,ad
 assert(hidden,states,suffixes)==(928,1792,992)
 summaries=j(P/'additive_summary.json');assert len(summaries)==len(ag)==1440
 for r in summaries:
  v=ag[r['corpus'],r['identity'],r['family'],r['view'],r['metric']];a=paired(v);close({'n':4,'mean':a['mean_delta_nll'],'ci95':a['ci95_delta_nll'],'per_window':v},r)
 snapshots=load(P/'attention_snapshots.pt');errs=defaultdict(float);sn=0
 for key,ls in snapshots.items():
  mode=key.rsplit('__',1)[1]
  for l,sp in ls.items():
   sn+=1;q,k,v=[sp[x].double()for x in ('q','k','v')];scores=q@k.transpose(-1,-2)/8;scores.masked_fill_(torch.ones(512,512,dtype=torch.bool).triu(1),float('-inf'));mask=sp['mask'];scores=scores.masked_fill(~mask,float('-inf'))if mask.dtype==torch.bool else scores+mask.double();prob=scores.softmax(-1);actual=(prob@v).transpose(1,2).reshape(1,512,768)
   assert torch.count_nonzero(v[:,:,0])==0
   a=prob[0,:,1:,1:].sum(-1);b=scores[0,:,1:,1:].logsumexp(-1)-scores[0,:,1:,0];u=(scores[:,:,1:,1:].softmax(-1)@v[:,:,1:])[0];state=sp['gate_state'];deliv=actual.clone();deliv[:,1:]=(state['target_mass'].double()[...,None]*u).transpose(0,1).reshape(1,511,768)
   for name,one,two in [('context',actual,sp['actual_context']),('b',b,state['b']),('mass',a,state['mass']),('mixture',u,sp['stable_mixture']),('delivered',deliv,sp['delivered_context'])]:errs[name]=max(errs[name],float((one-two.double()).abs().max()))
   direct=sp['actual_context'].clone()
   if l in spec(mode)[1]:direct[:,1:]=(direct[0].reshape(512,12,64).transpose(0,1)[:,1:]*(state['target_mass']/state['mass'])[...,None]).transpose(0,1).reshape(1,511,768)
   assert torch.equal(direct,sp['delivered_context'])and torch.equal(direct[:,0],sp['actual_context'][:,0])
 assert sn==4 and max(errs.values())<1e-4
 disk=j(P/'disk_checks.json');assert len(disk)==32
 for d in disk:assert d['output_bytes']+d['next_group_allowance_bytes']<=2046820352 and d['free_bytes']-d['next_group_allowance_bytes']>=2*1024**3
 assert sum(f.stat().st_size for f in P.rglob('*')if f.is_file())<=2*1024**3
 result.update(status='passed_saved_evidence_pending_readout',raw_hidden_states=hidden,raw_gate_states=states,suffix_controls=suffixes,runtime_gate_checks=guard,max_CPU_gate_formula_error=maperr,max_J_R_identity_error=jrerr,operation_maxima=dict(maxima),FP64_snapshot_errors=dict(errs),snapshot_states=sn,additive_summary_groups=1440,reported_resources={k:m[k]for k in ('wall_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','peak_process_rss_kib')},pretrained_loaded=False,gpu_used=False)
if __name__=='__main__':
 result=main();Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
