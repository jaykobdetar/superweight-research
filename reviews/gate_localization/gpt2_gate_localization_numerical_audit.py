"""Independent CPU audit of completed localization data. No runner/analyzer imports.
Shared reductions/hash utilities come from earlier independent fixed512 audit.
"""
import hashlib,json,math,resource
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from gpt2_gate_fixed512_numerical_audit import j,load,sha,th,close,metrics,dual
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'gpt2_gate_localization_run'
FROZEN={'gpt2_gate_localization.py':'eafab996ce06c697d081d8e0d06583cf9d3c0866ab2f27213cfcb4387503e238','analyze_gate_localization.py':'3998aed2c976c85999795f39d547db90270c2881f03725a720eb7e70269f2d28','GPT2_GATE_LOCALIZATION.md':'e97057f0f28d05f2d1d71baed68c012fdae10a286347700236f54e50c8fef415'}
LAYERS=tuple(range(4,12));FAMILIES=('remove_pc1','orthogonal_control','position');SETS=tuple(f'L{l}'for l in LAYERS)+('all',);MODES=('C','P','current','fullrank')+tuple(f+'_'+s for f in FAMILIES for s in SETS);COUNTS={'wiki_validation':4,'dubois':4,'hobbes':4,'optimizers':3}
def spec(mode):
 if mode in('C','P','current','fullrank','repeat','identity'):return None,()
 family,which=mode.rsplit('_',1);assert family in FAMILIES;return family,LAYERS if which=='all'else(int(which[1:]),)
def paired(x):
 x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(20261021);bs=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
 return dict(n=len(x),mean_delta_nll=float(x.mean()),ci95_delta_nll=np.quantile(bs,[.025,.975]).tolist(),per_window_delta_nll=x.tolist())
def formula(mode,l,b,a,m,d):
 f,layers=spec(mode);q=torch.arange(1,512,dtype=b.dtype).log()[:,None];s=(b.T-q-m['alpha'])@d['u'];du=s[:,None]*d['u'];dv=s[:,None]*d['v'];nu=du.double().norm(dim=-1);nv=dv.double().norm(dim=-1)
 if mode=='fullrank':tb=(m['alpha']+q+((b.T-q-m['alpha'])@m['basis_r'])@m['basis_r'].T).T;t=a
 elif l not in layers:tb,t=b,a
 else:tb=(b.T-du if f=='remove_pc1'else b.T-dv if f=='orthogonal_control'else m['alpha']+q).T;t=tb.sigmoid()
 return tb,t,{'s':s,'hypothetical_pc1_norm':nu,'hypothetical_orthogonal_norm':nv},float(((nu-nv).abs()/torch.maximum(torch.ones_like(nu),torch.maximum(nu,nv))).max())
def main():
 m=j(P/'manifest.json');assert m.get('status')==m.get('phase')=='completed'and m.get('completed_utc'),'Production incomplete; no scientific read'
 assert j(P/'audit.json')['status']=='passed','Analysis incomplete';torch.set_num_threads(1)
 assert(m['forward_attempts'],m['forward_passes'],m['records'],m['evaluation_forwards'],m['suffix_forwards'],m['calibration_forwards'])==(1868,1868,938,938,930,0)
 assert m['physical_execution_lengths']==[512]and not m['cross_length_comparison']and m['score_views']=={'full':[2,511],'early':[2,127],'later':[128,511]}
 for n,h in FROZEN.items():assert sha(ROOT/n)==m['source_hashes'][n]==h
 for n,h in m['source_hashes'].items():assert sha(ROOT/n)==sha(P/'frozen_sources'/n)==h
 assert m['model_hash_before']==m['model_hash_after']=='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527'and m['parameter_versions_unchanged']and m['calibration_maps_unchanged']
 assert m['precision_settings_before']==m['precision_settings_after'];assert j(ROOT/'gpt2_gate_subspace_run/manifest.json')['status']==j(ROOT/'gpt2_gate_subspace_v2_run/manifest.json')['status']=='failed'
 for n,h in m['input_dependencies'].items():assert sha(P/n)==sha(ROOT/'localization_inputs_v2'/n)==h
 assert j(ROOT/'review/gate_localization_design_audit.json')['status']=='passed'
 dep=j(P/'calibration_dependency.json');assert sha(P/'calibration_dependency.json')==m['calibration_dependency_sha256']
 for n,h in dep['copied_files'].items():assert sha(P/n)==sha(ROOT/'gpt2_gate_fixed512_run'/n)==h
 assert sha(P/'calibration_maps.pt')=='ed62f089168abeec63434e5f6dfa83ff8e7e139bc362e952456ea34859b5a098'and sha(P/'calibration_raw.pt')=='d41d5a91c699d30d5414a2916f7350a9093eda4b0004d8bb7b2fb458ebb31935'
 assert j(ROOT/'review/gpt2_gate_fixed512_numerical_audit.json')['status']=='passed';maps=load(P/'calibration_maps.pt');directions=load(P/'directions.pt')
 for l in LAYERS:
  u=maps[l]['basis_r'][:,0];u=u/u.norm();v=maps[l]['basis_qr'][:,0]-u*(u@maps[l]['basis_qr'][:,0]);assert v.norm()>=1e-6;v=v/v.norm();v=v*(-1 if v[v.abs().argmax()]<0 else 1)
  assert torch.equal(u,directions[l]['u'])and torch.equal(v,directions[l]['v'])
 assert tuple(j(P/'conditions.json'))==MODES
 raw=load(P/'per_token.pt');rows=[json.loads(x)for x in(P/'windows.jsonl').read_text().splitlines()];assert len(raw)==len(rows)==len({r['key']for r in rows})==938
 groups=defaultdict(list);segments=defaultdict(list);inputs=defaultdict(dict);closures=0
 for r in rows:
  key=r['key'];stem=key.rsplit('__',1)[0];z=raw[key];c=raw[stem+'__C'];p=raw[stem+'__P'];assert r['length']==r['input_tokens']==512 and r['target_start']==2 and r['score_index']==z['score_index']==1 and r['scored_tokens']==510 and not r['exposed_passage']
  assert z['token_sha256']==r['token_sha256']==c['token_sha256']==p['token_sha256']
  for field in('loss','pred','kl','kl_p','agreement_p'):assert z[field].shape==(511,)and torch.isfinite(z[field]).all()
  assert torch.equal(z['agreement_p'],z['pred'].eq(p['pred']));close(metrics(z,c,p,slice(1,None)),r)
  if r['trial']in('current','fullrank','repeat','identity'):
   ref=p if r['trial']in('current','fullrank')else c;assert z['final_hidden_sha256']==ref['final_hidden_sha256']
   for field in('loss','pred','kl','kl_p','agreement_p'):assert torch.equal(z[field][1:],ref[field][1:])
   closures+=1
  if r['trial']=='C':assert torch.count_nonzero(z['kl'])==0
  if r['trial']=='P':assert torch.count_nonzero(z['kl_p'])==0
  inputs[r['corpus'],r['identity'],r['window']][r['trial']]=r
  if r['trial']in MODES:
   g=(r['corpus'],r['identity'],r['trial']);groups[g].append(r)
   for name,sl in [('early_2_127',slice(1,127)),('later_128_511',slice(127,None))]:segments[g+(name,)].append(metrics(z,c,p,sl))
 assert len(groups)==248 and len(segments)==496 and len(inputs)==30 and closures==68
 for(c,i,w),rs in inputs.items():assert set(rs)==set(MODES)|({'repeat','identity'}if i=='natural'and w==0 else set())
 for fn,gr in [('summary.json',groups),('segment_summary.json',segments)]:
  saved=j(P/fn);assert len(saved)==len(gr)
  for r in saved:
   key=(r['corpus'],r['identity'],r['trial'])+((r['segment'],)if'segment'in r else());assert len(gr[key])==COUNTS[r['corpus']];close(dual(gr[key]),r)
 cg=defaultdict(list);cs=j(P/'paired_contrasts.json');assert len(cs)==810
 for r in cs:
  a,tail=r['trial'].split('_minus_');b='orthogonal_control_'+a.rsplit('_',1)[1]if tail=='orthogonal_control'else tail;rs=inputs[r['corpus'],r['identity'],r['window']];delta=rs[a]['nll']-rs[b]['nll'];close(delta,r['delta_nll']);cg[r['corpus'],r['identity'],r['trial']].append(delta)
 ps=j(P/'paired_summary.json');assert len(ps)==216
 for r in ps:close(paired(cg[r['corpus'],r['identity'],r['trial']]),r)
 ng=defaultdict(list);nr=j(P/'nonadditivity.json');assert len(nr)==90
 for r in nr:
  rs=inputs[r['corpus'],r['identity'],r['window']];f=r['family'];joint=rs[f+'_all']['nll']-rs['P']['nll'];singles=[rs[f+'_L'+str(l)]['nll']-rs['P']['nll']for l in LAYERS];delta=joint-sum(singles)
  close(dict(joint_delta_nll=joint,singleton_delta_nll=singles,sum_singleton_delta_nll=sum(singles),joint_minus_sum_delta_nll=delta),r);ng[r['corpus'],r['identity'],f].append(delta)
 ns=j(P/'nonadditivity_summary.json');assert len(ns)==24
 for r in ns:close(paired(ng[r['corpus'],r['identity'],r['family']]),r)
 result={'status':'checking','records':938,'full_summaries':248,'segment_summaries':496,'paired_contrasts':810,'paired_groups':216,'nonadditivity_records':90,'nonadditivity_groups':24,'raw_metric_closures':closures,'calibration_audit_reused_exact_bytes':True}
 return states(m,raw,rows,maps,directions,result)

def states(m,raw,rows,maps,directions,result):
 modes=MODES
 ops=j(P/'operation_checks.json');checks=j(P/'checks.json');sops=j(P/'suffix_operations.json');sc=j(P/'suffix_checks.json');assert len(ops)==938 and len(checks)==1688 and len(sops)==len(sc)==930
 assert all(v for c in checks for k,v in c.items()if k.endswith('_exact'));assert sum(c['trial'].endswith('_source_identity')for c in checks)==900
 gatecount=0;operation_max=defaultdict(float)
 for op in ops+sops:
  assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30 and op['max_context_error']<=1e-4
  for c in op['gate_checks'].values():
   gatecount+=1;assert c['finite']and c['source_context_exact']and c['unaffected_context_exact']and c['min_mass']>0 and c['norm_match_error']<=1e-6
   for k,v in c.items():
    if k.endswith('_error'):assert math.isfinite(v)and v<=1e-4;operation_max[k]=max(operation_max[k],v)
 assert gatecount==14400
 sources=load(P/'source_states.pt');assert set(sources)==set(raw);sourcecount=0
 for key,source in sources.items():
  if key.rsplit('__',1)[1]in('C','repeat','identity'):continue
  ref=sources[key.rsplit('__',1)[0]+'__P']
  for l,values in source['kv'].items():
   for k,v in values.items():assert torch.equal(v,ref['kv'][l][k])
  for l,v in source['q'].items():assert torch.equal(v,ref['q'][l])
  sourcecount+=1
 assert sourcecount==900
 gf=j(P/'gate_files.json');pf=j(P/'prefix_files.json');sf=j(P/'suffix_files.json');assert len(gf)==len(pf)==len(sf)==30
 tokenitems=load(P/'tokens.pt');tokenmap={(x['corpus'],x['window']):x['tokens']for x in tokenitems};bos=j(P/'data_manifest.json')['literals']['start_special']['id'];opindex={(x['corpus'],x['window'],x['identity'],x['trial']):x for x in ops};sindex={(x['corpus'],x['window'],x['identity'],x['trial']):x for x in sops};ci={(x['corpus'],x['window'],x['identity'],x['trial']):x for x in sc}
 maperror=0.;ncontrol=0;nreference=0;covstats={};gatedeliveredendpoints={'zeros':0,'ones':0};maxgain=0.
 for filename,h in gf.items():
  assert sha(P/'gates'/filename)==h;_,corpus,w,ns,identity=Path(filename).stem.split('__');w=int(w);assert ns=='512';stem=Path(filename).stem;pairname=f'gate__{corpus}__{w}__{identity}.pt';assert sha(P/'prefix'/pairname)==pf[pairname]and sha(P/'suffix'/pairname)==sf[pairname]
  gates=load(P/'gates'/filename);prefix=load(P/'prefix'/pairname);suffix=load(P/'suffix'/pairname);assert set(gates)==set(prefix)==set(modes)|({'repeat','identity'}if identity=='natural'and w==0 else set());assert set(suffix)==set(modes)
  x=tokenmap[corpus,w].clone()
  if identity=='bos':x[0]=bos
  y=x.clone();y[128:]=torch.roll(y[128:],1);assert torch.equal(x[:128],y[:128])and not torch.equal(x[128:],y[128:])
  for mode,pp in prefix.items():
   nreference+=1;assert set(pp)=={'hidden'}and pp['hidden'].shape==(1,128,768)and torch.isfinite(pp['hidden']).all()
   if mode in('current','fullrank','repeat','identity'):assert torch.equal(pp['hidden'],prefix['P'if mode in('current','fullrank')else'C']['hidden'])
  for group,collection in [('main',gates),('suffix',{k:v['gates']for k,v in suffix.items()})]:
   for mode,layers in collection.items():
    assert set(layers)==(set()if mode in('C','repeat','identity')else set(range(4,12)))
    if group=='main'and len(spec(mode)[1])==1:
     l=spec(mode)[1][0]
     for f in ('b','mass'):assert torch.equal(layers[l][f],gates['P'][l][f])
    for l,d in layers.items():
     b,a,t=d['b'],d['mass'],d['target_mass'];assert b.shape==a.shape==t.shape==(12,511)and torch.isfinite(b).all()and torch.isfinite(a).all()and(a>0).all()and torch.isfinite(t).all()
     tb,ta,nd,ne=formula(mode,l,b,a,{k:v.float()for k,v in maps[l].items()},{k:directions[l][k].float()for k in ('u','v')});assert ne<=1e-6
     for expected,actual in[(tb,d['target_b']),(ta,t)]:
      if expected is None:assert actual is None
      else:err=float((expected-actual).abs().max());maperror=max(maperror,err);assert err<1e-4
     if l in spec(mode)[1]:
      assert(t>=0).all()and(t<=1).all();gatedeliveredendpoints['zeros']+=int((t==0).sum());gatedeliveredendpoints['ones']+=int((t==1).sum())
     maxgain=max(maxgain,float((t/a).max()))
     if group=='main'and mode=='P':
      # Descriptive centered held-out covariance, separate from causal effects.
      xx=b[:,:510].T.double()-torch.arange(1,511,dtype=torch.float64).log()[:,None]
      key=(corpus,identity,l);count,sx,xxsum=covstats.get(key,(0,torch.zeros(12,dtype=torch.float64),torch.zeros(12,12,dtype=torch.float64)));covstats[key]=(count+510,sx+xx.sum(0),xxsum+xx.T@xx)
  for mode,d in suffix.items():
   ncontrol+=1;key=(corpus,w,identity,mode);chk=ci[key];op=opindex[key];sop=sindex[key];z=raw[stem+'__'+mode]
   assert chk['original_token_sha256']==th(x)==z['token_sha256']and chk['control_token_sha256']==th(y)==d['token_sha256'];assert all(v is True for k,v in chk.items()if k.endswith('_exact'))
   assert chk['logits_reference']==chk['logits_control']and chk['logits_reference']['shape']==[128,50257]and chk['logits_reference']['dtype']=='torch.float32'
   assert chk['reference_context_hashes']==op['prefix_context_hashes']==chk['control_context_hashes']==sop['prefix_context_hashes']and op['mask_records']==sop['mask_records']
   assert torch.equal(d['hidden'],prefix[mode]['hidden'])and d['score_index']==1 and d['scored_tokens']==126
   assert set(d['raw'])=={'loss','kl','kl_p','pred','agreement_p'}
   for k,v in d['raw'].items():assert len(v)==(127 if k=='loss'else 128)and torch.equal(v,z[k][:len(v)])
   for l,fields in d['gates'].items():
    for k,v in fields.items():assert(v is None and gates[mode][l][k]is None)or(v is not None and torch.equal(v[:,:127],gates[mode][l][k][:,:127]))
   for l,v in sources[stem+'__'+mode]['q'].items():assert torch.equal(v,d['source']['q'][l])
   for l,kv in sources[stem+'__'+mode]['kv'].items():
    for k,v in kv.items():assert torch.equal(v,d['source']['kv'][l][k])
  del gates,prefix,suffix
 assert nreference==938 and ncontrol==930
 result.update(main_gate_checks=7200,suffix_gate_checks=7200,source_identity_records=sourcecount,prefix_references=nreference,suffix_saved_array_closures=ncontrol,all_logit_context_runtime_hash_closures=930,max_cpu_map_error=maperror,operation_error_maxima=dict(operation_max),active_target_endpoints=gatedeliveredendpoints,max_gain=maxgain)
 normsummary=j(P/'norm_summary.json');assert len(normsummary)==5040;ni={(d['corpus'],d['window'],d['identity'],d['trial'],d['layer'],d['view']):d for d in normsummary};nf=j(P/'norm_files.json');assert len(nf)==30;ncount=0;nsingleton=0;normerror=0.;normguard=0.
 for name,h in nf.items():
  assert sha(P/'local_norms'/name)==h;_,c,w,n,i=Path(name).stem.split('__');w=int(w);norms=load(P/'local_norms'/name);gates=load(P/'gates'/name)
  for mode,layers in norms.items():
   f,affected=spec(mode)
   if len(affected)==1 and f in('remove_pc1','orthogonal_control'):
    l=affected[0];assert torch.equal(layers[l]['s'],norms['P'][l]['s']);nsingleton+=1
   for l,d in layers.items():
    ncount+=1;state=gates[mode][l];b,a=state['b'],state['mass'];tb,t,expected,ne=formula(mode,l,b,a,{k:v.float()for k,v in maps[l].items()},{k:directions[l][k].float()for k in('u','v')});normguard=max(normguard,ne);assert ne<=1e-6
    expected.update(logit_delta_norm=(state['target_b']-b).T.double().norm(dim=-1),mass_delta_norm=(state['target_mass']-a).T.double().norm(dim=-1))
    assert set(d)==set(expected)|{'context_delta_norm'}
    for k,v in d.items():assert v.shape==(511,)and torch.isfinite(v).all()
    for k,v in expected.items():err=float((v-d[k]).abs().max());normerror=max(normerror,err);assert err<=1e-4
    assert(d['context_delta_norm']>=0).all()
    if l not in affected:assert torch.count_nonzero(d['context_delta_norm'])==0
    if mode=='P'or l in affected:
     for view,sl in [('full',slice(0,510)),('early',slice(0,126)),('later',slice(126,510))]:
      expectedsummary={k:{'mean':float(v[sl].double().mean()),'rms':float(v[sl].double().square().mean().sqrt()),'max':float(v[sl].abs().max())}for k,v in d.items()};close(expectedsummary,ni[c,w,i,mode,l,view]);close(ne,ni[c,w,i,mode,l,view]['all_query_norm_match_guard_max'],1e-6)
  del norms,gates
 assert ncount==7200 and nsingleton==480
 result.update(norm_layer_states=ncount,singleton_directional_s_closures=nsingleton,norm_summary_rows=len(normsummary),max_cpu_norm_error=normerror,max_hypothetical_norm_guard=normguard)
 del sources,raw
 snaps=load(P/'attention_snapshots.pt');assert len(snaps)==26;nstate=0;maxima={k:0. for k in('b','mass','stable_mixture','actual_context','delivered_fp64','delivered_saved_formula','snapshot_context_norm')};fingerprints=0
 for key,layers in snaps.items():
  parts=key.split('__');corpus,w,identity=parts[1],int(parts[2]),parts[4];mode=parts[-1];index=sindex if'suffix'in parts else opindex;op=index[corpus,w,identity,mode];is_control='suffix'in parts
  normfile=f'gate__{corpus}__{w}__512__{identity}.pt';norms=load(P/'local_norms'/normfile) if not is_control else None
  for l,snap in layers.items():
   q,k,v=[snap[k].double()for k in('q','k','v')];n=q.shape[-2];assert n==512;scores=q@k.transpose(-1,-2)/8;scores.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),float('-inf'));mask=snap['mask'];scores=scores.masked_fill(~mask,float('-inf'))if mask.dtype==torch.bool else scores+mask.double()
   prob=scores.softmax(-1);actual=(prob@v).transpose(1,2).reshape(1,n,768);native=snap['actual_context'];maxima['actual_context']=max(maxima['actual_context'],float((actual-native.double()).abs().max()))
   if mode=='C':delivered=native;assert 'gate_state'not in snap
   else:
    assert torch.count_nonzero(v[:,:,0])==0;a=prob[0,:,1:,1:].sum(-1);b=scores[0,:,1:,1:].logsumexp(-1)-scores[0,:,1:,0];u=(scores[:,:,1:,1:].softmax(-1)@v[:,:,1:])[0];state=snap['gate_state'];target=state['target_mass'].double();expected64=actual.clone();expected64[:,1:]=(target[...,None]*u).transpose(0,1).reshape(1,n-1,768);delivered=snap['delivered_context']
    for label,one,two in[('b',b,state['b']),('mass',a,state['mass']),('stable_mixture',u,snap['stable_mixture']),('delivered_fp64',expected64,delivered)]:maxima[label]=max(maxima[label],float((one-two.double()).abs().max()))
    saved=native.clone()
    if l in spec(mode)[1]:
     c=native[0].reshape(n,12,64).transpose(0,1)[:,1:];mixed=c*(state['target_mass']/state['mass'])[...,None];saved[:,1:]=mixed.transpose(0,1).reshape(1,n-1,768)
    assert torch.equal(saved,delivered);maxima['delivered_saved_formula']=max(maxima['delivered_saved_formula'],float((saved-delivered).abs().max()));assert torch.equal(native[:,0],delivered[:,0])
    if norms is not None:
     cn=(delivered[:,1:]-native[:,1:])[0].double().norm(dim=-1);err=float((cn-norms[mode][l]['context_delta_norm']).abs().max());maxima['snapshot_context_norm']=max(maxima['snapshot_context_norm'],err);assert err<=2e-12 # CPU/GPU FP64 reduction order; FP32 delivered formula remains exact
   for kind,tensor in [('current',native),('delivered',delivered)]:
    fp=op['prefix_context_hashes'][str(l)][kind];assert fp['shape']==[1,128,768]and fp['dtype']=='torch.float32'and fp['sha256']==th(tensor[:,:128]);fingerprints+=1
   nstate+=1
  del norms
 assert nstate==52 and max(maxima.values())<1e-4
 disk=j(P/'disk_checks.json');assert len(disk)==30 and all(d['free_bytes']>=3*1024**3 for d in disk);assert m['disk_preflight']['passed']and m['disk_preflight']['free_before_bytes']>=7*1024**3
 result.update(status='passed',snapshot_states=nstate,FP64_snapshot_max_errors=maxima,snapshot_context_fingerprints=fingerprints,source_hashes=FROZEN,reported_resources={k:m[k]for k in('wall_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','peak_process_rss_kib')},review_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pretrained_loaded=False,gpu_used=False,limits='Successful full logits/contexts outside52 fixed snapshots remain runtime/hash evidence. Stored KL arrays reduced but no full logits to recompute every KL. Model hash closure is recorded, not independently reloaded. Tiny n4/n3 panel and correlated starts/score views; conditional bootstrap, no multiplicity correction. Hypothetical perturbation norms matched within own current state, not across joint trajectories. Nonadditivity is descriptive, not mediation.')
 assert result['review_peak_rss_kib']<3*1024**2
 return result
if __name__=='__main__':
 result=main();Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
