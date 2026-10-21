"""Memory-only complete runner-shaped serialization budget; no model/inference.

High-entropy finite arrays are size fixtures, not synthetic scientific evidence.
All32 input shards/four calibration shards are encoded with the actual codec;
the actual runner payload function and failure-payload AST define the schema.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,copy,hashlib,io,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import gpt2_gate_wiki_calibration as R
from gate_wiki_storage import dumps_npz,loads_npz,SnapshotStore
from gate_additive_failure import logit_pair,singleton_state,suffix_state
from run_transfer import sha

def main():
 start=time.time();torch.set_num_threads(1);rng=np.random.default_rng(20261029)
 out=Path(__file__).with_suffix('.json');assert not out.exists(),'Preserve earlier fixture'
 sources={n:sha(ROOT/n)for n in ('gpt2_gate_wiki_calibration.py','gate_wiki_storage.py','gate_wiki_calibration.py')}
 report={'status':'checking','model_forwards':0,'model_loaded':False,'gpu_used':False,'disk_payload_written':False,'source_sha256':sources}
 def array(shape,dtype=torch.float32):
  # Full finite exponent/mantissa entropy, with nonfinite exponent remapped.
  if dtype==torch.float32:
   u=rng.integers(0,2**32,size=shape,dtype=np.uint32);bad=(u&np.uint32(0x7f800000))==np.uint32(0x7f800000);u[bad]^=np.uint32(0x00800000);return torch.from_numpy(u.view(np.float32).copy())
  u=rng.integers(0,2**64,size=shape,dtype=np.uint64);bad=(u&np.uint64(0x7ff0000000000000))==np.uint64(0x7ff0000000000000);u[bad]^=np.uint64(0x0010000000000000);return torch.from_numpy(u.view(np.float64).copy())
 def digest():return hashlib.sha256(rng.bytes(32)).hexdigest()
 def fp(shape,dtype='torch.float32'):return {'sha256':digest(),'shape':list(shape),'dtype':dtype}
 def state(n=511):return {k:array((12,n))for k in ('b','mass','target_b','target_mass')}
 def source():return {'kv':{l:{k:array((768,))for k in('k','v')}for l in R.LAYERS},'q':{l:array((12,64))for l in R.LAYERS}}
 def metric(n=511):return {'loss':array((n,)),'kl':array((n,)),'pred':torch.from_numpy(rng.integers(0,50257,n,dtype=np.int64)),
  'kl_p':array((n,)),'agreement_p':torch.from_numpy(rng.integers(0,2,n,dtype=np.int8).astype(bool)), 'final_hidden_sha256':digest()}
 def norms(mode):
  affected=R.mode_spec(mode)[3];layers=R.LAYERS if mode=='P'else affected
  return {l:{**({'s':array((511,)),'hypothetical_pc1_norm':array((511,),torch.float64),'hypothetical_orthogonal_norm':array((511,),torch.float64)}if l in affected else{}),
   'logit_delta_norm':array((511,),torch.float64),'mass_delta_norm':array((511,),torch.float64),'context_delta_norm':array((511,),torch.float64)}for l in layers}
 actual_checks=json.loads((ROOT/'review/gate_wiki_context_fixture.json').read_text())['conditions']
 def operation(mode,base):
  gate_checks=copy.deepcopy(actual_checks[mode]['checks'])
  # Random decimal spellings avoid tiny all-zero JSON sizing assumptions.
  for d in gate_checks.values():
   for k,v in d.items():
    if isinstance(v,float):d[k]=float(rng.normal())
  return {**base,'trial':mode,'final_hidden_sha256':digest(),'projection_checks':8,'query_checks':8,'context_checks':8,'writer_locality_checks':30,
   'gate_checks':gate_checks,'mask_records':{l:({'kind':'implicit_causal'}if mode=='C'else{'kind':'explicit',**fp((1,1,512,512),'torch.bool')})for l in R.LAYERS},
   'max_context_error':float(rng.random()),'source_fingerprints':{'kv':{l:{k:fp((768,))for k in('k','v')}for l in R.LAYERS},'q':{l:fp((12,64))for l in R.LAYERS}},
   'prefix_context_hashes':{l:{k:fp((1,128,768))for k in('current','delivered')}for l in R.LAYERS},
   'current_context_hashes':{l:fp((1,512,768))for l in R.LAYERS},
   'gate_fingerprints':{}if mode=='C'else{l:{k:fp((12,511))for k in('b','mass','target_b','target_mass')}for l in R.LAYERS},
   'norm_fingerprints':{}if mode=='C'else{l:{k:fp((511,),'torch.float32'if k=='s'else'torch.float64')for k in(('s','hypothetical_pc1_norm','hypothetical_orthogonal_norm')if l in R.mode_spec(mode)[3]else())+('logit_delta_norm','mass_delta_norm','context_delta_norm')}for l in R.LAYERS}}
 def snapshot():return {'q':array((1,12,512,64)),'k':array((1,12,512,64)),'v':array((1,12,512,64)),
  'mask':torch.ones((1,1,512,512),dtype=torch.bool).tril(),'actual_context':array((1,512,768)),
  'stable_mixture':array((12,511,64)),'delivered_context':array((1,512,768)),'gate_state':state()}
 def snapset():
  store=SnapshotStore();base=snapshot();store.add('P',base)
  for mode in R.SNAPSHOT_MODES[1:]:
   other={**base,'gate_state':{**base['gate_state'],'target_b':array((12,511)),'target_mass':array((12,511))},'delivered_context':array((1,512,768))}
   store.add(mode,other)
  return store
 def measure(name,payload):
  data=dumps_npz(payload);result={'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  del data;return result
 try:
  shard_sizes=[];counts={'records':0,'checks':0,'operations':0,'suffix_checks':0,'suffix_operations':0,'norm_records':0,'snapshots':0,'readouts':0}
  for corpus,n in R.CORPUS_COUNTS.items():
   for window in range(n):
    for identity in('natural','bos'):
     base={'panel':'gate','corpus':corpus,'window':window,'identity':identity,'length':512,'exposed_passage':False};key=f'gate__{corpus}__{window}__512__{identity}'
     modes=list(R.CONDITIONS)+(['repeat','identity']if window==0 and identity=='natural'else[])
     raw={key+'__'+m:{'token_sha256':digest(),'score_index':1,**metric()}for m in modes};op=[];checks=[];sc=[];so=[];nd={};sm={}
     psource=source()
     for mode in modes:
      actual=mode if mode in R.CONDITIONS else'C';o=operation(actual,base);o['trial']=mode;op.append(o)
      if actual!='C':checks.append({**base,'trial':mode+'_source_identity','source_exact':True,'reference_fingerprints':R.fp_nested(psource),'trial_fingerprints':o['source_fingerprints']})
      if len(R.mode_spec(actual)[3])==1:checks.append({**base,'trial':mode+'_singleton','layer':10,'b_exact':True,'mass_exact':True,'context_exact':True,'coefficient_exact':True,'coefficient_max_abs_error':float(rng.random()),'reference_context':fp((1,512,768)),'trial_context':fp((1,512,768))})
      if mode in('current','fullrank','repeat','identity'):checks.append({**base,'trial':mode+'_closure',**{k+'_exact':True for k in('loss','kl','pred','kl_p','agreement_p','hidden','logits')},'logits_reference':fp((511,50257)),'logits_control':fp((511,50257))})
      if mode in R.CONDITIONS:
       nd[key+'__'+mode]=norms(mode);alt=operation(mode,base);so.append(alt)
       sm[key+'__'+mode]={'token_sha256':digest(),'original_token_sha256':digest(),'score_index':1,'scored_tokens':126,'hidden_prefix_fingerprint':fp((1,128,768)),**metric(128)};sm[key+'__'+mode]['loss']=array((127,));sm[key+'__'+mode].pop('final_hidden_sha256')
       sc.append({**base,'trial':mode,'original_token_sha256':digest(),'control_token_sha256':digest(),
        **{k+'_exact':True for k in('logits','hidden','gates','source','mask','contexts','loss','pred','kl','kl_p','agreement_p')},
        'logits_reference':fp((128,50257)),'logits_control':fp((128,50257)),
        'reference_context_hashes':o['prefix_context_hashes'],'control_context_hashes':alt['prefix_context_hashes']})
     is_snapshot=(corpus,window,identity)==R.SNAPSHOT_INPUT
     snaps=snapset().payload if is_snapshot else None
     readouts={mode:{'hidden':array((1,512,768)),'query_rows':torch.tensor(R.READOUT_ROWS),'logits':array((4,50257)),
      'targets':torch.from_numpy(rng.integers(0,50257,4,dtype=np.int64)),'token_sha256':digest(),'hidden_fingerprint':fp((1,512,768))}for mode in R.SNAPSHOT_MODES}if is_snapshot else{}
     payload=R.pack_input_payload(base,key,raw,checks,op,sc,so,nd,psource,snaps,readouts,sm)
     shard_sizes.append({**measure('inputs/'+key+'.npz',payload),'class':'snapshot'if is_snapshot else'ordinary'})
     for k,v in {'records':len(raw),'checks':len(checks),'operations':len(op),'suffix_checks':len(sc),'suffix_operations':len(so),'norm_records':len(nd),'snapshots':0 if snaps is None else len(snaps['states']),'readouts':len(readouts)}.items():counts[k]+=v
     del payload,raw,checks,op,sc,so,nd,psource,snaps,readouts,sm
  assert counts=={'records':776,'checks':1128,'operations':776,'suffix_checks':768,'suffix_operations':768,'norm_records':768,'snapshots':11,'readouts':11},counts
  calibration=[]
  for i in range(4):
   base={'phase':'calibration','corpus':'wiki_test','window':i,'identity':'natural','trial':'calibration'}
   raw={l:{'b':array((12,511)),'mass':array((12,511))}for l in R.LAYERS}
   calibration.append(measure(f'calibration/wiki_test__{i}.npz',{'raw':raw,'metrics':{'token_sha256':digest(),'kl_reference':'self/no-native-C',**metric()},'operation':operation('P',base)}))
  m=torch.load(ROOT/'gpt2_gate_fixed512_run/calibration_maps.pt',weights_only=True);w=torch.load(ROOT/'wiki_calibration_preparation/old_wiki_maps.pt',weights_only=True)
  s=json.loads((ROOT/'gpt2_gate_fixed512_run/calibration_spectra.json').read_text());sw=json.loads((ROOT/'wiki_calibration_preparation/old_wiki_spectra.json').read_text())
  from gate_wiki_calibration import fit_geometry
  maps={'O':w,'F':w,'M':m};spectra={'O':{int(k):v for k,v in sw.items()},'F':{int(k):v for k,v in sw.items()},'M':{int(k):v for k,v in s.items()}}
  directions,geometry=fit_geometry(maps,spectra)
  fit_sizes=[measure(name,obj)for name,obj in [('calibration_maps.npz',maps),('calibration_spectra.npz',spectra),('calibration_directions.npz',directions),('calibration_geometry.npz',geometry)]]
  input_bytes=sum(p.stat().st_size for p in (ROOT/'calibration_wiki_inputs').iterdir()if p.is_file())
  source_files={n:(ROOT/n).stat().st_size for n in R.SOURCE_NAMES if (ROOT/n).exists()}
  missing=[n for n in R.SOURCE_NAMES if not(ROOT/n).exists()]
  # Future analyzer/final protocol, indices/checkpoints, JSONL776 rows, final
  # reportable analyses all share an explicit16MiB bounded allowance.
  metadata_allowance=16*1024**2
  normal=sum(v['bytes']for v in shard_sizes+calibration+fit_sizes)+input_bytes+sum(source_files.values())+metadata_allowance
  report.update(counts=counts,input_shards=shard_sizes,calibration_shards=calibration,fit_files=fit_sizes,
   copied_input_bytes=input_bytes,copied_source_bytes=sum(source_files.values()),source_file_sizes=source_files,
   pending_source_names=missing,remaining_metadata_analysis_allowance_bytes=metadata_allowance,
   total_normal_with_allowance_bytes=normal,normal_limit_bytes=R.NORMAL_LIMIT,
   normal_margin_bytes=R.NORMAL_LIMIT-normal,max_snapshot_group_bytes=max(v['bytes']for v in shard_sizes if v['class']=='snapshot'),
   max_ordinary_group_bytes=max(v['bytes']for v in shard_sizes if v['class']=='ordinary'),
   max_calibration_group_bytes=max(v['bytes']for v in calibration))
  # Evaluate the exact runner failure-dictionary AST, with one feasible failure
  # kind at a time; never combine mutually exclusive immediately-raised paths.
  tree=ast.parse((ROOT/'gpt2_gate_wiki_calibration.py').read_text());exprs=[n.value for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='payload'for t in n.targets)and isinstance(n.value,ast.Dict)]
  assert len(exprs)==1;expression=compile(ast.Expression(exprs[0]),'<actual-runner-failure-payload>','eval')
  def bundle(prefix=False):
   return {'z':metric(127 if prefix else 511),'hidden':array((1,512,768)),
    'gates':{l:state()for l in R.LAYERS},'source':source(),
    'current_contexts':{l:array((1,512,768))for l in R.LAYERS},
    'prefix_contexts':{l:{k:array((1,128,768))for k in('current','delivered')}for l in R.LAYERS},
    'localization_norms':norms('remove_OO_all')}
  failure_sizes=[]
  for kind in('gate','reconstruction','singleton','suffix','identity','snapshot','metric','calibration'):
   current=bundle();base=bundle();snap=snapshot();gs=SimpleNamespace(failure_gate=None,failure_reconstruction=None,gate_cache=current['gates'],cache=current['source']['kv'],diag={l:{'source_q':current['source']['q'][l]}for l in R.LAYERS})
   logits=torch.randn(128 if kind in('suffix','metric')else 511,50257) # finite realistic logits for max-error diagnostics
   env={'gate':gs,'hidden':{'tensor':current['hidden']},'last_forward':{'phase':kind,'trial':'remove_OO_all'},'last_completed':{'phase':kind,'trial':'remove_OO_all'},'current_attempt_tokens':torch.arange(512,dtype=torch.long),'record_target':{'trial':'remove_OO_all'},'check_target':{'stage':kind},'singleton_failure':None,'suffix_failure':None,'identity_failure':None,'control_failure':{},'pending_logits':logits,'run':None,'meta':{'phase':'evaluation'},'calraw':{},'calmetrics':{},'calops':[],'frozen_maps':{k:R.maps_digest(v)for k,v in maps.items()},'error':AssertionError(kind),'logit_pair':logit_pair,'singleton_state':singleton_state,'suffix_state':suffix_state}
   if kind=='gate':gs.failure_gate={'raw_inputs':{k:snap[k]for k in('q','k','v','mask')},'state':snap['gate_state'],'stable_mixture':snap['stable_mixture'],'actual_context':snap['actual_context'],'delivered_context':snap['delivered_context'],'norms':current['localization_norms'][10],'check':actual_checks['remove_OO_all']['checks']['10']}
   elif kind=='reconstruction':gs.failure_reconstruction={**{k:snap[k]for k in('q','k','v','mask')},'actual_context':snap['actual_context'],'expected_context':snap['delivered_context']}
   elif kind=='singleton':env['singleton_failure']={'check':{'layer':10},'reference':base,'trial':current}
   elif kind=='suffix':env['suffix_failure']={'check':{},'reference':base,'control':current,'reference_logits':logits,'control_logits':logits+0.001,'original_tokens':torch.arange(512),'control_tokens':torch.arange(512)}
   elif kind=='identity':env['identity_failure']={'check':{},'left_hidden':current['hidden'],'right_hidden':base['hidden'],'logits':logit_pair(logits,logits+0.001)}
   elif kind=='snapshot':env['error']=SimpleNamespace(payload={'mode':'remove_OO_L10','mismatching_paths':[('q',)],'reference_actual':snapshot(),'trial_actual':snapshot()})
   elif kind=='metric':env['control_failure']={'prefix_logits':logits,'metrics':metric(127),'input_tokens':torch.arange(512)}
   else:
    env['meta']={'phase':'calibration'};env['calraw']={str(i):{l:{'b':array((12,511)),'mass':array((12,511))}for l in R.LAYERS}for i in range(4)};env['calmetrics']={str(i):metric()for i in range(4)}
   payload=eval(expression,{},env);buffer=io.BytesIO();torch.save(payload,buffer);n=buffer.tell();failure_sizes.append({'kind':kind,'bytes':n,'limit_bytes':R.FAILURE_LIMIT});assert n<=R.FAILURE_LIMIT,(kind,n)
   del buffer,payload,env,logits,current,base,snap,gs
  report['failure_capsules']=failure_sizes;report['maximum_failure_bytes']=max(v['bytes']for v in failure_sizes)
  assert normal<=R.NORMAL_LIMIT,('Normal payload exceeds proposed limit',normal,R.NORMAL_LIMIT)
  assert sources=={n:sha(ROOT/n)for n in sources},'Sources changed during fixture'
  report['status']='passed'
 except BaseException as error:report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
 finally:
  report.update(script_sha256=sha(Path(__file__)),wall_seconds=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report[k]for k in('status','total_normal_with_allowance_bytes','normal_margin_bytes','maximum_failure_bytes','peak_rss_kib')}))
if __name__=='__main__':main()
