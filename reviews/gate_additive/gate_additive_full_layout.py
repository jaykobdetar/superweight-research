"""Exact writer-layout resource fixture; no neural model, tokenizer or GPU.

Calls the current production orchestration and serialization, stubbing only forward
values/readout math/hash computations. Distinct persisted arrays own their storage.
This is resource/counter evidence, deliberately not a scientific numerical fixture.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
import copy,hashlib,json,resource,sys,tempfile,time,shutil
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(1)
import gpt2_gate_additive as M
from run_transfer import sha,dump
old=json.loads((ROOT/'gpt2_run/manifest.json').read_text())
prior=ROOT/'gpt2_gate_localization_run'
priorops=json.loads((prior/'operation_checks.json').read_text())
gatecheck=next(r for r in priorops if r['trial']=='P')['gate_checks']['4']
maskcheck=next(r for r in priorops if r['trial']=='P')['mask_records']['4']
del priorops
snapshots=torch.load(prior/'attention_snapshots.pt',weights_only=True,map_location='cpu',mmap=True)
prototype=next(v for k,v in snapshots.items()if k.endswith('__P'))[4]
def zeros(v):
 if isinstance(v,torch.Tensor):return torch.zeros(v.shape,dtype=v.dtype)
 if isinstance(v,dict):return {k:zeros(x)for k,x in v.items()}
 return copy.deepcopy(v)
# Make a compact shape prototype; do not retain mapped historical raw pages.
def shape_tree(v):
 if isinstance(v,torch.Tensor):return (tuple(v.shape),v.dtype)
 return {k:shape_tree(x)for k,x in v.items()}
snapshot_shapes=shape_tree(prototype);del snapshots,prototype

def from_shapes(v):
 if isinstance(v,tuple):return torch.zeros(v[0],dtype=v[1])
 return {k:from_shapes(x)for k,x in v.items()}
def fp(x):
 return {'sha256':hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()).hexdigest(),'shape':list(x.shape),'dtype':str(x.dtype)}
class Handle:
 def remove(self):pass
class Head:
 def __init__(self):self.weight=torch.zeros(1).expand(50257,768);self.bias=None
 def register_forward_pre_hook(self,cb):self.cb=cb;return Handle()
class FakeModel(torch.nn.Module):
 def __init__(self):
  super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1));self.lm_head=Head();self.transformer=SimpleNamespace(wte=SimpleNamespace(weight=self.lm_head.weight))
class Probe:
 def __init__(self,m,s,coords):self.selected=s;self.vectors={};self.locality=30;self.writer_zero_checks=0
 def reset_layers(self,**kwargs):pass
 def close(self):pass
class Gate:
 def __init__(self,m):self.m=m;m.gate=self;self.handles=[];self.failure_gate=None;self.failure_reconstruction=None
 def install_maps(self,*a):pass
 def install_directions(self,*a):pass
 def reset_gate(self,mode='P',zero=True,mask='causal',snapshot=False):
  self.mode=mode;self.projection_checks=self.query_checks=self.context_checks=8
  self.cache={l:{k:torch.zeros(768)for k in('k','v')}for l in M.LAYERS}
  self.diag={l:{'source_q':torch.zeros(12,64),'context_max_abs_error':1.2345678901234567e-5}for l in M.LAYERS}
  self.prefix_contexts={l:{k:torch.zeros(1,128,768)for k in('current','delivered')}for l in M.LAYERS}
  self.current_contexts={l:torch.zeros(1,512,768)for l in M.LAYERS}
  self.mask_records={l:copy.deepcopy(maskcheck)for l in M.LAYERS}
  self.gate_cache={l:{k:torch.zeros(12,511)for k in('b','mass','target_b','target_mass')}for l in M.LAYERS}if zero else{}
  self.gate_checks={l:copy.deepcopy(gatecheck)for l in M.LAYERS}if zero else{}
  self.localization_norms={l:{k:torch.zeros(511,dtype=torch.float32 if k=='s'else torch.float64)for k in('s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm')}for l in M.LAYERS}if zero else{}
  self.snapshots={l:from_shapes(snapshot_shapes)for l in(4,11)}if snapshot else{}
  for l,d in self.snapshots.items():d['gate_state']=self.gate_cache[l]
 def close(self):pass

def eval_stub(run,x,baseline=None,keep=False,control=False):
 assert len(x)==512;run.count+=1;run.m.lm_head.cb(None,(torch.zeros(1,512,768),))
 n=128 if control else 511
 z={'loss':torch.ones(127 if control else 511),'kl':torch.zeros(n),'pred':torch.zeros(n,dtype=torch.long)}
 if control:z.update(kl_p=torch.zeros(n),agreement_p=torch.ones(n,dtype=torch.bool))
 return z,torch.zeros(1).expand(n,50257),{'selected':{}},{}
def control_stub(run,x,baseline,pbaseline,failure):return eval_stub(run,x,control=True)
def p_metrics(z,ll,pl):z.update(kl_p=torch.zeros(len(z['pred'])),agreement_p=torch.ones(len(z['pred']),dtype=torch.bool))
class FakeReadout:
 def __init__(self,base,targets):
  self.base=base;self.targets=targets.clone();self.additive=torch.zeros(1,dtype=torch.float64).expand(511,50257)
  self.layers=[];self.single_losses={};self.loss_p=torch.ones(511,dtype=torch.float64)
 def add_singleton(self,l,z):assert l==M.LAYERS[len(self.layers)];self.layers.append(l);self.single_losses[l]=torch.ones(511,dtype=torch.float64)
 def finish(self,joint,native):
  assert self.layers==list(M.LAYERS)
  keys=('loss_p','loss_joint','loss_additive','sum_singleton_delta','J','R','joint_minus_sum','identity_error',
   'kl_joint_additive','kl_additive_joint','kl_p_additive','kl_native_additive','centered_logit_l2','centered_logit_rms','centered_logit_max_abs','joint_delta_centered_l2','additive_delta_centered_l2')
  d={k:torch.ones(511,dtype=torch.float64)if k.startswith('loss_')else torch.zeros(511,dtype=torch.float64)for k in keys}
  d['singleton_losses']=torch.ones(8,511,dtype=torch.float64)
  d.update({k:torch.ones(511,dtype=torch.bool)for k in('agreement_joint_additive','agreement_p_additive','agreement_native_additive')})
  return d
class Profile:
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def key_averages(self):return [SimpleNamespace(key='aten::_scaled_dot_product_efficient_attention',count=12)]

def counts(out):
 result={}
 for name in('gates','suffix','hidden','readout_rows','additive'):
  files=list((out/name).glob('*.pt'));result[name+'_files']=len(files)
  if name=='hidden':result['hidden_tensors']=sum(len(torch.load(p,weights_only=True,mmap=True))for p in files)
  if name in('gates','suffix'):
   result[name+'_states']=sum(sum(len(x if name=='gates'else x['gates'])for x in torch.load(p,weights_only=True,mmap=True).values())for p in files)
 snap=torch.load(out/'attention_snapshots.pt',weights_only=True,mmap=True);result['attention_states']=sum(map(len,snap.values()))
 raw=torch.load(out/'per_token.pt',weights_only=True,mmap=True);result['main_records']=len(raw)
 result['checks']=len(json.loads((out/'checks.json').read_text()))
 result['suffix_checks']=len(json.loads((out/'suffix_checks.json').read_text()))
 return result

started=time.time();before_free=shutil.disk_usage(ROOT).free
source_hashes={n:sha(ROOT/n)for n in('gpt2_gate_additive.py','gate_additive_readout.py','gate_additive_failure.py','analyze_gate_additive.py','GPT2_GATE_ADDITIVE_PROPOSAL.md','gate_additive_inputs.py')}
result={'status':'running','fixture':'actual_writer_full_layout','no_neural_model_or_tokenizer':True,'gpu_used':False,'stubbed_model_forwards':1992,'actual_model_forwards':0,'source_hashes':source_hashes}
with tempfile.TemporaryDirectory(prefix='gate-additive-full-layout-')as td:
 p=Path(td);cfg=p/'cfg.json';out=p/'run';dump(cfg,{'models':{'gpt2':{'revision':old['revision'],'path':'SYNTHETIC_NO_MODEL'}}})
 argv=['size_fixture','--config',str(cfg),'--inputs',str(ROOT/'additive_inputs'),'--calibration-reference',str(ROOT/'gpt2_gate_fixed512_run'),'--out',str(out)]
 print('SYNTHETIC_LAYOUT_DIRECTORY',out,flush=True)
 with ExitStack()as stack:
  for target,name,value in[(sys,'argv',argv),(M,'LayerWrites',Probe),(M,'GateLocalization',Gate),(M,'modelhash',lambda m:old['model_hash_before']),
      (M,'fingerprint',fp),(M,'tensor_hash',lambda t:fp(t)['sha256']),(M,'evaluate_control',control_stub),(M,'add_p_metrics',p_metrics),(M,'AdditiveReadout',FakeReadout),
      (M.EvidenceRunner,'ev',eval_stub),(torch.cuda,'reset_peak_memory_stats',lambda:None),(torch.cuda,'max_memory_allocated',lambda:0),
      (torch.cuda,'max_memory_reserved',lambda:0),(torch.profiler,'profile',lambda **kw:Profile()),(torch,'set_num_threads',lambda n:None)]:stack.enter_context(patch.object(target,name,value))
  stack.enter_context(patch.object(M.AutoModelForCausalLM,'from_pretrained',return_value=FakeModel()))
  try:M.main()
  except BaseException as error:
   result.update(status='failed',failure_type=type(error).__name__,failure_message=str(error))
   if(out/'manifest.json').exists():result['fixture_manifest']=json.loads((out/'manifest.json').read_text())
   raise
  finally:
   sizes={str(f.relative_to(out)):f.stat().st_size for f in out.rglob('*')if f.is_file()}
   result['total_serialized_bytes']=sum(sizes.values());result['file_sizes']=sizes
   result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
 manifest=json.loads((out/'manifest.json').read_text());result['counts']=counts(out)
 expected={'gates_files':32,'gates_states':1792,'suffix_files':32,'suffix_states':1792,'hidden_files':32,'hidden_tensors':928,'readout_rows_files':4,'additive_files':32,'attention_states':4,'main_records':1000,'checks':1800,'suffix_checks':992}
 assert result['counts']==expected,(result['counts'],expected)
 assert manifest['status']=='completed'and manifest['forward_passes']==1992 and manifest['forward_attempts']==1992
 result['component_bytes']={name:sum(v for k,v in sizes.items()if k.split('/')[0]==name)for name in sorted({k.split('/')[0]for k in sizes})}
 result.update(status='passed',cap_bytes=M.ESTIMATED_OUTPUT_BYTES,reserved_failure_bytes=M.CAPSULE_LIMIT,
  normal_cap_bytes=M.ESTIMATED_OUTPUT_BYTES-M.CAPSULE_LIMIT,
  margin_after_failure_reserve_bytes=M.ESTIMATED_OUTPUT_BYTES-M.CAPSULE_LIMIT-result['total_serialized_bytes'],
  minimum_free_recorded_bytes=min(d['free_bytes']for d in json.loads((out/'disk_checks.json').read_text())),
  free_at_end_bytes=shutil.disk_usage(out).free,source_hashes_unchanged=all(sha(ROOT/n)==h for n,h in source_hashes.items()))
 assert result['margin_after_failure_reserve_bytes']>=0 and result['source_hashes_unchanged']
 assert result['peak_rss_kib']<=3*1024**2
 result['limitations']='Serialization and orchestration resource fixture only: forward/logit/readout values and in-memory tensor hashes are stubs. Stored tensors own full uncompressed storage. Actual JSON float digit counts can vary; cap is enforced by production checkpoints. No claim about production model/RSS or scientific accuracy.'
result.update(temporary_output_removed=not p.exists(),free_before_bytes=before_free,free_after_cleanup_bytes=shutil.disk_usage(ROOT).free,wall_seconds=time.time()-started)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in('file_sizes','component_bytes')},indent=2))
