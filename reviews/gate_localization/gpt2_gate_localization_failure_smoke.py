"""Inject a record failure using fake model/probe; no pretrained model or GPU."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import json,sys,tempfile,resource
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(1)
import gpt2_gate_localization as M
from run_transfer import dump,sha
old=json.loads((ROOT/'gpt2_run/manifest.json').read_text())
class Handle:
 def remove(self):pass
class Ln:
 def register_forward_hook(self,cb):self.cb=cb;return Handle()
class FakeModel(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.transformer=SimpleNamespace(ln_f=Ln())
class Probe:
 def __init__(self,m,s,coords):self.selected=s;self.vectors={};self.locality=30;self.writer_zero_checks=0
 def reset_layers(self,**kwargs):pass
 def close(self):pass
class Gate:
 def __init__(self,m):self.m=m;m.gate=self;self.handles=[];self.failure_gate=None;self.failure_reconstruction=None
 def install_maps(self,*args):pass
 def install_directions(self,*args):pass
 def reset_gate(self,mode="P",zero=True,mask="causal",snapshot=False):
  self.mode=mode;self.projection_checks=self.query_checks=self.context_checks=8;self.gate_checks={l:{}for l in range(4,12)}if zero else{}
  self.cache={l:{k:torch.ones(1)for k in('k','v')}for l in range(4,12)}
  self.diag={l:{'source_q':torch.ones(1),'context_max_abs_error':0.}for l in range(4,12)}
  self.prefix_contexts={l:{k:torch.ones(1,128,768)for k in('current','delivered')}for l in range(4,12)}
  self.mask_records={l:{'kind':'implicit_causal'}for l in range(4,12)};self.gate_cache={};self.snapshots={};self.current_contexts={};self.localization_norms={}
 def close(self):pass
class FakeRunner:
 def __init__(self,m,p,out):self.m=m;self.p=p;self.out=out;self.count=0;self.raw={};self.traces={};self.checks=[]
 def ev(self,x,baseline=None,keep=False):
  assert len(x)==512;self.count+=1
  hidden=torch.full((1,512,768),1. if self.m.gate.mode=='C'else 2.)
  self.m.transformer.ln_f.cb(None,None,hidden)
  return {'loss':torch.ones(511),'kl':torch.zeros(511),'pred':torch.zeros(511,dtype=torch.long)},torch.zeros(511,97),{'selected':{}},{}
 def rec(self,key,*args,**kwargs):assert key.endswith('__C');raise RuntimeError('intentional first-C record failure after P forward')
 def checkpoint(self):
  torch.save(self.raw,self.out/'per_token.pt');torch.save(self.traces,self.out/'traces.pt');dump(self.out/'checks.json',self.checks)
with tempfile.TemporaryDirectory(prefix='gate_fixed512_failure_')as td:
 p=Path(td);cfg=p/'cfg.json';dump(cfg,{'models':{'gpt2':{'revision':old['revision'],'path':'synthetic-no-checkpoint'}}});out=p/'run'
 argv=['fixture','--config',str(cfg),'--inputs',str(ROOT/'localization_inputs_v2'),'--calibration-reference',str(ROOT/'gpt2_gate_subspace_run'),'--out',str(out)]
 with patch.object(sys,'argv',argv),patch.object(M.AutoModelForCausalLM,'from_pretrained',return_value=FakeModel()),patch.object(M,'modelhash',return_value=old['model_hash_before']),patch.object(M,'LayerWrites',Probe),patch.object(M,'GateLocalization',Gate),patch.object(M,'GateRunner',FakeRunner),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=123),patch.object(torch.cuda,'max_memory_reserved',return_value=456):
  try:M.main()
  except RuntimeError as error:assert str(error)=='intentional first-C record failure after P forward'
  else:raise AssertionError('injected failure did not propagate')
 fm=json.loads((out/'manifest.json').read_text());fs=torch.load(out/'failure_state.pt',weights_only=True,map_location='cpu')
 assert fm['status']==fm['phase']=='failed'and'completed_utc'not in fm
 assert fm['forward_passes']==fm['forward_attempts']==2 and fm['records']==0
 assert fm['last_forward_identity']['trial']=='P'and fm['active_record_target']['trial']=='C'and fm['active_check_target']['trial']=='C'
 assert fs['current_tensor_identity']['trial']=='P'and torch.equal(fs['current_hidden'],torch.full((1,512,768),2.))
 assert fs['pending_record_bundle']['hidden'].shape==(1,512,768)and torch.equal(fs['pending_record_bundle']['hidden'],torch.ones(1,512,768))
 assert fm['wall_seconds']>0 and fm['peak_gpu_allocated_bytes']==123 and fm['peak_gpu_reserved_bytes']==456 and fm['peak_process_rss_kib']>0
 # Pre-load disk rejection must never call model loading and must save zero-forward status.
 low=p/'low-disk';low_argv=argv[:-1]+[str(low)]
 with patch.object(sys,'argv',low_argv),patch.object(M.shutil,'disk_usage',return_value=SimpleNamespace(free=7*1024**3-1)),patch.object(M.AutoModelForCausalLM,'from_pretrained')as load_model:
  try:M.main()
  except AssertionError:pass
  else:raise AssertionError('low disk was accepted')
  load_model.assert_not_called()
 lm=json.loads((low/'manifest.json').read_text());assert lm['status']=='failed' and lm['phase']=='preflight' and lm['forward_attempts']==lm['forward_passes']==0 and lm['model_loaded']is False and lm['wall_seconds']>=0
 # The per-input floor is separately enforced inside the partial-output handler.
 partial=p/'floor';floor_argv=argv[:-1]+[str(partial)]
 with patch.object(sys,'argv',floor_argv),patch.object(M.shutil,'disk_usage',side_effect=[SimpleNamespace(free=8*1024**3),SimpleNamespace(free=3*1024**3-1)]),patch.object(M.AutoModelForCausalLM,'from_pretrained',return_value=FakeModel()),patch.object(M,'modelhash',return_value=old['model_hash_before']),patch.object(M,'LayerWrites',Probe),patch.object(M,'GateLocalization',Gate),patch.object(M,'GateRunner',FakeRunner),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(torch.cuda,'max_memory_reserved',return_value=0):
  try:M.main()
  except AssertionError as e:assert 'Disk reserve exhausted' in str(e)
  else:raise AssertionError('per-input reserve guard not enforced')
 pf=json.loads((partial/'manifest.json').read_text());assert pf['status']=='failed' and pf['active_failure_phase']=='disk_guard' and pf['forward_attempts']==pf['forward_passes']==0
 assert (partial/'failure_state.pt').exists()and (partial/'per_token.pt').exists()
 # An input mismatch and a model-loading exception are both zero-forward setup failures.
 for case in ('input_hash','model_load'):
  target=p/case;case_argv=argv[:-1]+[str(target)]
  with patch.object(sys,'argv',case_argv),patch.object(M.shutil,'disk_usage',return_value=SimpleNamespace(free=8*1024**3)),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(M.AutoModelForCausalLM,'from_pretrained',side_effect=RuntimeError('injected model load failure'))as loader:
   if case=='input_hash':
    with patch.dict(M.INPUT_HASHES,{'tokens.pt':'0'*64}):
     try:M.main()
     except AssertionError:pass
     else:raise AssertionError('input mismatch accepted')
    loader.assert_not_called()
   else:
    try:M.main()
    except RuntimeError as e:assert str(e)=='injected model load failure'
    else:raise AssertionError('model load exception missing')
    loader.assert_called_once()
  sm=json.loads((target/'manifest.json').read_text())
  assert sm['status']=='failed' and sm['active_failure_phase']=='setup' and sm['forward_attempts']==sm['forward_passes']==0 and sm['model_loaded']is False
 result={'status':'passed','preload_zero_forward_disk_failure':True,'per_input_reserve_failure_preserved':True,'fake_model_only':True,'pretrained_loaded':False,'gpu_used':False,'simulated_forwards':2,'all_physical_lengths_512':True,
  'input_hash_failure_preserved':True,'model_load_failure_preserved':True,
  'current_P_tensor_and_pending_C_record_distinguished':True,'partial_bundle_preserved':True,'failed_status_resources_and_count_verified':True,
  'source_hashes':{n:sha(ROOT/n)for n in('gpt2_gate_localization.py','analyze_gate_localization.py','GPT2_GATE_LOCALIZATION.md')},
  'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 assert result['peak_rss_kib']<3*1024**2
 Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
