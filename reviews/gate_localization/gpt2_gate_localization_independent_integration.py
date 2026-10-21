"""Independent physical512 random-weight CPU integration; no pretrained weights."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys,copy,tempfile,json,resource,hashlib
from unittest.mock import patch
import torch
from transformers import GPT2Config,GPT2LMHeadModel
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P));torch.set_num_threads(2)
from gpt2_gate_localization import GateLocalization,localization_map,mode_spec,CONDITIONS,LAYERS
from gpt2_gate_subspace import GateRunner,add_p_metrics
from gpt2_gate_subspace_v2 import evaluate_control
from gpt2_layer_writes import LayerWrites
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sourcehash=sha(P/'gpt2_gate_localization.py');torch.manual_seed(8173)
config=GPT2Config(n_layer=12,n_head=12,n_embd=768,n_inner=16,n_positions=512,vocab_size=79,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,use_cache=False);config._attn_implementation='sdpa'
model=GPT2LMHeadModel(config).eval().requires_grad_(False)
with torch.no_grad():
 for block in model.transformer.h:
  block.attn.c_attn.bias.copy_(torch.randn_like(block.attn.c_attn.bias)*.025);block.attn.c_proj.bias.copy_(torch.randn_like(block.attn.c_proj.bias)*.035)
x=torch.randint(0,79,(512,));y=x.clone();y[128:]=torch.roll(y[128:],1)
with torch.inference_mode():unhooked=model(x[None]).logits[0,:-1].clone()
selected={'layer':2,'row':447,'column':3,'weight':float(model.transformer.h[2].mlp.c_proj.weight[3,447])}
probe=LayerWrites(model,selected,[447,138]);gate=GateLocalization(model)
maps=torch.load(P/'gpt2_gate_fixed512_run/calibration_maps.pt',weights_only=True,map_location='cpu');ds=torch.load(P/'localization_inputs_v2/directions.pt',weights_only=True,map_location='cpu')
gate.install_maps(maps,'independent-frozen-calibration');gate.install_directions(ds,'independent-frozen-directions')
cap={};handle=model.transformer.ln_f.register_forward_hook(lambda m,a,o:cap.update(hidden=o.detach().clone()))
checks=[];kLError=0.;maxguard=0.
modes=['C','P','current','fullrank','remove_pc1_L4','orthogonal_control_L4','position_L4','remove_pc1_L11','orthogonal_control_L11','position_L11','remove_pc1_all','orthogonal_control_all','position_all']
with tempfile.TemporaryDirectory(prefix='gate-local-independent-')as td,patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
 run=GateRunner(model,probe,Path(td))
 def forward(tokens,mode,c=None,p=None,control=False):
  probe.reset_layers(E=False,blocked=[]);gate.reset_gate(mode,zero=mode!='C',mask='native'if mode=='C'else'causal',snapshot=mode=='remove_pc1_all')
  z,ll,_,_=evaluate_control(run,tokens,c,p,{})if control else run.ev(tokens,baseline=c,keep=True)
  if p is not None and not control:add_p_metrics(z,ll,p)
  return {'z':z,'ll':ll.clone(),'hidden':cap['hidden'].clone(),'source':copy.deepcopy(gate.cache),'q':{l:gate.diag[l]['source_q'].clone()for l in LAYERS},'gates':copy.deepcopy(gate.gate_cache),'contexts':copy.deepcopy(gate.current_contexts),'prefix':copy.deepcopy(gate.prefix_contexts),'norms':copy.deepcopy(gate.localization_norms),'checks':copy.deepcopy(gate.gate_checks)}
 c=forward(x,'C');assert torch.equal(c['ll'],unhooked);checks.append('Native C hooks reproduce no-hook physical512 logits exactly')
 p=forward(x,'P',c['ll']);add_p_metrics(p['z'],p['ll'],p['ll']);add_p_metrics(c['z'],c['ll'],p['ll'])
 for mode in modes:
  r=c if mode=='C'else p if mode=='P'else forward(x,mode,c['ll'],p['ll'])
  # Independent full-vocabulary CPU64 loss and KL, preserving target alignment.
  lp=r['ll'].double().log_softmax(-1);expected=-lp[torch.arange(511),x[1:]]
  assert torch.allclose(expected,r['z']['loss'].double(),rtol=0,atol=2e-6)
  ref=p['ll'].double().log_softmax(-1);kl=(ref.exp()*(ref-lp)).sum(-1);err=float((kl-r['z']['kl_p'].double()).abs().max());assert err<2e-6;kLError=max(kLError,err)
  if mode in ('current','fullrank'):assert torch.equal(r['ll'],p['ll'])and torch.equal(r['hidden'],p['hidden'])
  family,affected=mode_spec(mode)
  if mode!='C':
   assert all(torch.equal(v,p['source'][l][k])for l,d in r['source'].items()for k,v in d.items())
   assert all(torch.equal(v,p['q'][l])for l,v in r['q'].items())
   assert all(torch.count_nonzero(d['v'])==0 for d in r['source'].values())
   assert all(ch['source_context_exact']and ch['unaffected_context_exact']for ch in r['checks'].values())
   maxguard=max(maxguard,max(ch['norm_match_error']for ch in r['checks'].values()))
   for l in LAYERS:
    if l not in affected:assert torch.equal(r['prefix'][l]['current'],r['prefix'][l]['delivered'])
   if len(affected)==1:
    l=affected[0];assert torch.equal(r['contexts'][l],p['contexts'][l])
    assert all(torch.equal(r['gates'][l][k],p['gates'][l][k])for k in ('b','mass'))
    assert torch.equal(r['norms'][l]['s'],p['norms'][l]['s'])
  q=forward(y,mode,c['ll'],p['ll'],True)
  assert torch.equal(r['ll'][:128],q['ll'])and torch.equal(r['hidden'][:,:128],q['hidden'][:,:128])
  for k in ('loss','kl','kl_p','pred','agreement_p'):assert torch.equal(r['z'][k][1:127],q['z'][k][1:127])
  assert all(torch.equal(v[:,:127],q['gates'][l][k][:,:127])for l,d in r['gates'].items()for k,v in d.items())
  assert all(torch.equal(v,q['prefix'][l][k])for l,d in r['prefix'].items()for k,v in d.items())
  checks.append(mode+': source/query0, singleton prepatch if applicable, unaffected contexts, suffix and scoring')
  del r,q
 assert run.count==26
handle.remove();gate.close();probe.close();assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
assert sourcehash==sha(P/'gpt2_gate_localization.py'),'Runner changed during integration; rerun frozen bytes'
result={'status':'passed','runner_sha256':sourcehash,'random_model_forwards':27,'gate_runner_forwards':26,'physical_length':512,'changed_suffix_start':128,'checks':checks,'max_fp64_kl_error':kLError,'max_norm_guard_error':maxguard,'hook_cleanup':True,'nonzero_qkv_output_biases':True,'pretrained_loaded':False,'gpu_used':False,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
assert result['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
