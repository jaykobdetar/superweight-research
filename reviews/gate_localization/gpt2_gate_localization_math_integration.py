"""All31 modes on a random small-inner CPU GPT-2; never load pretrained weights."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import copy,json,resource,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import GPT2Config,GPT2LMHeadModel
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));torch.set_num_threads(1)
from gpt2_gate_localization import GateLocalization,CONDITIONS,LAYERS,localization_map,mode_spec,INPUT_HASHES
from gpt2_gate_subspace import GateRunner,add_p_metrics
from gpt2_layer_writes import LayerWrites
from run_transfer import sha
maps=torch.load(root/'gpt2_gate_fixed512_run/calibration_maps.pt',weights_only=True,map_location='cpu')
ds=torch.load(root/'localization_inputs_v2/directions.pt',weights_only=True,map_location='cpu')
gen=torch.Generator().manual_seed(1945);arithmetic=0;max_math=0.;maxnorm=0.
for l in LAYERS:
 b=torch.randn(12,511,generator=gen)+torch.arange(1,512).float().log()[None,:];a=b.sigmoid();mp={k:v.float()for k,v in maps[l].items()};dd={k:ds[l][k].float()for k in ('u','v')}
 for mode in CONDITIONS[1:]:
  tb,ta,le,me,nd,ne=localization_map(mode,l,b,a,mp,dd);family,affected=mode_spec(mode)
  # Independent NumPy FP64 reference applied to saved FP32 inputs/vectors.
  bb=b.T.double().numpy();qq=np.log(np.arange(1,512,dtype=np.float32)).astype(np.float64)[:,None]
  x=bb-qq-mp['alpha'].double().numpy();sv=x@dd['u'].double().numpy()
  if l in affected:
   want=bb-sv[:,None]*dd['u'].double().numpy()if family=='remove_pc1'else bb-sv[:,None]*dd['v'].double().numpy()if family=='orthogonal_control'else mp['alpha'].double().numpy()+qq
   err=float(np.max(np.abs(want-tb.T.double().numpy())));assert err<1e-5;max_math=max(max_math,err)
   assert torch.equal(ta,tb.sigmoid())
  elif mode!='fullrank':assert tb is b and ta is a
  else:assert ta is a and le<1e-4 and me<1e-4
  assert ne<1e-6;maxnorm=max(maxnorm,ne);arithmetic+=1
 # Extend the suffix while preserving prefix queries; map must be row-local.
 for mode in CONDITIONS[1:]:
  bb=b.clone();bb[:,127:]+=8
  t1=localization_map(mode,l,b,a,mp,dd);t2=localization_map(mode,l,bb,bb.sigmoid(),mp,dd)
  assert torch.equal(t1[0][:,:127],t2[0][:,:127])and torch.equal(t1[1][:,:127],t2[1][:,:127])

# Random model, nonzero Q/output biases.19 physical tokens keep this author test bounded;
# the full512 production orchestration/scoring is exercised separately by saved-data fixtures.
torch.manual_seed(929)
cfg=GPT2Config(n_layer=12,n_head=12,n_embd=768,n_inner=16,n_positions=512,vocab_size=97,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,use_cache=False)
cfg._attn_implementation='sdpa';m=GPT2LMHeadModel(cfg).eval().requires_grad_(False)
with torch.no_grad():
 for block in m.transformer.h:
  block.attn.c_attn.bias.copy_(torch.randn_like(block.attn.c_attn.bias)*.03)
  block.attn.c_proj.bias.copy_(torch.randn_like(block.attn.c_proj.bias)*.03)
selected={'layer':2,'row':447,'column':3,'weight':float(m.transformer.h[2].mlp.c_proj.weight[3,447])}
probe=LayerWrites(m,selected,[447,138]);gate=GateLocalization(m);gate.install_maps(maps,'synthetic-context-frozen-maps');gate.install_directions(ds,INPUT_HASHES['directions.pt'])
x=torch.randint(0,97,(19,));y=x.clone();y[10:]=torch.roll(y[10:],1);capture={}
h=m.transformer.ln_f.register_forward_hook(lambda mod,a,o:capture.update(hidden=o.detach().clone()))
sourcechecks=singletons=suffix=0;maxkl=0.
with tempfile.TemporaryDirectory(prefix='localization-math-')as td,patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
 run=GateRunner(m,probe,Path(td))
 def ev(tokens,mode,cl=None,pl=None):
  probe.reset_layers(E=False,blocked=[]);gate.reset_gate(mode,zero=mode!='C',mask='native'if mode=='C'else'causal')
  z,ll,cap,norm=run.ev(tokens,baseline=cl,keep=True)
  if pl is not None:add_p_metrics(z,ll,pl)
  return {'z':z,'logits':ll.clone(),'hidden':capture['hidden'].clone(),'source':copy.deepcopy(gate.cache),
    'gates':copy.deepcopy(gate.gate_cache),'contexts':copy.deepcopy(gate.current_contexts),'norms':copy.deepcopy(gate.localization_norms)}
 c=ev(x,'C');p=ev(x,'P',c['logits']);cl,pl=c['logits'],p['logits']
 for mode in CONDITIONS:
  r=c if mode=='C'else p if mode=='P'else ev(x,mode,cl,pl)
  if mode not in('C','P'):
   expected=(pl.double().log_softmax(-1).exp()*(pl.double().log_softmax(-1)-r['logits'].double().log_softmax(-1))).sum(-1)
   err=float((expected-r['z']['kl_p'].double()).abs().max());assert err<2e-6;maxkl=max(maxkl,err)
  if mode in('current','fullrank'):assert torch.equal(r['logits'],pl)and torch.equal(r['hidden'],p['hidden'])
  if mode!='C':
   assert all(torch.equal(v,p['source'][l][k])for l,d in r['source'].items()for k,v in d.items());sourcechecks+=1
   assert all(d['unaffected_context_exact']and d['norm_match_error']<=1e-6 for d in gate.gate_checks.values()) if mode not in ('C','P') else True
  family,affected=mode_spec(mode)
  if len(affected)==1:
   l=affected[0]
   assert torch.equal(r['contexts'][l],p['contexts'][l])and all(torch.equal(r['gates'][l][k],p['gates'][l][k])for k in('b','mass'))
   assert torch.equal(r['norms'][l]['s'],p['norms'][l]['s']);singletons+=1
  q=ev(y,mode,cl,pl)
  assert torch.equal(r['hidden'][:,:10],q['hidden'][:,:10])and torch.equal(r['logits'][:10],q['logits'][:10]);suffix+=1
  assert all(torch.equal(v[:,:9],q['gates'][l][k][:,:9])for l,d in r['gates'].items()for k,v in d.items())
 assert run.count==62
h.remove();gate.close();probe.close();assert all(not o._forward_hooks and not o._forward_pre_hooks for o in m.modules())
result={'status':'passed','pretrained_loaded':False,'gpu_used':False,'random_model_forwards':62,'physical_test_length':19,
 'math_cases':arithmetic,'row_local_map_cases':arithmetic,'max_numpy_fp64_map_error':max_math,'max_norm_error':maxnorm,
 'singleton_actual_prepatch_cases':singletons,'source_cases':sourcechecks,'exact_fixed_shape_prefix_cases':suffix,
 'max_kl_p_fp64_error':maxkl,'nonzero_q_and_output_biases':True,'hook_cleanup':True,
 'runner_sha256':sha(root/'gpt2_gate_localization.py'),'script_sha256':sha(__file__),
 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
assert result['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
