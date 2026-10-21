"""Independent random-weight CPU integration of physical512 additive readout."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys,tempfile,json,hashlib,resource,copy
from unittest.mock import patch
import torch
from transformers import GPT2Config,GPT2LMHeadModel
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P));torch.set_num_threads(2)
from gpt2_gate_additive import EvidenceRunner,compact_gates,READOUT_ROWS,LOSS64_FP32_SANITY_TOLERANCE
from gate_additive_readout import AdditiveReadout,loss64
from gpt2_gate_localization import GateLocalization,mode_spec,CONDITIONS,LAYERS
from gpt2_gate_subspace import add_p_metrics
from gpt2_gate_subspace_v2 import evaluate_control
from gpt2_layer_writes import LayerWrites
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={n:sha(P/n)for n in ('gpt2_gate_additive.py','gate_additive_readout.py','gpt2_gate_localization.py')}
torch.manual_seed(70913);config=GPT2Config(n_layer=12,n_head=12,n_embd=768,n_inner=16,n_positions=512,vocab_size=79,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,use_cache=False);config._attn_implementation='sdpa'
model=GPT2LMHeadModel(config).eval().requires_grad_(False)
with torch.no_grad():
 for b in model.transformer.h:
  b.attn.c_attn.bias.copy_(torch.randn_like(b.attn.c_attn.bias)*.025);b.attn.c_proj.bias.copy_(torch.randn_like(b.attn.c_proj.bias)*.035)
x=torch.randint(79,(512,));rotated=x.clone();rotated[128:]=torch.roll(rotated[128:],1)
with torch.inference_mode():native=model(x[None]).logits[0,:-1].clone()
selected={'layer':2,'row':447,'column':3,'weight':float(model.transformer.h[2].mlp.c_proj.weight[3,447])};probe=LayerWrites(model,selected,[447,138]);gate=GateLocalization(model)
gate.install_maps(torch.load(P/'gpt2_gate_fixed512_run/calibration_maps.pt',weights_only=True,map_location='cpu'),'frozen-fixture-maps');gate.install_directions(torch.load(P/'localization_inputs_v2/directions.pt',weights_only=True,map_location='cpu'),'frozen-fixture-directions')
captured={};h1=model.lm_head.register_forward_pre_hook(lambda m,a:captured.update(head=a[0].detach().clone()));h2=model.transformer.ln_f.register_forward_hook(lambda m,a,o:captured.update(norm=o.detach().clone()))
errors={'lm_head_FP64_max':0.,'loss64_vs_FP32_max':0.,'J_R_identity_max':0.};families={};hidden_states={};kept_main=kept_suffix=0
with tempfile.TemporaryDirectory(prefix='additive-independent-')as td,patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
 run=EvidenceRunner(model,probe,Path(td))
 def ev(tokens,mode,cb=None,pb=None,suffix=False):
  probe.reset_layers(E=False,blocked=[]);gate.reset_gate(mode,zero=mode!='C',mask='native'if mode=='C'else'causal',snapshot=False)
  z,ll,_,_=evaluate_control(run,tokens,cb,pb,{})if suffix else run.ev(tokens,baseline=cb,keep=True)
  if pb is not None and not suffix:add_p_metrics(z,ll,pb)
  assert torch.equal(captured['head'],captured['norm'])
  return z,ll.clone(),captured['head'].clone(),copy.deepcopy(gate.gate_cache)
 cz,cl,ch,cg=ev(x,'C');assert torch.equal(native,cl)
 pz,pl,ph,pg=ev(x,'P',cl);add_p_metrics(cz,cl,pl);add_p_metrics(pz,pl,pl)
 acc=None;single32={}
 for mode in CONDITIONS:
  z,ll,h,g=(cz,cl,ch,cg)if mode=='C'else(pz,pl,ph,pg)if mode=='P'else ev(x,mode,cl,pl)
  assert h.shape==(1,512,768)
  raw=ll[list(READOUT_ROWS)];reference=h[0,list(READOUT_ROWS)].double()@model.lm_head.weight.double().T
  err=(reference-raw.double()).abs();errors['lm_head_FP64_max']=max(errors['lm_head_FP64_max'],float(err.max()))
  d=768;u=2**-24;gamma=(2*d+2)*u/(1-(2*d+2)*u);bound=gamma*(h[0,list(READOUT_ROWS)].double().abs()@model.lm_head.weight.double().abs().T)+1e-12
  assert bool((err<=bound).all())
  discrepancy=float((loss64(ll,x[1:])-z['loss'].double()).abs().max());assert discrepancy<=LOSS64_FP32_SANITY_TOLERANCE;errors['loss64_vs_FP32_max']=max(errors['loss64_vs_FP32_max'],discrepancy)
  fam,affected=mode_spec(mode)
  kg=compact_gates(mode,g);kept_main+=len(kg);assert set(kg)==(set(LAYERS)if mode=='P'else set(affected))
  if mode not in ('current','fullrank'):hidden_states[mode]=h
  else:assert torch.equal(h,ph)and torch.equal(ll,pl)
  if fam:
   if len(affected)==1:
    if affected[0]==4:assert acc is None;acc=AdditiveReadout(pl,x[1:]);single32={}
    acc.add_singleton(affected[0],ll);single32[affected[0]]=z['loss']
   else:
    data=acc.finish(ll,cl);assert all(v.shape[-1]==511 for k,v in data.items()if torch.is_tensor(v))
    errors['J_R_identity_max']=max(errors['J_R_identity_max'],float(data['identity_error'].max()))
    assert max(float((data['singleton_losses'][j]-single32[l].double()).abs().max())for j,l in enumerate(LAYERS))<=LOSS64_FP32_SANITY_TOLERANCE
    assert data['J'][1:].numel()==510 and data['J'][1:127].numel()==126 and data['J'][127:].numel()==384
    families[fam]={'J_mean':float(data['J'][1:].mean()),'R_mean':float(data['R'][1:].mean()),'nonadd_mean':float(data['joint_minus_sum'][1:].mean())};acc=None
  zz,zl,zh,zg=ev(rotated,mode,cl,pl,True)
  assert torch.equal(ll[:128],zl)and torch.equal(h[:,:128],zh[:,:128])
  for k in ('loss','pred','kl','kl_p','agreement_p'):assert torch.equal(z[k][1:127],zz[k][1:127])
  sk=compact_gates(mode,zg,prefix=True);kept_suffix+=len(sk)
  for l,dct in kg.items():
   for k,v in dct.items():assert(v is None and sk[l][k]is None)or torch.equal(v[:,:127],sk[l][k])
  print('MODE_PASSED',mode,flush=True)
 assert run.count==62 and len(hidden_states)==29 and kept_main==kept_suffix==56 and len(families)==3
h1.remove();h2.remove();gate.close();probe.close();assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
out={'status':'passed','loaded_source_sha256':hashes,'current_source_sha256':{n:sha(P/n)for n in hashes},'random_model_forwards':63,'physical_length':512,'conditions':31,'suffix_controls':31,'post_LN_hidden_states':29,'compact_gate_states_main':kept_main,'compact_gate_states_suffix':kept_suffix,'readout_rows':list(READOUT_ROWS),'q0_excluded_primary':True,'errors':errors,'family_fixture_diagnostics':families,'hook_cleanup':True,'nonzero_QKV_output_biases':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pretrained_loaded':False,'gpu_used':False}
assert out['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
