"""CPU extracted probe hook and token-edit checks; no model/GPU load."""
import ast,json,os,resource
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];ns={'torch':torch,'F':F};node=next(n for n in ast.parse((ROOT/'newline_history.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='Probe');exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted Probe>','exec'),ns)
gen=torch.Generator().manual_seed(13);p=ns['Probe'].__new__(ns['Probe']);p.p=64;p.q=128;p.cache={};p.masks={};hidden=torch.randn(1,160,32,generator=gen)
base=torch.zeros(160,160);base.masked_fill_(torch.ones(160,160,dtype=torch.bool).triu(1),-torch.inf);base=base[None,None]
for edges in [[],[(0,128,64)],[(1,128,64)],[(0,128,64),(1,128,64)],[(0,128,63)]]:
 p.edges=edges
 for l in range(16):
  kwargs={'hidden_states':hidden,'attention_mask':torch.ones(1)};result=p.attn(l,(),kwargs);expected=base.clone()
  for ll,dest,src in edges:
   if ll==l:expected[:,:,dest,src]=-torch.inf
  assert torch.equal(result[1]['attention_mask'],expected)and kwargs['attention_mask'].shape==(1,)
 assert torch.equal(p.masks[160],base)
x=torch.randn(1,160,1712,generator=gen);w=torch.randn(1766,1712,generator=gen);m=SimpleNamespace(bias=None);o=F.linear(x,w);p.weight=w.clone();p.weight[:,1710]=0;alt=F.linear(x,p.weight)
for mode in ['clean','row','column']:
 for pos in [[64],[128],[64,128]]:
  p.mode=mode;p.positions=pos;y=p.output(m,(x,),o);expect=o.clone()
  if mode=='row':expect[:,pos,1764]=alt[:,pos,1764]
  elif mode=='column':expect[:,pos]=alt[:,pos]
  assert torch.equal(y,expect)and torch.equal(p.cache['activation'],x[0,:,1710])
v=torch.randn(1,160,2048,generator=gen)*20;assert p.value(None,(),v)is v;assert torch.equal(p.cache['H12_values'],v[0,[0,64,128],1536:1664].clamp(-8,8));p.context(None,(v,));assert torch.equal(p.cache['H12_contexts'],v[0,[0,64,128],1536:1664])
sets=torch.load(ROOT/'later_uses_run/tokens.pt',weights_only=True,map_location='cpu');n=0
for ws in sets.values():
 for xx in ws:
  original=xx.clone();base=xx.clone();prefix=base[:129];prefix[(prefix==187)|(prefix==535)]=209;base[0]=510
  for pid in [187,535,186,209,510,15,64]:
   for qid in [187,535]:
    edited=base.clone();edited[64]=pid;edited[128]=qid;assert torch.equal(edited[129:],xx[129:])and torch.equal(edited[130:],xx[130:]);assert not((edited[1:64]==187)|(edited[1:64]==535)).any();n+=1
  assert torch.equal(xx,original)
result={'all_80_layer_mask_checks':True,'shared_cached_causal_mask_unmodified':True,'all_nine_projection_localization_checks':True,'H12_clipped_value_context_slices_correct':True,'edited_input_variants_checked':n,'original_parent_tokens_unchanged':True,'identical_scored_targets':'130..511,382 tokens','expected_forwards':16*(14+24+4),'no_model_gpu_load':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
