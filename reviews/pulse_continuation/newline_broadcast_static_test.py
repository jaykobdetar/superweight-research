"""Extracted CPU masks/KV patches/state capture; no model/GPU load."""
import ast,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];ns={'torch':torch,'F':F}
for file,name in [('newline_history.py','Probe'),('newline_broadcast.py','Broadcast')]:
 node=next(n for n in ast.parse((ROOT/file).read_text()).body if isinstance(n,ast.ClassDef)and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted '+name+'>','exec'),ns)
p=ns['Broadcast'].__new__(ns['Broadcast']);p.p=64;p.q=128;p.edges=[];p.masks={};p.cache={};p.path_cache={};n=160;x=torch.zeros(1,n,2048,dtype=torch.float16);count=0
for pos in [64,128]:
 for layers in [[],list(range(2,16)),list(range(2,6)),list(range(6,12)),list(range(12,16))]:
  p.block_layers=layers;p.block_position=pos
  for l in range(16):
   kwargs={'hidden_states':x};mask=p.attn(l,(),kwargs)[1]['attention_mask'];expect=torch.zeros(1,1,n,n,dtype=x.dtype);expect.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf)
   if l in layers:expect[:,:,pos+1:,pos]=-torch.inf
   assert torch.equal(mask,expect)and 'attention_mask'not in kwargs;assert mask[0,0,pos,pos]==0;count+=1
base=torch.zeros(1,1,n,n,dtype=x.dtype);base.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf);assert torch.equal(p.masks[n],base)
gen=torch.Generator().manual_seed(8);donor={l:{k:torch.randn(1,2048,generator=gen).half()for k in 'kv'}for l in range(2,16)};original=torch.randn(1,n,2048,generator=gen).half()
patches=[{}, {'kinds':'v','zero':True}, {'kinds':'kv','zero_kinds':'v','donor':donor}]+[{'kinds':k,'donor':donor}for k in ['k','v','kv']]
for patch in patches:
 p.patch=patch
 for l in range(2,16):
  for k in 'kv':
   o=original.clone();result=p.project(l,k,o);expect=original.clone()
   if patch and k in patch['kinds']:expect[:,64]=0 if patch.get('zero') or k in patch.get('zero_kinds','') else donor[l][k]
   assert torch.equal(result,expect)and torch.equal(o,original);assert torch.equal(p.path_cache[l][k],expect[:,64]);result[:,64]=77;assert torch.equal(p.path_cache[l][k],expect[:,64])
a=torch.randn(1,n,2048,generator=gen);b=torch.randn(1,n,2048,generator=gen);assert p.norm(None,(a,),b)is b;assert torch.equal(p.cache['L2_input_p'],a[0,64])and torch.equal(p.cache['L2_normalized_p'],b[0,64])
result={'mask_cases_passed':count,'strictly_later_queries_only':True,'source_self_and_earlier_queries_retained':True,'shared_causal_cache_unchanged':True,'all_168_KV_patch_and_capture_checks':True,'captured_postpatch_preclip_KV_independent_clones':True,'norm_capture_position_correct':True,'expected_conditions_per_window':3+7+9+6+2,'expected_forwards':16*(3+7+9+6+2),'architectural_closure_count':16*5,'no_model_gpu_load':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
