"""CPU extracted factor/routing hook localization, no model import/load."""
import ast,json,os,resource
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];ns={'torch':torch}
for file,names in [('upstream.py',['Heads']),('upstream_mlp.py',['Routes','Factors'])]:
 tree=ast.parse((ROOT/file).read_text())
 for name in names:
  node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted '+name+'>','exec'),ns)
mask=SimpleNamespace(p=3);g=torch.Generator().manual_seed(7);o=torch.randn(1,7,8192,generator=g).half();f=ns['Factors'].__new__(ns['Factors']);f.mask=mask;f.cache={};f.patch={'gate':torch.tensor(2.,dtype=torch.float16)}
gate=f.factor('gate',o);expect=o.clone();expect[0,3,1710]=2;assert torch.equal(gate,expect)and torch.equal(o[:,0],gate[:,0]);assert torch.equal(f.factor('up',o),o)
r=ns['Routes'].__new__(ns['Routes']);r.mask=mask;r.patch={};r.cache={};v=torch.randn(1,7,2048,generator=g).half()*20;r.vhook(None,(),v);assert torch.equal(r.values,v[0,[0,3],1536:1664].clamp(-8,8))
x=torch.randn(1,7,2048,generator=g).half()
for mode in ['value0','valuep','zero']:
 r.mode=mode;y=r.hook(1,(x,))[0];expect=x.clone();expect[:,3,1536:1664]=0 if mode=='zero'else r.values[0 if mode=='value0'else 1]
 assert torch.equal(y,expect)and torch.equal(r.cache[1],expect[:,3]);assert torch.equal(r.hook(0,(x,))[0],x)
r.mode='selfmask';m=torch.zeros(1,1,7,7,dtype=torch.float16);m.masked_fill_(torch.ones(7,7,dtype=torch.bool).triu(1),-torch.inf)
changed=r.attn(None,(),{'attention_mask':m})[1]['attention_mask'];expect=m.expand(1,16,7,7).clone();expect[:,12,3,:3]=-torch.inf
assert torch.equal(changed,expect)and torch.isfinite(m[0,0,3,:3]).all()
result={'factor_patch_single_neuron_and_boundary_only':True,'current_values_clipped_correctly':True,'value_context_replacement_H12_boundary_only':True,'one_head_mask_H12_boundary_only':True,'original_inputs_and_masks_unmodified':True,'no_model_gpu_loaded':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
