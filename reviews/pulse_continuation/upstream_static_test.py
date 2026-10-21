"""Extracted-hook CPU tests; no runner import or model load."""
import ast,json,os,resource
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
def extract(file,name):
 tree=ast.parse((ROOT/file).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==name)
 ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted '+name+'>','exec'),ns);return ns[name]
Heads=extract('upstream.py','Heads');Masks=extract('empty_history.py','Masks')
obj=Heads.__new__(Heads);obj.mask=SimpleNamespace(p=3);obj.cache={};g=torch.Generator().manual_seed(20261021)
x=torch.randn(1,7,2048,generator=g).half();donors={l:torch.randn(1,2048,generator=g).half()for l in [0,1]};original=x.clone()
for hs in [[],[(0,0)],[(0,15),(1,2)],[(l,h)for l in [0,1]for h in range(16)]]:
 obj.patch={'heads':hs,'donor':donors}
 for l in [0,1]:
  y=obj.hook(l,(x,))[0];expected=x.clone()
  for ll,h in hs:
   if ll==l:expected[:,3,h*128:(h+1)*128]=donors[l][:,h*128:(h+1)*128]
  assert torch.equal(y,expected)and torch.equal(obj.cache[l],expected[:,3])and torch.equal(x,original)
  copy=obj.cache[l].clone();y=y.clone();y[:,3]=0;assert torch.equal(obj.cache[l],copy)
m=Masks.__new__(Masks);m.p=3;m.cache={};m.mode='explicit'
base=m.hook(0,(),{'hidden_states':x})[1]['attention_mask'];m.mode='L0_L1'
for l in [0,1,2,15]:
 z=m.hook(l,(),{'hidden_states':x})[1]['attention_mask'];expect=base.clone()
 if l in [0,1]:expect[:,:,3,:3]=-torch.inf
 assert torch.equal(z,expect)
old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',map_location='cpu',weights_only=True)
assert old['starts']['wiki']['development']==list(range(18432,26624,1024))
fresh=torch.load(ROOT/'confirmation_run/tokens.pt',map_location='cpu',weights_only=True)
for c,data in fresh.items():
 for xx in data['evaluation'][:4]:
  for tid in [510,15]:
   targets=[]
   for p in [128,256]:
    z=xx[256-p:].clone();z[0]=z[p]=tid;targets.append(z[p+2:]);assert len(targets[-1])==254
   assert torch.equal(targets[0],targets[1])and torch.equal(targets[0],xx[258:])
result={'head_patch_localization_pass':True,'all_context_channels_reconstruct_donor':True,'donor_and_input_immutability_pass':True,'mask_only_boundary_L0_L1_pass':True,'development_starts_match_saved_development':True,'matched_254_targets_across_lengths_and_identities':True,'model_or_gpu_loaded':False,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
