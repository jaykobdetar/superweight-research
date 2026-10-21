"""CPU-only dynamic-position H12 patch and token eligibility tests."""
import ast,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
def rotate_half(x):a,b=x.chunk(2,-1);return torch.cat([-b,a],-1)
ns={'torch':torch,'F':F,'rotate_half':rotate_half}
for file,name in [('newline_history.py','Probe'),('newline_heads.py','PathProbe'),('newline_natural.py','NaturalProbe')]:
 node=next(n for n in ast.parse((ROOT/file).read_text()).body if isinstance(n,ast.ClassDef)and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted '+name+'>','exec'),ns)
p=ns['NaturalProbe'].__new__(ns['NaturalProbe']);p.cache={};gen=torch.Generator().manual_seed(19);x=torch.randn(1,512,2048,generator=gen).half();original=x.clone();v=torch.randn(1,512,2048,generator=gen).half()*20;donor={'H12_contexts':torch.randn(3,128,generator=gen).half()}
for pp,qq in [(0,7),(3,19),(125,255)]:
 p.p=pp;p.q=qq;p.value(None,(),v)
 for kind in [None,'donor','value_p','value_q']:
  p.h12_patch=None if kind is None else {'kind':kind,'donor':donor};result=p.context(None,(x,));expected=x.clone()
  if kind:expected[0,qq,1536:1664]=donor['H12_contexts'][2]if kind=='donor'else v[0,pp if kind=='value_p'else qq,1536:1664].clamp(-8,8)
  assert result is None if kind is None else torch.equal(result[0],expected)
  assert torch.equal(x,original)and torch.equal(p.cache['H12_contexts'],x[0,[0,pp,qq],1536:1664])and torch.equal(p.cache['H12_applied_contexts'],expected[0,[0,pp,qq],1536:1664])
sets=torch.load(ROOT/'newline_heads_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];eligible=[];ineligible=[]
for c,ws in sets.items():
 for i,x in enumerate(ws):
  ps=(x==187).nonzero().flatten().tolist()
  if len(ps)>=2 and ps[1]<=255:
   pp,qq=ps[:2];sham=pp-1 if pp else qq-1;assert int(x[sham])!=187;eligible.append({'corpus':c,'window':i,'p':pp,'q':qq,'sham':sham,'score_start':qq+1,'scored_targets':510-qq})
  else:ineligible.append({'corpus':c,'window':i,'positions187':ps})
assert len(eligible)==12 and len(ineligible)==4 and all(z['corpus']=='wiki_validation'for z in ineligible);assert sum(z['p']==0 for z in eligible)==2
result={'dynamic_H12_patch_cases':12,'only_q_H12_changed':True,'native_and_applied_context_capture_correct':True,'original_inputs_unchanged':True,'eligible_cases':eligible,'ineligible_cases':ineligible,'expected_forwards':12*(4+2*(1+2+3)+3),'expected_activation_checks':12*9,'no_model_gpu_load':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
