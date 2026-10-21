"""Actual runner ev AST: pre-return vs post-return exception accounting; no model."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import ast,contextlib,hashlib,json,sys,copy
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import gpt2_gate_wiki_calibration as r
p=Path(r.__file__);initial=hashlib.sha256(p.read_bytes()).hexdigest();tree=ast.parse(p.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');node=copy.deepcopy(next(n for n in main.body if isinstance(n,ast.FunctionDef)and n.name=='ev'))
node.body=[ast.Global(names=n.names)if isinstance(n,ast.Nonlocal)else n for n in node.body];code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<review-actual-ev-AST>','exec')
checks=[]
for phase,suffix,returned in [('evaluation',False,False),('evaluation',False,True),('calibration',False,True),('suffix_control',True,True)]:
 run=SimpleNamespace(count=2,latest_logits=None)
 def fake(*args,**kwargs):
  if returned:run.count+=1
  raise RuntimeError('intentional fixture exception')
 run.ev=fake
 prior={'phase':'evaluation','trial':'previous'};active={'phase':phase,'trial':'attempt'}
 env={'attempts':2,'last_forward':dict(prior),'last_completed':dict(prior),'current_attempt_tokens':None,'check_target':{},'profiled':True,'pending_logits':None,'active_input':active,
 'probe':SimpleNamespace(reset_layers=lambda **kw:None),'gate':SimpleNamespace(reset_gate=lambda *args:None),'hidden':{},'control_failure':{},'run':run,'contextlib':contextlib,'torch':torch,'evaluate_control':fake,'meta':{'evaluation_forwards':0,'calibration_forwards':0,'suffix_forwards':0}}
 exec(code,env);x=torch.arange(512)
 try:env['ev'](x,suffix=suffix)
 except RuntimeError:pass
 else:raise AssertionError('fixture exception lost')
 assert env['attempts']==3 and run.count==2+int(returned)and env['last_forward']==active
 assert env['last_completed']==(active if returned else prior)
 assert torch.equal(env['current_attempt_tokens'],x);x[0]=-1;assert env['current_attempt_tokens'][0]==0
 target='suffix_forwards'if suffix else'calibration_forwards'if phase=='calibration'else'evaluation_forwards'
 assert env['meta'][target]==int(returned)and sum(env['meta'].values())==int(returned)
 checks.append({'phase':phase,'suffix':suffix,'model_returned_stub':returned,'correct':True})
# Exercise the real NumericRunner metric path with a tiny logits-returning stub.
class Stub:
 def parameters(self):yield torch.zeros(1)
 def __call__(self,*args,**kwargs):return SimpleNamespace(logits=torch.full((1,512,17),float('nan')))
obj=r.NumericRunner.__new__(r.NumericRunner);obj.p=SimpleNamespace(cache={},norms={});obj.m=Stub();obj.count=0
try:obj.ev(torch.zeros(512,dtype=torch.long),keep=True)
except AssertionError:pass
else:raise AssertionError('nonfinite scoring accepted')
assert obj.count==1 and obj.latest_logits.shape==(511,17)and torch.isnan(obj.latest_logits).all()
assert initial==hashlib.sha256(p.read_bytes()).hexdigest()
result={'status':'passed','runner_sha256':initial,'actual_nested_ev_accounting_cases':checks,'actual_numeric_runner_post_return_nonfinite_preserves_count_and_logits':True,'stub_logits_return_calls':1,'pretrained_or_random_model_forwards':0,'gpu_used':False,'scope':'No full runner setup or model execution; actual nested ev AST with failure stubs and actual NumericRunner metric path with fixed tiny logits.'}
(ROOT/'review/gate_wiki_forward_accounting_independent.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
