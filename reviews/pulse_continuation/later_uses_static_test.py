"""CPU extracted hook localization and frozen span checks; no model load."""
import ast,json,os,resource
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];ns={'torch':torch,'F':F}
tree=ast.parse((ROOT/'later_uses.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Neuron');exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted Neuron>','exec'),ns)
gen=torch.Generator().manual_seed(11);x=torch.randn(1,7,1712,generator=gen);weight=torch.randn(1766,1712,generator=gen);m=SimpleNamespace(bias=torch.randn(1766,generator=gen));o=F.linear(x,weight,m.bias);original=o.clone()
n=ns['Neuron'].__new__(ns['Neuron']);n.cache={};n.weight=weight.clone();n.weight[:,1710]=0;alt=F.linear(x,n.weight,m.bias)
for mode in ('clean','row','column'):
 for scope in ('source','later','all'):
  n.mode=mode;n.scope=scope;y=n.hook(m,(x,),o);expect=o.clone();pos={'source':0,'later':slice(1,None),'all':slice(None)}[scope]
  if mode=='row':expect[:,pos,1764]=alt[:,pos,1764]
  elif mode=='column':expect[:,pos]=alt[:,pos]
  assert torch.equal(y,expect)and torch.equal(o,original)and torch.equal(n.cache['activation'],x[0,:,1710])
spans={}
for c,d in json.loads((ROOT/'confirmation_run/data_manifest.json').read_text()).items():
 starts=[s+768 for s in d['evaluation_starts'][:4]];old=d['evaluation_starts']+d['calibration_starts'];assert all(s+512<=e or e+512<=s for s in starts for e in old);assert all(s+512<=t or t+512<=s for i,s in enumerate(starts)for t in starts[i+1:]);assert max(starts)+512<=d['token_count'];spans[c]=starts
result={'all_nine_mode_scope_localization_checks':True,'original_output_unchanged':True,'captured_preprojection_activation_exact':True,'fresh_spans_disjoint_from_all_original_confirmation_calibration_evaluation_and_each_other':spans,'score_targets':'positions2..511,510 targets','no_model_gpu_load':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
