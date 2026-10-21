"""Execute exact production-auditor AST guard blocks on valid/corrupt fixtures."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,ast,copy,json,hashlib,math,types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gate_wiki_calibration import CONDITIONS,mode_spec
from gpt2_gate_subspace import LAYERS
from run_transfer import th
p=ROOT/'audit_gate_wiki_results.py';source=p.read_text();digest=hashlib.sha256(source.encode()).hexdigest();assert digest=='ebae29f3c04c937aa8abf7dcbb608af91ee0a957f3ffface128e914e6b5f1686'
tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='audit')
loop=next(n for n in fn.body if isinstance(n,ast.For)and any(isinstance(z,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='expected_checks'for t in z.targets)for z in n.body))
start=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='expected_checks'for t in n.targets));end=next(i for i,n in enumerate(loop.body)if i>start and isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='mode')
main_nodes=loop.body[start:end]
sstart=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assert)and "set(d['suffix_metrics'])"in ast.unparse(n));send=next(i for i,n in enumerate(loop.body)if i>sstart and isinstance(n,ast.Assert)and "set(d['norms'])"in ast.unparse(n));suffix_nodes=loop.body[sstart:send]
gate_nodes=[next(n for n in fn.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='op')]
def code(nodes):return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'<exact-auditor-guard-AST>','exec')
codes={k:code(v)for k,v in [('main',main_nodes),('suffix',suffix_nodes),('gate',gate_nodes)]}
base={'corpus':'wiki_validation','window':0,'identity':'natural','length':512};checks=[]
for mode in CONDITIONS:
 if mode!='C':checks.append({**base,'trial':mode+'_source_identity','source_exact':True,'reference_fingerprints':{},'trial_fingerprints':{}})
 if len(mode_spec(mode)[3])==1:checks.append({**base,'trial':mode+'_singleton',**{k+'_exact':True for k in ['b','mass','context','coefficient']},'reference_context':{},'trial_context':{}})
for mode in ['current_closure','fullrank_closure','repeat','identity']:checks.append({**base,'trial':mode,**{k+'_exact':True for k in ['loss','kl','pred','kl_p','agreement_p','hidden','logits']},'logits_reference':{},'logits_control':{}})
x=torch.arange(512);y=x.clone();y[128:]=torch.roll(y[128:],1);raw={};sm={};sc=[];op={}
fields={'loss','pred','kl','kl_p','agreement_p'}
for mode in CONDITIONS:
 raw[mode]={k:torch.zeros(511,dtype=torch.long if k=='pred'else torch.bool if k=='agreement_p'else torch.float32)for k in fields}
 sm[mode]={'token_sha256':th(y),'score_index':1,'scored_tokens':126,**{k:v[:127 if k=='loss' else 128].clone()for k,v in raw[mode].items()}}
 sc.append({'trial':mode,'original_token_sha256':th(x),'control_token_sha256':th(y),**{k+'_exact':True for k in ['logits','hidden','gates','source','mask','contexts','loss','pred','kl','kl_p','agreement_p']},'logits_reference':{},'logits_control':{},'reference_context_hashes':{},'control_context_hashes':{}})
 op[mode]={'prefix_context_hashes':{},'mask_records':{}}
errorkeys=['conditional_error','native_reconstruction_error','mixed_reconstruction_error','sigmoid_context_error','fullrank_logit_error','fullrank_mass_error','fullrank_context_error','formula_error']
def gateop(mode):return {'trial':mode,'projection_checks':8,'query_checks':8,'context_checks':8,'writer_locality_checks':30,'max_context_error':0.,'gate_checks':{}if mode in ['C','repeat','identity']else{l:{'finite':True,'source_context_exact':True,'min_mass':.1,**{k:0. for k in errorkeys}}for l in LAYERS}}
env={'CONDITIONS':CONDITIONS,'mode_spec':mode_spec,'base':base,'d':{'checks':checks,'suffix_checks':sc,'suffix_metrics':sm},'raw':raw,'x':x,'y':y,'th':th,'torch':torch,'op':op,'sop':copy.deepcopy(op),'metric_fields':fields,'totals':{'suffix_records':0,'gate_checks':0},'LAYERS':LAYERS,'math':math,'extrema':{},'allops':[gateop(m)for m in ['calibration','C','repeat','identity','P','remove_OO_all']]}
def clone_env():return {k:v if isinstance(v,types.ModuleType) else copy.deepcopy(v)for k,v in env.items()}
for c in codes.values():exec(c,clone_env())
rejected=[]
def test(name,section,mutate):
 e=clone_env();mutate(e)
 try:exec(codes[section],e)
 except (AssertionError,KeyError):rejected.append(name)
 else:raise AssertionError(('corruption accepted',name))
test('duplicate_main_check','main',lambda e:e['d']['checks'].__setitem__(0,copy.deepcopy(e['d']['checks'][1])))
for typ,flag in [('source_identity','source_exact'),('singleton','coefficient_exact'),('closure','hidden_exact')]:
 test('missing_'+typ+'_flag','main',lambda e,t=typ,f=flag:next(ch for ch in e['d']['checks']if ch['trial'].endswith(t)).pop(f))
test('wrong_check_identity','main',lambda e:e['d']['checks'][0].__setitem__('window',1))
test('false_main_flag','main',lambda e:e['d']['checks'][0].__setitem__('source_exact',False))
test('duplicate_suffix_check','suffix',lambda e:e['d']['suffix_checks'].__setitem__(0,copy.deepcopy(e['d']['suffix_checks'][1])))
test('missing_suffix_flag','suffix',lambda e:e['d']['suffix_checks'][0].pop('contexts_exact'))
test('suffix_metric_changed','suffix',lambda e:e['d']['suffix_metrics']['C']['loss'].__setitem__(0,1.))
test('missing_suffix_mode','suffix',lambda e:e['d']['suffix_metrics'].pop('P'))
test('missing_gate_layer','gate',lambda e:e['allops'][0]['gate_checks'].pop(4))
test('extra_native_gate_layer','gate',lambda e:e['allops'][1]['gate_checks'].__setitem__(4,copy.deepcopy(e['allops'][0]['gate_checks'][4])))
test('missing_error_field','gate',lambda e:e['allops'][0]['gate_checks'][4].pop('conditional_error'))
test('over_guard_error','gate',lambda e:e['allops'][0]['gate_checks'][4].__setitem__('conditional_error',.001))
assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
out={'status':'passed','auditor_sha256':digest,'valid_exact_AST_blocks':list(codes),'negative_cases_rejected':rejected,'scope':'Focused exact-universe and required-flag tests. No synthetic full-model result or production-data audit claimed.','model_forwards':0,'gpu_used':False}
(ROOT/'review/gate_wiki_saved_audit_guards_independent.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
