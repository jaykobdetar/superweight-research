"""Frozen runner delta, input/source closure and zero-forward guard audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import ast,copy,hashlib,json,sys,tempfile,time,resource
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def literal(n):
 if isinstance(n,ast.BinOp)and isinstance(n.op,(ast.Mult,ast.Pow)):
  return literal(n.left)*literal(n.right)if isinstance(n.op,ast.Mult)else literal(n.left)**literal(n.right)
 if isinstance(n,ast.Dict):return {literal(k):literal(v)for k,v in zip(n.keys,n.values)}
 return ast.literal_eval(n)
def literals(tree):
 return {n.targets[0].id:literal(n.value)for n in tree.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('INPUT_HASHES','STORAGE_READY','GROUP_ALLOWANCES','SOURCE_NAMES','NORMAL_LIMIT','FAILURE_LIMIT')and n.targets[0].id not in ('NORMAL_LIMIT','FAILURE_LIMIT')}
def normalize(tree):
 tree=copy.deepcopy(tree)
 for n in tree.body:
  if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('INPUT_HASHES','STORAGE_READY','GROUP_ALLOWANCES','SOURCE_NAMES'):n.value=ast.Constant(None)
 return ast.dump(tree,include_attributes=False)
def main():
 started=time.time();freeze=json.loads((ROOT/'review/gate_wiki_prelaunch_freeze.json').read_text());before=ROOT/'review/gpt2_gate_wiki_calibration_before_freeze.py';current=ROOT/'gpt2_gate_wiki_calibration.py'
 assert sha(before)=='80559661cad5cc7205b4ce0041090e8037c97238709da3febdff089a84519b46'
 a,b=ast.parse(before.read_text()),ast.parse(current.read_text());assert normalize(a)==normalize(b)
 x,y=literals(a),literals(b);assert not x['INPUT_HASHES']and not x['STORAGE_READY']and not x['GROUP_ALLOWANCES'];assert y['STORAGE_READY']is True
 assert y['INPUT_HASHES']==freeze['input_hashes']and y['GROUP_ALLOWANCES']=={'calibration':1048576,'snapshot':60817408,'ordinary':4194304}
 assert [n for n in y['SOURCE_NAMES']if n not in x['SOURCE_NAMES']]==['audit_gate_wiki_results.py']
 assert [n for n in y['SOURCE_NAMES']if n!='audit_gate_wiki_results.py']==list(x['SOURCE_NAMES'])
 assert set(y['INPUT_HASHES'])=={p.name for p in (ROOT/'calibration_wiki_inputs').iterdir()if p.is_file()}
 for n,h in y['INPUT_HASHES'].items():assert sha(ROOT/'calibration_wiki_inputs'/n)==h
 for n,h in freeze['sources'].items():assert sha(ROOT/n)==h,(n,sha(ROOT/n),h)
 protocol=(ROOT/'GPT2_GATE_WIKI_CALIBRATION_REVISION.md').read_bytes();oldhash='0b437fcdd53ade96f9bf3562bf307edeae7bf62b9da002823179f8a58dfe9765';prefix=b'';match=None
 for line in protocol.splitlines(keepends=True):
  prefix+=line
  if hashlib.sha256(prefix).hexdigest()==oldhash:match=len(prefix);break
 assert match is not None,'Original protocol changed beyond append'
 fixture=json.loads((ROOT/'review/gate_wiki_full_layout_fixture.json').read_text());assert sha(ROOT/'review/gate_wiki_full_layout_fixture.json')==freeze['writer_json_sha256']and fixture['status']=='passed'
 assert fixture['source_sha256']['gpt2_gate_wiki_calibration.py']==sha(before)
 oldsource=sum(fixture['source_file_sizes'].values());newsource=sum((ROOT/n).stat().st_size for n in y['SOURCE_NAMES']);growth=newsource-oldsource;normal=fixture['total_normal_with_allowance_bytes']+growth
 assert normal<176*1024**2 and fixture['maximum_failure_bytes']<48*1024**2
 # Actual main, frozen gates unchanged. Loader replaced by a forbidden-action
 # tripwire and disk capacity stub only for tiny malformed-input fixtures.
 import gpt2_gate_wiki_calibration as r
 oldargv=sys.argv;oldusage=r.shutil.disk_usage;oldloader=r.AutoModelForCausalLM.from_pretrained;loader_calls=[]
 def blocked(*args,**kwargs):loader_calls.append(True);raise AssertionError('unexpected model-loader reach')
 r.AutoModelForCausalLM.from_pretrained=blocked;guard=[]
 try:
  with tempfile.TemporaryDirectory(dir='/tmp')as td:
   td=Path(td)
   for kind in ['missing_config','malformed_config','insufficient_disk','missing_input']:
    cfg=td/(kind+'.json');out=td/kind
    if kind!='missing_config':cfg.write_text('{'if kind=='malformed_config'else'{}')
    r.shutil.disk_usage=lambda p:SimpleNamespace(free=0 if kind=='insufficient_disk'else 8*1024**3)
    sys.argv=['review','--config',str(cfg),'--inputs',str(td/'absent_inputs'),'--calibration-reference',str(td/'absent_reference'),'--out',str(out)]
    try:r.main()
    except (AssertionError,FileNotFoundError,json.JSONDecodeError):pass
    else:raise AssertionError('invalid setup accepted')
    m=json.loads((out/'manifest.json').read_text());assert m['status']=='failed'and m['forward_attempts']==m['forward_passes']==m['records']==0 and m['model_loaded']is False
    assert m['last_attempted_forward_identity']==m['last_completed_forward_identity']=={}
    assert m['wall_seconds']>=0 and m['peak_gpu_allocated_bytes']==0 and 'completed_utc'not in m
    if kind!='insufficient_disk':assert(out/'failure_state.pt').exists()
    guard.append({'case':kind,'failure_type':m['failure_type'],'model_forwards':0,'failure_preserved':True})
 finally:sys.argv=oldargv;r.shutil.disk_usage=oldusage;r.AutoModelForCausalLM.from_pretrained=oldloader
 assert not loader_calls
 auditor_guard=json.loads((ROOT/'review/gate_wiki_saved_audit_guards_independent.json').read_text());assert auditor_guard['status']=='passed' and auditor_guard['auditor_sha256']==sha(ROOT/'audit_gate_wiki_results.py')
 result={'status':'passed_frozen_prelaunch_review','runner_sha256':sha(current),'freeze_sha256':sha(ROOT/'review/gate_wiki_prelaunch_freeze.json'),'runner_functions_AST_exact_vs_reviewed_pre_freeze':True,'only_allowed_four_constant_changes':True,'input_hashes_verified':len(y['INPUT_HASHES']),'source_hashes_verified':len(freeze['sources']),'runtime_source_files':len(y['SOURCE_NAMES']),'protocol_original_prefix_preserved':True,'source_size_growth_after_writer_bytes':growth,'adjusted_normal_with_16MiB_allowance_bytes':normal,'normal_margin_bytes':176*1024**2-normal,'maximum_measured_failure_bytes':fixture['maximum_failure_bytes'],'guard_cases':guard,'model_loader_calls':0,'model_forwards':0,'gpu_used':False,'wall_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (ROOT/'review/gate_wiki_freeze_independent.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
