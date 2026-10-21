"""No-model subprocess import and exact import-only refactor verification."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def funcs(p):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a=funcs(ROOT/'gate_calibration_factorial_maps.py');b=funcs(ROOT/'gate_factorial_algebra.py');assert a==b
assert funcs(ROOT/'gate_localization_inputs.py')['directions']==funcs(ROOT/'gate_direction_geometry.py')['directions']
s=(ROOT/'gate_wiki_calibration.py').read_text().replace('from gate_factorial_algebra import','from gate_calibration_factorial_maps import').replace('from gate_direction_geometry import','from gate_localization_inputs import')
assert hashlib.sha256(s.encode()).hexdigest()=='dcbd9eab9aa517122ee9a1564c971da6bc654bfe007c7c51be610b7a6b7cd4fc'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='review-sentinel-no-initialization',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
code='''import os,sys,json
before={k:os.environ.get(k)for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM')}
import gate_wiki_calibration,gate_wiki_storage,gate_factorial_algebra,gate_direction_geometry,torch
assert before=={k:os.environ.get(k)for k in before}
assert not torch.cuda.is_initialized()
assert 'gate_localization_inputs' not in sys.modules and 'gate_calibration_factorial_inputs' not in sys.modules
print(json.dumps({'runtime_import_environment_preserved':True,'cuda_initialized':False,'tokenizer_only_helpers_imported':False}))'''
r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,capture_output=True);assert r.returncode==0,(r.stdout,r.stderr)
result={'status':'passed','directions_AST_exact':True,'all_factorial_functions_AST_exact':True,'GateWiki_exact_original_bytes_after_reverting_only_two_import_lines':True,'fresh_subprocess':json.loads(r.stdout),'sources':{n:sha(ROOT/n)for n in ['gate_wiki_calibration.py','gate_wiki_storage.py','gate_factorial_algebra.py','gate_direction_geometry.py']},'model_forwards':0,'gpu_used':False}
(ROOT/'review/gate_wiki_import_independent.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
