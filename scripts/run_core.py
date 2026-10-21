"""Sequential core reproduction; one offline checkpoint load, no parameter writes."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
REPO=Path(__file__).resolve().parents[1]
JOBS=['column_factorial','same_layer_pair','later_uses','newline_history','newline_heads','newline_broadcast','newline_natural','newline_components','source_geometry']
def execute(work,ctx=None,jobs=None):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 work=Path(work).resolve();root=work/'experiments/pulse_continuation';sys.path[:0]=[str(root),str(root.parent/'pulse_causal_tranche')]
 # Isolated CLI imports are the supported path. The validation bridge reloads
 # these modules before execute, using the same already-resident checkpoint.
 from single_projection_core import hash_model
 identity=json.loads((work/'input_identity.json').read_text());torch.set_num_threads(4);torch.manual_seed(20261021)
 if ctx is None:
  model_path=Path(identity['model']);model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map='cuda',attn_implementation='sdpa',local_files_only=True);model.eval();model.requires_grad_(False);ctx={'model':model,'model_path':model_path,'original_hash':hash_model(model),'versions':{n:p._version for n,p in model.named_parameters()}}
 expected=json.loads((REPO/'manifests/checkpoint.json').read_text());assert ctx['original_hash']==expected['loaded_fp16_parameter_sha256'],'Wrong checkpoint/dtype'
 tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True);completed=[];start=time.time()
 for name in jobs or JOBS:
  out=root/(name+'_run');assert not (out/'windows.jsonl').exists(),f'Existing result: {out}; use a fresh prepared work directory.'
  spec=importlib.util.spec_from_file_location('reproduce_'+name,root/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  if name in ('later_uses','newline_heads'):
   def prepare(context,destination,name=name):
    saved=torch.load(work/'prepared'/('later.pt' if name=='later_uses' else 'heads.pt'),weights_only=True,map_location='cpu');torch.save(saved,destination/'tokens.pt');(destination/'data_manifest.json').write_text((work/'prepared/data_manifest.json').read_text());return (saved,tok) if name=='later_uses' else (saved['development'],saved['evaluation'])
   m.prepare=prepare
  m.run(ctx,out,{})
  assert hash_model(ctx['model'])==ctx['original_hash'];assert ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()}
  completed.append(name);(work/'reproduction_progress.json').write_text(json.dumps({'completed':completed,'wall_seconds':time.time()-start},indent=2)+'\n');print('REPRODUCED',name,flush=True)
 return completed
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--jobs',nargs='+',choices=JOBS);a=ap.parse_args();execute(a.work,jobs=a.jobs)
