"""CPU-only component output/localization and residual-order checks."""
import ast,itertools,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
# Extract only independent component/norm methods; no network or pretrained module instantiated.
cls=next(n for n in ast.parse((ROOT/'newline_components.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='Components');cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in ['component','norm1']];ns={'torch':torch};exec(compile(ast.Module(body=[cls],type_ignores=[]),'<extracted Components>','exec'),ns)
p=ns['Components']();p.q=17;p.swaps={};p.components={};p.cache={};gen=torch.Generator().manual_seed(27);original={n:torch.randn(1,35,2048,generator=gen).half()for n in ['A0','M0','A1']};donors={w:{n:torch.randn(2048,generator=gen).half()for n in original}for w in 'CE'};initial=torch.randn(1,35,2048,generator=gen).half()
for code in itertools.product('CE',repeat=3):
 p.swaps={n:donors[w][n]for n,w in zip(original,code)};x=initial.clone()
 for n in ['A0','M0','A1']:
  o=original[n].clone();y=p.component(n,o);expect=o.clone();expect[0,p.q]=p.swaps[n];assert torch.equal(y,expect)and torch.equal(o,original[n]);assert torch.equal(p.components[n]['native'],original[n][0,p.q])and torch.equal(p.components[n]['applied'],p.swaps[n]);x=x+y
 expect=((initial[0,p.q]+p.swaps['A0'])+p.swaps['M0'])+p.swaps['A1'];assert torch.equal(x[0,p.q],expect)
 p.norm1(None,(x,),x.float());assert torch.equal(p.cache['before_mlp1'],x[0,p.q])and torch.equal(p.cache['normalized_mlp1'],x[0,p.q].float())
p.swaps={};assert p.component('A0',original['A0'])is original['A0']
elig=json.loads((ROOT/'newline_natural_run/eligibility.json').read_text());assert sum(z['eligible']for z in elig)==12
# Local native decoder source confirms attention result addition, then post-attention norm and MLP result addition.
source=Path('/EXTERNAL/superweight-venv/lib/python3.12/site-packages/transformers/models/olmo/modeling_olmo.py').read_text();decoder=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)and n.name=='OlmoDecoderLayer');forward=next(n for n in decoder.body if isinstance(n,ast.FunctionDef)and n.name=='forward');s=ast.unparse(forward);assert s.index('self.self_attn(')<s.index('hidden_states = residual + hidden_states')<s.index('self.post_attention_layernorm(hidden_states)')<s.index('self.mlp(hidden_states)')
result={'all_eight_frozen_combinations_localize_q_only':True,'native_and_applied_component_captures_exact':True,'original_component_outputs_unmodified':True,'FP16_residual_addition_order_verified':True,'native_decoder_hook_order_verified':True,'MLP1_pre_post_norm_capture_exact':True,'contexts':16+12,'conditions_per_context':2+8+6,'expected_forwards':28*16,'expected_closures':28*4,'no_model_gpu_load':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
