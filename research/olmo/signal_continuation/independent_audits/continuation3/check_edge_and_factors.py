import ast,json
from pathlib import Path
import numpy as np
P=Path('/workspace/shared/olmo_signal_production/continuation');O=Path(__file__).resolve().parent
S=Path('/workspace/shared/olmo_computational_role/downstream')
z=np.load(P/'h14_prefix/traces.npz')
# Read only two existing rows, no model forward.
rows=[]
for kind in ['gate','up']:
 with (S/'weights'/f'model.layers.0.mlp.{kind}_proj.weight.f32').open('rb') as f:
  f.seek(1091*2048*4);rows.append(np.frombuffer(f.read(2048*4),dtype='<f4').astype(np.float64))
record=json.loads((P/'h14_prefix/m0_1091_factors.json').read_text());errors=[]
for x in record:
 h=lambda w:z[x['panel']+'__'+('newline_R' if w=='R' else 'newline_R_edge_H14')+'__m0_normalized'].astype(np.float64)
 g=rows[0]@h(x['gate_donor']);u=rows[1]@h(x['up_donor']);a=g/(1+np.exp(-g))*u
 errors.append(max(abs(x['gate']-g),abs(x['up']-u),abs(x['activation']-a)))
assert max(errors)<1e-12
node=next(x for x in ast.parse((P/'collect_h14_prefix.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='blocked_edge')
env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated edge hook>','exec'),env)
rng=np.random.default_rng(12864);probs=rng.random((16,129,129),dtype=np.float32);probs/=probs.sum(-1,keepdims=True)
values=rng.normal(size=(16,129,128)).astype(np.float32);context=rng.normal(size=(129,2048)).astype(np.float32)
orig=[x.copy() for x in [probs,values,context]];hook=env['blocked_edge'](14);actual=hook(0,context,probs,values)
expected=context.copy();p=probs[14,128].astype(np.float64);p[64]=0;p/=p.sum();expected[128,14*128:15*128]=p@values[14].astype(np.float64)
mask=np.ones(context.shape,dtype=bool);mask[128,14*128:15*128]=False
assert np.array_equal(actual[mask],context[mask]);assert hook(1,context,probs,values) is context
assert all(np.array_equal(a,b) for a,b in zip([probs,values,context],orig))
out={'factor_2x2_rows':len(record),'factor_max_error':max(errors),'factor_row_elements_read':4096,'edge_hook_other_coordinates_exact':True,'edge_hook_other_layer_identity':True,'edge_hook_inputs_unchanged':True,'edge_hook_fp64_formula_max_error':float(abs(actual-expected).max()),'edge_record':hook.records}
(O/'edge_and_factors_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
