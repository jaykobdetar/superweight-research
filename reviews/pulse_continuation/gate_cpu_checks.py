"""Synthetic CPU test of gate decomposition and residualized context patches."""
import ast,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1);torch.manual_seed(74)
ROOT=Path(__file__).resolve().parents[1];source=(ROOT/'gate.py').read_text()
class StubLocal:
    def context(self,*args):return None
def rotate_half(x):
    a,b=x.chunk(2,dim=-1);return torch.cat([-b,a],dim=-1)
nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['parts','Gate']]
env={'torch':torch,'math':math,'Local':StubLocal,'rotate_half':rotate_half}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<Gate AST>','exec'),env)
parts,Gate=env['parts'],env['Gate'];g=Gate.__new__(Gate)
cache={k:torch.randn(1,5,2048,dtype=torch.float16) for k in ['q','k','v']}
cache.update(cos=torch.ones(1,5,128),sin=torch.zeros(1,5,128))
z=parts(cache,15);assert z['a'][0]==1 and torch.equal(z['y'][0],z['v'])
assert torch.equal(z['u'][0],torch.zeros(128))
for p in range(1,5):
    q=cache['q'][0,p,1920:].float();k=cache['k'][0,1:p+1,1920:].float();v=cache['v'][0,1:p+1,1920:].float()
    expected=(q@k.T/math.sqrt(128)).softmax(-1)@v
    assert torch.allclose(z['u'][p],expected,atol=1e-6)
context=torch.randn(1,5,2048,dtype=torch.float16)
context[0,:,1920:]=(z['y']+.003).half()
g.capture=False;g.gpu={k:cache[k] for k in ['q','k','v']};g.rope=(cache['cos'],cache['sin'])
g.patch={'type':'gate','head':15,'gate':'native','value':'native'}
assert torch.equal(g.context(None,(context,))[0],context)
g.patch={'type':'gate','head':15,'gate':'native','value':'native','reconstruct':True}
direct=g.context(None,(context,))[0]
assert torch.equal(direct[0,:,1920:],z['y'].half())
assert torch.equal(direct[:,:,:1920],context[:,:,:1920])
g.patch={'type':'gate','head':15,'gate':'one','value':'zero'}
zero=g.context(None,(context,))[0]
assert torch.equal(zero[0,:,1920:],(context[0,:,1920:].float()-z['y']).half())
assert bool(zero[0,:,1920:].ne(0).any())
g.patch={'type':'gate','head':15,'gate':'zero','value':'native','reconstruct':True}
removed=g.context(None,(context,))[0]
assert torch.equal(removed[0,0,1920:],z['v'].half())
assert torch.equal(removed[0,1:,1920:],z['u'][1:].half())
result={'gate_sha256':hashlib.sha256(source.encode()).hexdigest(),'residualized_identity_bitwise':True,'direct_reconstruction_localized':True,'conditional_mixture_and_first_query_pass':True,'zero_head_leaves_reconstruction_residual':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
