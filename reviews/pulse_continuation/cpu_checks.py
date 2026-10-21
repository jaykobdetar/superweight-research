"""Synthetic CPU-only Local hook, normalization, and KL formula checks."""
import ast, hashlib, json, os, resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1)
torch.manual_seed(182)
root = Path(__file__).resolve().parents[1]
source = (root/'factorial.py').read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Local')
env = {'torch':torch}
exec(compile(ast.Module(body=[node],type_ignores=[]), '<Local AST>', 'exec'),env)
local = env['Local'].__new__(env['Local']);local.capture=False;local.cache={}
x = torch.randn(1,5,2048,dtype=torch.float16)
donor = torch.randn_like(x)
for kind in ['q','k','v']:
    local.patch={'type':'qkv','head':15,'kinds':kind,'donor':{kind:donor}}
    expected=x.clone()
    if kind=='q':expected[:,:,1920:]=donor[:,:,1920:]
    else:expected[:,:1,1920:]=donor[:,:1,1920:]
    assert torch.equal(local.proj(kind,x),expected)
    wrong={'q':'k','k':'v','v':'q'}[kind]
    assert torch.equal(local.proj(wrong,x),x)
local.patch={};local.capture=True
out=local.proj('k',x.clone());saved=local.cache['k'].clone();out.clamp_(-.1,.1)
assert torch.equal(saved,local.cache['k'])
local.capture=False
normalized=torch.nn.functional.layer_norm(x.float(),(2048,),eps=1e-5).half()
other_input=donor[:,:1]
other_norm=torch.nn.functional.layer_norm(other_input.float(),(2048,),eps=1e-5).half()
for mode in ['coordinate','complement','full','donor_scale']:
    local.patch={'type':'norm','mode':mode,'donor':{'input':other_input,'norm':other_norm}}
    actual=local.norm(None,(x,),normalized)
    assert torch.equal(actual[:,1:],normalized[:,1:])
    if mode=='coordinate':
        expected=normalized.clone();expected[:,0,1764]=other_norm[:,0,1764]
    elif mode=='complement':
        expected=normalized.clone();expected[:,0]=other_norm[:,0];expected[:,0,1764]=normalized[:,0,1764]
    elif mode=='full':
        expected=normalized.clone();expected[:,0]=other_norm[:,0]
    else:
        expected=normalized.clone();sigma=other_input[:,0].float().var(-1,unbiased=False).add(1e-5).sqrt()
        xx=x[:,0].float();expected[:,0]=((xx-xx.mean(-1,keepdim=True))/sigma[:,None]).half()
    assert torch.equal(actual,expected)
clean=torch.randn(7,31,dtype=torch.float32);altered=torch.randn_like(clean)
lb=clean.log_softmax(-1);lp=altered.log_softmax(-1)
implemented=(lb.exp()*(lb-lp)).sum(-1)
reference=torch.distributions.kl_divergence(torch.distributions.Categorical(logits=clean),torch.distributions.Categorical(logits=altered))
assert torch.allclose(implemented,reference,atol=1e-6)
result={'factorial_sha256':hashlib.sha256(source.encode()).hexdigest(),'qkv_localization':True,'preclip_capture_clone':True,'all_normalization_modes':True,'kl_clean_to_condition_orientation':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
