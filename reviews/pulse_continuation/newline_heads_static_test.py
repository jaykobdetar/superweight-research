"""CPU-only extracted local-attention reconstruction, masks and split tests."""
import ast,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
def rotate_half(x):
 a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)
ns={'torch':torch,'F':F,'rotate_half':rotate_half}
for filename,names in [('newline_history.py',['Probe']),('newline_heads.py',['PathProbe','edited'])]:
 tree=ast.parse((ROOT/filename).read_text())
 for name in names:
  node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted '+name+'>','exec'),ns)
p=ns['PathProbe'].__new__(ns['PathProbe']);p.p=64;p.q=128;p.edges=[];p.head_edges=[];p.masks={};p.cache={};n=160;gen=torch.Generator().manual_seed(31)
p.current={k:torch.randn(1,n,2048,generator=gen).half()*5 for k in 'qkv'};saved={k:v.clone()for k,v in p.current.items()};ang=torch.randn(1,n,64,generator=gen);ang=torch.cat([ang,ang],-1);p.rope=(ang.cos(),ang.sin());p.context_patch=None
hidden=torch.zeros(1,n,2048,dtype=torch.float16)
for edges in [[],[(7,128,64)],[(2,128,63)],[(h,128,64)for h in range(16)]]:
 p.head_edges=edges
 for l in [0,1]:
  mask=p.attn(l,(),{'hidden_states':hidden,'position_embeddings':p.rope})[1]['attention_mask'];expect=torch.zeros(1,16 if edges and l==0 else 1,n,n,dtype=torch.float16);expect.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf)
  if l==0:
   for h,dest,source in edges:expect[:,h,dest,source]=-torch.inf
  assert torch.equal(mask,expect)
def native_parts(raw):
 q,k,v=[raw[z].clamp(-8,8).view(1,n,16,128).transpose(1,2)for z in 'qkv'];co,si=[x.unsqueeze(1)for x in p.rope];q=(q*co+rotate_half(q)*si).to(q.dtype);k=(k*co+rotate_half(k)*si).to(k.dtype);return q,k,v
p.head_edges=[];q,k,v=native_parts(p.current);o=F.scaled_dot_product_attention(q,k,v,attn_mask=p.masks[n],is_causal=False).transpose(1,2).reshape(1,n,2048)
donor={'L0_'+k:torch.randn(3,2048,generator=gen).half()*6 for k in 'qkv'};errors=[]
for edges in [[],[(7,128,64)]]:
 p.head_edges=edges
 for kinds in ['', 'k','v','kv','q']:
  p.context_patch={'heads':[7],'kinds':kinds,'donor':donor};y=p.context0(None,(o,))[0];expected=o.clone();raw={k:z.clone()for k,z in p.current.items()}
  for kind in kinds:raw[kind][:,128 if kind=='q' else 64,896:1024]=donor['L0_'+kind][2 if kind=='q' else 1,896:1024]
  qq,kk,vv=native_parts(raw);mask=p.masks[n][:,:,128:129].expand(1,16,1,n).clone()
  for h,dest,source in edges:mask[:,h,0,source]=-torch.inf
  ctx=F.scaled_dot_product_attention(qq[:,:,128:129],kk,vv,attn_mask=mask,is_causal=False).transpose(1,2).reshape(1,1,2048);expected[:,128,896:1024]=ctx[:,0,896:1024]
  assert torch.equal(y,expected);assert all(torch.equal(p.current[k],saved[k])for k in 'qkv');errors.append(float((y-expected).abs().max()))
  prob=((qq[:,:,128:129].float()@kk.float().transpose(-1,-2))/(128**.5)+mask.float()).softmax(-1);assert torch.equal(p.cache['L0_mass_0_p_q'],prob[0,:,0,[0,64,128]])
prior=json.loads((ROOT/'confirmation_run/data_manifest.json').read_text());later=json.loads((ROOT/'later_uses_run/data_manifest.json').read_text());spans={}
for c,d in prior.items():
 starts=[4096,5120,6144,7168]if c=='optimizers'else[16384,18432,20480,22528];excluded=d['calibration_starts']+d['evaluation_starts']+later[c]['evaluation_starts'];assert all(abs(s-e)>=512 for s in starts for e in excluded);spans[c]=starts
old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');assert all(len(old['starts'][c]['development'])==8 for c in ['wiki','hume'])
for c in ['wiki','hume']:
 for s in old['starts'][c]['development']:
  xx=old['corpora'][c][s:s+512];x=ns['edited'](xx);assert x[0]==510 and x[64]==x[128]==187 and torch.equal(x[129:],xx[129:])
result={'native_clipping_FP32_RoPE_final_FP16_cast_matches':True,'all_ten_local_recomputation_patches_match_independent_SDPA':True,'patch_only_one_head_at_q':True,'raw_projections_unchanged':True,'head_specific_masks_correct':True,'diagnostic_masses_replay_exact':True,'development_windows':16,'heldout_disjoint_starts':spans,'expected_forward_count':16*18+16*(3+2*(4+10)+6),'maximum_local_reference_error':max(errors),'no_model_gpu_load':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
