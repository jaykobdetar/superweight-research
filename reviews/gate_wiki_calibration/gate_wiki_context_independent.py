"""Independent no-model CPU formula, lossless-codec and adversarial audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,hashlib,json,time,resource,copy,io,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gate_wiki_calibration import CONDITIONS,SNAPSHOT_MODES,mode_spec,wiki_map
from gate_wiki_storage import SnapshotStore,SnapshotMismatch,SHARED_PATHS,flatten,restore_snapshot,dumps_npz,loads_npz,write_npz,tensor_identity
from gate_direction_geometry import directions
from gpt2_gate_subspace import LAYERS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reject(name,fn,checks):
 try:fn()
 except (AssertionError,KeyError,ValueError,TypeError,RuntimeError):checks.append(name)
 else:raise AssertionError(('accepted invalid input',name))
def snapshot():
 return {'q':torch.randn(1,12,7,64),'k':torch.randn(1,12,7,64),'v':torch.randn(1,12,7,64),'mask':torch.ones(1,1,7,7,dtype=torch.bool).tril(),
 'actual_context':torch.randn(1,7,768),'stable_mixture':torch.randn(12,6,64),'gate_state':{'b':torch.randn(12,6),'mass':torch.rand(12,6),'target_b':torch.randn(12,6),'target_mass':torch.rand(12,6)},'delivered_context':torch.randn(1,7,768)}
def main():
 start=time.time();torch.set_num_threads(1);torch.manual_seed(20261023)
 sources={n:sha(ROOT/n)for n in ['gate_wiki_calibration.py','gate_wiki_storage.py','gate_factorial_algebra.py','gate_direction_geometry.py','analyze_gate_wiki_calibration.py']}
 assert sources['gate_wiki_calibration.py']=='62a4719b9c0d50b72d23263b98f32b4dca7ef7172b36df8f5deff6b0a0463619'
 assert sources['gate_wiki_storage.py']=='83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06'
 old=torch.load(ROOT/'wiki_calibration_preparation/old_wiki_maps.pt',weights_only=True,map_location='cpu');mix=torch.load(ROOT/'gpt2_gate_fixed512_run/calibration_maps.pt',weights_only=True,map_location='cpu')
 maps={'O':old,'F':mix,'M':mix};dirs={k:directions(v)for k,v in maps.items()};cases=0;unaffected=0;max_error=0.;center_error=0.;norm_error=0.
 assert len(CONDITIONS)==24 and len(set(CONDITIONS))==24 and len(SNAPSHOT_MODES)==11
 for dtype in [torch.float32,torch.float64]:
  for layer in LAYERS:
   local={label:{k:v.to(dtype)for k,v in mm[layer].items()}for label,mm in maps.items()};ds={label:{k:dd[layer][k].to(dtype)for k in ('u','v')}for label,dd in dirs.items()};b=torch.randn(12,511,dtype=dtype)*3;mass=b.sigmoid()
   for mode in CONDITIONS:
    if mode=='fullrank':continue
    family,center,basis,affected=mode_spec(mode);bb,aa,*_=wiki_map(mode,layer,b,mass,local,ds)
    if layer not in affected:
     assert torch.equal(bb,b)and torch.equal(aa,mass);unaffected+=1
    else:
     # Independently written explicit per-query coefficient/direction equations.
     coeff=torch.stack([torch.dot(b[:,q]-torch.log(torch.tensor(q+1,dtype=dtype))-local[center]['alpha'],ds[basis]['u'])for q in range(511)])
     direction=ds[basis]['u'if family=='remove'else'v'];expected=b-direction[:,None]*coeff[None,:]
     error=float((bb-expected).abs().max());max_error=max(max_error,error);assert torch.allclose(bb,expected,rtol=0,atol=1e-5 if dtype==torch.float32 else 2e-14)
     assert torch.equal(aa,bb.sigmoid())
     a=(coeff[:,None]*ds[basis]['u']).double().norm(dim=-1);v=(coeff[:,None]*ds[basis]['v']).double().norm(dim=-1);ne=float(((a-v).abs()/torch.maximum(torch.ones_like(a),torch.maximum(a,v))).max());norm_error=max(norm_error,ne);assert ne<=1e-6
    cases+=1
   if layer==10:
    for family in ['remove','orthogonal']:
     for basis in ['O','F']:
      first=wiki_map(f'{family}_O{basis}_L10',10,b,mass,local,ds)[0];second=wiki_map(f'{family}_F{basis}_L10',10,b,mass,local,ds)[0]
      displacement=torch.dot(local['F']['alpha']-local['O']['alpha'],ds[basis]['u'])*ds[basis]['u'if family=='remove'else'v']
      error=float(((second-first)-displacement[:,None]).abs().max());center_error=max(center_error,error);assert error<=1e-4
 base=snapshot();store=SnapshotStore();store.add('P',base)
 for i,mode in enumerate(SNAPSHOT_MODES[1:]):
  trial=copy.deepcopy(base);trial['delivered_context']+=i+1;trial['gate_state']['target_b']+=i+1;trial['gate_state']['target_mass']=trial['gate_state']['target_b'].sigmoid();store.add(mode,trial)
 data=dumps_npz(store.payload);restored=loads_npz(data);leaves=0
 for mode in SNAPSHOT_MODES:
  expected=flatten(restore_snapshot(store.payload,mode));actual=flatten(restore_snapshot(restored,mode));assert expected.keys()==actual.keys()
  for k in expected:assert tensor_identity(expected[k])==tensor_identity(actual[k]);leaves+=1
 assert all(len(restored['states'][mode]['unique'])==3 for mode in SNAPSHOT_MODES[1:])
 checks=[];mode=SNAPSHOT_MODES[1]
 for name in ['alias_reference_mode','alias_reference_path','alias_hash','alias_dtype','alias_shape','missing_alias','extra_alias','unique_alias_collision','unique_missing','unique_extra','identity_hash','base_mutation','trial_base_flag']:
  bad=copy.deepcopy(restored);st=bad['states'][mode];p=SHARED_PATHS[0]
  if name=='alias_reference_mode':st['aliases'][p]['reference_mode']=mode
  elif name=='alias_reference_path':st['aliases'][p]['reference_path']=('k',)
  elif name=='alias_hash':st['aliases'][p]['sha256']='0'*64
  elif name=='alias_dtype':st['aliases'][p]['dtype']='torch.float64'
  elif name=='alias_shape':st['aliases'][p]['shape']=[1]
  elif name=='missing_alias':del st['aliases'][p]
  elif name=='extra_alias':st['aliases'][('extra',)]=copy.deepcopy(st['aliases'][p])
  elif name=='unique_alias_collision':st['unique'][p]=base['q']
  elif name=='unique_missing':del st['unique'][('delivered_context',)]
  elif name=='unique_extra':st['unique'][('extra',)]=torch.ones(1)
  elif name=='identity_hash':st['identities'][('delivered_context',)]['sha256']='0'*64
  elif name=='base_mutation':bad['base']['q'][0,0,0,0]+=1
  else:st['base']=True
  reject(name,lambda:restore_snapshot(bad,mode),checks)
 for name in ['value','dtype','shape']:
  bad=copy.deepcopy(base)
  if name=='value':bad['q'][0,0,0,0]+=1
  elif name=='dtype':bad['q']=bad['q'].double()
  else:bad['q']=bad['q'].reshape(-1)
  trial=SnapshotStore();trial.add('P',base)
  try:trial.add('invalid',bad)
  except SnapshotMismatch as e:
   assert tensor_identity(e.payload['trial_actual']['q'])==tensor_identity(bad['q'])and tensor_identity(e.payload['reference_actual']['q'])==tensor_identity(base['q']);checks.append('incoming_'+name)
  else:raise AssertionError(name)
 # Corrupt an archive tensor without updating its identity; reject an unused
 # array, a repeated tensor reference, and a duplicate dictionary key.
 with np.load(io.BytesIO(dumps_npz({'x':torch.arange(4),'y':torch.arange(3)})),allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
 for name in ['array_bytes','unused_array','reused_tensor_reference','duplicate_dictionary_key']:
  aa=copy.deepcopy(arrays);header=json.loads(aa['metadata'].tobytes())
  if name=='array_bytes':aa['a0'][0]=99
  elif name=='unused_array':aa['extra']=np.ones(1)
  elif name=='reused_tensor_reference':header['object']['items'][1][1]=copy.deepcopy(header['object']['items'][0][1])
  else:header['object']['items'][1][0]=copy.deepcopy(header['object']['items'][0][0])
  aa['metadata']=np.frombuffer(json.dumps(header).encode(),dtype=np.uint8);stream=io.BytesIO();np.savez_compressed(stream,**aa);reject(name,lambda:loads_npz(stream.getvalue()),checks)
 with tempfile.TemporaryDirectory(dir='/tmp')as td:
  p=Path(td)/'small.npz';reject('cap_before_write',lambda:write_npz(p,store.payload,1),checks);assert not p.exists();write_npz(p,store.payload);before=sha(p);reject('existing_immutable',lambda:write_npz(p,store.payload),checks);assert sha(p)==before
 result={'status':'passed','sources':sources,'audit_script_sha256':sha(__file__),'map_cases':cases,'unaffected_exact_cases':unaffected,'maximum_explicit_formula_abs_error':max_error,'maximum_common_state_center_error':center_error,'maximum_relative_norm_error':norm_error,'snapshot_states':11,'snapshot_tensor_leaves_roundtripped':leaves,'encoded_synthetic_bytes':len(data),'negative_cases_rejected':checks,'pretrained_or_random_model_forwards':0,'gpu_used':False,'scope':'Uses frozen real maps as synthetic old/fresh fixtures; no fresh calibration or model execution. Context integration evidence reviewed separately.','wall_seconds':time.time()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (ROOT/'review/gate_wiki_context_independent.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
