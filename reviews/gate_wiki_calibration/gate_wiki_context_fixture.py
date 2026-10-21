"""No pretrained weights/data execution: CPU synthetic maps + random GPT2 hooks."""
import sys,time,json,resource,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from gate_wiki_calibration import *
from gpt2_gate_localization import localization_map
from gpt2_gate_subspace import maps_digest
from gate_wiki_storage import *
from run_transfer import sha,dump

def main():
 start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021)
 m8=torch.load(ROOT/'gpt2_gate_fixed512_run/calibration_maps.pt',weights_only=True)
 old=torch.load(ROOT/'wiki_calibration_preparation/old_wiki_maps.pt',weights_only=True)
 dm=directions(m8);do=directions(old);maps={'O':old,'F':m8,'M':m8};ds={'O':do,'F':dm,'M':dm}
 exact=0
 for l in LAYERS:
  local={k:v.float()for k,v in m8[l].items()};d={k:dm[l][k].float()for k in ('u','v')}
  b=torch.randn(12,511)*2;mass=b.sigmoid()
  for family,previous in [('remove','remove_pc1'),('orthogonal','orthogonal_control')]:
   for which in SETS:
    new=wiki_map(family+'_M8_'+which,l,b,mass,{'M':local},{'M':d})
    ref=localization_map(previous+'_'+which,l,b,mass,local,d)
    assert all(torch.equal(new[i],ref[i])for i in (0,1));exact+=1
 # Random config is marked explicitly, contains no pretrained checkpoint.
 cfg=GPT2Config(n_layer=12,n_head=12,n_embd=768,n_positions=512,n_ctx=512,vocab_size=257,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)
 cfg._attn_implementation='sdpa';model=GPT2LMHeadModel(cfg).eval();model.requires_grad_(False)
 gate=GateWiki(model);gate.install_sets(maps,ds,'synthetic-fixture-reused-maps')
 x=torch.randint(0,257,(1,512));base=None;snapstore=SnapshotStore();results={};prefix=None;negative=[]
 with torch.inference_mode():
  for mode in CONDITIONS:
   gate.reset_gate(mode,zero=mode!='C',mask='native'if mode=='C'else'causal',snapshot=mode in SNAPSHOT_MODES)
   out=model(x,use_cache=False).logits
   assert gate.context_checks==gate.query_checks==gate.projection_checks==8
   assert set(gate.snapshots)==({10}if mode in SNAPSHOT_MODES else set())
   if mode=='P':
    base=out.clone();prefix={l:gate.current_contexts[l].clone()for l in LAYERS};ps={l:{k:v.clone()for k,v in s.items()}for l,s in gate.gate_cache.items()}
   elif mode in ('current','fullrank'):assert torch.equal(base,out)
   if mode.endswith('_L10'):
    assert torch.equal(gate.current_contexts[10],prefix[10]);assert torch.equal(gate.gate_cache[10]['b'],ps[10]['b'])
   if mode in SNAPSHOT_MODES:snapstore.add(mode,gate.snapshots[10])
   results[mode]={'checks':gate.gate_checks,'snapshots':list(gate.snapshots)}
   if mode not in ('C','P'):assert torch.equal(out[:,0],base[:,0])
  # One full exact-prefix control tests hook combination and physical shape.
  altered=x.clone();altered[:,128:]=torch.roll(altered[:,128:],1,1)
  gate.reset_gate('remove_OF_all');first=model(x,use_cache=False).logits
  gate.reset_gate('remove_OF_all');second=model(altered,use_cache=False).logits
  assert torch.equal(first[:,:128],second[:,:128])
 gate.close()
 raw=snapstore.payload;packed=dumps_npz(raw);restored=loads_npz(packed)
 for mode in SNAPSHOT_MODES:
  a=flatten(restore_snapshot(raw,mode));b=flatten(restore_snapshot(restored,mode));assert a.keys()==b.keys()
  for k in a:assert torch.equal(a[k],b[k])
 for corruption in ('alias_hash','alias_reference','identity_hash','unique_collision','missing_alias'):
  bad=copy.deepcopy(restored);state=bad['states'][SNAPSHOT_MODES[1]];p=SHARED_PATHS[0]
  if corruption=='alias_hash':state['aliases'][p]['sha256']='0'*64
  elif corruption=='alias_reference':state['aliases'][p]['reference_path']=('k',)
  elif corruption=='identity_hash':state['identities'][('delivered_context',)]['sha256']='0'*64
  elif corruption=='unique_collision':state['unique'][p]=torch.zeros_like(raw['base']['q'])
  else:del state['aliases'][p]
  try:restore_snapshot(bad,SNAPSHOT_MODES[1])
  except AssertionError:negative.append(corruption)
  else:raise AssertionError(('corrupt alias accepted',corruption))
 for corruption in ('value','shape','dtype'):
  bad=copy.deepcopy(raw['base'])
  if corruption=='value':
   bad['q'][0,0,0,0]+=1
   assert not torch.equal(bad['q'],raw['base']['q'])
  elif corruption=='shape':bad['q']=bad['q'].reshape(12,512,64)
  else:bad['q']=bad['q'].double()
  trial=SnapshotStore();trial.add('P',raw['base'])
  try:trial.add('invalid',bad)
  except SnapshotMismatch as error:
   assert torch.equal(error.payload['trial_actual']['q'],bad['q']);assert torch.equal(error.payload['reference_actual']['q'],raw['base']['q']);negative.append('incoming_'+corruption)
  else:raise AssertionError('incoming difference silently aliased')
 result={'status':'passed','pretrained_model_loaded':False,'synthetic_random_cpu_model_forwards':26,'gpu_used':False,'M8_maps_exact_old_localization_cases':exact,'conditions':results,'snapshot_states':len(raw['states']),'snapshot_encoded_bytes':len(packed),'negative_cases_rejected':negative,'exact_current_fullrank_q0_singleton_prefix':True,'sources':{n:sha(ROOT/n)for n in ('gate_wiki_calibration.py','gate_wiki_storage.py')},'wall_seconds':time.time()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 dump(ROOT/'review/gate_wiki_context_fixture.json',result);print({k:v for k,v in result.items()if k!='conditions'})
if __name__=='__main__':main()
