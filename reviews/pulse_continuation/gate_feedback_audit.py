"""CPU-only independent replay of controlled source-gate swaps."""
import ast,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'gate_feedback_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']and meta['forward_passes']==768
for field,p in [('script_sha256',ROOT/'gate_feedback.py'),('tokens_sha256',ROOT/'residual_paths_run/tokens.pt'),('mean_cache_sha256',ROOT/'cache_run/calibration.pt'),('gate_calibration_sha256',ROOT/'global_gates_run/calibration.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT);assert result['rows']==768 and result['groups']==96
old=torch.load(ROOT/'global_gates_run/per_token.pt',map_location='cpu',weights_only=True);old={k:{z['window']:z for z in v}for k,v in old.items()}
quiet=torch.load(ROOT/'quiet_anchor_run/per_token.pt',map_location='cpu',weights_only=True);quiet={k:{z['window']:z for z in v}for k,v in quiet.items()}
held=torch.load(ROOT/'gate_subspaces_run/heldout_native_gates.pt',map_location='cpu',weights_only=True)
donors=torch.load(OUT/'native_gate_donors.pt',map_location='cpu',weights_only=True);assert len(donors)==32
checks=json.loads((OUT/'reconstruction_checks.json').read_text());assert len(checks)==128
controls={};effects={};ratios={};donor_changes={}
for c in ['wiki','hume','plato','python']:
 g='gate_paths_'+c
 for i in range(8):
  for k in ['loss','pred','kl']:
   assert torch.equal(raw[g+'__zero_native'][i][k],quiet['quiet_'+c+'__K_mean_V_zero_clean'][i][k])
   assert torch.equal(raw[g+'__zero_cross_all'][i][k],raw[g+'__zero_cross_all_deleted'][i][k])
   for w in ['mean','zero']:assert torch.equal(raw[g+'__'+w+'_cross_L2'][i][k],raw[g+'__'+w+'_current'][i][k])
   for t,ot in [('clean','clean'),('mean_native','mean_KV'),('mean_current','oracle_clean'),('mean_global_wiki','wiki_global_clean'),('mean_global_hume','hume_global_clean')]:assert torch.equal(raw[g+'__'+t][i][k],old['gates_'+c+'__'+ot][i][k])
  for l in range(2,16):
   for w in ['mean','zero']:
    a=donors[c,i][w][l];assert a.shape==(16,512)and torch.isfinite(a).all()and(a>=0).all()and(a<=1).all()and torch.equal(a[:,0],torch.ones(16))
   assert torch.equal(donors[c,i]['mean'][l],held[c,i][l])
  assert torch.equal(donors[c,i]['mean'][2],donors[c,i]['zero'][2])
 for w in ['mean','zero']:
  for t in ['current','same_all']:controls[c+'__'+w+'_'+t]=compare(raw,groups,g,w+'_'+t,w+'_native')
  for t in ['cross_all','cross_early','cross_middle','cross_late','cross_L2','cross_L6','global_wiki','global_hume']:effects[c+'__'+w+'_'+t]=effect(groups,g,w+'_'+t,w+'_native')
 for key,rows in groups.items():
  if key.startswith(g+'__'):
   assert all(z['score_start']==1 and z['scored_tokens']==510 for z in rows.values())
   ratios[c+'__'+key.split('__')[1]]=math.exp(effect(groups,g,key.split('__')[1],'clean')['mean'])
 donor_changes[c]={str(l):{'mean_absolute_delta':float(torch.stack([(donors[c,i]['zero'][l][:,1:]-donors[c,i]['mean'][l][:,1:]).abs().mean()for i in range(8)]).mean()),'mean_signed_delta':float(torch.stack([(donors[c,i]['zero'][l][:,1:]-donors[c,i]['mean'][l][:,1:]).mean()for i in range(8)]).mean())}for l in range(2,16)}
for z in checks:
 c,i,w,t=z['corpus'],z['window'],z['world'],z['control'];a=raw['gate_paths_'+c+'__'+w+'_'+t][i];b=raw['gate_paths_'+c+'__'+w+'_native'][i]
 assert abs(float((a['loss']-b['loss']).double().mean())-z['delta_nll_to_native'])<1e-12
 assert float(a['pred'].eq(b['pred']).double().mean())==z['agreement_to_native']
# Execute only the extracted small context hook on synthetic CPU tensors, never import model runners.
tree=ast.parse((ROOT/'gate_feedback.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='DonorGates')
class Stub:
 def context(self,l,inputs):return ('fallback',)
ns={'Gates':Stub,'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted context>','exec'),ns);obj=ns['DonorGates']()
gen=torch.Generator().manual_seed(7);o=torch.randn(1,5,2048,generator=gen).half();cur=torch.rand(16,5,generator=gen);don=torch.rand(16,5,generator=gen);v=torch.randn(1,1,2048,generator=gen).half()
obj.donor_active=True;obj.donor={2:don};obj.mean={2:{'v':v}};obj.probabilities=lambda l:cur
for use in [False,True]:
 obj.donor_layers=[2]if use else [];obj.gpu={2:{}};y=obj.context(2,(o,))[0];a=don if use else cur
 expect=((1-a[:,1:].T[None,:,:,None])*o[:,1:].reshape(1,4,16,128).float()+a[:,1:].T[None,:,:,None]*v.reshape(1,1,16,128).float()).reshape(1,4,2048).half()
 assert torch.equal(y[:,0],o[:,0])and torch.equal(y[:,1:],expect)and 2 not in obj.gpu
result.update(controls=controls,effects_vs_recipient_native=effects,ppl_ratios_to_clean=ratios,native_donor_gate_changes=donor_changes,prior_mean_world_anchors_exact=True,mean_native_gate_donors_exact=True,zero_cross_all_source_state_invariance_exact=True,synthetic_localization_and_formula_pass=True,reconstruction_checks_replayed=128,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['controls','effects_vs_recipient_native','ppl_ratios_to_clean','native_donor_gate_changes']},indent=2))
