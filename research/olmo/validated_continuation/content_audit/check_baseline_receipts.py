import hashlib,json
from pathlib import Path
import numpy as np
R=Path('/workspace/shared/olmo_computational_role/content_donors');S=R/'results/preregistered_v1';O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((R/'frozen_manifest.json').read_text());meta=json.loads((S/'run_meta.json').read_text());ident=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
for key,rel in [('manifest_sha256','frozen_manifest.json'),('protocol_sha256','contrast_protocol.json'),('amendment_sha256','preoutput_amendment_01.json'),('content_runner_sha256','run_content_donors.py'),('algebra_sha256','donor_algebra.py')]:assert meta[key]==sha(R/rel)
checks=[];groups=[]
worlds=('native','postboundary_mask0','postboundary_maskp','postboundary_mask0p')
for rec in m['sequences']:
 for label in (0,1):
  rows={}
  for world in worlds:
   key=f"{rec['pair_id']}__{label}__{world}";p=S/(key+'.json')
   if not p.exists():continue
   j=json.loads(p.read_text());assert j['key']==key and j['run_identity']==ident and j['manifest_sha256']==meta['manifest_sha256']
   assert j['cache_sha256']==sha(S/'cache'/(key+'.npz'))
   mm=j['metrics'];logits=np.array(mm['label_logits'],dtype=float);lp=np.array(mm['label_log_probs'])
   assert abs(mm['correct_minus_foil']-(logits[label]-logits[1-label]))<1e-5
   assert abs(mm['correct_label_loss']+lp[label])<1e-10
   assert abs((logits-lp)[0]-(logits-lp)[1])<1e-10
   assert mm['observed_target_logit_error']<1e-3
   assert len(mm['prefix_hashes'])==16
   with np.load(S/'cache'/(key+'.npz')) as a:
    assert a['P'].shape==(16,16,rec['n'],rec['n'])
    assert all(a[k].shape==(16,16,128) for k in ['v0','vp','vr'])
    for L in range(2,16):
     for source in ([0] if world=='postboundary_mask0' else [rec['p']] if world=='postboundary_maskp' else [0,rec['p']] if world=='postboundary_mask0p' else []):
      assert np.count_nonzero(a['P'][L,:,rec['p']+1:,source])==0
   rows[world]=mm;checks.append({'key':key,'cache_sha256_verified':True,'score_fields_consistent':True,'actual_masked_P_entries_zero':True,'manual_target_logit_error':mm['observed_target_logit_error']})
  if len(rows)==4:
   assert all(x['prefix_hashes']==rows['native']['prefix_hashes'] for x in rows.values())
   groups.append({'pair_id':rec['pair_id'],'label':label,'all_four_world_prefix_hashes_exact':True,'native_p_access_cost':rows['postboundary_maskp']['correct_label_loss']-rows['native']['correct_label_loss'],'mask0_p_access_cost':rows['postboundary_mask0p']['correct_label_loss']-rows['postboundary_mask0']['correct_label_loss']})
out={'completed_receipts_checked':len(checks),'planned_baselines':64,'complete':len(checks)==64,'completed_four_world_groups_checked':len(groups),'max_manual_target_logit_error':max(x['manual_target_logit_error'] for x in checks),'checks':checks,'four_world_groups':groups,'no_model_execution':True}
(O/'baseline_receipt_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['checks','four_world_groups']},indent=2))
