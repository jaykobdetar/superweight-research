"""Read-only audit of tokenizer-only reconstruction outputs; no tokenizer/model load."""
from pathlib import Path
import hashlib,json,resource
import torch
P=Path(__file__).resolve().parents[1];R=P/'localization_inputs_rebuild_v2';O=P/'localization_inputs_v2';F=P/'localization_inputs_rebuild'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
m=j(R/'reconstruction.json');assert m['status']=='passed'and m['model_forwards']==0 and not m['model_loaded']and not m['gpu_used']
assert m['source_sha256']==sha(P/'prepare_localization_replay_inputs.py')=='ed0e2b31703a7aab0d4ef65803d1ea47f10a0a867953560315dddf09f0714703'
files=dict(m['new_output_sha256']);files['preflight.json']=m['historical_preflight_sha256'];assert len(files)==7
for n,h in files.items():
 assert sha(R/n)==sha(O/n)==h
 assert not(R/n).is_symlink()and (R/n).stat().st_ino!=(O/n).stat().st_ino
rows=j(R/'data_manifest.json')['passages'];inv=j(R/'exclusion_inventory.json');checks=0
for r in rows:
 for old in inv['evaluation_and_olmo_calibration'][r['corpus']]:
  assert max(r['char_start'],old['char_start'])>=min(r['used_char_end'],old['char_end']);checks+=1
assert checks==m['historical_interval_checks']==1192
new=load(R/'tokens.pt');old=load(F/'tokens.pt');assert len(new)==len(old)==15
assert sha(R/'tokens.pt')!=sha(F/'tokens.pt')
for a,b in zip(new,old):assert a['corpus']==b['corpus']and a['window']==b['window']and torch.equal(a['tokens'],b['tokens'])
a=(F/'prepare_localization_replay_inputs_initial.py').read_text();b=(P/'prepare_localization_replay_inputs.py').read_text()
# Confirm functional code change is canonical string-object reuse before torch.save.
assert b.replace(" canonical_corpora={c:c for c in ('wiki_validation','dubois','hobbes','optimizers')}\n",'').replace("c,w,a=canonical_corpora[record['corpus']],record['window'],record['char_start']","c,w,a=record['corpus'],record['window'],record['char_start']")==a
out={'status':'passed','helper_sha256':m['source_sha256'],'seven_outputs_byte_identical':files,'all_separate_inodes':True,'historical_interval_checks':checks,'initial_serialization_failure_preserved':True,'initial_and_corrected15_token_tensors_exact':True,'only_source_delta_canonical_string_reuse':True,'model_or_tokenizer_loaded_by_auditor':False,'gpu_used':False,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'recorded_reconstruction_resources':{k:m[k]for k in('wall_seconds','peak_rss_kib')},'limits':'Historical selection/exclusion proof is copied with frozen hashes, not regenerated. Local source/tokenizer reconstruction was performed by parent; this auditor verifies resulting bytes, interval checks and source logic.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
