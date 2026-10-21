"""Reproduce gate localization with 8 calibration +1,868 study forwards.

Earlier model experiments are unnecessary. --prepare-only stages code/metadata
and uses only tokenizers. A separately supplied verified calibration reference
avoids new calibration forwards; its files are copied and checked explicitly.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from run_gate_study import ROOT,sha,read,save,copy_exact

def validate_calibration(m,identity,rebuilt):
 assert m.get('status')=='completed','Calibration must have completed'
 assert m['calibration_maps_tensor_digest']==identity['calibration_maps_tensor_digest']
 assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] is True,'Calibration model closure failed'
 if rebuilt:
  assert m['kind']=='calibration_only_rebuild'
  assert m['expected_forwards']==m['forward_passes']==m['calibration_forwards']==8
  assert m['evaluation_forwards']==0
  assert m['calibration_raw_sha256']==identity['calibration_raw_sha256']
  assert m['calibration_maps_sha256']==identity['calibration_maps_sha256']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--calibration-reference',type=Path);args=ap.parse_args()
 started=time.time();work=args.work.resolve();dest=work/'pulse_transfer';env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 subprocess.run([sys.executable,str(ROOT/'scripts/run_gate_study.py'),'--config',str(args.config.resolve()),'--work',str(work),'--prepare-only'],env=env,check=True)
 identity=read(ROOT/'manifests/localization_reproduction.json');metadata=dest/'localization_reproduction_metadata'
 for n,h in identity['metadata_sha256'].items():
  source=ROOT/'results/gate_localization/input_reconstruction'/n;assert sha(source)==h;copy_exact(source,metadata/n)
 save(work/'localization_preparation.json',{'status':'passed','model_forwards':0,'metadata_sha256':identity['metadata_sha256'],'source_identity_sha256':sha(ROOT/'manifests/localization_reproduction.json'),'note':'Historical selection evidence staged. Frozen private tokens/directions are rebuilt after calibration is available; no previous model study is rerun.'})
 if args.prepare_only:print('PREPARED_LOCALIZATION_WORKSPACE',dest);return
 cfg=dest/'inputs.local.json';cal=dest/'localization_calibration';inputs=dest/'localization_inputs_v2';out=dest/'gpt2_gate_localization_run'
 assert not any(p.exists()for p in(cal,inputs,out)),'Use a fresh workspace; no overwrite or automatic resume'
 def invoke(script,*a):subprocess.run([sys.executable,str(dest/script),*map(str,a)],cwd=dest,env=env,check=True)
 newcal=8
 if args.calibration_reference:
  reference=args.calibration_reference.resolve();cal.mkdir();m=read(reference/'manifest.json')
  validate_calibration(m,identity,False)
  for n in identity['calibration_dependency_files']:copy_exact(reference/n,cal/n)
  newcal=0
 else:invoke('rebuild_gate_calibration.py','--config',cfg,'--out',cal)
 validate_calibration(read(cal/'manifest.json'),identity,not args.calibration_reference)
 assert sha(cal/'calibration_raw.pt')==identity['calibration_raw_sha256']
 assert sha(cal/'calibration_maps.pt')==identity['calibration_maps_sha256']
 invoke('prepare_localization_replay_inputs.py','--config',cfg,'--metadata',metadata,'--calibration',cal,'--out',inputs)
 evidence=read(inputs/'reconstruction.json');assert evidence['status']=='passed'and evidence['model_forwards']==0
 invoke('gpt2_gate_localization.py','--config',cfg,'--inputs',inputs,'--calibration-reference',cal,'--out',out)
 invoke('analyze_gate_localization.py',out)
 m=read(out/'manifest.json');a=read(out/'audit.json')
 assert m['status']==m['phase']=='completed'and m['forward_passes']==m['forward_attempts']==1868
 assert m['records']==m['evaluation_forwards']==938 and m['suffix_forwards']==930 and m['calibration_forwards']==0
 assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] is True
 assert m['calibration_maps_unchanged'] is True and m['calibration_maps_tensor_digest']==identity['calibration_maps_tensor_digest']
 assert m['precision_settings_before']==m['precision_settings_after']
 assert a['status']=='passed'and a['records']==938 and a['forwards']==1868 and a['calibration_forwards']==0
 assert a['suffix_audit']['records']==930 and a['total_new_gate_context_checks']==14400
 save(work/'localization_reproduction.json',{'status':'completed','new_calibration_forwards':newcal,'new_study_forwards':1868,'new_total_forwards':newcal+1868,'calibration_manifest_sha256':sha(cal/'manifest.json'),'study_manifest_sha256':sha(out/'manifest.json'),'study_audit_sha256':sha(out/'audit.json'),'wall_seconds':time.time()-started,'note':'Completed reproduction is not an exact comparison with the reference, and not an external independent replication claim. Compare exported per-token numerical results separately.'})
 print('COMPLETED_LOCALIZATION_REPRODUCTION',out)
if __name__=='__main__':main()
