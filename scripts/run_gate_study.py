"""Reproduce the fixed-512 gate study without rerunning previous model batches.

Private input IDs and activations are created only in the chosen workspace.
The caller explicitly requests eight calibration and 1,352 study forwards.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
PARENTS=('gpt2_groups_run','gpt2_writes_run','gpt2_layer_writes_run',
         'gpt2_regrowth_kv_run','gpt2_value_geometry_run','gpt2_query_source_run',
         'gpt2_query_mixture_run')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def copy_exact(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert sha(source)==sha(target),('Existing staged bytes differ',target)
    else:shutil.copyfile(source,target)

def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args();started=time.time();work=args.work.resolve()
    if work.is_relative_to(ROOT):
        rel=work.relative_to(ROOT)
        assert rel.parts and rel.parts[0].startswith('.reproduce'), 'Inside the repository use an ignored .reproduce... workspace'
    dest=work/'pulse_transfer'
    identity=read(ROOT/'manifests/gate_reproduction.json')
    supplied=read(args.config)
    models=read(ROOT/'manifests/transfer_models.json')
    sources=read(ROOT/'manifests/input_sources.json')['sources']
    cfg={'models':{},'sources':{}}
    for name,expected in sources.items():
        path=Path(supplied['sources'][name]).expanduser().resolve()
        assert sha(path)==expected['sha256'],('Source hash mismatch',name)
        cfg['sources'][name]=str(path)
    for name in ('gpt2','llama'):
        m=supplied['models'][name];expected=models[name]
        assert m['model_id']==expected['model_id'] and m['revision']==expected['revision']
        path=Path(m['path']).expanduser().resolve()
        files=expected['checkpoint_files'] if name=='gpt2' else {
            k:expected['checkpoint_files'][k]for k in ('config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json')}
        for filename,d in files.items():
            p=path/filename;assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'],(name,filename,'Checkpoint/tokenizer hash mismatch')
        cfg['models'][name]={'model_id':m['model_id'],'revision':m['revision'],'path':str(path)}
    code=read(ROOT/'manifests/transfer_source_export.json')['files']
    assert all(n in code for n in ('prepare_gate_replay_inputs.py','rebuild_gate_calibration.py','gpt2_gate_fixed512.py','analyze_gate_fixed512.py'))
    dest.mkdir(parents=True,exist_ok=True)
    for name,h in code.items():
        assert Path(name).name==name, 'Source names must be basenames'
        source=ROOT/'experiments/pulse_transfer'/name
        assert sha(source)==h,('Exported source changed',name)
        copy_exact(source,dest/name)
    metadata=[('gpt2_run',n)for n in ('manifest.json','selection.json','data_manifest.json')]
    metadata += [('llama_run','data_manifest.json')]+[(n,'data_manifest.json')for n in PARENTS]
    expected_seeds=identity['seed_metadata_sha256']
    assert set(expected_seeds)=={name+'/'+file for name,file in metadata}, 'Seed metadata inventory differs'
    for name,file in metadata:
        source=ROOT/'results/pulse_transfer'/name/file
        assert sha(source)==expected_seeds[name+'/'+file],('Seed metadata hash mismatch',name,file)
        copy_exact(source,dest/name/file)
    config=dest/'inputs.local.json'
    if config.exists():assert read(config)==cfg,'Do not mix input configurations'
    else:save(config,cfg)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    def invoke(script,*arguments):
        subprocess.run([sys.executable,str(dest/script),*map(str,arguments)],cwd=dest,env=env,check=True)
    prior=dest/'llama_replacements_run'
    if prior.exists():
        evidence=read(prior/'input_reconstruction.json')
        assert evidence['status']=='passed' and evidence['model_forwards']==0 and evidence['passages']==8
        assert evidence['model_loaded'] is False and evidence['gpu_used'] is False
        assert evidence['tokenizer_files']=={k:models['llama']['checkpoint_files'][k]['sha256']
            for k in ('config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json')}
        assert sha(prior/'calibration_inputs.json')==identity['historical_llama_calibration_inputs_sha256']==evidence['calibration_inputs_sha256']
        assert sha(prior/'calibration_tokens.pt')==evidence['calibration_tokens_sha256']
    else:
        invoke('prepare_gate_replay_inputs.py','--config',config,
               '--models-manifest',ROOT/'manifests/transfer_models.json',
               '--parent',dest/'llama_run','--out',prior,
               '--expected-input-sha256',identity['historical_llama_calibration_inputs_sha256'])
    save(work/'gate_preparation.json',{'status':'passed','model_forwards':0,'code_sha256':code,
        'config_sha256':sha(config),'source_identity_sha256':sha(ROOT/'manifests/gate_reproduction.json'),
        'seed_metadata':{name+'/'+file:sha(dest/name/file)for name,file in metadata},
        'input_reconstruction_sha256':sha(prior/'input_reconstruction.json'),
        'note':'Reuses only verified prior metadata. No previous model experiment or Llama weight load.'})
    if args.prepare_only:
        print('PREPARED_GATE_WORKSPACE',dest,flush=True);return
    cal=dest/'gpt2_gate_calibration_rebuild';out=dest/'gpt2_gate_fixed512_run'
    assert not cal.exists() and not out.exists(),'Calibration/study output already exists; no overwrite or automatic resume'
    invoke('rebuild_gate_calibration.py','--config',config,'--out',cal)
    cm=read(cal/'manifest.json')
    assert cm['status']=='completed' and cm['forward_passes']==8 and cm['evaluation_forwards']==0
    for file,field in (('calibration_raw.pt','calibration_raw_sha256'),('calibration_maps.pt','calibration_maps_sha256')):
        assert sha(cal/file)==identity[field]==cm[field],('Strict reference mismatch; preserve this rebuild separately',file)
    assert cm['calibration_maps_tensor_digest']==identity['calibration_maps_tensor_digest']
    invoke('gpt2_gate_fixed512.py','--config',config,'--calibration-reference',cal,'--out',out)
    invoke('analyze_gate_fixed512.py',out)
    m=read(out/'manifest.json');audit=read(out/'audit.json')
    assert m['status']==m['phase']=='completed' and m['forward_passes']==m['forward_attempts']==1352
    assert m['records']==m['evaluation_forwards']==680 and m['suffix_forwards']==672 and m['calibration_forwards']==0
    assert audit['status']=='passed' and audit['records']==680 and audit['forwards']==1352
    assert audit['total_new_gate_context_checks']==10240 and audit['suffix_audit']['records']==672
    save(work/'gate_reproduction.json',{'status':'completed','new_calibration_forwards':8,
        'new_study_forwards':1352,'new_total_forwards':1360,'wall_seconds':time.time()-started,
        'calibration_manifest_sha256':sha(cal/'manifest.json'),'study_manifest_sha256':sha(out/'manifest.json'),
        'study_audit_sha256':sha(out/'audit.json'),
        'note':'Completion is not an exact comparison to published results. Run the numeric comparison separately.'})
    print('COMPLETED_GATE_REPRODUCTION',out,flush=True)
if __name__=='__main__':main()
