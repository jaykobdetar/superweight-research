"""Stage a private offline workspace and run nine sequential transfer batches."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGES=[
 ('gpt2_run',['run_transfer.py','--model','gpt2'],['analyze_transfer.py']),
 ('llama_run',['run_transfer.py','--model','llama'],['analyze_transfer.py']),
 ('gpt2_groups_run',['gpt2_groups.py'],['analyze_followup.py','--kind','groups']),
 ('llama_source_kv_run',['llama_source_kv.py'],['analyze_followup.py','--kind','kv']),
 ('gpt2_writes_run',['gpt2_writes.py'],['analyze_writes.py']),
 ('llama_replacements_run',['llama_replacements.py'],['analyze_replacements.py']),
 ('gpt2_layer_writes_run',['gpt2_layer_writes.py'],['analyze_layer_writes.py']),
 ('gpt2_regrowth_kv_run',['gpt2_regrowth_kv.py'],['analyze_regrowth_kv.py']),
 ('gpt2_value_geometry_run',['gpt2_value_geometry.py'],['analyze_value_geometry.py'])]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--stage',choices=[x[0] for x in STAGES]);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());work=args.work.resolve();dest=work/'pulse_transfer'
    source_manifest=json.loads((ROOT/'manifests/input_sources.json').read_text())['sources']
    models=json.loads((ROOT/'manifests/transfer_models.json').read_text())
    for name,expected in source_manifest.items():
        p=Path(cfg['sources'][name]).expanduser().resolve();assert sha(p)==expected['sha256'],(name,'source hash mismatch')
        cfg['sources'][name]=str(p)
    for name,expected in models.items():
        model=cfg['models'][name];path=Path(model['path']).expanduser().resolve()
        assert model['model_id']==expected['model_id'] and model['revision']==expected['revision']
        for filename,identity in expected['checkpoint_files'].items():
            p=path/filename;assert p.stat().st_size==identity['bytes'] and sha(p)==identity['sha256'],(name,filename,'checkpoint mismatch')
        model['path']=str(path)
    dest.mkdir(parents=True,exist_ok=True)
    for source in sorted((ROOT/'experiments/pulse_transfer').iterdir()):
        if source.suffix not in ('.py','.md'):continue
        target=dest/source.name
        if target.exists():assert sha(source)==sha(target),('staged code changed',target)
        else:shutil.copyfile(source,target)
    config=dest/'inputs.local.json'
    if config.exists():assert json.loads(config.read_text())==cfg,'Do not mix input configurations in one workspace'
    else:config.write_text(json.dumps(cfg,indent=2)+'\n')
    print('Verified inputs and staged private workspace:',dest,flush=True)
    if args.prepare_only:return
    for name,cmd,analysis in STAGES:
        if args.stage and name!=args.stage:continue
        out=dest/name
        if out.exists():raise RuntimeError(f'{out} exists; use a fresh workspace or choose a not-yet-started stage. No overwrite or silent resume.')
        subprocess.run([sys.executable,str(dest/cmd[0]),*cmd[1:],'--config',str(config),'--out',str(out)],check=True,cwd=dest)
        subprocess.run([sys.executable,str(dest/analysis[0]),str(out),*analysis[1:]],check=True,cwd=dest)

if __name__=='__main__':main()
