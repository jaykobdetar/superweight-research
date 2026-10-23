from pathlib import Path
import time,subprocess,json,sys,os
ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'protocol.json').read_text())
env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':''}
def run(script,*args):
 print('START',script,*args,flush=True);subprocess.run([sys.executable,str(ROOT/'checks'/script),*args],env=env,check=True)
while not (ROOT/'sources/main/acquisition.json').exists():time.sleep(10)
run('run_prefix.py','main','--half');run('run_prefix.py','main');run('validate_endpoint.py')
for r in P['checkpoints']:
 name=r['name']
 if name=='main':continue
 run('acquire_prefix.py',name);run('run_prefix.py',name)
print('ALL EIGHT CHECKPOINTS COMPLETE',flush=True)
