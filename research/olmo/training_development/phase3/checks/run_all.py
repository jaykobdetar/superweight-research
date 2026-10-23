from pathlib import Path
import os,json,sys,subprocess
R=Path(__file__).resolve().parents[1];P=json.loads((R/'protocol.json').read_text());env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':''}
for ref in P['checkpoints']:
 for script in ['acquire_prefix.py','run_prefix.py']:
  print('START',script,ref['name'],flush=True);subprocess.run([sys.executable,str(R/'checks'/script),ref['name']],env=env,check=True)
print('ALL FOUR ADAPTIVE CHECKPOINTS COMPLETE',flush=True)
