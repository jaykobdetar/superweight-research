import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));torch.set_num_threads(2)
from gate_wiki_storage import read_npz
p=R/'gpt2_gate_wiki_calibration_run';best={}
for f in sorted((p/'inputs').glob('*.npz')):
 d=read_npz(f)
 for phase,ops in [('main',d['operations']),('suffix',d['suffix_operations'])]:
  for op in ops:
   for l,g in op['gate_checks'].items():
    for k in ['max_gain','min_mass','conditional_error']:
     v=g[k]
     if k not in best or (v<best[k]['value']if k=='min_mass'else v>best[k]['value']):best[k]={'value':v,'phase':phase,'metadata':d['metadata'],'mode':op['trial'],'layer':l}
(R/'review/gate_wiki_extrema_locations.json').write_text(json.dumps(best,indent=2));print(json.dumps(best,indent=2))
