"""Input-geometry-only diagnosis of the failed optimizer greedy selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import hashlib,json,resource,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from gate_localization_inputs import sha,overlap,th
torch.set_num_threads(1)
out=root/'gpt2_gate_localization_inputs';failure=json.loads((out/'preflight.json').read_text());assert failure['status']=='failed'
inventory=json.loads((out/'exclusion_inventory.json').read_text());spans=inventory['evaluation_and_olmo_calibration']['optimizers']
cfg=json.loads((root/'inputs.local.json').read_text());old=json.loads((root/'gpt2_run/data_manifest.json').read_text())
texts=[]
for k in('sgd','adam','adamw'):
    p=Path(cfg['sources'][k]);assert sha(p)==old['sources'][k]['sha256'];texts.append(p.read_text())
text='\n\n'.join(texts);tok=AutoTokenizer.from_pretrained(cfg['models']['gpt2']['path'],local_files_only=True)
eligible=[]
for j in range(1001):
    a=j*(len(text)-8192)//1000;e=tok(text[a:a+8192],add_special_tokens=False,return_offsets_mapping=True)
    assert len(e['input_ids'])>=512;b=a+max(v[1]for v in e['offset_mapping'][:512])
    if not any(overlap(a,b,r['char_start'],r['char_end'])for r in spans):
        eligible.append({'numerator':j,'start':a,'end':b,'token_sha256':th(torch.tensor(e['input_ids'][:512]))})
packing=[];end=-1
for r in sorted(eligible,key=lambda r:(r['end'],r['start'],r['numerator'])):
    if r['start']>=end:packing.append(r);end=r['end']
# Exact dynamic program: four increasing windows, nonoverlap, minimum summed
# distance to the original ordered quartile targets; lexicographic numerator tie.
targets=(125,375,625,875);states={}
for i,r in enumerate(eligible):states[i]=(abs(r['numerator']-targets[0]),(r['numerator'],),(i,))
for target in targets[1:]:
    nxt={}
    for i,r in enumerate(eligible):
        options=[(v[0]+abs(r['numerator']-target),v[1]+(r['numerator'],),v[2]+(i,))
                 for j,v in states.items()if eligible[j]['end']<=r['start']]
        if options:nxt[i]=min(options)
    states=nxt
best=min(states.values())if states else None
result={'status':'completed','model_forwards':0,'model_loaded':False,'gpu_used':False,
    'failed_preflight_sha256':sha(out/'preflight.json'),'exclusion_inventory_sha256':sha(out/'exclusion_inventory.json'),
    'script_sha256':sha(__file__),'eligible_grid_candidates':len(eligible),'maximum_nonoverlapping_grid_windows':len(packing),
    'maximum_packing':packing,'best_four_by_quartile_distance':None if best is None else[eligible[i]for i in best[2]],
    'best_total_numerator_distance':None if best is None else best[0],
    'scope':'Only optimizer input geometry; no changes to failed artifacts or selection rule.',
    'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
assert result['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
