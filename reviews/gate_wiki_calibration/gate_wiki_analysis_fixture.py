"""Deterministic synthetic factorial contrasts; no source/model data."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from analyze_gate_wiki_calibration import *
from gate_wiki_calibration import mode_spec
from run_transfer import dump,sha
shards=[]
for corpus in ('wiki_validation','utilitarianism','ethics','optimizers_new'):
 for identity in ('natural','bos'):
  for window in range(4):
   modes=list(CONDITIONS)+(['repeat','identity']if identity=='natural'and window==0 else [])
   raw={};scale=window+1
   for mode in modes:
    delta=0.
    if mode in CONDITIONS:
     family,center,basis,layers=mode_spec(mode)
     if family:
      delta=(9. if center=='M'else 1.*(center=='F')+2.*(basis=='F')+3.*(center==basis=='F'))*scale
      if family=='orthogonal':delta*=.5
      if len(layers)==8:delta*=2
    raw[f'gate__{corpus}__{window}__512__{identity}__'+mode]={'token_sha256':'1'*64,'final_hidden_sha256':'0'*64,'score_index':1,'loss':torch.full((511,),1.+delta),'pred':torch.zeros(511,dtype=torch.int64),'kl':torch.zeros(511),'kl_p':torch.zeros(511),'agreement_p':torch.ones(511,dtype=torch.bool)}
   shards.append({'metadata':{'corpus':corpus,'identity':identity,'window':window},'per_token':raw,'norms':{}})
r=summarize(shards);assert len(r['summary'])==576 and len(r['contrasts'])==912
for d in r['contrasts']:
 if d['family']not in ('remove','orthogonal'):continue
 expected={'center_at_O':1,'center_at_F':4,'direction_at_O':2,'direction_at_F':5,'interaction':3,'M8_minus_OO':9,'M8_minus_FF':3}[d['name']]
 expected*=.5 if d['family']=='orthogonal'else 1
 expected*=2 if d['layer_set']=='all'else 1
 assert d['delta_nll']['per_window']==[expected*w for w in (1,2,3,4)]
 expectedstat=paired([expected*w for w in (1,2,3,4)])
 assert d['delta_nll']==expectedstat
# Paired subtraction cancels arbitrary common window variation exactly.
a=paired([2,4,6,8]);b=paired([1,2,3,4]);assert a['mean']==2*b['mean']and a['ci95']==[2*x for x in b['ci95']]
dump(ROOT/'review/gate_wiki_analysis_fixture.json',{'status':'passed','summary_rows':576,'contrast_rows':912,'all_factorial_directions_interactions_and_M8_contrasts_exact':True,'bootstrap_common_index_scaling_exact':True,'model_forwards':0,'source_sha256':sha(ROOT/'analyze_gate_wiki_calibration.py')})
print('passed')
