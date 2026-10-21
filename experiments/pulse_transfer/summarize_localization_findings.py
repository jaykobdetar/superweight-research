"""Produce exhaustive numeric ranges and sign summaries; no new fitting."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'gpt2_gate_localization_run'
assert json.loads((RUN/'audit.json').read_text())['status']=='passed'
s=json.loads((RUN/'summary.json').read_text());p=json.loads((RUN/'paired_summary.json').read_text());n=json.loads((RUN/'nonadditivity_summary.json').read_text())
cs=['wiki_validation','dubois','hobbes','optimizers'];ids=['natural','bos'];families=['remove_pc1','orthogonal_control','position'];sets=[f'L{l}'for l in range(4,12)]+['all']
def row(c,i,t):return next(r for r in s if(r['corpus'],r['identity'],r['trial'])==(c,i,t))
def compact(r):return {k:r[k]for k in ('p_ppl_ratio','p_ppl_ratio_ci95','p_mean_agreement','p_mean_kl','ppl_ratio','mean_agreement')}
result={'status':'descriptive_summary_pending_independent_claim_review','all_layer_groups':[],'singleton_groups':[],'nonadditivity':n,'baseline_P':[]}
for c in cs:
 for i in ids:
  result['baseline_P'].append({'corpus':c,'identity':i,**compact(row(c,i,'P'))})
  result['all_layer_groups'].append({'corpus':c,'identity':i,**{f:compact(row(c,i,f+'_all'))for f in families}})
  for l in sets[:-1]:
   r=row(c,i,'remove_pc1_'+l);control=next(z for z in p if(z['corpus'],z['identity'],z['trial'])==(c,i,'remove_pc1_'+l+'_minus_orthogonal_control'))
   result['singleton_groups'].append({'corpus':c,'identity':i,'layer':l,**{f:compact(row(c,i,f+'_'+l))for f in families},'removal_minus_control':control})
result['paired_contrast_sign_counts']={suffix:{'groups':len(rr),'positive_means':sum(r['mean_delta_nll']>0 for r in rr),'negative_means':sum(r['mean_delta_nll']<0 for r in rr),'positive_CIs':sum(r['ci95_delta_nll'][0]>0 for r in rr),'negative_CIs':sum(r['ci95_delta_nll'][1]<0 for r in rr)}for suffix in ('_minus_P','_minus_orthogonal_control')for rr in [[r for r in p if r['trial'].endswith(suffix)]]}
result['selection_warning']='Every singleton is retained. Any ranking/next-layer choice made from these outcomes uses this panel as development evidence.'
(ROOT/'review/localization_findings_ranges.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result['all_layer_groups']:
 print(r['corpus'],r['identity'],' '.join(f+': PPL/P %.5f agreement %.4f'%(r[f]['p_ppl_ratio'],r[f]['p_mean_agreement'])for f in families))
print('ALL_LAYER_NONADDITIVITY')
for r in n:print(r['corpus'],r['identity'],r['family'],r['mean_delta_nll'],r['ci95_delta_nll'])
