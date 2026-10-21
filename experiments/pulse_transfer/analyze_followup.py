"""Audit exact stored metrics and derive paired grouped/KV contrasts."""
import argparse,json,math,resource
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from analyze_transfer import aggregate

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--kind',choices=('groups','kv'),required=True);a=ap.parse_args();p=a.run;rows=[json.loads(l) for l in (p/'windows.jsonl').read_text().splitlines()];raw=torch.load(p/'per_token.pt',weights_only=True,map_location='cpu');traces=torch.load(p/'traces.pt',weights_only=True,map_location='cpu');m=json.loads((p/'manifest.json').read_text());expected={'groups':680,'kv':1056}[a.kind];assert len(rows)==len(raw)==len(traces)==len({r['key'] for r in rows})==m['forward_passes']==m['records']==expected;assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'];err=0.;groups=defaultdict(list);byinput=defaultdict(dict)
 for r in rows:
  key=r['key'];pref='__'.join(key.split('__')[:5]);z=raw[key];b=raw[pref+'__'+('clean' if a.kind=='groups' else 'C')];j=1;assert r['score_index']==z['score_index']==j and r['scored_tokens']==510;assert z['token_sha256']==b['token_sha256']==r['token_sha256'];vals={'nll':float(z['loss'][j:].double().mean()),'baseline_nll':float(b['loss'][j:].double().mean()),'agreement':float(z['pred'][j:].eq(b['pred'][j:]).double().mean()),'kl':float(z['kl'][j:].double().mean())}
  for k,v in vals.items():assert math.isfinite(v);err=max(err,abs(v-r[k]));assert abs(v-r[k])<1e-12
  groups[r['corpus'],r['identity'],r['trial']].append(r);byinput[r['corpus'],r['identity'],r['window']][r['trial']]=r
 checks=json.loads((p/'checks.json').read_text());assert len(checks)==(8 if a.kind=='groups' else 416)
 for ch in checks:
  for k in ('loss_exact','pred_exact','query_later_exact'):
   if k in ch:assert ch[k]
 summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rs)} for (c,i,t),rs in groups.items() if t not in ('repeat','oracle')];dump(p/'summary.json',summary);contrasts=[]
 for (c,i,w),rr in byinput.items():
  if a.kind=='groups':
   for mode in ('row','column'):
    singles=sum(rr[f'{mode}_{j}']['nll']-rr['clean']['nll'] for j in (0,1,2));joint=rr[mode+'_012']['nll']-rr['clean']['nll'];contrasts.append({'corpus':c,'identity':i,'window':w,'trial':mode+'_joint_minus_sum_singles','delta_nll':joint-singles,'joint_delta_nll':joint,'sum_single_delta_nll':singles})
   for mode in ('0','1','2','01','02','12','012'):contrasts.append({'corpus':c,'identity':i,'window':w,'trial':'column_minus_row_'+mode,'delta_nll':rr['column_'+mode]['nll']-rr['row_'+mode]['nll']})
  else:
   for world in 'CRN':
    for mode in ('currentK','cleanK'):contrasts.append({'corpus':c,'identity':i,'window':w,'trial':world+'_zeroV_'+mode+'_minus_world','delta_nll':rr[world+'_zeroV_'+mode]['nll']-rr[world]['nll']})
    contrasts.append({'corpus':c,'identity':i,'window':w,'trial':world+'_block_minus_matched_mask','delta_nll':rr[world+'_mask_blocked']['nll']-rr[world+'_mask_causal']['nll']})
    contrasts.append({'corpus':c,'identity':i,'window':w,'trial':world+'_mask_reference_drift','delta_nll':rr[world+'_mask_causal']['nll']-rr[world]['nll']})
 dump(p/'paired_contrasts.json',contrasts);cg=defaultdict(list)
 for r in contrasts:cg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
 cs=[]
 for (c,i,t),ds in cg.items():
  ds=np.array(ds);rng=np.random.default_rng(20261021);bs=ds[rng.integers(0,len(ds),(2000,len(ds)))].mean(1);cs.append({'corpus':c,'identity':i,'trial':t,'n':len(ds),'mean_delta_nll':float(ds.mean()),'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist(),'exp_mean_delta_nll':float(np.exp(ds.mean()))})
 dump(p/'paired_summary.json',cs);dump(p/'audit.json',{'status':'passed','records':len(rows),'max_scalar_error':err,'checks':len(checks),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 for r in summary:
  if r['identity']=='natural' and r['trial'] in (('row_0','row_012','column_012','mlp_coordinate','residual_selected','residual_both','residual_all') if a.kind=='groups' else ('R','N','N_KC_VN','N_KN_VC','C_zeroV_currentK','C_zeroV_cleanK','C_mask_blocked')):print(r['corpus'],r['trial'],round(r['ppl_ratio'],5),round(r['mean_agreement'],5))
if __name__=='__main__':main()
