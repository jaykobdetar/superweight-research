"""CPU-only exact numeric audit and descriptive transfer summaries."""
import argparse,json,math,resource
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def aggregate(rows):
 rng=np.random.default_rng(20261021);d=np.array([r['nll']-r['baseline_nll'] for r in rows]);bs=d[rng.integers(0,len(d),(2000,len(d)))].mean(1);return {'windows':len(rows),'mean_delta_nll':float(d.mean()),'ppl_ratio':float(np.exp(d.mean())),'ppl_ratio_ci95':np.exp(np.quantile(bs,[.025,.975])).tolist(),'mean_agreement':float(np.mean([r['agreement'] for r in rows])),'mean_kl':float(np.mean([r['kl'] for r in rows])),'per_window_delta_nll':d.tolist()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();p=a.run;rows=[json.loads(l) for l in (p/'windows.jsonl').read_text().splitlines()];raw=torch.load(p/'per_token.pt',map_location='cpu',weights_only=True);traces=torch.load(p/'traces.pt',map_location='cpu',weights_only=True);keys=[r['key'] for r in rows];assert len(keys)==len(set(keys)) and set(keys)==set(raw)==set(traces);manifest=json.loads((p/'manifest.json').read_text());assert manifest['records']==len(rows) and manifest['model_hash_before']==manifest['model_hash_after'] and manifest['parameter_versions_unchanged'];err=0.;groups=defaultdict(list)
 for r in rows:
  key=r['key'];z=raw[key];base=raw[key if r['panel']=='history' else key.split('__')[0]+'__'+'__'.join(key.split('__')[1:5])+'__clean'];j=r['score_index'];assert j== (129 if r['panel']=='history' else 1);assert z['score_index']==j and z['token_sha256']==r['token_sha256']==base['token_sha256'];assert len(z['loss'])==r['input_tokens']-1 and len(z['loss'][j:])==r['scored_tokens'];values={'nll':float(z['loss'][j:].double().mean()),'baseline_nll':float(base['loss'][j:].double().mean()),'agreement':float(z['pred'][j:].eq(base['pred'][j:]).double().mean()),'kl':float(z['kl'][j:].double().mean())}
  for k,v in values.items():assert math.isfinite(v);err=max(err,abs(v-r[k]));assert abs(v-r[k])<1e-12
  if r['panel']=='source0':groups[(r['corpus'],r['length'],r['identity'],r['trial'])].append(r)
 assert len(rows)==976 and manifest['forward_passes']==984,(len(rows),manifest['forward_passes']);checks=json.loads((p/'checks.json').read_text());assert len(checks)==16 and all(x['loss_exact'] and x['pred_exact'] for x in checks)
 summary=[{'corpus':c,'length':n,'identity':i,'trial':t,**aggregate(rr)} for (c,n,i,t),rr in groups.items() if t not in ('repeat','oracle')];write(p/'summary.json',summary)
 history=[]
 for c in sorted({r['corpus'] for r in rows}):
  for n in (256,512):
   rs={v:sorted([r for r in rows if r['panel']=='history' and r['corpus']==c and r['length']==n and r['trial']==v],key=lambda r:r['window']) for v in ('U','R','R_double')};assert all(len(x)==4 for x in rs.values());means={v:np.median([abs(r['q_contribution']) for r in x]) for v,x in rs.items()};norms=[float(traces[r['key']]['capture']['output_source_before'].float().norm()) for r in rs['U']];suppression=means['R']/max(means['U'],1e-30);relative=means['U']/np.median(norms)
   history.append({'corpus':c,'length':n,'median_abs_q_contribution':{k:float(v) for k,v in means.items()},'R_to_U_ratio':float(suppression),'U_contribution_to_median_q_MLP_norm':float(relative),'meets_per_corpus_screen':bool(suppression<=.1 and relative>=.1),'per_window_q_contribution':{v:[r['q_contribution'] for r in x] for v,x in rs.items()},'suffix_delta_nll_R_minus_U':[r['nll']-u['nll'] for r,u in zip(rs['R'],rs['U'])],'suffix_delta_nll_double_minus_U':[r['nll']-u['nll'] for r,u in zip(rs['R_double'],rs['U'])]})
 write(p/'history_summary.json',history);write(p/'audit.json',{'records':len(rows),'forwards':manifest['forward_passes'],'closure_checks':len(checks),'max_scalar_error':err,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'status':'passed'})
 print('NATURAL 512')
 for r in summary:
  if r['identity']=='natural' and r['length']==512 and r['trial'] in ('row','column'):print(r['corpus'],r['trial'],round(r['ppl_ratio'],5),round(r['mean_agreement'],5))
 print('HISTORY 512')
 for r in history:
  if r['length']==512:print(r['corpus'],r['median_abs_q_contribution'],r['R_to_U_ratio'],r['meets_per_corpus_screen'])
if __name__=='__main__':main()
