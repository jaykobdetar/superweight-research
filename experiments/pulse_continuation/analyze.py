"""Paired-window bootstrap summaries for completed continuation runs."""
import csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
def ci(x):
 x=np.asarray(x,float);rng=np.random.default_rng(20261021);m=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
 return float(x.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))
def write(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def analyze(out):
 meta=json.loads((out/'manifest.json').read_text());assert 'completed_utc' in meta,'Wait for run completion'
 groups=defaultdict(list)
 for line in (out/'windows.jsonl').read_text().splitlines():
  r=json.loads(line);groups[r['group'],r['trial']].append(r)
 rows=[];contrasts=[]
 for (group,trial),rs in groups.items():
  assert len({r['window'] for r in rs})==len(rs)
  d,l,h=ci([r['nll']-r['baseline_nll'] for r in rs]);agr,al,ah=ci([r['agreement'] for r in rs])
  kl,klow,khigh=ci([r['kl_clean_to_condition'] for r in rs])
  rows.append({'group':group,'trial':trial,'windows':len(rs),'nll':np.mean([r['nll'] for r in rs]),'ppl_ratio':math.exp(d),'ratio_low':math.exp(l),'ratio_high':math.exp(h),
   'delta_nll':d,'delta_low':l,'delta_high':h,'agreement':agr,'agreement_low':al,'agreement_high':ah,'kl':kl,'kl_low':klow,'kl_high':khigh})
  for reference in ('clean','deleted'):
   ref={r['window']:r for r in groups.get((group,reference),[])}
   if not ref:continue
   assert all(r['token_sha256']==ref[r['window']]['token_sha256'] for r in rs)
   d,l,h=ci([r['nll']-ref[r['window']]['nll'] for r in rs])
   contrasts.append({'group':group,'trial':trial,'reference':reference,'delta_nll':d,'low':l,'high':h,'ppl_ratio':math.exp(d)})
 write(out/'summary.csv',rows);write(out/'contrasts.csv',contrasts)
 save={'rows':sum(map(len,groups.values())),'conditions':len(groups),'pointwise_bootstrap_draws':2000,'seed':20261021,'unit':'matched context windows; fixed calibration'}
 (out/'analysis_metadata.json').write_text(json.dumps(save,indent=2)+'\n');print(save)
 for r in rows:
  if r['group'].startswith('natural') and (r['trial'].startswith('norm') or r['trial'].startswith('H15_') or r['trial'] in ('clean','deleted')):
   print(r['group'],r['trial'],f"ratio {r['ppl_ratio']:.6g} agreement {r['agreement']:.4f} KL {r['kl']:.5g}")
 return rows
if __name__=='__main__':analyze(Path(sys.argv[1]))
