#!/usr/bin/env python3
"""Deterministic paired window analysis; no model load or GPU access."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

def write_csv(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(x):
 x=np.asarray(x,float);rng=np.random.default_rng(20261021)
 boot=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
 return float(x.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))
def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
 groups=defaultdict(list)
 for line in (out/'windows.jsonl').read_text().splitlines():
  r=json.loads(line);groups[(r['group'],r['trial'])].append(r)
 result=[];contrasts=[]
 for (group,trial),rs in groups.items():
  rs=sorted(rs,key=lambda r:r['window']);assert len({r['window'] for r in rs})==len(rs)
  diff=[r['nll']-r['baseline_nll'] for r in rs];m,lo,hi=interval(diff)
  row={'group':group,'trial':trial,'windows':len(rs),'mean_nll':np.mean([r['nll'] for r in rs]),
   'ppl':math.exp(np.mean([r['nll'] for r in rs])),'delta_nll':m,'delta_nll_low':lo,'delta_nll_high':hi,
   'ppl_ratio':math.exp(m),'ratio_low':math.exp(lo),'ratio_high':math.exp(hi),'agreement':np.mean([r['agreement'] for r in rs])}
  for field in ('first127_delta_nll','later9_delta_nll','suffix512_delta_nll'):
   vals=[r[field] for r in rs if field in r];row[field]=np.mean(vals) if vals else ''
  result.append(row)
  refgroup='causal' if group in ('directions','replacements') else group
  for ref in ('clean','deleted','pulse'):
   reference={r['window']:r for r in groups.get((refgroup,ref),[])}
   if not reference:continue
   vals=[r['nll']-reference[r['window']]['nll'] for r in rs]
   d,l,h=interval(vals)
   contrasts.append({'group':group,'trial':trial,'reference':ref,'windows':len(rs),'delta_nll':d,'low':l,'high':h,'ppl_ratio':math.exp(d)})
 write_csv(out/'summary.csv',result)
 if contrasts:write_csv(out/'contrasts.csv',contrasts)
 (out/'analysis_metadata.json').write_text(json.dumps({'rows':sum(map(len,groups.values())),'conditions':len(groups),
  'bootstrap_resamples':2000,'seed':20261021,'unit':'matched context window','interval':'pointwise percentile 95%; fixed calibration and fitted subspaces; not multiplicity corrected'},indent=2)+'\n')
 print(json.dumps({'rows':sum(map(len,groups.values())),'conditions':len(groups)}))
 for row in result:
  if (row['group']=='causal' and (row['trial'].endswith('context_all_damage') or row['trial'].endswith('context_all_rescue') or row['trial'] in ('clean','deleted','pulse'))) or row['group']=='replacements':
   print(row['group'],row['trial'],round(row['ppl_ratio'],6),round(row['agreement'],5))
if __name__=='__main__':main()
