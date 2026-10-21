"""Paired 2x2 component interaction and whole-column versus row-removal contrasts."""
import json,sys,math
from pathlib import Path
from collections import defaultdict
from analyze import ci,write
out=Path(sys.argv[1]);groups=defaultdict(dict)
for line in (out/'windows.jsonl').read_text().splitlines():
 r=json.loads(line);groups[r['group']].setdefault(r['trial'],{})[r['window']]=r
rows=[]
for g,conds in groups.items():
 ids=sorted(conds['clean']);assert all(sorted(v)==ids for v in conds.values())
 for name,coef in [('joint_minus_individuals',{'delete_column':1,'delete_row':-1,'delete_complement':-1,'clean':1}),('column_vs_row',{'delete_column':1,'delete_row':-1}),('complement_vs_clean',{'delete_complement':1,'clean':-1})]:
  vals=[sum(w*conds[t][i]['nll'] for t,w in coef.items()) for i in ids];a,l,h=ci(vals);rows.append({'group':g,'contrast':name,'windows':len(ids),'delta_nll':a,'low':l,'high':h,'ratio':math.exp(a)})
write(out/'interaction_contrasts.csv',rows)
