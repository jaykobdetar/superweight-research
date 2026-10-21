"""Matched geometry effects and numerical reconstruction checks."""
import json,sys
from pathlib import Path
from collections import defaultdict
from analyze import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);d=defaultdict(dict);rows=[]
for line in (out/'windows.jsonl').read_text().splitlines():
 a=json.loads(line);d[a['group'],a['trial']][a['window']]=a
for g in sorted({g for g,t in d}):
 terms={
 'col_angle_on_row_direction':[('angle_N_direction_R',1),('angle_R_direction_R',-1)],
 'col_direction_on_row_angle':[('angle_R_direction_N',1),('angle_R_direction_R',-1)],
 'angle_direction_interaction':[('angle_N_direction_N',1),('angle_N_direction_R',-1),('angle_R_direction_N',-1),('angle_R_direction_R',1)],
 'clean_angle_on_row_direction':[('angle_C_direction_R',1),('angle_R_direction_R',-1)],
 'clean_direction_on_row_angle':[('angle_R_direction_C',1),('angle_R_direction_R',-1)],
 'axis_minus_column':[('negative_axis_only',1),('column',-1)]}
 for a in 'CN':
  for seed in range(20261021,20261024):terms[f'random_{a}_{seed}_minus_own']=[(f'angle_{a}_random_{seed}',1),(f'angle_{a}_direction_{a}',-1)]
 for name,ts in terms.items():
  ids=sorted(d[g,ts[0][0]]);v,l,h=ci([sum(c*d[g,t][i]['nll'] for t,c in ts) for i in ids]);rows.append({'group':g,'contrast':name,'windows':len(ids),'terms':json.dumps(ts),'delta_nll':v,'low':l,'high':h})
write(out/'geometry_contrasts.csv',rows)
