"""Unedited-context pulse and matched output contrasts."""
import json,sys
from pathlib import Path
from collections import defaultdict
from analyze import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);pulses=defaultdict(dict);records=defaultdict(dict)
for r in json.loads((out/'pulse_observations.json').read_text()):pulses[r['group'],r['trial']][r['window']]=r
for line in (out/'windows.jsonl').read_text().splitlines():
 r=json.loads(line);records[r['group'],r['trial']][r['window']]=r
rows=[];contrasts=[]
for (g,t),rs in pulses.items():
 row={'group':g,'trial':t,'windows':len(rs)}
 for k in ('p_pulse','q_pulse','source_pulse'):
  a,l,h=ci([rs[i][k] for i in sorted(rs)]);row[k]=a;row[k+'_low']=l;row[k+'_high']=h
 rows.append(row)
terms={'H14_edge_minus_control':[('edge',1),('control',-1)],'H14_edge_minus_sham':[('edge',1),('sham',-1)],'H12_induction':[('clean_H12_donor',1),('clean',-1)],'H12_reversal':[('edge_H12_donor',1),('edge',-1)],'inducing_q_with_p_deleted':[('edge_row_p',1),('clean_row_p',-1)],'induced_q_row_cost_after_p_deleted':[('edge_row_both',1),('edge_row_p',-1)],'edge_by_q_deletion_after_p_deleted':[('edge_row_both',1),('edge_row_p',-1),('clean_row_both',-1),('clean_row_p',1)],'row_q_cost_in_edge_world':[('edge_row_q',1),('edge',-1)],'column_q_cost_in_edge_world':[('edge_column_q',1),('edge',-1)]}
for g in sorted(set(g for g,t in records)):
 for name,ts in terms.items():
  ids=sorted(records[g,ts[0][0]]);row={'group':g,'contrast':name,'terms':json.dumps(ts),'windows':len(ids)}
  for k in ('nll','q_pulse'):
   table=records if k=='nll' else pulses;a,l,h=ci([sum(v*table[g,t][i][k] for t,v in ts) for i in ids]);row[k+'_difference']=a;row[k+'_low']=l;row[k+'_high']=h
  contrasts.append(row)
write(out/'pulse_summary.csv',rows);write(out/'natural_contrasts.csv',contrasts)
