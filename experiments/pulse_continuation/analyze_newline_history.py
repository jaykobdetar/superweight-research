"""Output summaries and paired token-history effects on probe pulse amplitude."""
import json,sys
from pathlib import Path
from collections import defaultdict
from analyze import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);groups=defaultdict(dict)
for r in json.loads((out/'pulse_observations.json').read_text()):groups[r['group'],r['trial']][r['window']]=r
rows=[];pairs=[]
for (g,t),rs in groups.items():
 row={'group':g,'trial':t,'windows':len(rs)}
 for k in ('p_pulse','q_pulse','source_pulse'):
  a,l,h=ci([rs[i][k] for i in sorted(rs)]);row[k]=a;row[k+'_low']=l;row[k+'_high']=h
 rows.append(row)
 if t!='clean':
  ref=groups[g,'clean'];a,l,h=ci([rs[i]['q_pulse']-ref[i]['q_pulse'] for i in sorted(rs)]);pairs.append({'group':g,'trial':t,'reference':'clean','q_pulse_difference':a,'low':l,'high':h})
 elif '_p209_' not in g:
  prefix,tail=g.rsplit('_p',1);_,qid=tail.split('_q');rg=prefix+'_p209_q'+qid;ref=groups[rg,'clean'];a,l,h=ci([rs[i]['q_pulse']-ref[i]['q_pulse'] for i in sorted(rs)]);pairs.append({'group':g,'trial':'clean','reference':rg,'q_pulse_difference':a,'low':l,'high':h})
write(out/'pulse_summary.csv',rows);write(out/'pulse_contrasts.csv',pairs)
