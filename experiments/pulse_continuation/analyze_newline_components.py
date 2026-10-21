"""Paired residual-component factorial contrasts; no refitting or selection."""
import itertools,json,sys
from pathlib import Path
from collections import defaultdict
from analyze import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);f=defaultdict(dict);r=defaultdict(dict)
for a in json.loads((out/'factor_observations.json').read_text()):f[a['group'],a['trial']][a['window']]=a
for line in (out/'windows.jsonl').read_text().splitlines():
 a=json.loads(line);r[a['group'],a['trial']][a['window']]=a
rows=[];contrasts=[]
for (g,t),d in f.items():
 a={'group':g,'trial':t,'windows':len(d)}
 for k in ('q_pulse','q_gate','q_up'):
  v,l,h=ci([d[i][k] for i in sorted(d)]);a.update({k:v,k+'_low':l,k+'_high':h})
 rows.append(a)
def contrast(g,name,terms):
 ids=sorted(f[g,terms[0][0]]);a={'group':g,'contrast':name,'terms':json.dumps(terms),'windows':len(ids)}
 for k in ('nll','q_pulse','q_gate','q_up'):
  d=r if k=='nll' else f;v,l,h=ci([sum(c*d[g,t][i][k] for t,c in terms) for i in ids]);a.update({k:v,k+'_low':l,k+'_high':h})
 contrasts.append(a)
for g in sorted({g for g,t in f}):
 for n in range(1,4):
  for inds in itertools.combinations(range(3),n):
   terms=[]
   for bits in itertools.product((0,1),repeat=n):
    code=['C']*3
    for i,b in zip(inds,bits):code[i]='E' if b else 'C'
    terms.append(('mix_'+''.join(code),(-1)**(n-sum(bits))))
   contrast(g,'factorial_'+'_'.join(('A0','M0','A1')[i] for i in inds),terms)
 for base,donor,ref in [('C','E','clean'),('E','C','edge')]:
  for name in ('A0','M0','A1'):contrast(g,base+'_from_'+donor+'_'+name+'_minus_base',[(base+'_from_'+donor+'_'+name,1),(ref,-1)])
 contrast(g,'edge_minus_clean',[('edge',1),('clean',-1)])
write(out/'factor_summary.csv',rows);write(out/'component_contrasts.csv',contrasts)
