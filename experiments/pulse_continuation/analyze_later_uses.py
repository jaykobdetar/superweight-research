"""Source/later scope interaction on fresh matched contexts."""
import json,sys,math
from pathlib import Path
from collections import defaultdict
from analyze import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);groups=defaultdict(dict)
for line in (out/'windows.jsonl').read_text().splitlines():
 r=json.loads(line);groups[r['group']].setdefault(r['trial'],{})[r['window']]=r
rows=[]
for g,cs in groups.items():
 ids=sorted(cs['clean'])
 for component in ('row','column'):
  for label,coefs in [('all_minus_source',[(component+'_all',1),(component+'_source',-1)]),('source_later_interaction',[(component+'_all',1),(component+'_source',-1),(component+'_later',-1),('clean',1)])]:
   vals=[sum(w*cs[t][i]['nll'] for t,w in coefs) for i in ids];a,l,h=ci(vals);rows.append({'group':g,'component':component,'contrast':label,'delta_nll':a,'low':l,'high':h,'ratio':math.exp(a)})
 for scope in ('source','later','all'):
  vals=[cs['column_'+scope][i]['nll']-cs['row_'+scope][i]['nll'] for i in ids];a,l,h=ci(vals);rows.append({'group':g,'component':'column_vs_row','contrast':scope,'delta_nll':a,'low':l,'high':h,'ratio':math.exp(a)})
write(out/'scope_interactions.csv',rows)
