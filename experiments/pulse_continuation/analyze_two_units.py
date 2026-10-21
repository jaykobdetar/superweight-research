"""Conditional second-unit effects and scope comparisons, paired by window."""
import json,sys,math
from collections import defaultdict
from pathlib import Path
from analyze import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);groups=defaultdict(dict)
for s in (out/'windows.jsonl').read_text().splitlines():
 r=json.loads(s);groups[r['group']].setdefault(r['trial'],{})[r['window']]=r
rows=[]
def effect(g,a,b,label):
 d=groups[g];vals=[d[a][i]['nll']-d[b][i]['nll'] for i in sorted(d[a])];m,l,h=ci(vals);rows.append({'group':g,'contrast':label,'delta_nll':m,'low':l,'high':h,'ppl_ratio':math.exp(m)})
for g in groups:
 for a in ('clean','row','column'):
  ref='clean' if a=='clean' else f'first_{a}_second_clean'
  for b in ('row','column'):effect(g,f'first_{a}_second_{b}',ref,'second_'+b+'_given_first_'+a)
 for b in ('row','column'):
  def trial(a,bb):return 'clean' if a==bb=='clean' else f'first_{a}_second_{bb}'
  d=groups[g];vals=[d[trial('column',b)][i]['nll']-d[trial('column','clean')][i]['nll']-d[trial('clean',b)][i]['nll']+d['clean'][i]['nll'] for i in sorted(d['clean'])];m,l,h=ci(vals);rows.append({'group':g,'contrast':'second_'+b+'_dependence_change_after_first_column','delta_nll':m,'low':l,'high':h,'ppl_ratio':math.exp(m)})
write(out/'conditional_contrasts.csv',rows)
scope=[]
for g,d in groups.items():
 if not g.endswith('_source'):continue
 other=groups[g[:-7]+'_all']
 for t in d:
  vals=[other[t][i]['nll']-d[t][i]['nll'] for i in sorted(d[t])];m,l,h=ci(vals);scope.append({'group':g[:-7],'trial':t,'all_minus_source_delta':m,'low':l,'high':h})
write(out/'scope_contrasts.csv',scope)
