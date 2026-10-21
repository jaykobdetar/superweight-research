"""Paired pulse, component and conditional-compensation summaries."""
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
def contrast(g,name,terms):
 ids=sorted(records[g,terms[0][0]]);row={'group':g,'contrast':name,'terms':json.dumps(terms),'windows':len(ids)}
 for k in ('nll','q_pulse'):
  table=records if k=='nll' else pulses;a,l,h=ci([sum(coef*table[g,t][i][k] for t,coef in terms) for i in ids]);row[k+'_difference']=a;row[k+'_low']=l;row[k+'_high']=h
 contrasts.append(row)
for g in sorted(set(g for g,t in records)):
 if not g.startswith('heads_') or g.endswith('_Q535'):continue
 for k in ('oracle','k','v','kv','q'):contrast(g,'selected_minus_control_'+k,[('selected_'+k,1),('control_'+k,-1)])
 for t in ('edge_selected','edge_control','edge_all','sham_selected'):
  contrast(g,t+'_minus_clean',[(t,1),('clean',-1)])
 if g.endswith('_R'):
  contrast(g,'inducing_q_with_p_deleted',[('edge_delete_p',1),('delete_p',-1)])
  contrast(g,'induced_q_row_cost_after_p_deleted',[('edge_delete_both',1),('edge_delete_p',-1)])
  contrast(g,'q_row_cost_after_p_deleted_without_induction',[('delete_both',1),('delete_p',-1)])
  contrast(g,'edge_by_q_deletion_after_p_deleted',[('edge_delete_both',1),('edge_delete_p',-1),('delete_both',-1),('delete_p',1)])
  contrast(g,'edge_by_p_deletion',[('edge_delete_p',1),('edge_selected',-1),('delete_p',-1),('clean',1)])
write(out/'pulse_summary.csv',rows);write(out/'pathway_contrasts.csv',contrasts)
