"""Paired source-broadcast contrasts, including explicit K/V interaction."""
import json,sys
from pathlib import Path
from collections import defaultdict
from analyze import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);groups=defaultdict(dict)
for line in (out/'windows.jsonl').read_text().splitlines():
 r=json.loads(line);groups[r['group'],r['trial']][r['window']]=r
rows=[]
terms={'row_key_value_damage_interaction':[('clean_from_row_kv',1),('clean_from_row_k',-1),('clean_from_row_v',-1),('clean',1)],'column_donor_restoration_interaction':[('row_from_column_kv',1),('row_from_column_k',-1),('row_from_column_v',-1),('row',1)],'fixed_minus_current_key_zeroV':[('cleanK_zero_p_values',1),('clean_zero_p_values',-1)],'column_keys_into_row_minus_whole_column':[('row_from_column_k',1),('column',-1)]}
for name in ('all','early','middle','late'):
 terms['p_block_'+name+'_by_row_deletion']=[('row_block_p_'+name,1),('clean_block_p_'+name,-1),('row',-1),('clean',1)]
 terms['remaining_row_cost_after_p_block_'+name]=[('row_block_p_'+name,1),('clean_block_p_'+name,-1)]
for g in sorted(set(g for g,t in groups)):
 for name,ts in terms.items():
  ids=sorted(groups[g,ts[0][0]]);a,l,h=ci([sum(v*groups[g,t][i]['nll'] for t,v in ts) for i in ids]);rows.append({'group':g,'contrast':name,'terms':json.dumps(ts),'windows':len(ids),'delta_nll':a,'low':l,'high':h})
write(out/'broadcast_contrasts.csv',rows)
