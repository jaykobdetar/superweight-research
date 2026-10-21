"""Prompt-level summaries of a fixed64-token greedy generation panel."""
import collections,json,sys
from pathlib import Path
from analyze import ci,write
out=Path(sys.argv[1]);meta=json.loads((out/'manifest.json').read_text());assert 'completed_utc' in meta
groups=collections.defaultdict(list)
for line in (out/'results.jsonl').read_text().splitlines():
 r=json.loads(line);groups[r['corpus'],r['variant']].append(r);groups['pooled',r['variant']].append(r)
rows=[]
for (c,v),rs in groups.items():
 row={'corpus':c,'variant':v,'prompts':len(rs),'identical_sequences':sum(r['identical_sequence'] for r in rs)}
 for k in ('free_agreement','matching_prefix_length','matched_history_agreement','matched_history_kl','reference_delta_nll'):
  val,low,high=ci([r[k] for r in rs]);row[k]=val;row[k+'_low']=low;row[k+'_high']=high
 rows.append(row)
write(out/'summary.csv',rows);print(json.dumps({'rows':len(rows),'pointwise_bootstrap_draws':2000,'unit':'prompts; fixed frozen templates; pooled mixes four fixed corpora'}))
