"""Matched ordinary/isolated output and scalar-factor summaries."""
import json,sys
from pathlib import Path
from collections import defaultdict
from analyze_upstream import analyze,ci,write
out=Path(sys.argv[1]);analyze(out);groups=defaultdict(list)
for r in json.loads((out/'factor_observations.json').read_text()):groups[r['group'],r['trial']].append(r)
rows=[]
for (g,t),rs in groups.items():
 row={'group':g,'trial':t,'windows':len(rs)}
 for key in ('gate','up','intermediate','pulse'):
  a,l,h=ci([r[key] for r in rs]);row[key]=a;row[key+'_low']=l;row[key+'_high']=h
 rows.append(row)
write(out/'factor_summary.csv',rows)
