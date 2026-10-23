"""Reaggregate the exported development screen; no model execution."""
import json,pathlib,collections
root=pathlib.Path(__file__).resolve().parents[1]/'snapshot'
s=json.loads((root/'results/pulse_continuation/newline_heads_run/selection.json').read_text());m=collections.defaultdict(list)
for row in s['screen']:m[row['head']].append(row['fraction'])
r=sorted([{'head':h,'mean_fraction':sum(xs)/len(xs)} for h,xs in m.items()],key=lambda a:-a['mean_fraction'])
assert r==s['ranking'] and r[0]['head']==s['selected']==14
print(json.dumps({'ranking_exact':True,'selected':r[0],'development_windows_per_head':{h:len(xs) for h,xs in m.items()}},indent=2))
