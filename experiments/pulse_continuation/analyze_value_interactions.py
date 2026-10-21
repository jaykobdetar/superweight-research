"""Finite-intervention additivity checks; no new model work or fitted selection."""
import collections,json
from pathlib import Path
from analyze import ci,write
ROOT=Path(__file__).resolve().parent;out=ROOT/'value_layers_run';groups=collections.defaultdict(dict)
for line in (out/'windows.jsonl').read_text().splitlines():
 r=json.loads(line);groups[r['group']].setdefault(r['trial'],{})[r['window']]=r['nll']
rows=[]
for group,rs in groups.items():
 def a(t):return [rs[t][i] for i in sorted(rs[t])]
 base=a('mean_values');zero=a('zero_values');singles=[a(f'remove_L{l}') for l in range(2,16)]
 all_damage=[z-b for z,b in zip(zero,base)];sum_damage=[sum(x[i]-base[i] for x in singles) for i in range(len(base))]
 third=[a('mean_values')[i]-a('remove_early')[i]-a('remove_middle')[i]-a('remove_late')[i]+a('restore_early')[i]+a('restore_middle')[i]+a('restore_late')[i]-a('zero_values')[i] for i in range(len(base))]
 for name,values in [('all_values_removed',all_damage),('sum_of_individual_removals',sum_damage),('joint_minus_sum_individual',[x-y for x,y in zip(all_damage,sum_damage)]),('three_region_third_order_contrast',third)]:
  mean,low,high=ci(values);rows.append({'group':group,'contrast':name,'mean_nats':mean,'low':low,'high':high,'windows':len(values),'window_values':json.dumps(values)})
write(out/'interaction_contrasts.csv',rows)
print(json.dumps([r for r in rows if r['contrast']=='joint_minus_sum_individual'],indent=2))
