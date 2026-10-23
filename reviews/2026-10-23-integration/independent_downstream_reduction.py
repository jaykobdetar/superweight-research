"""Independent scalar reduction of frozen Q1 exports. No model or original helpers."""
from pathlib import Path
import math,json,sys
import numpy as np
R=Path(sys.argv[1]);C=R/'continuation'
p=json.loads((C/'downstream/frozen_protocol.json').read_text())
s=json.loads((C/'results/downstream_summary.json').read_text());it=json.loads((C/'results/downstream_interactions.json').read_text())
losses={};agreement={};n=0;errors=[];controls=[]
for folder in ['downstream','axis_controls']:
 for f in (C/folder).glob('*.npz'):
  with np.load(f,allow_pickle=False) as a:
   if 'loss' not in a:continue
   v=list(map(float,a['loss']));assert len(v)==255 and all(math.isfinite(x) for x in v)
   losses[f.stem]=v;agreement[f.stem]=list(map(float,a['argmax_agrees_with_native']));n+=1
assert n==28
mean=lambda x:math.fsum(x)/len(x)
for panel in p['panels']:
 name=panel['name'];later=panel['first_later187'];g=panel['group_name']
 masks={'after_source':list(range(1,255)),'source_query_only':[0],'before_next_newline':list(range(1,later)),'after_next_newline':list(range(later+1,255))}
 baseline=losses[name+'__native_no_hook']
 for row in [x for x in s if x['panel']==name]:
  key=name+'__'+row['treatment'];ids=masks[row['scope']];v=losses[key]
  dn=mean([v[i]-baseline[i] for i in ids]);checks={'mean_nll':mean([v[i] for i in ids]),'delta_nll':dn,'ppl_ratio':math.exp(dn),'argmax_agreement':mean([agreement[key][i] for i in ids])}
  assert len(ids)==row['query_count']
  for field,value in checks.items():errors.append(abs(value-row[field]))
 for row in [x for x in it if x['panel']==name]:
  ids=masks[row['scope']]
  def contrast(terms):return mean([math.fsum(sign*losses[name+'__'+t][i] for t,sign in terms) for i in ids])
  errors.append(abs(contrast([(g+'_drop_with1710',1),('drop1710',-1),(g+'_drop_only',-1),('native_no_hook',1)])-row['primary_by_group_interaction']))
  if 'axis_by_complement_interaction' in row:
   errors.append(abs(contrast([(g+'_drop_with1710',1),(g+'_axis_remove_with1710',-1),(g+'_complement_remove_with1710',-1),('drop1710',1)])-row['axis_by_complement_interaction']))
 for a,b in [('native_no_hook','reconstruct_native'),('drop1710',g+'_same_world_addback')]:
  dif=[abs(x-y) for x,y in zip(losses[name+'__'+a],losses[name+'__'+b])]
  assert max(dif)<.001 and mean(dif)<.0001
  controls.append({'panel':name,'pair':[a,b],'maximum_abs_nll_difference':max(dif),'mean_abs_nll_difference':mean(dif)})
assert max(errors)<1e-12
print(json.dumps({'date_utc':'2026-10-23','scope':'Independently authored scalar reaggregation of all exported downstream loss summaries and factorials. Not independent inference, hook audit, or head14 routing validation.','forward_outputs':n,'summary_rows':len(s),'interaction_rows':len(it),'numeric_fields_checked':len(errors),'maximum_difference':max(errors),'controls':controls,'all_pass':True},indent=2))
