"""Reanalyze frozen public H14 and H12 interventions; no new model forward."""
from pathlib import Path
import json,csv,itertools,numpy as np
R=Path(__file__).resolve().parent;D=R/'repo/results/pulse_continuation';O=R/'results'
def ranges(rows,fields):return {k:{'min':float(np.min([x[k] for x in rows])),'median':float(np.median([x[k] for x in rows])),'max':float(np.max([x[k] for x in rows]))} for k in fields}
a=json.loads((D/'newline_components_run/factor_observations.json').read_text());by={(x['group'],x['window'],x['trial']):x for x in a};pairs=sorted(set((x['group'],x['window']) for x in a))
summary={t:{'n':sum(x['trial']==t for x in a),**ranges([x for x in a if x['trial']==t],['q_pulse','q_gate','q_up'])} for t in sorted(set(x['trial'] for x in a))}
inter=[];factor=[]
for group,win in pairs:
 get=lambda bits:by[group,win,'mix_'+bits]['q_pulse'];r={'group':group,'window':win,'baseline':get('CCC')}
 for mask in range(1,8):
  effect=0.
  for sub in range(8):
   if sub&~mask:continue
   b=''.join('E' if sub&(1<<j) else 'C' for j in range(3));effect+=(-1)**(mask.bit_count()-sub.bit_count())*get(b)
  r['interaction_'+''.join(['A0','M0','A1'][j] for j in range(3) if mask&(1<<j))]=effect
 r['closure_error']=get('EEE')-(r['baseline']+sum(v for k,v in r.items() if k.startswith('interaction')));assert abs(r['closure_error'])<1e-12;inter.append(r)
 # Neuron-level factor replay from recorded native factors, not a full forward.
 for gd,ud in itertools.product(['clean','edge'],repeat=2):
  gg=by[group,win,gd]['q_gate'];uu=by[group,win,ud]['q_up'];c=0.63232421875*gg*np.exp(-np.logaddexp(0,-gg))*uu
  factor.append({'group':group,'window':win,'gate_donor':gd,'up_donor':ud,'replayed_scalar_real_arithmetic':float(c)})
checks=[]
for x in a:
 g=np.float32(x['q_gate']);u=np.float16(x['q_up']);s=np.float16(g*np.exp(-np.logaddexp(np.float32(0),-g)));p=float(np.float32(np.float16(s*u)))*0.63232421875
 checks.append(abs(p-x['q_pulse']))
b=json.loads((D/'upstream_mlp_run/factor_observations.json').read_text());h12={t:{'n':sum(x['trial']==t for x in b),**ranges([x for x in b if x['trial']==t],['gate','up','pulse'])} for t in sorted(set(x['trial'] for x in b))};bb={(x['group'],x['window'],x['trial']):x for x in b};bp=sorted(set((x['group'],x['window']) for x in b))
assert all(bb[g,w,'H12_valuep']['pulse']==bb[g,w,'H12_selfmask']['pulse'] for g,w in bp)
assert all(bb[g,w,'H12_value0']['pulse']==bb[g,w,'ordinary_H12']['pulse'] for g,w in bp)
assert all(bb[g,w,'ordinary_swap_gate_up']['pulse']==bb[g,w,'isolated']['pulse'] for g,w in bp)
audit={'h14_cases':len(pairs),'h14_rows':len(a),'h14_fp16_factor_replay_max_abs_pulse_error':max(checks),'h14_interaction_closure_max_abs':max(abs(x['closure_error']) for x in inter),'h12_cases':len(bp),'h12_rows':len(b),'h12_valuep_selfmask_all_pulses_exact':True,'h12_value0_isolated_donor_all_pulses_exact':True,'h12_both_factors_reconstruct_isolated_pulse':True}
for name,obj in [('historical_h14_component_summary',summary),('historical_h12_summary',h12),('historical_path_checks',audit),('historical_h14_interactions',inter),('historical_h14_factor_replay',factor)]:
 (O/(name+'.json')).write_text(json.dumps(obj,indent=2))
 if isinstance(obj,list):
  with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(obj[0]));w.writeheader();w.writerows(obj)
print(json.dumps(audit,indent=2));print('M0A1',ranges(inter,['interaction_M0A1']))
