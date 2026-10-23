"""Audit effective donor activations and summarize the completed prefix tests."""
from pathlib import Path
import csv,json,hashlib
import numpy as np

R=Path(__file__).resolve().parent;O=R/'m0_donors';P=json.loads((O/'protocol.json').read_text())
old=np.load(R/'h14_prefix/traces.npz')
w=np.memmap('/workspace/shared/olmo_computational_role/downstream/weights/model.layers.0.mlp.down_proj.weight.f32',mode='r',dtype='<f4',shape=(2048,8192))
rows=[];audit=[];effective={};factorials=[]
for panel in P['panels']:
    name=panel['name'];baseline={}
    for world in ('R','E'):
        if panel['split']=='discovery':
            c='newline_R' if world=='R' else 'newline_R_edge_H14'
            baseline[world]={k:old[name+'__'+c+'__'+k] for k in ('m0_activations','m0_output','l1_normalized')}
        else:
            z=np.load(O/(name+'__'+world+'__native.npz'))
            baseline[world]={k:z[k] for k in ('m0_activations','m0_output','l1_normalized')}
    for world in ('R','E'):
        donor=baseline['E' if world=='R' else 'R'];rec=baseline[world]
        world_rows={}
        for treatment,group in P['treatments'].items():
            stem=name+'__'+world+'__'+treatment
            j=json.loads((O/(stem+'.json')).read_text());path=O/(stem+'.npz')
            assert j['result_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            z=np.load(path)
            assert np.array_equal(z['m0_activations'],rec['m0_activations'])
            a=rec['m0_activations'].copy()
            if group is None:
                a=donor['m0_activations'].copy();predicted=donor['m0_output'].astype(np.float64)
            else:
                a[group]=donor['m0_activations'][group]
                predicted=rec['m0_output'].astype(np.float64)
                if group:
                    predicted+=w[:,group].astype(np.float64)@(donor['m0_activations'][group].astype(np.float64)-rec['m0_activations'][group])
            err=float(np.max(abs(predicted-z['m0_output'])))
            assert err<1e-4,(stem,err)
            effective[stem]=a
            ae=dict(panel=name,world=world,treatment=treatment,
                    recorded_activations_are_native_exact=True,
                    effective_activation_to_output_max_abs_error=err)
            if treatment=='sham_reproject':
                ae['normalized_identity_max_abs_error']=float(np.max(abs(z['l1_normalized']-rec['l1_normalized'])))
                assert ae['normalized_identity_max_abs_error']<5e-4
            audit.append(ae)
            row=dict(panel=name,split=panel['split'],world=world,treatment=treatment,
                primary_gate=j['features']['1710']['gate'],primary_up=j['features']['1710']['up'],
                primary_raw_write=j['features']['1710']['raw_write'],
                alternate_pair_axis=j['features']['3950']['axis_write']+j['features']['4786']['axis_write'],
                column_alignment=j['column_alignment'])
            rows.append(row);world_rows[treatment]=row
        for metric in ('primary_raw_write','alternate_pair_axis','column_alignment'):
            interaction=(world_rows['top4'][metric]-world_rows['single1091'][metric]
                         -world_rows['other_top3'][metric]+world_rows['sham_reproject'][metric])
            factorials.append(dict(panel=name,split=panel['split'],world=world,metric=metric,
                                   interaction=float(interaction)))

for name,data in [('donor_summary',rows),('donor_factorial_interactions',factorials)]:
    (O/(name+'.json')).write_text(json.dumps(data,indent=2))
    with (O/(name+'.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
np.savez_compressed(O/'effective_activations.npz',**effective)
receipt=dict(all_pass=True,intervention_count=len(rows),
    maximum_effective_output_error=max(x['effective_activation_to_output_max_abs_error'] for x in audit),
    maximum_sham_normalized_error=max(x.get('normalized_identity_max_abs_error',0) for x in audit),
    activation_schema='In original per-run NPZ, m0_activations are native pre-hook activations. effective_activations.npz contains the actual donor-swapped activations; full-M0-output donor uses all opposite-world activations as its equivalent donor representation, up to documented FP32 projection rounding.',
    checks=audit,
    interpretation='Conditional interventions in two fixed upstream worlds. Full-M0 output donation is a module-level comparator, not assumed recovery of the opposite full world. No loss or training claim.')
(O/'effective_activation_audit.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='checks'},indent=2))
for split in ('discovery','context_holdout'):
    print(split)
    for world in ('R','E'):
        for treatment in P['treatments']:
            selection=[x for x in rows if x['split']==split and x['world']==world and x['treatment']==treatment]
            print(world,treatment,{k:[min(x[k] for x in selection),max(x[k] for x in selection)]
                                  for k in ('primary_raw_write','alternate_pair_axis','column_alignment')})
