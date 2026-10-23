import runpy,io,contextlib,json,numpy as np
from pathlib import Path
with contextlib.redirect_stdout(io.StringIO()):a=runpy.run_path('/workspace/shared/olmo_signal_audit/audit.py')
R=a['R'];D=a['D'];N=a['N'];J=a['J'];K=a['K'];dw=a['dw'];act=a['act'];carry=a['carry'];labels=a['labels'];axis=a['axis'];mlp=a['mlp'];g=a['g'];u=a['u'];other=a['other'];column=a['column'];clean=a['clean'];row=a['row'];primary=a['primary'];geom=a['geom']
# Independent masked coefficient dot product, rather than subtraction from a rounded full output.
def masked(i,drop=(),keep=None):
 co=act[i].copy()
 if keep is not None:
  mask=np.ones(N,dtype=bool);mask[keep]=False;co[mask]=0
 co[list(drop)]=0
 v=np.empty(D)
 for lo in range(0,D,128):v[lo:lo+128]=dw[lo:lo+128].astype(np.float64)@co
 return v-v.mean()+carry[i]
checks={};allerr={}
for r in json.loads((R/'results/alternate_writer_interventions.json').read_text()):
 i=labels.index([r['panel'],r['condition']]);name=r['intervention'];spec={'column_deleted':((J,),None),'column_drop7579':((J,7579),None),'only7579_plus_carry':((),[7579]),'column_drop_source_set':((J,7579,7845,6947,2637),None),'source_set_only_plus_carry':((),[7579,7845,6947,2637]),'column_drop_query_pair':((J,3950,4786),None),'query_pair_only_plus_carry':((),[3950,4786]),'column_drop_newline_positive_trio':((J,2476,2637,7845),None),'carry_only':((),[])}[name];v=masked(i,*spec);m=geom(v);m['orthogonal_norm']=float(np.linalg.norm(v-(v@axis)*axis))
 for key,val in m.items():allerr[key]=max(allerr.get(key,0),abs(val-r[key]))
checks['all_189_masked_neuron_interventions_max_abs_deltas']=allerr
allerr={}
for r in json.loads((R/'results/source_geometry_interventions.json').read_text()):
 i=labels.index([r['panel'],r['condition']]);c=carry[i];o=other[i];p=primary[i]
 v={'clean':clean[i],'row_deleted':row[i],'column_deleted':column[i],'column_no_carry_axis':column[i]-(c@axis)*axis,'column_no_other_mlp_axis':column[i]-(o@axis)*axis,'column_no_both_axis':column[i]-((c+o)@axis)*axis,'primary_only':p}[r['intervention']]
 m=geom(v);m['orthogonal_norm']=float(np.linalg.norm(v-(v@axis)*axis));m['normalized_coordinate1764']=float(v[K]/np.sqrt(np.mean(v*v)+1e-5));m['axis_projection']=m.pop('axis');m['centered_norm']=m.pop('norm')
 for key,val in m.items():allerr[key]=max(allerr.get(key,0),abs(val-r[key]))
checks['all_147_geometry_conditions_max_abs_deltas']=allerr
errs=[]
for r in json.loads((R/'results/h14_factorial.json').read_text()):
 ir=labels.index([r['panel'],'newline_R']);ie=labels.index([r['panel'],'newline_R_edge_H14']);gg=g[ir if r['gate_donor']=='R' else ie,J];uu=u[ir if r['up_donor']=='R' else ie,J];contribution=gg/(1+np.exp(-gg))*uu*dw[K,J];errs.append(abs(contribution-r['contribution']))
checks['all_16_scalar_factorial_max_abs_delta']=max(errs)
allerr={}
for r in json.loads((R/'results/h14_nonprimary_component_swaps.json').read_text()):
 ir=labels.index([r['panel'],'newline_R']);ie=labels.index([r['panel'],'newline_R_edge_H14']);ci=ir if r['carry_donor']=='R' else ie;mi=ir if r['other_mlp_donor']=='R' else ie;v=carry[ci]+other[mi];m=geom(v);m['orthogonal_norm']=float(np.linalg.norm(v-(v@axis)*axis))
 for key,val in m.items():allerr[key]=max(allerr.get(key,0),abs(val-r[key]))
checks['all_16_component_swaps_max_abs_deltas']=allerr
Path('/workspace/shared/olmo_signal_audit/interventions.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
