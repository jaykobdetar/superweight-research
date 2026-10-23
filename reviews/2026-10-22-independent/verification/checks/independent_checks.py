"""Independent review calculations on the immutable public snapshot; no model imports."""
import collections,csv,hashlib,itertools,json,math,pathlib,sys
import numpy as np
root=pathlib.Path(__file__).resolve().parents[1]/'snapshot'; out=pathlib.Path(__file__).resolve().parent
analysis=json.loads((root/'results/calibration_world_comparison/calibration_world_evaluation_analysis.json').read_text())
look={}; n=0; maximum_metric_error=0.; observed=[]
for p in sorted((root/'results/calibration_world_comparison/inputs').glob('*.npz')):
 ledger=json.loads(p.with_suffix('.json').read_text()); m=ledger['metadata']; c,i,w=m['corpus'],m['identity'],m['window']
 with np.load(p,allow_pickle=False) as z:
  for key in z.files:
   if not key.startswith('metrics__') or not key.endswith('__loss'):continue
   arm=key.split('__')[1]
   for view,sl in [('full',slice(1,511)),('early',slice(1,127)),('later',slice(127,511))]:
    look[(c,i,w,view,'000__'+arm)]={dest:float(z['metrics__'+arm+'__'+source][sl].astype(np.float64).mean()) for dest,source in [('nll','loss'),('kl_baseline','kl_boundary'),('agreement_baseline','agreement_boundary')]}
for row in analysis['window_metrics']+analysis['anchors']:
 m=look[tuple(row[k] for k in ['corpus','identity','window','view','mode'])]
 for field,value in m.items():
  e=abs(value-row[field]);maximum_metric_error=max(e,maximum_metric_error);assert e==0;(n:=n+1)
rows=json.loads((root/'results/calibration_world_comparison_report/tables/all_64_primary.json').read_text())['rows']
rng=np.random.Generator(np.random.PCG64(20261021));indices=rng.integers(0,4,size=(2000,4),dtype=np.int64)
all_indices=np.array(list(itertools.product(range(4),repeat=4)))
all_signs=np.array(list(itertools.product([-1,1],repeat=4)))
changes=[];looflips=[];all_rows=[];worst=0
for row in rows:
 xs=np.array([math.fsum(k*look[row['corpus'],row['identity'],w,'later',arm]['nll'] for arm,k in row['terms'].items()) for w in range(4)],dtype=np.float64)
 assert xs.tolist()==row['passage_values_nats']
 ci=np.quantile(xs[indices].mean(1),[.025,.975]); mean=float(xs.mean()); assert mean==row['mean_nats']; assert ci.tolist()==[row['ci_low_nats'],row['ci_high_nats']]
 exactci=np.quantile(xs[all_indices].mean(1),[.025,.975]); loo=np.array([(xs.sum()-x)/3 for x in xs]); kind=lambda a:1 if a[0]>0 else -1 if a[1]<0 else 0
 name=[row['contrast'],row['corpus'],row['identity']]
 if kind(ci)!=kind(exactci):changes.append({'row':name,'original_ci':ci.tolist(),'exact_bootstrap_ci':exactci.tolist()})
 if ((loo*mean)<0).any():looflips.append(name)
 a={'row':name,'mean':mean,'passages':xs.tolist(),'ci2000':ci.tolist(),'ci_all256':exactci.tolist(),'loo_means':loo.tolist(),'passage_signs_match_mean':int(np.count_nonzero(xs*mean>0)), 'signflip_reference_p':float(np.mean(np.abs((all_signs*xs).mean(1))>=abs(mean)-1e-18))}
 all_rows.append(a)
 worst=max(worst,float(np.abs(ci-exactci).max()))
summary={'scope':'Independent raw exported scalar reduction and hand-written T23 contrast/bootstrap reconstruction; sign-flip reference requires symmetry/exchangeability and is not a randomized-design p-value. No model, hidden states, input/prediction IDs or private evidence were accessed.', 'numpy':np.__version__,'scalar_values_recomputed':n,'max_metric_error':maximum_metric_error,'all64_primary_point_and_frozen_CI_exact':True,'bootstrap_index_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),'nominal_zero_containing_intervals':sum(r['ci_low_nats']<=0<=r['ci_high_nats'] for r in rows),'max_absolute_endpoint_change':max(abs(r['mean_nats']) for r in rows if r['contrast']=='joint_CC_minus_PP'),'exhaustive256_zero_containing_intervals':sum(x['ci_all256'][0]<=0<=x['ci_all256'][1] for x in all_rows),'max_CI_endpoint_difference_exhaustive_vs2000':worst,'zero_inclusion_changes':changes,'leave_one_out_sign_change_rows':looflips,'all_rows':all_rows}
# Independently reaggregate the main OLMo row/column comparison and geometry examples.
olmo=[]
for run in ['column_factorial_run','source_geometry_run']:
 records=[json.loads(l) for l in (root/'results/pulse_continuation'/run/'windows.jsonl').read_text().splitlines()]
 groups=collections.defaultdict(list)
 for r in records:groups[(r['group'],r['trial'])].append(r)
 for (g,t),rs in groups.items():
  if ('natural' in g and g.startswith('first_') and t in ['delete_row','delete_column']) or (run=='source_geometry_run' and g.startswith('first_') and t in ['angle_C_direction_R','angle_N_direction_R','angle_R_direction_N','negative_axis_only']):
   olmo.append({'run':run,'group':g,'trial':t,'n':len(rs),'ppl_ratio':math.exp(sum(r['nll']-r['baseline_nll'] for r in rs)/len(rs)),'individual_ratios':[math.exp(r['nll']-r['baseline_nll']) for r in rs]})
summary['olmo_selected_reaggregations']=olmo
(out/'independent_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['all_rows','olmo_selected_reaggregations']},indent=2));print(json.dumps(olmo,indent=2))
