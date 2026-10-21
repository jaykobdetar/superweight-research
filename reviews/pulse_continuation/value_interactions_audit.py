"""CPU replay of finite-intervention value additivity and three-region contrasts."""
import csv,itertools,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'value_layers_run'
raw=torch.load(OUT/'per_token.pt',map_location='cpu',weights_only=True);raw={k:{z['window']:z for z in v}for k,v in raw.items()}
rows=list(csv.DictReader((OUT/'interaction_contrasts.csv').open()));assert len(rows)==16
results={}
for c in ['wiki','hume','plato','python']:
 g='values_'+c
 def losses(t):
  vals=[]
  for i in range(8):
   z=raw[g+'__'+t][i];assert z['token_sha256']==raw[g+'__mean_values'][i]['token_sha256'];vals.append(float(z['loss'].double().mean()))
  return vals
 mean=losses('mean_values');zero=losses('zero_values');single=[losses(f'remove_L{l}')for l in range(2,16)]
 joint=[z-b for z,b in zip(zero,mean)];summed=[sum(s[i]-mean[i]for s in single)for i in range(8)]
 corner={(0,0,0):'zero_values',(1,0,0):'restore_early',(0,1,0):'restore_middle',(0,0,1):'restore_late',(1,1,0):'remove_late',(1,0,1):'remove_middle',(0,1,1):'remove_early',(1,1,1):'mean_values'}
 third=[sum((-1)**(3-sum(bits))*losses(trial)[i]for bits,trial in corner.items())for i in range(8)]
 vals={'all_values_removed':joint,'sum_of_individual_removals':summed,'joint_minus_sum_individual':[a-b for a,b in zip(joint,summed)],'three_region_third_order_contrast':third}
 for name,a in vals.items():
  row=next(r for r in rows if r['group']==g and r['contrast']==name);saved=json.loads(row['window_values']);assert int(row['windows'])==len(saved)==len(a)==8
  assert max(abs(x-y)for x,y in zip(a,saved))<1e-12
  ci=interval(a)
  for field,k in [('mean_nats','mean'),('low','low'),('high','high')]:assert abs(float(row[field])-ci[k])<1e-12
  results[c+'__'+name]={**ci,'positive_windows':sum(v>0 for v in a),'window_values':a}
out={'rows_replayed':16,'raw_token_means_and_hash_alignment_verified':True,'all_bootstrap_intervals_replayed':True,'third_order_coding':'positive indicators mean region values present; reversing to removal indicators reverses the third-order sign','contrasts':results,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='contrasts'},indent=2))
