"""Independent Report12 table and selected narrative quantitative checks."""
import json,re,hashlib,resource
from pathlib import Path
import torch,numpy as np
P=Path(__file__).resolve().parents[1];run=P/'gpt2_gate_localization_run';s=json.loads((run/'summary.json').read_text());ps=json.loads((run/'paired_summary.json').read_text());ns=json.loads((run/'nonadditivity_summary.json').read_text());text=(P/'REPORT12.md').read_text()
rs={(r['corpus'],r['identity'],r['trial']):r for r in s};pair={(r['corpus'],r['identity'],r['trial']):r for r in ps};count=0;family=identity=None
for line in text.splitlines():
 if line=='### PC1 removal':family='remove_pc1'
 elif line=='### Orthogonal control':family='orthogonal_control'
 elif line=='### Mean + log(position)':family='position'
 elif line=='Natural starts:':identity='natural'
 elif line=='BOS replacement:':identity='bos'
 elif re.match(r'\| (L[0-9]+|all) \|',line):
  cells=[v.strip()for v in line.split('|')[1:-1]]
  for c,value in zip(('wiki_validation','dubois','hobbes','optimizers'),cells[1:]):assert value==f"{rs[c,identity,family+'_'+cells[0]]['p_ppl_ratio']:.4f}";count+=1
assert count==216
assert all(rs[c,i,'remove_pc1_'+l]['p_ppl_ratio_ci95'][1]<1 for c in('dubois','hobbes')for i in('natural','bos')for l in('L9','L10'))
for c in('wiki_validation','dubois','hobbes'):
 for i in('natural','bos'):
  ci=pair[c,i,'remove_pc1_L10_minus_orthogonal_control']['ci95_delta_nll'];assert ci[0]>0 if c=='wiki_validation'else ci[1]<0
for i in('natural','bos'):assert pair['dubois',i,'remove_pc1_L5_minus_orthogonal_control']['ci95_delta_nll'][0]>0
assert sum(r['mean_delta_nll']>0 for r in ns)==24 and sum(r['ci95_delta_nll'][0]>0 for r in ns)==22
norms=json.loads((run/'norm_summary.json').read_text());means={}
for l in(4,10):
 for f in('remove_pc1','orthogonal_control'):
  zz=[r for r in norms if(r['corpus'],r['identity'],r['trial'],r['layer'],r['view'])==('wiki_validation','natural',f+'_L'+str(l),l,'full')];assert len(zz)==4
  means[f+'_L'+str(l)]={k:float(np.mean([z[k]['mean']for z in zz]))for k in('logit_delta_norm','context_delta_norm')}
d=torch.load(run/'directions.pt',weights_only=True,map_location='cpu');shares={l:float(d[l]['u'].square().max())for l in d};assert int(d[4]['u'].square().argmax())==11
roundclaims={'L4_loading_pct':100*shares[4],'later_loading_pct_min':100*min(v for l,v in shares.items()if l!=4),'later_loading_pct_max':100*max(v for l,v in shares.items()if l!=4)}
ident=json.loads((run/'start_identity_summary.json').read_text());assert len(ident['natural_seen_in_calibration'])==15 and not any(ident['natural_seen_in_calibration'])and not ident['bos_seen_in_calibration']
primary=[r for r in s if r['trial']=='P'];joint=[r for r in s if r['trial']=='remove_pc1_all'];position=[r for r in s if r['trial']=='position_all']
ranges={'P_C_agreement':[100*min(r['mean_agreement']for r in primary),100*max(r['mean_agreement']for r in primary)],'joint_C_agreement':[100*min(r['mean_agreement']for r in joint),100*max(r['mean_agreement']for r in joint)],'position_P_PPL':[min(r['p_ppl_ratio']for r in position),max(r['p_ppl_ratio']for r in position)]}
out={'status':'passed','table_cells':count,'L9_L10_book_P_intervals_below_zero':8,'L10_directional_contrast_intervals_signed':6,'L5_DuBois_control_better_intervals':2,'nonadditivity_positive_means':24,'nonadditivity_positive_CIs':22,'L4_L10_norm_means':means,'geometry_shares_percent':roundclaims,'narrative_ranges':ranges,'all15_natural_and_BOS_start_IDs_absent_calibration':True,'REPORT12_sha256':hashlib.sha256((P/'REPORT12.md').read_bytes()).hexdigest(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pretrained_loaded':False,'gpu_used':False}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
