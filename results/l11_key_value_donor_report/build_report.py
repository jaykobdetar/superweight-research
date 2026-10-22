"""Report18 candidate from exact frozen JSON/text only; no NPZ/model access.

Reproduce into a FRESH directory:
  python -B pulse_transfer/l11_key_value_donor_report_candidate_v4/build_report.py --root pulse_transfer --out pulse_transfer/report18_rebuild_v1
No publication or Library action is performed. Raw payloads are never opened.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,csv,hashlib,json,math,re,resource,shutil,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'review'/'report18_matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

PINS={
'gpt2_l11_key_value_donor_run_r3/l11_key_value_donor_analysis.json':'ef66f96d212781694ec7ea423710ecb9f3fa3d1344a9f72877e49bfc3e3b0dfa',
'gpt2_l11_key_value_donor_run_r3/manifest.json':'602d85764807d4678e6b61e5aa2e42244b29680a5b457c6e2bf230f7adf1d288',
'gpt2_l11_key_value_donor_run_r3/numeric_inventory.json':'65f8f6c2636b189e899348d053cf3f7f2a0e931482741a83b38d2236710c5b8f',
'review/t22_numerical_audit_v8.json':'9434fcde0b818b5db80c898bdc2a1ae75321b1bd398b3332fe6fc9a054cf8b1b',
'review/t22_projection_audit.json':'0866028b140935b9095a07e01cb738b5301c20e10d22d59dc6f9972b42ae6e41',
'review/t22_readout_audit.json':'5335d5375f7036aaa853011137875d6abe17b174722f4519aa7bde5320d0b222',
'review/t22_primary_literature_recheck_v1.json':'3c7f2e9c58f173f8ce2a13707c4f4f08c8981453d6e8d2bd9e24f66c865efdb6',
'GPT2_L11_KEY_VALUE_DONOR_PROPOSAL_v2.md':'6238ea83f978705b0069e3b476f3e867ba95f08fbae2864d6e92827b88e2f159',
'T22_HYPOTHESIS_LEDGER.md':'d332227db88885d2ac78766f7b3fb745f69fbeb5790cef667005fa69a5a72902',
'review/t22_first_input_exact_replay_gate_v8_actual.json':'e4dc025c3450cf815acbb430b70d1643f492d886c0512b0ab4450c8db1a1ae3d',
'review/t22_first_input_exact_replay_gate_v9_actual.json':'350203b78708eace3b2c322bcde2ff86ff42b36ca7adcb1d1b42904e12d6bd1d',
'review/t22_first_input_replay_actual_independent_acceptance_v9.json':'5f909f9a31aab84f0ea366e2c1e5487921a6bf1754722b053bb4bfd0d3d9362f',
'review/t22_N_actual_acceptance_v1_bookkeeping_failure.json':'2782c6dc4f24f9de8d2ec01425192ef917fb30badd871606de084a0cf0822d70',
'review/t22_numerical_audit_actual_independent_acceptance_v8.json':'34ce10f216a741e12670b6384b3520e5883acab88880d1f5d77922b47d381ff5',
't22_context_decomposition_run_v2/report.json':'59acb8a6b979c80272ee262a516ee01fd03c0af96c6a024168fba7004d70cf9e',
'review/t22_scientific_claims_independent_review_v1.json':'831cb56bd990493f49cb74571fa55e696dfc6a8e5b542b400b9f6ec635d64ce8',
'review/t22_context_saved_array_independent_audit_v1.json':'26f81e8a7c6d9dfa9d94e192ba76029d2c0c5dc394c8c6372490c8e7d3da22f9',
'review/t22_context_descriptive_scientific_acceptance_v1.json':'36d5fc0dfb8f5df10b069d17479702269b4985f0d203c3afb14f19e363029615'}
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
LABELS={'wiki_validation':'Wiki','utilitarianism':'Utilitarianism','ethics':'Ethics','optimizers_new':'Code'}
STRATA=tuple((c,i) for c in CORPORA for i in ('natural','bos'))
COLORS={'K':'#2878A5','V':'#D87825','KV':'#353535'}
ANALYSIS='gpt2_l11_key_value_donor_run_r3/l11_key_value_donor_analysis.json'
MANIFEST='gpt2_l11_key_value_donor_run_r3/manifest.json'
INV='gpt2_l11_key_value_donor_run_r3/numeric_inventory.json'
NFILE='review/t22_numerical_audit_v8.json';PFILE='review/t22_projection_audit.json';RFILE='review/t22_readout_audit.json'


def need(x,msg):
 if not x:raise AssertionError(msg)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()


def json_write(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def write(path,text):
 with Path(path).open('x') as f:f.write(text)


def csv_write(path,rows,fields):
 with Path(path).open('x',newline='') as f:
  writer=csv.DictWriter(f,fields);writer.writeheader()
  for row in rows:
   writer.writerow({k:json.dumps(row[k],separators=(',',':'),allow_nan=False) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in fields})


def human_text(text):
 # Presentation spacing must never edit fenced commands or inline code.
 parts=re.split(r'(```.*?```|`[^`\n]*`)',text,flags=re.S)
 for i in range(0,len(parts),2):
  value=re.sub(r'\b([A-Za-z]{3,})(?=\d)',r'\1 ',parts[i])
  parts[i]=re.sub(r'(\d)(KiB|MiB|GiB)\b',r'\1 \2',value)
 return ''.join(parts)


def ci_text(row,scale=1):return f"{row['mean']*scale:+.3f} [{row['ci95'][0]*scale:+.3f}, {row['ci95'][1]*scale:+.3f}]"


def expanded(rows):
 result=[]
 for row in rows:
  need(row['n']==4 and len(row['per_window'])==4,'four passage values')
  result.append({**row,'ci_low':row['ci95'][0],'ci_high':row['ci95'][1],**{f'passage_{j}':x for j,x in enumerate(row['per_window'])}})
 return result


def lookup(a,c,i,d,p,metric):
 return next(z for z in a['contrasts'] if (z['corpus'],z['identity'],z['view'],z['metric'],z['name'])==(c,i,'later',metric,f'000__{d}_{p}_effect'))


def validate_inputs(data):
 a=data[ANALYSIS];m=data[MANIFEST];inv=data[INV];n=data[NFILE];p=data[PFILE];r=data[RFILE]
 need(m['status']=='completed' and m['all_final_guards_passed'] is True,'complete guarded study')
 need(a['scope']['raw_manifest_sha256']==PINS[MANIFEST] and a['scope']['raw_inventory_sha256']==PINS[INV],'analysis raw bindings')
 need(a['scope']['source_hashes']=={Path(k).name:v['sha256'] for k,v in inv.items()} and len(inv)==32,'all32 analysis inventory bindings')
 need(n['status']==p['status']==r['status']=='passed','all primary audits passed')
 need(n['audited_manifest_sha256']==p['run_manifest_sha256']==r['run_manifest_sha256']==PINS[MANIFEST],'same audited manifest')
 need(n['analysis_sha256']==PINS[ANALYSIS] and n['raw_inventory_sha256']==p['inventory_sha256']==r['inventory_sha256']==PINS[INV],'audit analysis/inventory binding')
 need([m[k] for k in ('forward_attempts','model_outputs_observed','forward_passes','fully_processed_forwards')]==[776]*4,'retained776 counts')
 need([m[k] for k in ('evaluation_forwards','suffix_forwards','anchor_forwards')]==[384,384,8],'retained categories')
 need([len(a[k]) for k in ('window_metrics','anchors','summaries','contrasts','paired_start_interactions')]==[1152,24,288,1368,684],'complete result universe')
 need(sum(x['primary'] for x in a['contrasts'])==128,'all128 primary comparisons')
 need(n['arrays_verified']==33933 and n['controls']['KV_closure']==128,'audit array/control counts')
 need(p['projection_comparisons']==28 and p['coordinates_checked']==18087936 and r['counts']['full_vocabulary_rows']==48,'P/R counts')
 need(data['review/t22_first_input_exact_replay_gate_v9_actual.json']['status']=='passed_exact_first_input_replay_gate','replay gate PASS')
 context=data['t22_context_decomposition_run_v2/report.json'];need(context['run_manifest_sha256']==PINS[MANIFEST] and context['raw_inventory_sha256']==PINS[INV],'context same run')
 accepted=data['review/t22_context_descriptive_scientific_acceptance_v1.json'];need(accepted['status']=='passed_descriptive_context_arithmetic_acceptance_not_equality_claim' and accepted['context_report_sha256']==PINS['t22_context_decomposition_run_v2/report.json'] and accepted['independent_audit_sha256']==PINS['review/t22_context_saved_array_independent_audit_v1.json'],'context accepted descriptive scope')
 return a,m,n,p,r,context


def safe_readout_rows(r):
 keys=('query','saved_fp32_kl','fp64_kl_saved_gpu_logits','signed_kl_error','absolute_kl_error','absolute_reference_kl','absolute_error_to_signal','reference_is_exact_zero')
 rows=[]
 for arm,state in r['states'].items():
  for row in state['small_KL_saved_gpu_logit_diagnostics']:
   need(set(row)==set(keys),'typed selected KL-only audit row')
   rows.append({'mode':arm,**{k:row[k] for k in keys}})
 need(len(rows)==48,'all48 selected KL-only rows')
 return rows


def savefig(fig,out,name):
 fig.savefig(out/'figures'/f'{name}.png',dpi=180,bbox_inches='tight',metadata={'Software':'Report18 frozen-JSON builder'})
 fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight',metadata={'Date':None,'Creator':'Report18 frozen-JSON builder'})
 plt.close(fig)


def diagram(out):
 fig,ax=plt.subplots(figsize=(12,5.4));ax.set_xlim(0,12);ax.set_ylim(0,5.4);ax.axis('off')
 def box(x,y,w,h,text,color='#edf3f6'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#59717e'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
 def arrow(x1,y1,x2,y2,label=None,color='#455a64'):
  ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,lw=1.6,color=color))
  if label:ax.text((x1+x2)/2,(y1+y2)/2+.15,label,ha='center',fontsize=9,color=color)
 box(.2,3.55,2.3,1.05,'Same fixed input\n512 tokens')
 box(3.05,3.55,3.1,1.05,'Donor: early L10 write edit\nPC or orth; q1–126','#fcebdc')
 box(7,3.55,4.55,1.05,'Actual donor L11 earlier K/V\nLater L11 input Q/K/V stays baseline\n(and later upstream residual)' )
 arrow(2.65,4.08,2.95,4.08);arrow(6.3,4.08,6.85,4.08)
 box(.2,1.3,2.3,1.1,'Recipient upstream\nfully baseline')
 box(3.05,1.3,3.1,1.1,'Baseline L11 projection\nQ, source/later K/V fixed')
 box(7,1.3,4.55,1.1,'Splice earlier K/V before SDPA\nCheck outputs q127–511\nScore targets 128–511 (q511 unscored)','#e2eef3')
 arrow(2.65,1.85,2.95,1.85);arrow(6.3,1.85,6.85,1.85)
 arrow(9.25,3.4,9.25,2.55,'exact donor bytes','#b56724')
 ax.text(.2,4.95,'Finite donor/recipient intervention — native boundary 000; all 12 heads',fontsize=13,weight='bold')
 ax.text(.2,.53,'Combined K+V: all later attention contexts, writes, hidden states and full-vocabulary logits equal donor bytes.',fontsize=10)
 ax.text(.2,.15,'128 closures pass before single-path interpretation. Earlier outputs need not equal the donor; L11 K/V doses are unequal.',fontsize=10)
 savefig(fig,out,'causal_donor_recipient')


def effects_figure(a,out,metric,scale,name,xlabel):
 fig,axes=plt.subplots(1,2,figsize=(12,6.3),sharex=True,sharey=(metric!='kl_baseline'))
 for ax,d,title in zip(axes,('PC','orth'),('PC-removal donor','Orthogonal donor')):
  for j,path in enumerate(('K','V','KV')):
   rows=[lookup(a,c,i,d,path,metric) for c,i in STRATA];mean=np.array([z['mean'] for z in rows])*scale
   low=np.array([z['ci95'][0] for z in rows])*scale;high=np.array([z['ci95'][1] for z in rows])*scale
   need(np.all(low<=mean) and np.all(mean<=high),'mean inside displayed CI')
   y=np.arange(8)[::-1]+(j-1)*.19
   ax.errorbar(mean,y,xerr=np.vstack((mean-low,high-mean)),fmt=('o','s','D')[j],ms=4,capsize=2,lw=1.2,color=COLORS[path],label=path+' only' if path!='KV' else 'K+V')
  ax.axvline(0,color='#777',lw=.8);ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_xlabel(xlabel);ax.legend(frameon=False,loc='lower right' if metric=='nll' else 'upper right',fontsize=9)
 axes[0].set_yticks(np.arange(8)[::-1]);axes[0].set_yticklabels([LABELS[c]+' / '+('BOS' if i=='bos' else 'natural') for c,i in STRATA])
 if metric=='kl_baseline':
  axes[0].tick_params(axis='y',labelleft=True)
  axes[1].set_yticks(np.arange(8)[::-1]);axes[1].tick_params(axis='y',labelleft=False)
  for ax in axes:ax.set_ylim(-.5,7.5)
 fig.suptitle('Later targets 128–511: every source/start stratum',fontsize=14)
 fig.text(.5,.015,'Means and nominal conditional 95% paired passage-cluster bootstrap CIs; n=4 passages/source. Correlated, unadjusted; unequal L11 doses.',ha='center',fontsize=9)
 if metric=='kl_baseline':fig.subplots_adjust(left=.23,right=.98,bottom=.15,top=.84,wspace=.08)
 else:fig.tight_layout(rect=(0,.04,1,.96))
 savefig(fig,out,name)


def kl_error_figure(rows,out):
 fig,axes=plt.subplots(1,2,figsize=(11,4.2))
 x=np.array([z['absolute_reference_kl'] for z in rows])*1e6;y=np.array([z['absolute_kl_error'] for z in rows])*1e6
 axes[0].scatter(x,y,s=24,c='#416c83',alpha=.8);axes[0].set_xscale('symlog',linthresh=.1);axes[0].set_xlabel('Saved-logit FP64 KL (micro-nats; symlog x)');axes[0].set_ylabel('|saved FP32 KL − FP64 KL| (micro-nats)');axes[0].grid(alpha=.2)
 axes[0].set_title('Only 48 selected vocabulary rows')
 axes[1].axis('off');axes[1].text(0,.96,'Numerical scope',weight='bold',fontsize=13,va='top')
 text=('Wiki window0, natural start;12 states ×4 rows.\n\nMaximum absolute discrepancy:3.05994e−7 nats.\nMaximum defined error/reference ratio:0.441326.\nEight exact-zero references: ratios are null.\n\nThese rows do not bound every panel row.\nA 1e−4 consistency pass does not establish\nsensitivity to effects near1e−6.\n\nNo tiny-interaction ranking follows from this plot.')
 axes[1].text(0,.84,human_text(text),fontsize=10,va='top',linespacing=1.5)
 fig.tight_layout();savefig(fig,out,'selected_kl_error_scope')


def build_tables(a,r,context,out):
 t=out/'tables';counts={}
 fields=['corpus','identity','view','metric','name','primary','n','mean','ci_low','ci_high','passage_0','passage_1','passage_2','passage_3','terms']
 for key,name in [('contrasts','all_1368_contrasts'),('paired_start_interactions','all_684_paired_start_rows')]:
  rows=expanded(a[key]);csv_write(t/(name+'.csv'),rows,fields);json_write(t/(name+'.json'),a[key]);counts[name]=len(rows)
 primary=[x for x in a['contrasts'] if x['primary']];csv_write(t/'all_128_primary.csv',expanded(primary),fields);json_write(t/'all_128_primary.json',primary);counts['all_128_primary']=len(primary)
 lines=['# All128 predeclared primary comparisons','', 'NLL/KL in nats per target. Nominal unadjusted conditional 95% CIs; four paired passage clusters. Full precision and all four passage values are in the adjacent CSV/JSON.','', '|Source|Start|Metric|Estimand|Mean|95% CI|Passages0,1,2,3|','|---|---|---|---|---:|---|---|']
 for z in primary:lines.append('|'+ '|'.join([LABELS[z['corpus']],z['identity'],z['metric'],z['name'].removeprefix('000__'),f"{z['mean']:.9g}",f"[{z['ci95'][0]:.9g}, {z['ci95'][1]:.9g}]",', '.join(f'{v:.9g}' for v in z['per_window'])])+'|')
 write(t/'all_128_primary.md',human_text('\n'.join(lines))+'\n')
 for key in ('window_metrics','anchors'):
  csv_write(t/(key+'.csv'),a[key],['corpus','window','identity','view','mode','nll','kl_baseline','agreement_baseline']);json_write(t/(key+'.json'),a[key]);counts[key]=len(a[key])
 summaries=[]
 for z in a['summaries']:
  for metric,value in z['metrics'].items():summaries.append({**{k:z[k] for k in ('corpus','identity','view','mode')},'metric':metric,**value})
 csv_write(t/'all_864_arm_metric_summaries.csv',expanded(summaries),['corpus','identity','view','mode','metric','n','mean','ci_low','ci_high','passage_0','passage_1','passage_2','passage_3']);json_write(t/'all_288_arm_summaries.json',a['summaries']);counts['arm_metric_summaries']=len(summaries)
 for key in ('gate_norm_window_summaries','write_norm_window_summaries','l11_norm_window_summaries','matching_quality'):
  json_write(t/(key+'.json'),a[key]);counts[key]=len(a[key])
 later=[]
 for c,i in STRATA:
  for d in ('PC','orth'):
   z={'corpus':c,'identity':i,'direction':d}
   for p in ('K','V','KV'):
    q=lookup(a,c,i,d,p,'nll');z.update({p+'_mean':q['mean'],p+'_ci_low':q['ci95'][0],p+'_ci_high':q['ci95'][1]})
   later.append(z)
 csv_write(t/'later_NLL_all_16_direction_strata.csv',later,['corpus','identity','direction']+[p+s for p in ('K','V','KV') for s in ('_mean','_ci_low','_ci_high')])
 dose=[]
 for z in a['l11_norm_window_summaries']:
  if z['view']=='full' and z['mode'] in ('000__remove_write_matched_early','000__orth_write_matched_early'):
   e=z['active_early_values'];k=e['k_delta_baseline_norm']['rms'];v=e['v_delta_baseline_norm']['rms'];need(k>0,'positive donor K dose')
   dose.append({**{key:z[key] for key in ('corpus','window','identity','mode')},'K_active_RMS':k,'V_active_RMS':v,'V_to_K_RMS_ratio':v/k})
 need(len(dose)==64,'64MAIN donor dose records');csv_write(t/'all_64_MAIN_L11_dose_records.csv',dose,list(dose[0]));counts['MAIN_L11_dose_records']=64
 readout=safe_readout_rows(r);csv_write(t/'selected_48_KL_error_rows.csv',readout,list(readout[0]));json_write(t/'selected_48_KL_error_rows.json',readout);counts['selected_KL_error_rows']=len(readout)
 context_rows=[]
 for d,s in context['summaries'].items():
  for key,v in s.items():context_rows.append({'direction':d,'diagnostic':key,**v})
 csv_write(t/'optional_context_all_descriptive_summaries.csv',context_rows,['direction','diagnostic','cells','nonfinite_or_undefined','min','median','p95','max']);counts['optional_context_summary_rows']=len(context_rows)
 return counts,readout


def numerical_claims(a,r,context):
 singles=[z for z in a['contrasts'] if z['view']=='later' and z['metric']=='kl_baseline' and z['name'] in ['000__'+d+'_'+p+'_effect' for d in ('PC','orth') for p in ('K','V')]]
 differences=[z for z in a['contrasts'] if z['view']=='later' and z['metric']=='kl_baseline' and z['name'] in ('000__PC_K_minus_V','000__orth_K_minus_V')]
 nll=[z for z in a['contrasts'] if z['view']=='later' and z['metric']=='nll' and z['name'] in ['000__'+d+'_'+p+'_effect' for d in ('PC','orth') for p in ('K','V')]]
 klrows=safe_readout_rows(r)
 claims={'later_single_path_KL':{'stratum_means':len(singles),'positive_means':sum(z['mean']>0 for z in singles),'positive_CI_lower_bounds':sum(z['ci95'][0]>0 for z in singles),'positive_passage_values':sum(x>0 for z in singles for x in z['per_window']),'passage_values':sum(len(z['per_window']) for z in singles),'mean_min':min(z['mean'] for z in singles),'mean_max':max(z['mean'] for z in singles)},'later_K_minus_V_KL':{'negative_means':sum(z['mean']<0 for z in differences),'comparisons':len(differences),'nominal_CIs_excluding_zero':sum(z['ci95'][0]>0 or z['ci95'][1]<0 for z in differences),'unequal_L11_doses':True},'later_single_path_NLL':{'means':len(nll),'positive':sum(z['mean']>0 for z in nll),'negative':sum(z['mean']<0 for z in nll),'CIs_including_zero':sum(z['ci95'][0]<=0<=z['ci95'][1] for z in nll)},'all_primary_NLL_CIs_including_zero':sum(z['ci95'][0]<=0<=z['ci95'][1] for z in a['contrasts'] if z['primary'] and z['metric']=='nll'),'selected_KL_error':{'rows':len(klrows),'max_absolute_error':max(z['absolute_kl_error'] for z in klrows),'max_defined_relative_error':max(z['absolute_error_to_signal'] for z in klrows if z['absolute_error_to_signal'] is not None),'null_ratios':sum(z['absolute_error_to_signal'] is None for z in klrows),'whole_panel_bound':False},'coverage':{'primary':128,'contrasts':1368,'paired_start_rows':684,'retained_forwards':776,'cumulative_forwards':803,'independent_full_panel_replication':False},'optional_context':{'status':'accepted_saved_array_descriptive_arithmetic_not_primitive_regeneration','all_cells':9240,'arrays_private_not_opened':313,'report_sha256':PINS['t22_context_decomposition_run_v2/report.json']}}
 dose=[z['active_early_values']['v_delta_baseline_norm']['rms']/z['active_early_values']['k_delta_baseline_norm']['rms'] for z in a['l11_norm_window_summaries'] if z['view']=='full' and z['mode'] in ('000__remove_write_matched_early','000__orth_write_matched_early')]
 claims['MAIN_L11_dose']={'records':len(dose),'individual_K_V_RMS_values':2*len(dose),'V_to_K_ratio_min':min(dose),'V_to_K_ratio_max':max(dose),'not_same_as_64_main_suffix_matching_pairs':True}
 claims['primary_interval_counts']={metric:{'exclude_zero':sum(z['ci95'][0]>0 or z['ci95'][1]<0 for z in a['contrasts'] if z['primary'] and z['metric']==metric),'include_zero':sum(z['ci95'][0]<=0<=z['ci95'][1] for z in a['contrasts'] if z['primary'] and z['metric']==metric)} for metric in ('nll','kl_baseline')}
 need(claims['primary_interval_counts']=={'nll':{'exclude_zero':26,'include_zero':38},'kl_baseline':{'exclude_zero':46,'include_zero':18}},'all128 intervals counted')
 need(claims['later_single_path_KL']['positive_means']==claims['later_single_path_KL']['positive_CI_lower_bounds']==32 and claims['later_single_path_KL']['positive_passage_values']==128,'full KL pattern checked')
 need(claims['later_K_minus_V_KL']['nominal_CIs_excluding_zero']==8 and claims['later_single_path_NLL']=={'means':32,'positive':17,'negative':15,'CIs_including_zero':15},'full NLL/difference pattern checked')
 return claims


def report(a,m,n,p,r,context,claims,table_counts):
 body='''# Report18 candidate — earlier L11 keys and values transmit a chosen early-write perturbation

**Candidate for scientific and hygiene review. Publication remains held; this is not a claim of independent full-panel replication.**

The completed T22 panel finds finite later-output consequences from transplanting either earlier keys or earlier values into a baseline GPT-2 recipient. Both directions and all eight source/start strata show positive mean baseline-to-recipient KL for both single paths. Combined key+value recipients reproduce their actual donors byte-for-byte at every later context, projected attention write, block/final hidden state and full-vocabulary logit. All128 required closure events pass. Signed NLL effects differ across directions, paths and passages; distributional change is not uniformly a loss increase or improvement. These results identify transmission of these chosen finite perturbations, not intrinsic key/value importance or native-task necessity.

## What was intervened on

We used the pinned GPT-2 checkpoint in FP32 SDPA, batch1, no cache, and physical length512. Layer indices are zero-based: L11 is the final attention block. Sixteen fresh passages—four each from Wiki validation, Utilitarianism, Ethics and Python optimizer code—were paired with natural and token0-only BOS-replacement starts. The accepted panel has32 identities; freshness concerns the declared777-hash experimental history and source-span/calibration-buffer exclusions, not GPT-2 pretraining overlap. Selection made831 decisions, accepting16 and rejecting815, before outcomes.

The two donors reuse one frozen T18 fit: PC-removal and one fixed orthogonal direction. Full L10 projected-write proposals supply the existing querywise FP64 interpolation recipe; the actual rounded edits are copied only at earlier queries1–126. The two L10 donor write changes are approximately norm matched. Their resulting L11 key and value changes are **not** dose matched. Across64 MAIN donor records (32 identities×2 directions;128 K/V RMS values), active earlier V/K RMS ratios range1.27769–1.59731. These64 dose comparisons differ from the64 MAIN/SUFFIX PC–orth L10 matching pairs. The fit originated in a different calibration world with source-position values zeroed at L4–L11; no native refit or direction ensemble was added here.

![Donor and recipient causal design](figures/causal_donor_recipient.png)

A recipient follows the complete baseline upstream trajectory and retains every query vector. Only earlier L11 K, V, or both are copied from the corresponding actual donor; source0 and later K/V remain baseline. Thus the recipient does not rerun the early-write edit. Each input has12 scientific arms, MAIN then a fixed rotated-SUFFIX control per arm. Both directions and variants must pass combined-KV closure before any single-path arm is interpreted. Natural passage0 additionally receives unhooked-repeat and identity-writer anchors. The retained study comprises384 MAIN,384 SUFFIX and8 anchor forwards, with no calibration forwards.

## Distributional transmission and signed loss

All32 K-only/V-only later-KL stratum means and their nominal95% intervals are positive; all128 underlying passage values are positive. Means range from5.33481e−6 to3.39704e−5 nats/target. Here KL means the frozen FP32 scorer's baseline-to-arm divergence, not a redefined metric. This supports finite distributional transmission through both transplanted paths under the fixed recipient conditions.

![Later NLL in every direction/source/start stratum](figures/later_NLL_all_strata.png)

NLL supplies a more qualified picture. Among32 single-path stratum means,17 are positive and15 negative;15 intervals include zero. PC value transplants raise mean later NLL in all eight strata, whereas orthogonal value transplants lower it in all eight. Key effects change sign across sources and starts. For example, natural Utilitarianism has opposite PC key/value signs, and natural Ethics has opposing orthogonal key/value signs; their combined effects can cancel substantially. The complete16-row table below includes all K/V/KV NLL results, including weak and reversed effects. A negative NLL difference is a lower loss on these targets, not a general benefit claim.

![Later KL in every direction/source/start stratum](figures/later_KL_all_strata.png)

V mean KL exceeds K mean KL in all16 direction/stratum comparisons, but only8 of16 nominal K−V intervals exclude zero. The natural L11 doses differ, so neither this ordering nor a larger response identifies intrinsic V dominance. All19 fixed coefficient definitions are retained across three views and three metrics. The128 co-primary comparisons include single-path effects, K−V and factorial interactions in later NLL/KL; all are exported. Of 64 primary NLL intervals,26 exclude zero;46 of64 primary KL intervals exclude zero. In total72 of128 exclude zero and56 include it—descriptive counts, not a study-success criterion. Nonadditivity in loss or KL can arise from attention, MLP/normalization and probability/loss nonlinearities, and cancellation; it is not a mediation fraction or a newly isolated physical mechanism.

## Numerical evidence and what it does not certify

The independent saved-array numerical audit passed33,933 arrays and all frozen statistics/control checks. The projection audit passed28 relations over18,087,936 coordinates: natural L10 projections and L11 original c_attn/context projections, with explicit FP32 arithmetic assumptions. Transplanted QKV is checked as a byte splice, never misrepresented as the output of the unchanged projection. The readout audit passed12 states and48 selected full-vocabulary rows. These checks corroborate the declared implementation and saved evidence, not the whole-model causal interpretation.

![Selected-row KL scoring error and its scope](figures/selected_kl_error_scope.png)

For those48 rows, saved FP32 KL versus stable FP64 KL computed from the **saved GPU logits** differs by at most3.05994e−7 nats. The maximum defined absolute-error/reference ratio is0.441326; eight exact-zero references have null ratios. This arithmetic comparison is distinct from the separate FP64 hidden×embedding readout reconstruction. A 1e−4 saved-metric consistency pass does not establish sensitivity at 1e−6, and selected rows do not bound every unsaved panel row. Signed canonical values are retained without clipping. We therefore avoid ranking very small interactions, treating a small/zero effect as absence, or assigning whole-panel resolution from these diagnostics.

## Uncertainty, provenance and preserved failures

Each source contributes only four passage clusters. The2,000 paired bootstrap draws use seed20261021 and the same passage indices across starts, arms, views, metrics and contrasts. Intervals are nominal, correlated, conditional on the one fixed fit and unadjusted for multiplicity. There is no any-one-interval success rule, calibration-fit uncertainty estimate, equivalence test or independent-replication interpretation. Starts share text; directions share the same construction; the four sources are not four interchangeable population replications. Every contrast and paired-start row retains all four passage values.

The completed A/B segments took308.9699s and308.8087s. Maximum observed segment RSS was2,461,524KiB and GPU allocated peak1,145,661,440B, under prospectively reviewed4GiB primary-process cap and R0=2240 MiB admission and6GiB preload host headroom. Both final parameter/source/precision guards pass. The retained panel is776 forwards, while cumulative acquisition is803: two preserved resource-stopped prefixes contributed1 and26 returns, followed by a clean776-forward panel. Failed prefixes are not reused as scientific samples or retroactively given missing final guards.

A separate exact replay compared the completed first input against the preserved26-forward prefix: all1,366 saved numeric-array occurrences match byte-for-byte, and26 recorded full-hidden/full-vocabulary-logit fingerprints match exactly. The first comparator failed because its expected anchor phase was wrongly labeled MAIN; the reviewed one-literal correction used `anchor`, preserved the failure, and did not change equality criteria or rerun the model. An independent N-acceptance harness also had a preserved schema overconstraint; its correction required no N rerun. These are infrastructure failures, not failed scientific hypotheses. Older infrastructure evidence also has explicit gaps: a tiny-CPU fixture left an unfinished10-call raw prefix unsaved, and the first writer witness did not save rejected4,385,576-byte ordinary entropy. Neither is missing retained-panel output. The scientific record instead leaves uniform loss benefit, intrinsic path dominance, native-pulse/task necessity and calibration-world transfer unsupported or untested.

## Optional selected-context analysis — accepted descriptive scope

A separately pre-ledgered calculation covers only the predetermined Wiki0 natural panel: both directions,12 heads and385 later queries (9,240 cells). Independent saved-array auditing accepted313 arrays,308 field checks over32,076,120 coordinates and1,464 summary fields, including recomputation of the saved delta-attention×delta-value product. It did not independently regenerate primitive softmax or captured SDPA. The established identity `(Ckv−Ck)−(Cv−Cb)=(Ad−Ab)(Vd−Vb)` describes the reconstructed context factorial, with actual-minus-reconstructed residuals explicitly retained.

Median measured four-arm residual-sum/product ratios are0.003237 (PC) and0.002400 (orth), but maxima exceed100,000: the smallest cells remain unresolved. Captured-interaction relative diagnostics have2 PC and3 orth exact-zero denominators. The named baseline-reference allocation/conditional-mixture decomposition has typical actual relative residuals near2e−5, but two zero denominators per direction and some tiny-signal rows have relative residuals near1. Term medians and their ratio are not causal fractions; this supplies no new gate intervention or unique attribution. All descriptive summaries are exported; private activation arrays were not opened for this report.

## Relation to prior work and open tests

K/V and path patching are established in [Interpretability in the Wild](https://arxiv.org/html/2211.00593v1) and [Localizing Model Behavior With Path Patching](https://arxiv.org/html/2304.05969v1); [activation-patching best practices](https://arxiv.org/html/2309.16042v2) already emphasize metric/method dependence. [The Super Weight](https://arxiv.org/html/2411.07191v2), [Massive Activations](https://arxiv.org/html/2402.17762v2), [sink/gate work](https://arxiv.org/html/2602.01203v3) and [GPT-2 sink circuitry](https://arxiv.org/html/2604.14722v1) provide related activation/gating context. The primary URLs were rechecked before this report; this is not an exhaustive novelty search. T22's candidate contribution is this fixed-fit, fresh-panel finite transmission evidence with exact donor/recipient closure and disclosed numerical/provenance limits—not a new patching or attention-algebra method.

Fresh native calibration, independently fixed direction controls and held-out tests of any proposed gate-versus-mixture intervention remain open. No head was isolated, no rank-one mechanism established, and no original native pulse, task necessity, universal transfer, generation fidelity or speed claim was tested. Publication and Library delivery remain separate from this review candidate.
'''
 appendix=['\n## Comprehensive appendix\n','The prose and figures use rounded display values. CSV/JSON tables preserve the frozen numbers and passage vectors. No primary, sign-reversed, zero-crossing or paired-start result is omitted.\n','### All later K/V/KV NLL effects\n','Units: millinats/target (1e−3 nats). Mean [nominal95% CI]; n=4 passage clusters. Positive is higher loss.\n','|Source|Start|Direction|K only|V only|K+V|','|---|---|---|---:|---:|---:|']
 for c,i in STRATA:
  for d in ('PC','orth'):appendix.append('|'+ '|'.join([LABELS[c],i,d]+[ci_text(lookup(a,c,i,d,p,'nll'),1000) for p in ('K','V','KV')])+'|')
 appendix+=['\n### Complete machine-readable results\n','|Table|Rows|Coverage|','|---|---:|---|',
 '|[All primary comparisons](tables/all_128_primary.md) · [CSV](tables/all_128_primary.csv) · [JSON](tables/all_128_primary.json)|128|All later co-primary estimates, CIs and four passage values|',
 '|[All contrasts](tables/all_1368_contrasts.csv) · [JSON](tables/all_1368_contrasts.json)|1368|19 definitions ×4 sources ×2 starts ×3 views ×3 metrics; all passage values|',
 '|[All paired-start rows](tables/all_684_paired_start_rows.csv) · [JSON](tables/all_684_paired_start_rows.json)|684|BOS-minus-natural interactions; all passage values|',
 '|[Window metrics](tables/window_metrics.csv)|1152|Every arm/input/view, three metric columns|',
 '|[Anchor views](tables/anchors.csv)|24|Both anchor kinds, four natural passage0 inputs, three views|',
 '|[Arm metric summaries](tables/all_864_arm_metric_summaries.csv)|864|All means/CIs/passage values; [288 wide JSON summaries](tables/all_288_arm_summaries.json) also include transforms|',
 '|[Gate](tables/gate_norm_window_summaries.json), [write](tables/write_norm_window_summaries.json), [L11](tables/l11_norm_window_summaries.json) norms|1152 each|Every window/arm/view; source and active-early scopes retained|',
 '|[Matching diagnostics](tables/matching_quality.json)|64|Both variants,32 inputs; approximate actual donor matching|',
 '|[All MAIN L11 doses](tables/all_64_MAIN_L11_dose_records.csv)|64|Two donor directions×32 identities; K/V RMS and V/K ratio,128 individual RMS values|',
 '|[Selected KL errors](tables/selected_48_KL_error_rows.csv)|48|Typed scalar diagnostics only; no predictions/logits|',
 f"|[Optional context summaries](tables/optional_context_all_descriptive_summaries.csv)|{table_counts['optional_context_summary_rows']}|All accepted descriptive summary fields; no activation arrays|",
 '\n### Optional context summary, accepted descriptive scope\n','Each direction has4,620 head/query cells. Captured-interaction denominators are zero in2 PC/3 orth cells; gate actual-delta denominators in2 cells per direction. These are distinct counts. Undefined ratios are excluded from finite summaries and counted explicitly.\n','|Direction|Arm residual sum/product median; max|Actual-product relative error median; max|Allocation L2 median|Mixture L2 median|Gate actual relative residual median; undefined|','|---|---|---|---:|---:|---|']
 for d,label in [('remove','PC'),('orth','orth')]:
  s=context['summaries'][d];a1=s['measured_arm_error_to_product'];a2=s['interaction_actual_error_to_product'];g=s['gate_actual_error_relative']
  appendix.append(f"|{label}|{a1['median']:.6g}; {a1['max']:.6g}|{a2['median']:.6g}; {a2['max']:.6g}|{s['allocation_term_L2']['median']:.6g}|{s['mixture_term_L2']['median']:.6g}|{g['median']:.6g}; {g['nonfinite_or_undefined']}|")
 appendix+=['\n### Rebuild and integrity\n','Run the copied builder with Python, NumPy and Matplotlib in one thread, supplying the local research root and a **new** output directory:\n','```bash\npython -B pulse_transfer/l11_key_value_donor_report_candidate_v4/build_report.py --root pulse_transfer --out pulse_transfer/report18_rebuild_v1\n```\n','The builder accepts only the hash-pinned JSON/text inputs recorded in `build_manifest.json`; it does not decode a raw NPZ, load a tokenizer/checkpoint/model, change frozen sources, or publish anything. `claims.json` records exact claim counts/scales for independent review. `file_inventory.json` commits every candidate file except itself. This report builder is not the separate full scientific reproduction workflow.\n']
 body=human_text(body);appendix=[human_text(x) for x in appendix]
 return body+'\n'.join(appendix)+'\n',len(re.findall(r"\b[\w−]+\b",body))



INHERITED_V3_INVENTORY_SHA='8934d1d3d47990857feb17049005adbbdf782bcf6677858d0f2b28a1508c4953'
CORRECTION_REVIEW_SHA='a44cd614b7d4ae2efe71350cc4b48137bcb575a5ce02c41532d5bb07cb5882d4'


def presentation_only(args):
 """Copy reviewed numeric tables, rerender only the two corrected figures."""
 started=time.monotonic();root=args.root.resolve();src=args.presentation_from.resolve();out=args.out.resolve()
 need(not out.exists() and not out.is_relative_to(src),'fresh independent presentation directory')
 need(sha(src/'file_inventory.json')==INHERITED_V3_INVENTORY_SHA,'exact reviewed v3inventory')
 inv=json.loads((src/'file_inventory.json').read_text());actual={p.relative_to(src).as_posix() for p in src.rglob('*') if p.is_file()}
 need(actual==set(inv)|{'file_inventory.json'},'complete inherited35file membership')
 for name,entry in inv.items():
  p=src/name;need(not p.is_symlink() and p.resolve().is_relative_to(src) and sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],'immutable inherited file '+name)
 need(sha(root/'review/t22_report18_v3_independent_scientific_table_figure_privacy_review.json')==CORRECTION_REVIEW_SHA,'five-correction source review')
 need(sha(root/ANALYSIS)==PINS[ANALYSIS],'same frozen analysis for unchanged KL plot values')
 # Only this safe aggregate JSON is read for plotting; no statistics recomputed.
 a=json.loads((root/ANALYSIS).read_text());old_text=(src/'candidate.md').read_text()
 old_wording='all 1,366 numeric occurrences and 26 full hidden/logit outputs match exactly.'
 new_wording='all 1,366 saved numeric-array occurrences match byte-for-byte, and 26 recorded full-hidden/full-vocabulary-logit fingerprints match exactly.'
 old_command='python -B build_report.py --root pulse_transfer --out pulse_transfer/report 18_rebuild_v1'
 new_command='python -B pulse_transfer/l11_key_value_donor_report_candidate_v4/build_report.py --root pulse_transfer --out pulse_transfer/report18_rebuild_v1'
 need(old_text.count(old_wording)==old_text.count(old_command)==1,'exact two prose corrections')
 text=old_text.replace(old_wording,new_wording).replace(old_command,new_command)
 changed={'build_report.py','candidate.md','claims.json','build_manifest.json','file_inventory.json'}|{f'figures/{name}.{ext}' for name in ('causal_donor_recipient','later_KL_all_strata') for ext in ('png','svg')}
 out.mkdir();(out/'figures').mkdir();(out/'tables').mkdir()
 try:
  for name in inv:
   if name not in changed:shutil.copyfile(src/name,out/name)
  shutil.copyfile(__file__,out/'build_report.py');write(out/'candidate.md',text)
  plt.rcParams.update({'font.size':10,'svg.hashsalt':'report18-fixed-json-v1','axes.spines.top':False,'axes.spines.right':False})
  diagram(out);effects_figure(a,out,'kl_baseline',1e6,'later_KL_all_strata','Baseline-to-arm KL (micro-nats/target)')
  words=len(re.findall(r"\b[\w−]+\b",text.split('## Comprehensive appendix')[0]));need(1200<=words<=1800,'main word range')
  claims=json.loads((src/'claims.json').read_text());claims['main_report_word_count']=words;json_write(out/'claims.json',claims)
  meta=json.loads((src/'build_manifest.json').read_text());meta.update(builder_sha256=sha(__file__),main_word_count=words,wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,presentation_only_ancestry={'v3_inventory_sha256':INHERITED_V3_INVENTORY_SHA,'review_sha256':CORRECTION_REVIEW_SHA,'corrections':['R1','R2','R3','R4','R5'],'statistics_or_audit_recomputed':False,'tables_copied_byte_exact':True,'unchanged_NLL_and_error_figures_copied_byte_exact':True})
  need(meta['peak_RSS_bytes']<=3*1024**3,'presentation process cap')
  json_write(out/'build_manifest.json',meta)
  need(all(sha(out/name)==entry['sha256'] for name,entry in inv.items() if name.startswith('tables/')),'all22 tables byteexact')
  for name in ('later_NLL_all_strata','selected_kl_error_scope'):
   for ext in ('png','svg'):need(sha(out/'figures'/f'{name}.{ext}')==inv[f'figures/{name}.{ext}']['sha256'],'unchanged other figure')
  need(sha(src/'file_inventory.json')==INHERITED_V3_INVENTORY_SHA,'inherited inventory end recheck')
  for name,entry in inv.items():need(sha(src/name)==entry['sha256'],'inherited file end recheck')
  files={p.relative_to(out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()};json_write(out/'file_inventory.json',files)
  print(json.dumps({'status':'completed_presentation_only_v4','builder_sha256':sha(__file__),'candidate_sha256':sha(out/'candidate.md'),'inventory_sha256':sha(out/'file_inventory.json'),'files':len(files)+1,'words':words,'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':meta['peak_RSS_bytes']}))
 except BaseException as error:
  json_write(out/'build_failure.json',{'status':'failed_presentation_correction','error':repr(error),'builder_sha256':sha(__file__),'no_statistical_or_audit_recomputation':True});raise


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--presentation-from',type=Path);args=ap.parse_args()
 if args.presentation_from is not None:return presentation_only(args)
 root=args.root.resolve();out=args.out.resolve();started=time.monotonic()
 need(not out.exists(),'fresh output directory required');need(not out.is_relative_to(root/'gpt2_l11_key_value_donor_run_r3') and not out.is_relative_to(root/'t22_context_decomposition_run_v2'),'output outside immutable runs')
 data={}
 for name,digest in PINS.items():
  path=root/name;need(path.suffix in ('.json','.md') and sha(path)==digest,'exact JSON/text input '+name)
  data[name]=json.loads(path.read_text()) if path.suffix=='.json' else path.read_text()
 a,m,n,p,r,context=validate_inputs(data);claims=numerical_claims(a,r,context)
 out.mkdir();(out/'figures').mkdir();(out/'tables').mkdir();shutil.copyfile(__file__,out/'build_report.py')
 try:
  plt.rcParams.update({'font.size':10,'svg.hashsalt':'report18-fixed-json-v1','axes.spines.top':False,'axes.spines.right':False})
  counts,readout=build_tables(a,r,context,out);diagram(out)
  effects_figure(a,out,'nll',1000,'later_NLL_all_strata','ΔNLL (millinats/target; positive = higher loss)')
  effects_figure(a,out,'kl_baseline',1e6,'later_KL_all_strata','Baseline-to-arm KL (micro-nats/target)')
  kl_error_figure(readout,out);text,words=report(a,m,n,p,r,context,claims,counts);write(out/'candidate.md',text)
  need(1200<=words<=1800,('main report word range',words));claims['main_report_word_count']=words;json_write(out/'claims.json',claims)
  need([counts[k] for k in ('all_128_primary','all_1368_contrasts','all_684_paired_start_rows')]==[128,1368,684],'exhaustive tables')
  for name,digest in PINS.items():need(sha(root/name)==digest,'input end recheck '+name)
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;need(peak<=3*1024**3,'builder RSS<=3GiB')
  for path in out.rglob('*'):
   if path.is_file() and path.suffix in ('.md','.csv','.json','.py','.svg'):
    need(not re.search(r'/(?:home|Users)/|[A-Za-z]:\\\\Users\\\\',path.read_text()),'no private host path in candidate '+path.name)
  json_write(out/'build_manifest.json',{'status':'completed_report18_candidate_pending_independent_review','builder_sha256':sha(__file__),'input_sha256':PINS,'table_counts':counts,'main_word_count':words,'context_acceptance':'passed saved-array descriptive arithmetic and scientific scope; not equality or primitive regeneration','raw_payload_decodes':0,'model_tokenizer_checkpoint_GPU_calls':0,'frozen_source_edits':0,'deletions':0,'publication_actions':0,'matplotlib_cache':'Persistent review/report18_matplotlib_cache outside candidate; no temporary-cache cleanup requested.','wall_seconds':time.monotonic()-started,'peak_RSS_bytes':peak,'privacy_scope':'Only frozen aggregate/contrast/norm/control/error scalar outputs and allowed hashes. No input/prediction IDs, source text, weights, full gates/activations/hidden/logits or private host paths.'})
  files={p.relative_to(out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()};json_write(out/'file_inventory.json',files)
  print(json.dumps({'status':'completed_report18_candidate','out':out.name,'builder_sha256':sha(__file__),'inventory_sha256':sha(out/'file_inventory.json'),'files':len(files)+1,'bytes':sum(x['bytes'] for x in files.values())+(out/'file_inventory.json').stat().st_size,'main_words':words,'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':peak}))
 except BaseException as error:
  json_write(out/'build_failure.json',{'status':'failed_report18_candidate_build','builder_sha256':sha(__file__),'error_type':type(error).__name__,'error':str(error),'wall_seconds':time.monotonic()-started,'no_model_or_raw_payload_decode':True});raise


if __name__=='__main__':main()
