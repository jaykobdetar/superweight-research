"""Report19 JSON-only candidate builder. SOURCE ONLY pending root build GO.
From the task root, after approval:
python3.12 -B pulse_transfer/calibration_world_report_candidate_v2/build_report.py --inputs pulse_transfer/calibration_world_report_candidate_v2/data/report_presentation_inputs.json --out pulse_transfer/report19_rebuild_v2
No model, NPZ, tokenizer, weights, input IDs, raw activation or publication access.
"""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_n]='1'
import argparse,base64,csv,hashlib,html,json,math,re,resource,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ANALYSIS='gpt2_calibration_world_evaluation_run_v1/calibration_world_evaluation_analysis.json'
MANIFEST='gpt2_calibration_world_evaluation_run_v1/manifest.json'
INVENTORY='gpt2_calibration_world_evaluation_run_v1/numeric_inventory.json'
PRIMARY='review/t23_complete_primary_readout_v1.json'
GEOMETRY='review/t23_secondary_saved_fit_geometry_v1.json'
NFILE='review/t23_evaluation_numerical_actual_v2.json'
PFILE='review/t23_selected_projection_actual_v1.json'
RFILE='review/t23_selected_readout_actual_v1.json'
SCIENCE='review/t23_independent_scientific_review_v1.json'
LITERATURE='review/t23_primary_literature_recheck_v1.json'
PINS={
ANALYSIS:'8dc01b8911cd95c08c9ba9c46c95a00e0cf5c54177d5bc70f7eda635bd1797b5',
MANIFEST:'1b2e1f8f4c36355b297b45461c3b34143d42b30a348ec438d07a0f3a665c1382',
INVENTORY:'851bd22f549a205fe53581d5c65b75974565bb8b4ef0814fc3b269d5e8e1bb05',
PRIMARY:'ff44aa2b740ee2a0fb1ad49edf99d3285cae5d7ff3e62996aaf759899145ad61',
GEOMETRY:'5b57beb4e63fae6cf471e0b26dec149895f6766c01864c196c9d22cd94c7319f',
NFILE:'d2f81f3cd07b76792c17ab31ea4c09d1579fc6450668ae4e5a0b78f7c31125cc',
PFILE:'b18c4093b987238a08c843670ebe3295d0f991765b2a59f1b886ddd42012e64d',
RFILE:'89b8084cf33b55c25c7c653b6d92c86e7315f280a6c975dc23550da701523f24',
'review/t23_selected_PR_actual_v1.json':'34fa96d1d2437c5eae78198f0825c6ba1dd8e49d3e7babe36ecd54dd9fcd1ba7',
'review/t23_calibration_actual_result_independent_acceptance_v1.json':'87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300',
LITERATURE:'0f7062041552cb61f99fe5fc17b8fb5d9d22b35911bc68fbeaf3c25d76db2a0f',
SCIENCE:'ff4a58720d74a6f93d03c5bd29e0750db8ba4f17a24c06c94e42762423e8ca60'}
TEMPLATE_PINS={'report_template.md': 'eb4b9504c3d4729873d6a07ce1154aa0ade7066737c34b24650dbae5af4642b7', 'methods_template.md': 'c7fcf97050abb0c8556ab418cf64017dfcea5701c352df34dc1365c048b7f198'}
BUNDLE_SHA='eee96c1a34d885316ea3e085fc49cf894baa38c50b23635741d6832d939a5ad5'
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
LABELS={'wiki_validation':'Wiki','utilitarianism':'Utilitarianism','ethics':'Ethics','optimizers_new':'Code'}
STRATA=tuple((c,i) for c in CORPORA for i in ('natural','bos'))
CONTRASTS=('D_PP','D_PC','D_CP','D_CC','joint_CC_minus_PP','center_main','direction_pair_main','interaction')
TITLES=('D_PP','D_PC','D_CP','D_CC','D_CC − D_PP','Center main effect','Direction-pair main effect','Interaction')
PRIVATE=re.compile(r'/(?:home|Users)/|[A-Za-z]:\\Users\\')
FORBIDDEN={'tokens','token_ids','input_ids','pred','prediction','predictions','target_ids','targets','qkv','hidden','logits','credentials','access_token','api_key','left_id','right_id'}

def need(value,message):
 if not value:raise AssertionError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def write(path,text):
 with Path(path).open('x') as f:f.write(text)
def put(path,value):write(path,json.dumps(value,indent=2,allow_nan=False)+'\n')
def safe(value):
 if isinstance(value,dict):
  for k,v in value.items():need(isinstance(k,str) and k.lower() not in FORBIDDEN and not PRIVATE.search(k),'no private/raw key');safe(v)
 elif isinstance(value,(list,tuple)):
  for x in value:safe(x)
 elif isinstance(value,str):need(not PRIVATE.search(value),'no personal locator')
 elif isinstance(value,float):need(math.isfinite(value),'finite public number')
 else:need(value is None or type(value) in (bool,int),'public scalar type')

def load(inputs):
 need(inputs.is_file() and not inputs.is_symlink() and sha(inputs)==BUNDLE_SHA,'exact public presentation bundle')
 bundle=json.loads(inputs.read_text());need(bundle['schema']=='T23_Report19_public_presentation_inputs_v1' and bundle['original_json_sha256']==PINS,'original commitments kept separate from public projections')
 safe(bundle['presentation']);analysis=bundle['analysis'];need(analysis['sha256']==PINS[ANALYSIS] and analysis['encoding']=='base64_exact_original_UTF8_JSON_bytes','analysis byte commitment')
 analysis_bytes=base64.b64decode(analysis['base64'],validate=True);need(hashlib.sha256(analysis_bytes).hexdigest()==PINS[ANALYSIS],'exact decoded original analysis bytes')
 data=dict(bundle['presentation']);need(set(data)==set(PINS)-{ANALYSIS},'finite projected presentation roster');data[ANALYSIS]=json.loads(analysis_bytes);safe(data)
 for name,h in TEMPLATE_PINS.items():need(sha(HERE/name)==h,'frozen readable template '+name)
 a,m,n,p,r=(data[x] for x in (ANALYSIS,MANIFEST,NFILE,PFILE,RFILE))
 need(m['status']=='completed' and m['all_final_guards_passed'] is True and m['forward_passes']==1160,'completed full evaluation')
 need(a['scope']['raw_manifest_sha256']==PINS[MANIFEST] and a['scope']['raw_inventory_sha256']==PINS[INVENTORY],'analysis raw commitments')
 for x in (n,p,r):need(x['status']=='passed','projected NPR pass')
 need(n['audited_manifest_sha256']==p['run_manifest_sha256']==r['run_manifest_sha256']==PINS[MANIFEST] and n['analysis_sha256']==PINS[ANALYSIS],'same-run NPR commitments')
 need(n['counts']['arrays']==41871 and p['projection_comparisons']==19 and p['coordinates_checked']==7471104 and r['counts']['full_vocabulary_rows']==72,'finite saved-audit scope')
 return data,analysis_bytes

def primaries(data):
 a=data[ANALYSIS];rows=[z for z in a['contrasts'] if z['primary']]
 lookup={(z['name'].removeprefix('000__'),z['corpus'],z['identity']):z for z in rows}
 need(len(rows)==len(lookup)==64 and set(lookup)=={(n,c,i) for n in CONTRASTS for c,i in STRATA},'all64prespecified primaries')
 ordered=[]
 for name in CONTRASTS:
  for c,i in STRATA:
   z=lookup[name,c,i];need(z['metric']=='nll' and z['view']=='later' and z['n']==4 and len(z['per_window'])==4,'primary definition/four passage values')
   ordered.append({'contrast':name,'corpus':c,'identity':i,'n':4,'mean_nats':z['mean'],'ci_low_nats':z['ci95'][0],'ci_high_nats':z['ci95'][1],
    'mean_millinats':1000*z['mean'],'ci_low_millinats':1000*z['ci95'][0],'ci_high_millinats':1000*z['ci95'][1],
    'passage_values_nats':z['per_window'],'passage_values_millinats':[1000*x for x in z['per_window']],'terms':z['terms']})
 expected=data[PRIMARY];need(expected['analysis_sha256']==PINS[ANALYSIS] and expected['nominal_primary_CI_including_zero']==51,'independent primary readout binding')
 need(sum(z['ci_low_nats']<=0<=z['ci_high_nats'] for z in ordered)==51,'all-primary descriptive count')
 return ordered

def tables(rows,out):
 put(out/'tables/all_64_primary.json',{'status':'DRAFT_complete_prespecified_primary_table','units':'nats and millinats per later target; unadjusted conditional95% intervals,4passages','rows':rows})
 fields=['contrast','corpus','identity','n','mean_nats','ci_low_nats','ci_high_nats','mean_millinats','ci_low_millinats','ci_high_millinats',*[f'passage_{j}_nats' for j in range(4)],'passage_values_millinats','terms']
 with (out/'tables/all_64_primary.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fields);writer.writeheader()
  for row in rows:
   z={**row,**{f'passage_{j}_nats':x for j,x in enumerate(row['passage_values_nats'])}};writer.writerow({k:json.dumps(z[k],separators=(',',':')) if isinstance(z[k],(dict,list)) else z[k] for k in fields})
 md=['| Contrast | Source | Start | Mean [95% CI], millinats/target |','|---|---|---|---|'];ht=['<table><thead><tr><th>Contrast</th><th>Source</th><th>Start</th><th>Mean [95% CI], millinats/target</th></tr></thead><tbody>']
 for z in rows:
  v=f"{z['mean_millinats']:+.6f} [{z['ci_low_millinats']:+.6f}, {z['ci_high_millinats']:+.6f}]";values=(z['contrast'],LABELS[z['corpus']],z['identity'],v)
  md.append('| '+' | '.join(values)+' |');ht.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in values)+'</tr>')
 ht.append('</tbody></table>');write(out/'tables/all_64_primary.md','\n'.join(md)+'\n');return '\n'.join(md),'\n'.join(ht)

def figures(rows,out):
 # Only this future build function imports plotting modules; no model package.
 os.environ.setdefault('MPLCONFIGDIR',str(out.parent/'review'/'report19_matplotlib_cache'))
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
 plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.hashsalt':'Report19_fixed_frozen_JSON_v1','axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight',metadata={'Date':None,'Creator':'Report19 JSON-only builder'});fig.savefig(out/'figures'/f'{name}.png',dpi=180,bbox_inches='tight',metadata={'Software':'Report19 JSON-only builder'});plt.close(fig)
 fig,axes=plt.subplots(4,2,figsize=(13,16));labels=[LABELS[c]+' / '+i for c,i in STRATA]
 for ax,name,title in zip(axes.ravel(),CONTRASTS,TITLES):
  z=[r for r in rows if r['contrast']==name];means=np.array([r['mean_millinats'] for r in z]);lo=np.array([r['ci_low_millinats'] for r in z]);hi=np.array([r['ci_high_millinats'] for r in z]);y=np.arange(8)
  # Draw endpoint segments directly; bootstrap intervals need not contain means.
  ax.hlines(y,lo,hi,color='#2f6f95',linewidth=1.5);ax.plot(means,y,'o',color='#2f6f95',markersize=4);ax.axvline(0,color='#666',linewidth=.7)
  ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Millinats per later target');ax.grid(axis='x',alpha=.18)
 fig.suptitle('All 64 primary comparisons — signed later NLL',fontsize=16,y=.995);fig.tight_layout(rect=(0,.035,1,.98));fig.text(.5,.008,'Four passage clusters; paired nominal 95% CIs conditional on fixed calibration/FP32 scorer; numerical error not in bootstrap.\nPanel-specific x scales. Every interval and four passage values are retained in the tables.',ha='center',fontsize=10);save(fig,'all_64_primary_forest')
 fig,ax=plt.subplots(figsize=(13,7.8));ax.set_xlim(0,13);ax.set_ylim(0,8);ax.axis('off')
 def box(x,y,w,h,text,color='#eef4f8',size=11):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor=color,edgecolor='#667'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
 def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=13,color='#555'))
 ax.text(6.5,7.7,'Same four calibration texts; no evaluation refit',ha='center',fontsize=16)
 box(.3,5.8,2.8,1.1,'PE: source values zero\nexplicit causal mask');box(3.5,5.8,2.8,1.1,'PN: source values zero\nnative mask')
 box(6.7,5.8,2.8,1.1,'CE: native values\nexplicit causal mask','#f5f0e6');box(9.9,5.8,2.8,1.1,'CN: native values\nnative mask','#f5f0e6')
 ax.text(6.5,5.3,'Only PE and CE supply the primary parameters; PN/CN check mask routes',ha='center',fontsize=11)
 box(.7,3.5,5.1,1.1,'Center × direction pair (both u and v)\nPP = PE,PE       PC = PE,CE\nCP = CE,PE       CC = CE,CE',size=12)
 box(6.6,3.5,5.7,1.1,'Each cell: actual full remove(u) and orth(v) writes\nthen match to its own per-query minimum norm\nMatching is within cell, not a common cross-cell dose',size=11)
 arrow(1.7,5.7,2.4,4.75);arrow(8.1,5.7,4.3,4.75);arrow(5.9,4.05,6.4,4.05)
 box(1.2,1.55,4.5,1.1,'Native-value, native-mask evaluation\nDeliver matched post-c_proj write\nonly at earlier queries 1–126',size=12)
 box(7.1,1.55,4.7,1.1,'Score later targets 128–511\nfrom queries 127–510\nD = remove − orth, and fixed factorial contrasts',size=12)
 arrow(9.3,3.4,4.2,2.85);arrow(5.85,2.1,6.9,2.1)
 ax.text(6.5,.65,'Inactive queries copy each variant’s native baseline write. Query 511 has no next-token target.\nLayer 10 fit differences can arise upstream at layers 4–9; no claim that L10 values create their own incoming Q/K.',ha='center',fontsize=11)
 save(fig,'calibration_world_schematic');return matplotlib.__version__,np.__version__

def claims(data,rows):
 p=data[PRIMARY];r=data[RFILE];geometry=next(x for x in data[GEOMETRY]['geometry_rows'] if x['left']=='PE' and x['right']=='CE' and x['layer']==10)
 return {'status':'DRAFT_descriptive_claims_pending_final_report_review','all_64_primary_rows':True,'CI_including_zero':p['nominal_primary_CI_including_zero'],'primary_by_contrast':p['primary_by_contrast'],'maximum_absolute_joint_mean_nats':p['maximum_absolute_joint_calibration_shift'],'L10_center_L2_distance':geometry['center_L2_distance'],'L10_u_cosine':geometry['canonical_direction_pair']['u']['signed_dot'],'L10_v_cosine':geometry['canonical_direction_pair']['v']['signed_dot'],'selected_readout_maximum_absolute_KL_error':r['small_KL_summary']['maximum_absolute_kl_error'],'selected_readout_maximum_defined_KL_error_signal_ratio':p['selected_readout_maximum_defined_KL_error_signal_ratio'],'selected_readout_maximum_NLL_error':p['selected_readout_maximum_saved_NLL_error'],'selected_exact_zero_FP64_KL_reference_count':r['small_KL_summary']['exact_zero_fp64_kl_count'],'maximum_actual_pair_relative_RMS_gap':4.1807359229341295e-8,'CC_over_PP_active_target_RMS_ratio_range':[1.00044852047,1.01219646995],'dose_ratio_records':64,'calibration_forwards':16,'evaluation_forwards':1160,'input_identities':32,'experimental_history_hash_guard':841,'candidate_decisions':1090,'candidate_rejections':1074,'accepted_passages':16,'N_arrays':41871,'P_relations':19,'P_coordinates':7471104,'R_rows':72,'new_bootstrap_or_fit':False,'model_replay':False,'portable_reproduction_and_public_export':'PENDING','second_T22_full776_replay':'UNPERFORMED','future_ensemble_call_count':{'status':'design_pending_no_schedule'}}

def render_html(markdown,table_html,out,analysis_bytes):
 def inline(text):
  text=html.escape(text);text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text);text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text);return text
 chunks=[];paragraph=[];code=[];inside=False;table=False
 def flush():
  if paragraph:chunks.append('<p>'+inline(' '.join(paragraph))+'</p>');paragraph.clear()
 for line in markdown.splitlines():
  if line.startswith('```'):
   flush()
   if inside:chunks.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>');code=[]
   inside=not inside;continue
  if inside:code.append(line);continue
  if line.startswith('|'):
   flush()
   if not table:chunks.append('<div class="table-scroll">'+table_html+'</div>');table=True
   continue
  image=re.fullmatch(r'!\[([^]]+)\]\(figures/([a-z0-9_]+)\.svg\)',line)
  if image:
   flush();svg=(out/'figures'/f'{image[2]}.svg').read_text();chunks.append('<figure>'+svg[svg.index('<svg'):]+f'<figcaption>{html.escape(image[1])}</figcaption></figure>');continue
  if line.startswith('#'):
   flush();level=len(line)-len(line.lstrip('#'));chunks.append(f'<h{level}>'+inline(line[level:].strip())+f'</h{level}>');continue
  if not line.strip():flush()
  else:paragraph.append(line)
 flush();need(not inside,'balanced code block')
 chunks.append('<p>All downloads above are embedded in this HTML; companion files are optional.</p>')
 for relative,mime in (('tables/all_64_primary.csv','text/csv'),('tables/all_64_primary.json','application/json'),('methods.md','text/markdown'),('data/full_frozen_analysis.json','application/json')):
  encoded=base64.b64encode((out/relative).read_bytes()).decode();download='download="'+Path(relative).name+'" href="data:'+mime+';base64,'+encoded+'"'
  chunks=[chunk.replace('href="'+relative+'"',download) for chunk in chunks]
 css='body{font:17px/1.65 system-ui,sans-serif;max-width:1000px;margin:auto;padding:1em;color:#18212b}figure{margin:1.5em 0}svg{width:100%;height:auto}figcaption{font-size:.9em;color:#465460}.table-scroll{overflow:auto}table{border-collapse:collapse;font-size:.85em}th,td{padding:.5em;border:1px solid #ccd;text-align:left;white-space:nowrap}pre{overflow:auto;background:#f2f4f6;padding:1em}h1{line-height:1.2}a{color:#165d89}'
 return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Report19 draft</title><style>'+css+'</style><main>'+''.join(chunks)+'</main></html>'

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();started=time.monotonic();inputs=args.inputs.resolve();out=args.out.resolve()
 need(not out.exists() and not out.is_symlink(),'fresh immutable build output');need(not out.is_relative_to(HERE.resolve()) and not out.is_relative_to(inputs.parent) and all(not p.is_symlink() for p in out.parents),'output outside immutable public source/input trees')
 data,analysis_bytes=load(inputs);rows=primaries(data);out.mkdir();(out/'figures').mkdir();(out/'tables').mkdir();(out/'data').mkdir()
 try:
  text=HERE.joinpath('report_template.md').read_text();table_md,table_html=tables(rows,out);versions=figures(rows,out)
  literature='### Primary literature\n\n'+'\n\n'.join('['+x['title']+']('+x['url']+'). '+x['relevance'] for x in data[LITERATURE]['sources'])
  provenance='### Frozen evidence commitments\n\n'+'\n\n'.join('`'+name+'` — SHA256 `'+h+'`.' for name,h in PINS.items())
  artifacts='### Data and build artifacts\n\n[All 64 primary rows, CSV](tables/all_64_primary.csv) · [JSON including four passage values](tables/all_64_primary.json) · [Exact complete frozen analysis](data/full_frozen_analysis.json) · [Methods and rebuild command](methods.md).'
  replacements={'SCHEMATIC':'![Calibration worlds, parameter cells and early-write comparison](figures/calibration_world_schematic.svg)','FOREST':'![All 64 primary means and nominal intervals; panel-specific horizontal scales](figures/all_64_primary_forest.svg)','PRIMARY_TABLE':table_md,'LITERATURE':literature,'ARTIFACTS':artifacts,'PROVENANCE':provenance}
  for name,value in replacements.items():need(text.count('{{'+name+'}}')==1,'one readable template insertion');text=text.replace('{{'+name+'}}',value)
  need('{{' not in text,'no unresolved template fields');main_words=len(re.findall(r"\b[\w−]+\b",text.split('### Primary literature')[0]));need(1250<=main_words<=1900,('manageable main report length',main_words))
  need(hashlib.sha256(analysis_bytes).hexdigest()==PINS[ANALYSIS],'full frozen analysis exact bytes')
  with (out/'data/full_frozen_analysis.json').open('xb') as f:f.write(analysis_bytes)
  public_claims=claims(data,rows);public_claims['main_word_count']=main_words;safe(public_claims);put(out/'claims.json',public_claims)
  write(out/'methods.md',(HERE/'methods_template.md').read_text());write(out/'candidate.md',text);write(out/'candidate.html',render_html(text,table_html,out,analysis_bytes))
  for name in ('build_report.py','report_template.md','methods_template.md'):
   with (out/name).open('xb') as f:f.write((HERE/name).read_bytes())
  need(sha(inputs)==BUNDLE_SHA,'immutable public bundle end recheck')
  with (out/'data/report_presentation_inputs.json').open('xb') as f:f.write(inputs.read_bytes())
  for name,h in TEMPLATE_PINS.items():need(sha(HERE/name)==h,'immutable template end recheck')
  for path in out.rglob('*'):
   if path.is_file() and path.suffix in ('.py','.md','.html','.json','.csv','.svg'):need(not PRIVATE.search(path.read_text()),'no personal locator in output '+path.name)
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;need(peak<=3*1024**3 and 'torch' not in sys.modules,'JSON plotting only within3GiB')
  put(out/'build_manifest.json',{'status':'DRAFT_report19_candidate_pending_final_root_scientific_review','builder_sha256':sha(__file__),'template_sha256':TEMPLATE_PINS,'original_JSON_commitments_not_runtime_file_dependencies':PINS,'public_presentation_bundle_sha256':BUNDLE_SHA,'main_word_count':main_words,'table_rows':64,'figure_panels':{'primary_forest':8,'strata_per_panel':8},'complete_analysis_exact_byte_copy':True,'self_contained_HTML_embeds_all_analysis_and_three_small_downloads':True,'aggregate_presentation_not_independent_audit':True,'numpy_version':versions[1],'matplotlib_version':versions[0],'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':peak,'raw_payload_decodes':0,'checkpoint_or_model_or_tokenizer_or_GPU_calls':0,'new_bootstrap_or_fit':False,'publication_actions':0,'source_edits':0,'deletions':0,'reproduction_export_status':'PENDING','second_T22_full776_replay':'UNPERFORMED'})
  inventory={p.relative_to(out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()};put(out/'file_inventory.json',inventory)
  print(json.dumps({'status':'DRAFT_built','candidate_sha256':sha(out/'candidate.md'),'inventory_sha256':sha(out/'file_inventory.json'),'files':len(inventory)+1,'words':main_words,'peak_RSS_bytes':peak}))
 except BaseException as error:
  put(out/'build_failure.json',{'status':'failed_report_build','error_type':type(error).__name__,'error':str(error),'builder_sha256':sha(__file__),'no_model_or_raw_array_access':True});raise
if __name__=='__main__':main()
