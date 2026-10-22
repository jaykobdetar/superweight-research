#!/usr/bin/env python3
"""Rebuild T20 figures/tables from hash-bound aggregate analysis; no model imports.
Outputs must be new paths; existing artifacts are never overwritten or removed.
"""
import argparse, hashlib, json, os, pathlib, resource, sys, time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parents[1]
BINDINGS={
 'results/post_write_norm_matching/post_write_norm_analysis.json':'3fa2df289866b3e6e1914c21372f70a1da9c7ae99f213c4039935b0dd45c31eb',
 'results/post_write_norm_matching/execution.json':'bb46a4f94e7276e79563a9962b1977298fd915667b4b7dbce64ef402a2279164',
 'results/post_write_norm_matching/audits/numerical.json':'518860c3aed6d5571ec0653249c167ef482ac0529f37696dc642ad2e10dcfd3f',
 'results/post_write_norm_matching/audits/projection.json':'d865e341591eebe7995ce2c8b02fd862c808854bfba920fe7e095eb958399097',
 'results/post_write_norm_matching/audits/readout.json':'178fb5562a472f3a7ecb470459164704050fda207d954fe610749a6c63ac0922',
}
CORPORA=[('wiki_validation','Wiki'),('utilitarianism','Utilitarianism'),('ethics','Ethics'),('optimizers_new','Code')]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,s):
 with pathlib.Path(p).open('x') as f:f.write(s)
def main():
 start=time.time();parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=pathlib.Path,default=ROOT/'.reproduce-write-figures')
 args=parser.parse_args();out=args.out.resolve()
 if out.exists():raise FileExistsError(f'Preserve existing output; choose a fresh --out: {out}')
 data={}
 for name,digest in BINDINGS.items():
  p=ROOT/name
  if sha(p)!=digest:raise ValueError(f'Bound input changed: {name}')
  data[name]=json.loads(p.read_text())
 a=data[next(iter(BINDINGS))];m=data['results/post_write_norm_matching/execution.json']
 for name in list(BINDINGS)[2:]:
  assert data[name]['status']=='passed'
  assert data[name]['run_manifest_sha256']=='bb46a4f94e7276e79563a9962b1977298fd915667b4b7dbce64ef402a2279164'
 assert m['forward_passes']==m['fully_processed_forwards']==776
 c={(r['corpus'],r['identity'],r['view'],r['name']):r for r in a['contrasts']}
 summaries={(r['corpus'],r['identity'],r['view'],r['mode']):r for r in a['summaries']}
 rows=[]
 for corpus,label in CORPORA:
  for identity in ['natural','bos']:
   row={'corpus':corpus,'label':label,'identity':identity}
   for key,name in [('full','000__full_remove_minus_orth'),('matched','000__matched_remove_minus_orth'),('endpoint111','111__matched_remove_minus_orth'),('endpoint_difference','endpoint_111_minus_000__matched_remove_minus_orth'),('matching_difference','000__matching_difference_of_differences')]:
    row[key]=c[corpus,identity,'full',name]
   row['matched_pc_ppl_vs_c']=summaries[corpus,identity,'full','000__remove_write_matched']['metrics']['ppl_ratio_c']
   row['early']=c[corpus,identity,'early','000__matched_remove_minus_orth']
   row['later']=c[corpus,identity,'later','000__matched_remove_minus_orth']
   rows.append(row)
 assert len(rows)==8
 assert all((r['matched']['ci95'][0]>0 if r['corpus']=='wiki_validation' else r['matched']['ci95'][1]<0) for r in rows)
 q=a['matching_quality'];assert len(q)==128
 for r in q:
  for d in r['domains'].values():assert d['relative_RMS_gap']<=1e-5
 out.mkdir(parents=True,exist_ok=False)
 # Persistent local font cache, deliberately retained; no temporary directory cleanup.
 cache=out/'font_cache';cache.mkdir(exist_ok=True)
 os.environ['MPLCONFIGDIR']=str(cache)
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'T20_Report16_v1','figure.facecolor':'white','savefig.facecolor':'white'})
 outputs=[]
 def save(fig,name):
  for ext in ['png','svg']:
   p=out/f'{name}.{ext}';assert not p.exists()
   kwargs={'dpi':180} if ext=='png' else {'metadata':{'Date':None}}
   fig.savefig(p,bbox_inches='tight',**kwargs);outputs.append(p)
  plt.close(fig)
 # Diagram: actual projected writes, not a context-space norm control.
 fig,ax=plt.subplots(figsize=(13.8,7.1));ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
 def box(x,y,w,h,s,color='#edf2f7'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor=color,edgecolor='#8b98aa',linewidth=1.1));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=10,linespacing=1.4)
 def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,color='#4a5568',linewidth=1.5))
 ax.text(.1,6.65,'T20 | Match the change in the actual L10 attention write',fontsize=18,weight='bold')
 box(.2,3.1,2.4,1.25,'Same incoming state\none boundary / query\n000: native V₀\n111: zero V₀')
 box(3.2,4.4,3.4,1.25,'Baseline context → actual c_proj\nBaseline write y₀\n(includes bias + FP32 rounding)')
 box(3.2,1.6,3.4,1.8,'Two same-state full edits\nPC removal  /  fixed orthogonal\ncontexts → actual c_proj\nFull writes yPC and yorth')
 arrow(2.7,4,3.1,4.9);arrow(2.7,3.4,3.1,2.5)
 box(7.3,3.05,3.1,2.65,'CPU FP64, per query\nΔj = yj − y₀; nj = ‖Δj‖₂\nt = min(nPC, north)\nλj = t / nj   (0 if nj = 0)\nŷj = FP32(y₀ + λj Δj)','#e9f4fb')
 arrow(6.7,4.9,7.2,4.7);arrow(6.7,2.6,7.2,3.7)
 box(11.1,3.5,2.5,1.65,'Actual delivered FP32 write\nDevice copy, byte verified\n→ remaining live network\n→ next-token NLL','#e6f3ed')
 arrow(10.5,4.3,11,4.3)
 box(.2,.45,13.4,1.25,'Byte branches: λ=0 copies y₀; λ=1 copies yj; source query q₀ copies baseline.\nInterior: separate FP64 multiply/add, one FP32 cast.\nAcceptance uses norms of BOTH actual delivered changes, not proposed contexts.','#fff3df')
 ax.text(.2,.08,'One shared frozen map F; one fixed orthogonal direction. No map refit. Matching rescales each full operation toward its own baseline.',fontsize=10,color='#394555')
 save(fig,'01_actual_write_operation')
 # Paired forest with conditional, passage-cluster intervals.
 fig,ax=plt.subplots(figsize=(11.4,7));ys=np.arange(8)[::-1]
 for key,offset,color,marker,label in [('full',.13,'#8b98aa','o','Full operation'),('matched',-.13,'#146b9e','s','Actual-write matched')]:
  means=np.array([r[key]['mean'] for r in rows]);lo=np.array([r[key]['ci95'][0] for r in rows]);hi=np.array([r[key]['ci95'][1] for r in rows])
  ax.errorbar(means,ys+offset,xerr=[means-lo,hi-means],fmt=marker,color=color,ecolor=color,capsize=3,markersize=6,label=label,lw=1.8)
 ax.axvline(0,color='#303b49',lw=1);ax.set_yticks(ys,[f"{r['label']} · {'natural' if r['identity']=='natural' else 'BOS replacement'}" for r in rows]);ax.set_ylim(-.6,7.7)
 ax.grid(axis='x',alpha=.2);ax.set_xlabel('Mean NLL(PC removal) − NLL(fixed orthogonal), nats per target')
 ax.set_title('Native values (000): sign pattern survives actual-write norm matching',loc='left',weight='bold',fontsize=15,pad=16)
 ax.legend(loc='upper left',bbox_to_anchor=(0,-.19),ncol=2,frameon=False)
 fig.text(.18,.025,'Negative: PC removal has lower loss. Positive: fixed orthogonal has lower loss.\nConditional 95% paired bootstrap intervals: 4 passages per source, 2,000 draws; no multiplicity correction.',fontsize=10)
 fig.subplots_adjust(left=.25,bottom=.24,top=.9);save(fig,'02_native_full_vs_matched_forest')
 # Record-level actual quality distribution, not a pooled query distribution.
 fig,ax=plt.subplots(figsize=(11.3,5.8))
 vals={}
 for domain,color,style,label in [('all_nonsource','#b95a20','-','All nonsource queries 1–511'),('primary_scored','#146b9e','--','Primary scored queries 1–510')]:
  x=np.sort([r['domains'][domain]['relative_RMS_gap'] for r in q]);vals[domain]=x.tolist()
  ax.step(x,np.arange(1,129)/128,where='post',lw=2.3,color=color,linestyle=style,label=label)
 ax.axvline(1e-5,color='#9c2332',ls=':',lw=2,label='Frozen acceptance threshold: 10⁻⁵')
 ax.annotate('Maximum across both domains\n7.0766 × 10⁻⁸',xy=(max(vals['primary_scored']),1),xytext=(1.3e-7,.7),arrowprops={'arrowstyle':'->','color':'#394555'},fontsize=11)
 ax.set_xscale('log');ax.set_xlim(1.5e-8,1.6e-5);ax.set_ylim(0,1.06);ax.set_yticks(np.linspace(0,1,5))
 ax.set_xlabel('RMS(actual PC norm − actual orthogonal norm) / RMS(target norm)')
 ax.set_ylabel('Fraction of 128 variant records');ax.grid(alpha=.18)
 ax.set_title('Measured FP32 norm mismatch: every variant passes both RMS guards',loc='left',weight='bold',fontsize=14,pad=15)
 ax.legend(loc='lower right',frameon=False,fontsize=10)
 fig.text(.11,.015,'128 correlated records = 32 inputs × 2 value boundaries × main / suffix. Each point summarizes one query domain.\nThese are approximate norms; individual tiny-target query relative gaps can exceed the RMS threshold.',fontsize=10)
 fig.subplots_adjust(bottom=.24,top=.88);save(fig,'03_actual_norm_quality')
 def fmt(r):return f"{r['mean']:+.6f} [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]"
 table='| Source | Start | Full PC − orth | Write-matched PC − orth |\n|---|---|---:|---:|\n'
 endpoint='| Source | Start | 111 matched PC − orth | (111 − 000) matched contrast |\n|---|---|---:|---:|\n'
 absolute='| Source | Start | Matched PC / native baseline perplexity |\n|---|---|---:|\n'
 for r in rows:
  label=f"{r['label']} | {r['identity']}";table+=f"| {label} | {fmt(r['full'])} | {fmt(r['matched'])} |\n"
  endpoint+=f"| {label} | {fmt(r['endpoint111'])} | {fmt(r['endpoint_difference'])} |\n"
  pp=r['matched_pc_ppl_vs_c'];absolute+=f"| {label} | {pp['value']:.6f} [{pp['ci95'][0]:.6f}, {pp['ci95'][1]:.6f}] |\n"
 timing='| Source | Start | Early targets 2–127 | Later targets 128–511 |\n|---|---|---:|---:|\n'
 difference='| Source | Start | Matched contrast − full contrast |\n|---|---|---:|\n'
 for r in rows:
  label=f"{r['label']} | {r['identity']}"
  timing+=f"| {label} | {fmt(r['early'])} | {fmt(r['later'])} |\n"
  difference+=f"| {label} | {fmt(r['matching_difference'])} |\n"
 starts=[r for r in a['paired_start_interactions'] if r['view']=='full' and r['name']=='bos_minus_natural__000__matched_remove_minus_orth']
 start_table='| Source | BOS-minus-natural matched contrast |\n|---|---:|\n'
 for corpus,label in CORPORA:
  r=next(r for r in starts if r['corpus']==corpus)
  start_table+=f"| {label} | {fmt(r)} |\n"
 write(out/'tables.md',table+'\n'+endpoint+'\n'+absolute+'\n'+timing+'\n'+difference+'\n'+start_table)
 facts={'rows':rows,'paired_start_contrasts':starts,'matching_quality':q,'relative_RMS_values':vals,'threshold':1e-5,'units':'NLL nats per target; norms in model activation units; relative RMS dimensionless','bootstrap_scope':a['statistical_scope']}
 write(out/'figure_data.json',json.dumps(facts,indent=2,allow_nan=False)+'\n')
 outputs.extend([out/'tables.md',out/'figure_data.json'])
 provenance={'status':'draft_figures_generated','builder_sha256':sha(__file__),'inputs':BINDINGS,'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs},'model_forwards':0,'gpu_used':False,'deletions':0,'wall_seconds':time.time()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Saved aggregate tables and actual per-record RMS quality only; does not rerun numerical/projection/readout audits or models.'}
 write(out/'provenance.json',json.dumps(provenance,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':provenance['status'],'output':str(out),'peak_rss_kib':provenance['peak_rss_kib']}))
if __name__=='__main__':main()
