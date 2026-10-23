from pathlib import Path
import json,csv,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT.parent/'phase2/protocol.json').read_text());out=ROOT/'deliverables';out.mkdir(exist_ok=True)
rs=[]
files=[*sorted((ROOT.parent/'phase2/results').glob('*/float32.json')),*sorted((ROOT.parent/'phase3/results').glob('*/float32.json'))]
assert len(files)==12, 'Wait for all four adaptive checkpoints'
for f in files:
 r=json.loads(f.read_text());r['cohort']='initial' if 'phase2' in f.parts else 'adaptive';r['step']=None if r['name']=='main' else int(r['name'].split('-')[0][4:]);rs.append(r)
rs.sort(key=lambda r:r['step'] if r['step'] is not None else 10**10)
long=[];summary=[]
for r in rs:
 by={(o['panel'],o['condition']):o for o in r['observations']}
 for o in r['observations']:
  for j,position in enumerate(o['positions']):
   for k,neuron in enumerate(P['neurons']):
    long.append({'checkpoint':r['name'],'commit':r['revision'],'selection_cohort':r['cohort'],'panel':o['panel'],'condition':o['condition'],'position':position,'neuron':neuron,'gate':o['gate'][j][k],'up':o['up'][j][k],'activation':o['activation'][j][k],'scalar_contribution':o['contribution'][j][k],'query_H14_attention_to_p64':o['l0_attention_q_to_0_p_q'][14][1],'query_H12_attention_to_source0':o['l1_attention_q_to_0_p_q'][12][0],'pre_mlp1_centered_norm':o['pre_mlp1_centered_norm'][j]})
 for k,neuron in enumerate(P['neurons']):
  row={'checkpoint':r['name'],'commit':r['revision'],'selection_cohort':r['cohort'],'step':r['step'],'neuron':neuron,'single_The_contribution':r['single_tokens']['510']['contribution'][0][k]}
  for cond in P['worlds']:
   values=[by[(p['name'],cond)]['contribution'][-1][k] for p in P['panels']];row[cond+'_median']=float(np.median(values));row[cond+'_min']=min(values);row[cond+'_max']=max(values)
  h14=[];h12=[];h0=[];h14mass=[];h14massu=[]
  for p in P['panels']:
   def v(c):return by[(p['name'],c)]['contribution'][-1][k]
   d=v('newline_U')-v('newline_R');h14.append((v('newline_R_edge_H14')-v('newline_R'))/d if abs(d)>1 else None)
   d=v('empty_isolated')-v('empty_ordinary');h12.append((v('empty_patch_H12')-v('empty_ordinary'))/d if abs(d)>1 else None);h0.append((v('empty_patch_H0')-v('empty_ordinary'))/d if abs(d)>1 else None)
   h14mass.append(by[(p['name'],'newline_R')]['l0_attention_q_to_0_p_q'][14][1]);h14massu.append(by[(p['name'],'newline_U')]['l0_attention_q_to_0_p_q'][14][1])
  for label,vs in [('H14_restoration_fraction',h14),('H12_restoration_fraction',h12),('H0_restoration_fraction',h0)]:
   good=[v for v in vs if v is not None];row[label+'_median']=float(np.median(good)) if good else None;row[label+'_valid_panels']=len(good)
  row['H14_R_attention_p_median']=float(np.median(h14mass));row['H14_U_attention_p_median']=float(np.median(h14massu));summary.append(row)
for name,rows in [('per_probe_values.csv',long),('checkpoint_summary.csv',summary)]:
 if rows:
  with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'checkpoint_summary.json').write_text(json.dumps(summary,indent=2))
print('CHECKPOINTS',len(rs),'PROBE ROWS',len(long))
for r in summary:
 if r['neuron']==1710:print(r['checkpoint'],'source',round(r['single_The_contribution'],5),'U',round(r['newline_U_median'],5),'R',round(r['newline_R_median'],5),'H14edge',round(r['newline_R_edge_H14_median'],5),'attnp',round(r['H14_R_attention_p_median'],4),'H12',r['H12_restoration_fraction_median'])
if len(rs)<2:raise SystemExit()
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8));x=np.arange(len(rs));labs=[str(r['step']//1000)+'k'+('*' if r['cohort']=='adaptive' else '') if r['step'] is not None else 'pinned\nfinal' for r in rs]
colors=['#1767a3','#d47a24','#548b58','#8b5796']
for n,c in zip(P['neurons'],colors):
 ys=[next(r['single_The_contribution'] for r in summary if r['checkpoint']==z['name'] and r['neuron']==n) for z in rs];axs[0,0].plot(x,ys,'-o',ms=4,label=str(n),c=c)
axs[0,0].set_yscale('symlog',linthresh=1);axs[0,0].set_title('Single-token The: signed contributions to row 1764');axs[0,0].set_ylabel('Contribution (symlog)');axs[0,0].legend(title='Neuron',fontsize=8)
primary=[r for r in summary if r['neuron']==1710]
for key,label,c in [('newline_U_median','No earlier newline','#1767a3'),('newline_R_median','Earlier newline','#d47a24'),('newline_R_edge_H14_median','Earlier newline, H14 edge removed','#548b58')]:axs[0,1].plot(x,[r[key] for r in primary],'-o',label=label,c=c)
axs[0,1].set_title('Primary contribution at later newline, median of 4 probes');axs[0,1].set_ylabel('Signed contribution');axs[0,1].legend(fontsize=8)
for key,label,c in [('H14_R_attention_p_median','Earlier newline at p64','#1767a3'),('H14_U_attention_p_median','Space at p64','#d47a24')]:axs[1,0].plot(x,[r[key] for r in primary],'-o',label=label,c=c)
axs[1,0].set_ylim(-.03,1.03);axs[1,0].set_title('L0/H14 attention from q128 to p64');axs[1,0].set_ylabel('Attention mass');axs[1,0].legend(fontsize=8)
for key,label,c in [('H12_restoration_fraction_median','L1/H12 donor patch','#1767a3'),('H0_restoration_fraction_median','L1/H0 control donor patch','#d47a24')]:axs[1,1].plot(x,[r[key] if r[key] is not None else np.nan for r in primary],'-o',label=label,c=c)
axs[1,1].axhline(1,color='#999',ls='--',lw=1);axs[1,1].set_title('Empty-history donor restoration of primary contribution');axs[1,1].set_ylabel('Fraction of isolated minus ordinary gap');axs[1,1].legend(fontsize=8)
for a in axs.flat:a.set_xticks(x,labs,rotation=45,ha='right',fontsize=8);a.grid(alpha=.18);a.set_xlabel('Checkpoint (unequally spaced in training)')
fig.suptitle('OLMo-1B: development of the final-identified early-layer pathways',fontsize=16,fontweight='bold')
fig.text(.045,.015,'Eight initial points plus four adaptive points (*); unchanged synthetic probes, n=4. No loss, gradients, optimizer state or training intervention.\nRestoration fractions omitted when the signed denominator has magnitude <=1. Final is the exact pinned model, not an assigned step.',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(out/'functional_trajectory.png',dpi=170);fig.savefig(out/'functional_trajectory.pdf');plt.close(fig)
# The five source-token probes were fixed before inference. Keep all of them,
# rather than selecting The after seeing its clean mature pulse.
sources=[];tokens={'510':'The','15':'period','187':'newline','535':'double newline','209':'space'}
for r in rs:
 for tid,label in tokens.items():
  o=r['single_tokens'][tid]
  for k,n in enumerate(P['neurons']):
   sources.append({'checkpoint':r['name'],'commit':r['revision'],'selection_cohort':r['cohort'],'step':r['step'],'token_id':int(tid),'token_label':label,'neuron':n,**{v:o[v][0][k] for v in ['gate','up','activation','contribution']}})
with (out/'source_token_values.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(sources[0]));w.writeheader();w.writerows(sources)
fig,axs=plt.subplots(2,2,figsize=(12,8));tokcolors=['#1767a3','#d47a24','#548b58','#8b5796','#b3484e']
for ax,n in zip(axs.flat,P['neurons']):
 for (tid,label),c in zip(tokens.items(),tokcolors):
  ys=[next(z['contribution'] for z in sources if z['checkpoint']==r['name'] and z['neuron']==n and z['token_id']==int(tid)) for r in rs]
  ax.plot(x,ys,'-o',ms=4,label=label,c=c)
 ax.set_yscale('symlog',linthresh=1 if n!=8041 else .001);ax.set_title('Neuron '+str(n));ax.set_ylabel('Signed contribution (symlog)');ax.set_xticks(x,labs,rotation=45,ha='right',fontsize=8);ax.set_xlabel('Checkpoint (unequally spaced in training)');ax.grid(alpha=.18);ax.legend(fontsize=8)
fig.suptitle('The source pulse is token-dependent during development',fontsize=16,fontweight='bold')
fig.text(.045,.015,'Five prespecified single-token inputs; * marks adaptively selected checkpoints. Each panel shows the neuron’s signed contribution to output coordinate 1764.\nAxes have independent scales; the paper-listed neuron 8041 remains tiny here. These five tokens do not establish universality.',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(out/'source_token_specialization.png',dpi=170);fig.savefig(out/'source_token_specialization.pdf');plt.close(fig)
