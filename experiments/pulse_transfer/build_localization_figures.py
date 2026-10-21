"""Plot every predeclared layer/family; condition selection never uses outcomes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'gpt2_gate_localization_run');ap.add_argument('--out',type=Path,default=ROOT/'figures12');args=ap.parse_args()
assert json.loads((args.run/'audit.json').read_text())['status']=='passed'
rows=json.loads((args.run/'summary.json').read_text());nonadd=json.loads((args.run/'nonadditivity_summary.json').read_text())
corpora=['wiki_validation','dubois','hobbes','optimizers'];names=['Wiki validation (n=4)','Du Bois (n=4)','Hobbes (n=4)','Optimizer code (n=3)']
identities=['natural','bos'];sets=[f'L{l}'for l in range(4,12)]+['all'];x=np.array(list(range(8))+[9])
families=[('remove_pc1','Remove PC1','#236aab'),('orthogonal_control','Matched orthogonal control','#aa7532'),('position','Mean + log(position)','#8c4491')]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});args.out.mkdir(parents=True,exist_ok=True)
def row(c,i,t):return next(r for r in rows if(r['corpus'],r['identity'],r['trial'])==(c,i,t))
def save(fig,name):
 fig.savefig(args.out/(name+'.png'),dpi=165);fig.savefig(args.out/(name+'.svg'));plt.close(fig)
for metric,ci,ylabel,filename in [('p_ppl_ratio','p_ppl_ratio_ci95','Perplexity / zero-V reference P','localization_loss'),('p_mean_agreement',None,'Argmax agreement with reference P','localization_agreement')]:
 fig,axes=plt.subplots(4,2,figsize=(11,11))
 for a,(c,name)in enumerate(zip(corpora,names)):
  for b,i in enumerate(identities):
   ax=axes[a,b]
   for fi,(family,label,color)in enumerate(families):
    rr=[row(c,i,family+'_'+s)for s in sets];yy=np.array([r[metric]for r in rr]);xx=x+(fi-1)*.12
    if ci:
     bounds=np.array([r[ci]for r in rr]);err=np.maximum(0,np.stack([yy-bounds[:,0],bounds[:,1]-yy]));ax.errorbar(xx,yy,yerr=err,fmt='o',color=color,label=label,capsize=2,ms=3)
    else:ax.plot(xx,yy,'o',color=color,label=label,ms=3)
    ax.plot(xx[:8],yy[:8],color=color,lw=.7,alpha=.5)
   ax.axvline(8,color='#9da5aa',ls=':',lw=.8);ax.axhline(1,color='#687681',lw=.6);ax.set_xticks(x);ax.set_xticklabels(sets);ax.set_title(name+' · '+i);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
   if ci:
    ax.set_yscale('log');ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
   else:ax.set_ylim(min(.75,min(row(c,i,f+'_'+s)[metric]for f,_,_ in families for s in sets)-.02),1.005)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.966))
 fig.suptitle('Singleton layers and joint intervention; '+('paired bootstrap 95% intervals'if ci else 'means'),y=.995);fig.tight_layout(rect=(0,0,1,.94));save(fig,filename)
fig,axes=plt.subplots(4,2,figsize=(10,9))
for a,(c,name)in enumerate(zip(corpora,names)):
 for b,i in enumerate(identities):
  ax=axes[a,b]
  for fi,(family,label,color)in enumerate(families):
   r=next(r for r in nonadd if(r['corpus'],r['identity'],r['family'])==(c,i,family));y=r['mean_delta_nll'];lo,hi=r['ci95_delta_nll'];ax.errorbar(fi,y,yerr=np.maximum(0,[[y-lo],[hi-y]]),fmt='o',color=color,capsize=3)
  ax.axhline(0,color='#687681',lw=.8);ax.set_xticks(range(3));ax.set_xticklabels(['PC1 removal','Orthogonal','Position']);ax.set_title(name+' · '+i);ax.set_ylabel('Joint − Σ singleton ΔNLL');ax.grid(axis='y',alpha=.15)
fig.suptitle('Descriptive nonadditivity, not a mediation estimate; paired 95% intervals',y=.995);fig.tight_layout(rect=(0,0,1,.97));save(fig,'localization_nonadditivity')
fig,ax=plt.subplots(figsize=(10.5,5.8));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
box=dict(boxstyle='round,pad=.5',fc='#edf3f7',ec='#708898')
ax.text(.2,5.3,'Reference P: source V at position 0 is zero in L4–L11',fontsize=13,weight='bold')
ax.text(1.6,3.9,'Current queries and keys\n\nb = log(non-source mass / source mass)',ha='center',va='center',bbox=box,fontsize=10)
ax.text(5.2,3.9,'One selected layer, or all eight\n\ns = (b − log(q) − α) · u\n\nPC1: b′ = b − s u\nControl: b′ = b − s v\nPosition: b′ = α + log(q)',ha='center',va='center',bbox=box,fontsize=10)
ax.text(8.6,3.9,'a′ = sigmoid(b′)\n\nc′ = c × (a′ / a)',ha='center',va='center',bbox=box,fontsize=11)
for a,b in [(3.1,3.5),(6.9,7.7)]:ax.annotate('',xy=(b,3.9),xytext=(a,3.9),arrowprops={'arrowstyle':'->','lw':1.5})
ax.text(.2,1.65,'u and v are frozen, orthogonal unit vectors. Both directional arms use their own current s.\nAt a singleton layer, the incoming state is identical to P; joint arms may diverge upstream.\nMatching logit norms does not match mass changes, context changes, or prediction changes.',fontsize=10,va='center')
ax.text(.2,.45,'Query 0 and the current conditional non-source value mixture are preserved at each patched boundary.\nDownstream layers remain free to respond. All runs use 512 tokens; C is the separate native reference.',fontsize=10,va='center')
fig.tight_layout();save(fig,'localization_mechanism')
print('Saved four figures in',args.out)
