"""Render all predeclared PCA/QR ranks; no outcome-based condition selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'gpt2_gate_fixed512_run');ap.add_argument('--out',type=Path,default=ROOT/'figures11');args=ap.parse_args()
RUN=args.run
assert json.loads((RUN/'audit.json').read_text())['status']=='passed'
rows=json.loads((RUN/'unexposed_summary.json').read_text())
corpora=['wiki_validation','dubois','hobbes','optimizers']
labels=['Wiki validation (n=3)','Du Bois (n=4)','Hobbes (n=4)','Optimizer code (n=4)']
conditions=[('pca_keep','PCA retained','#236aab','-'),('qr_keep','Random retained','#236aab','--'),
            ('pca_remove','PCA removed','#b65030','-'),('qr_remove','Random removed','#b65030','--')]
def row(c,i,t,n=512):
    return next(r for r in rows if (r['corpus'],r['identity'],r['trial'],r['length'])==(c,i,t,n))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
for metric,ci,ylabel,name in [('p_ppl_ratio','p_ppl_ratio_ci95','Perplexity / zero-V reference P','gate_rank_loss'),
                             ('p_mean_agreement',None,'Argmax agreement with zero-V P','gate_rank_agreement')]:
    fig,axes=plt.subplots(4,2,figsize=(10.5,11))
    for a,(c,label) in enumerate(zip(corpora,labels)):
        for b,identity in enumerate(['natural','bos']):
            ax=axes[a,b]
            for trial,title,color,ls in conditions:
                rr=[row(c,identity,trial+str(k))for k in (1,3,6)]
                yy=np.array([r[metric]for r in rr])
                if ci:
                    bounds=np.array([r[ci]for r in rr]);err=np.maximum(0,np.stack([yy-bounds[:,0],bounds[:,1]-yy]))
                    ax.errorbar([1,3,6],yy,yerr=err,fmt='o',linestyle=ls,color=color,label=title,capsize=2,ms=4)
                else:ax.plot([1,3,6],yy,'o',linestyle=ls,color=color,label=title,ms=4)
            ax.axhline(row(c,identity,'position')[metric],color='#786396',ls=':',lw=1.3,label='Position-only')
            ax.axhline(1,color='#687681',lw=.6,alpha=.7)
            ax.set_title(label+' · '+identity,fontsize=10)
            ax.set_xlabel('Number of directions retained / removed')
            ax.set_ylabel(ylabel);ax.set_xticks([1,3,6]);ax.grid(axis='y',alpha=.15)
            if ci:ax.set_yscale('log')
            else:ax.set_ylim(0,1.03)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,ncol=5,loc='upper center',bbox_to_anchor=(.5,.963),fontsize=9)
    fig.suptitle('512-token inputs; fixed unexposed-passage view'+ ('; paired bootstrap 95% intervals'if ci else ''),fontsize=11,y=.994)
    fig.tight_layout(rect=(0,0,1,.93))
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)
