"""Static, data-bound T18 figures; no model or inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from gate_wiki_storage import read_npz

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'gpt2_gate_wiki_calibration_run'
OUT=ROOT/'figures14';OUT.mkdir(exist_ok=True)
A=json.loads((RUN/'wiki_factorial_analysis.json').read_text())
IX={(r['corpus'],r['identity'],r['view'],r['mode']):r for r in A['summary']}
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
LABELS=('Wiki validation','Utilitarianism','Ethics','Optimizer code')
COLORS={'OO':'#147d92','OF':'#7a4dad','FO':'#cf821b','FF':'#b33e5c','M8':'#66717b'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(13,8),sharey='row')
for row,which in enumerate(('L10','all')):
 for col,start in enumerate(('natural','bos')):
    ax=axes[row,col]
    for j,cell in enumerate(('OO','OF','FO','FF','M8')):
        rows=[IX[c,start,'full','remove_'+cell+'_'+which] for c in CORPORA]
        y=np.array([r['ppl_ratio_p']['point'] for r in rows]);ci=np.array([r['ppl_ratio_p']['ci95'] for r in rows])
        ax.errorbar(np.arange(4)+(j-2)*.13,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o',ms=4,capsize=2,color=COLORS[cell],label=cell)
    ax.axhline(1,color='#454545',lw=1,ls='--');ax.set_xticks(range(4),LABELS,rotation=12)
    ax.set_title(('L10 only' if which=='L10' else 'Joint L4–L11')+' · '+start)
    if col==0:ax.set_ylabel('Perplexity / P reference')
handles,legend_labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,legend_labels,ncols=5,fontsize=10,loc='lower center',bbox_to_anchor=(.5,-.04))
fig.suptitle('Fresh calibration preserves the scoped L10 pattern\nFour paired passages per corpus; conditional 95% bootstrap intervals',y=1.01)
fig.tight_layout();save(fig,'calibration_effects')

fig,axes=plt.subplots(1,2,figsize=(13,4.7),sharey=True)
for ax,which in zip(axes,('L10','all')):
 for k,start in enumerate(('natural','bos')):
  for j,cell in enumerate(('OO','FF')):
   rows=[next(r for r in A['contrasts'] if(r['corpus'],r['identity'],r['view'],r['layer_set'],r['name'])==(c,start,'full',which,'remove_minus_orthogonal_'+cell)) for c in CORPORA]
   y=np.array([r['delta_nll']['mean'] for r in rows]);ci=np.array([r['delta_nll']['ci95'] for r in rows])
   ax.errorbar(np.arange(4)+(k*2+j-1.5)*.14,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o' if start=='natural' else 's',ms=4,capsize=2,color=COLORS[cell],label=cell+' / '+start)
 ax.axhline(0,color='#454545',lw=1,ls='--');ax.set_xticks(range(4),LABELS,rotation=12);ax.set_title('L10 only' if which=='L10' else 'Joint L4–L11')
axes[0].set_ylabel('PC1 removal − orthogonal change\nMean NLL difference (nats/target)');axes[0].legend(fontsize=9)
fig.suptitle('Direction matters, but the control is matched only in gate-logit norm',y=1.01)
fig.tight_layout();save(fig,'calibration_controls')

g=read_npz(RUN/'calibration_geometry.npz')
fig,axes=plt.subplots(1,2,figsize=(11,4))
layers=list(range(4,12))
axes[0].plot(layers,[g[l]['absolute_pc1_cosine']for l in layers],'o-',color=COLORS['OO'])
axes[0].set_ylim(.90,1.003);axes[0].set_ylabel('|Old · fresh PC1|');axes[0].set_xlabel('Layer (zero-based)')
axes[1].bar(np.array(layers)-.16,[g[l]['O_relative_leading_gap']for l in layers],width=.32,label='Old Wiki4',color=COLORS['OO'])
axes[1].bar(np.array(layers)+.16,[g[l]['F_relative_leading_gap']for l in layers],width=.32,label='Fresh Wiki4',color=COLORS['FF'])
axes[1].set_ylabel('(eigenvalue 1 − 2) / eigenvalue 1');axes[1].set_xlabel('Layer (zero-based)');axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1))
fig.suptitle('Stable leading directions in these two fixed calibrations',y=1.01);fig.tight_layout();save(fig,'calibration_geometry')

fig,ax=plt.subplots(figsize=(12,5.5));ax.set_xlim(0,12);ax.set_ylim(0,5.5);ax.axis('off')
def box(x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#5c6e7a'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
box(.2,3.7,3.1,1.2,'Calibration only\nOld Wiki4 / fresh Wiki4\nFit center α and direction u','#d9eef2')
box(4.0,3.7,3.8,1.2,'Independent choices\nOO · OF · FO · FF\nFirst letter: center; second: direction','#eee4f5')
box(8.5,3.7,3.1,1.2,'Held-out evaluation\n16 passages × 2 starts\nAlways 512 positions','#e3ecd9')
box(.2,1.3,3.1,1.3,'P reference\nSource value at position 0\nset to zero in L4–L11','#e7e9ed')
box(4.0,1.3,3.8,1.3,'s = (b − log(q) − α) · u\nPC1 removal: b′ = b − s u\nControl: b′ = b − s v (v orthogonal to u)','#f4e4d2')
box(8.5,1.3,3.1,1.3,'Change allocation for q > 0\na′ = sigmoid(b′)\nc′ = c × a′/a','#e3ecd9')
for start,end in [((3.4,4.3),(3.9,4.3)),((7.9,4.3),(8.4,4.3)),((5.9,3.6),(5.9,2.75)),((3.4,1.95),(3.9,1.95)),((7.9,1.95),(8.4,1.95))]:
 ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#455964','lw':1.8})
ax.text(6,.45,'Local non-source value mixture is preserved. Downstream computation remains live.\nEach control matches logit norm at its own incoming state, not context change or loss.',ha='center',va='center',fontsize=11)
ax.set_title('What the calibration factorial actually intervenes on',fontsize=16,pad=10);save(fig,'calibration_method')
print('Built four T18 figure pairs')
