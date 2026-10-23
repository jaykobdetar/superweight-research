from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent;O=R/'results'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
data=json.loads((R/'m0_donors/donor_summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),sharey=True)
treatments=['sham_reproject','single1091','other_top3','top4','full_m0_output']
labels=['Native\n(reprojected)','1091\nonly','Other 3\nonly','All 4','Whole MLP\noutput']
for ax,world,title in zip(axes,['R','E'],['Induce the pulse in the history-suppressed world','Suppress the pulse in the H14-edge-blocked world']):
    for i,t in enumerate(treatments):
        for split,marker,color,shift in [('discovery','o','#176C9C',-.07),('context_holdout','^','#D67B20',.07)]:
            ys=[-x['primary_raw_write'] for x in data if x['world']==world and x['treatment']==t and x['split']==split]
            ax.scatter(np.full(len(ys),i+shift),ys,marker=marker,color=color,s=32,alpha=.8,
                       label='Discovery contexts' if i==0 and split=='discovery' else ('Frozen context holdouts' if i==0 else None))
    ax.axhline(0,color='#888888',lw=.8);ax.set_xticks(range(5),labels);ax.set_title(title,fontsize=11)
    ax.set_xlabel('Opposite-world donor activation(s), at L0 / q=128',labelpad=10)
axes[0].set_ylabel('Negative of L1 primary write (−write)\n(larger = stronger negative pulse)')
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('A small L0 neuron group conditionally carries the H14 pulse',fontsize=13,y=.99)
fig.text(.5,.005,'Four neurons: 1091, 3419, 6535, 4798. Prefix causal tests, not downstream loss or a unique minimal-circuit claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.95))
for ext in ('png','pdf'):fig.savefig(O/('m0_causal_carrier.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)

data=json.loads((O/'downstream_summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.4,4.2))
names=['dubois__32768__natural187','optimizers__12000__natural187']
treatments=['drop1710','newline_positive_axis_remove_with1710','newline_positive_complement_remove_with1710','newline_positive_drop_with1710']
labels=['1710 column\nremoved','Also remove\ntrio axis','Also remove\ntrio complement','Also remove\nwhole trio']
for ax,name,title in zip(axes,names,['Du Bois prefix','Optimizer-code prefix']):
    choose=lambda t:next(x for x in data if x['panel']==name and x['scope']=='after_source' and x['treatment']==t)
    ys=[choose(t)['mean_nll'] for t in treatments]
    bars=ax.bar(range(4),ys,color=['#8A939E','#6494B5','#DDA369','#1C8078'],width=.66)
    ax.axhline(choose('native_no_hook')['mean_nll'],color='#353535',ls='--',lw=1,label='Native model')
    for b,y in zip(bars,ys):ax.text(b.get_x()+b.get_width()/2,y+.06,f'{y:.3f}',ha='center',fontsize=9)
    ax.set_xticks(range(4),labels);ax.set_ylim(0,max(ys)+.65);ax.set_title(title);ax.set_ylabel('Mean next-token NLL (lower is better)')
    ax.legend(frameon=False,fontsize=9,loc='upper right')
fig.suptitle('Newline-source rescue needs a coupled direction change',fontsize=13,y=.99)
fig.text(.5,.005,'Same 254 targets in each prefix. Trio: 2476, 2637, 7845. Axis-only removal is not sufficient for the joint rescue.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.95))
for ext in ('png','pdf'):fig.savefig(O/('cancellation_behavior_controls.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
