from pathlib import Path
import csv,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'deliverables/source_cross_time.csv')));names=sorted({r['readout_checkpoint'] for r in rows},key=lambda n:10**10 if n=='main' else int(n.split('-')[0][4:]));labels=[str(int(n.split('-')[0][4:])//1000)+'k' if n!='main' else 'pinned final' for n in names]
fig,axs=plt.subplots(1,2,figsize=(11.5,4.8));norm=SymLogNorm(linthresh=1,vmin=-300,vmax=300)
for ax,tok,label in zip(axs,[510,15],['The','period']):
 a=np.array([[float(next(r['contribution'] for r in rows if r['readout_checkpoint']==w and r['prefix_checkpoint']==h and int(r['neuron'])==1710 and int(r['token_id'])==tok)) for h in names] for w in names]);im=ax.imshow(a,cmap='RdBu_r',norm=norm,aspect='equal');ax.set_xticks(range(len(names)),labels,rotation=45,ha='right',fontsize=8);ax.set_yticks(range(len(names)),labels,fontsize=8);ax.set_title('Single-token '+label);ax.set_xlabel('Prefix state checkpoint');ax.set_ylabel('Gate/up/down readout checkpoint')
 for i in range(len(names)):
  ax.plot([i-.48,i+.48,i+.48,i-.48,i-.48],[i-.48,i-.48,i+.48,i+.48,i-.48],color='black',lw=.65)
fig.subplots_adjust(left=.08,right=.87,bottom=.29,top=.84,wspace=.35);cax=fig.add_axes([.9,.25,.018,.52]);fig.colorbar(im,cax=cax,label='Primary scalar contribution (symlog)')
fig.suptitle('Cross-time readouts are not interchangeable',fontsize=15,fontweight='bold');fig.text(.06,.025,'Exploratory native-coordinate hybrids, not training interventions. Outlined diagonal: natural checkpoint combinations.\nOff-diagonal cells combine an earlier/later prefix state with another checkpoint’s selected neuron readout.',fontsize=9,color='#555');fig.savefig(R/'deliverables/source_cross_time.png',dpi=180);plt.close(fig)
