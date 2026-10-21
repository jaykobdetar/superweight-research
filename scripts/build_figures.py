"""Build original explanatory diagrams and scientific plots from exported tables."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(14,7));ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
boxes=[(0.3,4.7,3.5,1.2,'Earlier newline K and V\nL0/H14 at later query q\nSingle-edge + donor controls'),(5.1,4.7,3.5,1.2,'Layer-0 MLP output M0\nConditional relay\nLayer-1 attention amplifies'),(10.0,4.7,3.5,1.2,'Layer-1 neuron1710\nSiLU(gate) × up\nPulse at coordinate1764'),(10.0,1.4,3.5,1.4,'Source vector geometry\nRow vs entire-column removal\nAxis alignment + remainder'),(5.1,1.4,3.5,1.4,'Downstream source K / V\nCoupled donor compatibility\nContext-dependent routing'),(0.3,1.4,3.5,1.4,'Later-token predictions\nLoss, KL, argmax agreement\nGeneration checks')]
for x,y,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor='#edf4fa',edgecolor='#245979',linewidth=1.5));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
for start,end in [((3.95,5.3),(4.95,5.3)),((8.75,5.3),(9.85,5.3)),((11.75,4.55),(11.75,2.95)),((9.85,2.1),(8.75,2.1)),((4.95,2.1),(3.95,2.1))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':2,'color':'#245979'})
ax.text(7,6.7,'A scoped causal account in OLMo-1B-0724-hf',ha='center',fontsize=18,weight='bold');ax.text(7,.55,'Arrows summarize conditional interventions; they do not establish a unique minimal circuit.\nThe separate empty-history L1/H12 route does not transfer the newline-history pulse state.',ha='center',fontsize=11);fig.savefig(OUT/'mechanism.png',dpi=160,bbox_inches='tight');plt.close(fig)
def read(name):return list(csv.DictReader((ROOT/'results/pulse_continuation'/name/'summary.csv').open()))
a=read('same_layer_pair_run');cs=['wiki_validation','dubois','hobbes','optimizers'];labs=['Wiki','Du Bois','Hobbes','Code'];fig,ax=plt.subplots(figsize=(8,4.5));xs=np.arange(4)
for offset,t,label,color in [(-.19,'first_row_second_clean','One output coordinate','#b64842'),(.19,'first_column_second_clean','Entire neuron output','#28758b')]:
 rs=[next(r for r in a if r['group']=='pair_'+c and r['trial']==t) for c in cs];v=np.array([float(r['ppl_ratio']) for r in rs]);lo=np.array([float(r['ratio_low']) for r in rs]);hi=np.array([float(r['ratio_high']) for r in rs]);ax.bar(xs+offset,v,.36,label=label,color=color,yerr=[v-lo,hi-v],capsize=3)
ax.set_yscale('log');ax.set_xticks(xs,labs);ax.set_ylabel('Perplexity ratio to native (log scale)');ax.axhline(1,color='gray',lw=.8);ax.legend(frameon=False);ax.set_title('Source0 deletion: matched inputs and target spans');fig.tight_layout();fig.savefig(OUT/'row_column.png',dpi=160);plt.close(fig)
a=read('source_geometry_run');ts=['row','column','angle_C_direction_R','angle_N_direction_R','angle_R_direction_N','negative_axis_only'];labels=['Row deleted','Column deleted','C alignment / R direction','N alignment / R direction','R alignment / N direction','Negative axis only'];z=np.array([[float(next(r for r in a if r['group']=='first_'+c and r['trial']==t)['ppl_ratio']) for c in cs] for t in ts]);fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(np.log10(z),cmap='magma_r',vmin=0,vmax=3.3,aspect='auto');ax.set_xticks(range(4),labs);ax.set_yticks(range(6),labels)
for i in range(6):
 for j in range(4):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',color='white' if z[i,j]>50 else 'black')
fig.colorbar(im,ax=ax,label='log10 perplexity ratio');ax.set_title('Source geometry donor combinations\nAlignment includes donor mean and radius');fig.tight_layout();fig.savefig(OUT/'geometry.png',dpi=160);plt.close(fig)
print('Built3 figures')
