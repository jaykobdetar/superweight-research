from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';a=json.loads((O/'source_decomposition.json').read_text())[:5];c=json.loads((O/'cancellation.json').read_text())[:5]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(10,8),gridspec_kw={'height_ratios':[1.2,1]},layout='constrained')
x=np.arange(5);labs=['The (510)','Period (15)','Newline (187)','Double newline (535)','Space (209)'];ax=axs[0]
ax.bar(x-.15,[r['negative_other_sum'] for r in c],width=.3,label='Negative alternate-neuron writes',color='#235A82');ax.bar(x+.15,[r['positive_other_sum'] for r in c],width=.3,label='Positive alternate-neuron writes',color='#DDA259')
ax.scatter(x,[r['remaining_axis'] for r in a],marker='D',s=42,color='#222222',label='Net axis including incoming residual',zorder=3);ax.axhline(0,color='#8B9296',linewidth=.8);ax.set_xticks(x,labs);ax.set_ylabel('Centered coordinate-1764 axis projection');ax.set_title('Alternate neurons dominate the surviving coordinate-1764 write',loc='left',fontweight='bold',pad=14);ax.grid(axis='y',alpha=.15);ax.set_ylim(-13,10.2);ax.legend(loc='upper left',frameon=False,fontsize=9);ax.annotate('Nearly exact cancellation\n−7.919 + 7.795',xy=(2,-.10),xytext=(2.55,4.8),arrowprops={'arrowstyle':'->','color':'#444'},fontsize=10)
ax=axs[1]
for dx,key,label,col in [(-.24,'clean_alignment','Native','#235A82'),(0,'row_alignment','Delete only (1764,1710)','#DDA259'),(.24,'column_alignment','Delete whole neuron 1710','#448878')]:ax.bar(x+dx,[r[key] for r in a],width=.23,label=label,color=col)
ax.axhline(0,color='#8B9296',linewidth=.8);ax.set_xticks(x,labs);ax.set_ylim(-1.05,.38);ax.set_ylabel('Signed alignment with coordinate 1764');ax.set_title('Scalar-edge and whole-column removal produce different source geometry',loc='left',fontweight='bold',pad=14);ax.grid(axis='y',alpha=.15);ax.legend(loc='upper center',frameon=False,fontsize=9,ncol=3)
fig.suptitle('OLMo-1B-0724-hf: local production, not a downstream rescue test',fontsize=15,fontweight='bold')
fig.supxlabel('Five cached source-position-0 inputs; pinned FP32 L1 MLP replay. No later-layer inference.',fontsize=10)
fig.savefig(O/'signal_production.png',dpi=190);fig.savefig(O/'signal_production.pdf')
