from pathlib import Path
import csv,json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
C=Path(__file__).resolve().parents[1];D=C/'deliverables';plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
r=list(csv.DictReader((D/'query_cross_time.csv').open()));names=[x['name'] for x in json.loads((C/'query_compatibility_addendum.json').read_text())['checkpoints']];labs=['final' if n=='main' else str(int(n.split('-')[0][4:])//1000)+'k' for n in names]
fig,axs=plt.subplots(1,2,figsize=(11.5,5));norm=SymLogNorm(linthresh=1,vmin=-500,vmax=500)
for ax,cond,title in zip(axs,['newline_U','newline_R_edge_H14'],['No earlier newline','Earlier newline with H14 edge removed']):
 z=np.array([[np.median([float(o['contribution']) for o in r if o['readout_checkpoint']==w and o['prefix_checkpoint']==p and o['condition']==cond and o['neuron']=='1710']) for p in names] for w in names]);im=ax.imshow(z,cmap='RdBu_r',norm=norm)
 ax.set_xticks(range(6),labs,rotation=45,ha='right');ax.set_yticks(range(6),labs);ax.set_title(title);ax.set_xlabel('Prefix state checkpoint');ax.set_ylabel('Gate up down readout checkpoint')
 for k in range(6):ax.add_patch(plt.Rectangle((k-.5,k-.5),1,1,fill=False,ec='#333',lw=.8))
fig.subplots_adjust(left=.09,right=.86,top=.84,bottom=.24,wspace=.38);cax=fig.add_axes([.90,.27,.018,.46]);fig.colorbar(im,cax=cax,label='Primary contribution (symlog)');fig.suptitle('Later prefix states and readouts change together',fontsize=14,fontweight='bold');fig.text(.06,.035,'Exploratory native-coordinate hybrids, median of the same four probes. Outlined diagonal: natural checkpoint combinations.\nOff-diagonal states are not natural checkpoints and do not establish a training cause.',fontsize=8,color='#555');fig.savefig(D/'query_compatibility.png',dpi=180);plt.close(fig)
r=list(csv.DictReader((D/'source_token_values.csv').open()));names=[]
for x in r:
 if x['checkpoint'] not in names:names.append(x['checkpoint'])
labs=['final' if n=='main' else str(int(n.split('-')[0][4:])//1000)+'k'+('*' if n in [p['name'] for p in json.loads((C.parent/'phase3/protocol.json').read_text())['checkpoints']] else '') for n in names]
fig,ax=plt.subplots(figsize=(11.5,3.5))
for tok,label in [(510,'The'),(15,'period'),(187,'newline'),(535,'double newline'),(209,'space')]:ax.plot(range(12),[next(float(o['contribution']) for o in r if o['checkpoint']==n and int(o['neuron'])==1710 and int(o['token_id'])==tok) for n in names],'-o',ms=4,label=label)
ax.set_yscale('symlog',linthresh=1);ax.set_xticks(range(12),labs,rotation=40,ha='right');ax.set_ylabel('Primary scalar contribution');ax.grid(alpha=.18);ax.legend(ncol=5,loc='upper center',fontsize=8);ax.set_title('The primary neuron does not acquire all token responses monotonically',fontweight='bold');fig.subplots_adjust(left=.09,right=.97,top=.87,bottom=.23);fig.text(.06,.015,'Five fixed single-token probes; * marks adaptive checkpoints. Unequal training intervals. No claim of corpus-wide coverage.',fontsize=8,color='#555');fig.savefig(D/'primary_source_tokens.png',dpi=180);plt.close(fig)
