"""Render preserved native-only diagnostic; no model or experiment edits."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'native_prefix_diagnostic_run');ap.add_argument('--out',type=Path,default=ROOT/'figures11');args=ap.parse_args()
rows=json.loads((args.run/'comparisons.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
colors={'default':'#2167a2','math':'#b05425'}
for backend,label in [('default','Default memory-efficient SDPA'),('math','Forced math SDPA')]:
 r=next(x for x in rows if x['reference']==backend+'__short128' and x['trial']==backend+'__full512')
 values=[r['layer_states'][f'L{i}.output']['max_abs']for i in range(12)]
 axes[0].plot(range(12),values,'o-',color=colors[backend],label=label,ms=4)
 axes[1].plot(range(12),[r['layer_states'][f'L{i}.output']['relative_l2']for i in range(12)],'o-',color=colors[backend],ms=4)
axes[0].set_yscale('log');axes[1].set_yscale('log')
axes[0].set_ylabel('Maximum absolute residual difference')
axes[1].set_ylabel('Residual relative L2 difference')
for ax in axes:
 ax.set_xlabel('Block output, layer (zero-based)');ax.set_xticks(range(12));ax.grid(alpha=.15)
 ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8,loc='upper left')
fig.suptitle('Same 128-token prefix, two execution lengths: one native GPT-2 input',fontsize=12)
out=args.out;out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'prefix_numerics.png',dpi=170);fig.savefig(out/'prefix_numerics.svg');plt.close(fig)
