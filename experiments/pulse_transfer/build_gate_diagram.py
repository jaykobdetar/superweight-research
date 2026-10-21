"""Explanatory diagram of the prospective, query-local gate intervention."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'figures11');args=ap.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(12,5.5),constrained_layout=True);ax.set_xlim(0,12);ax.set_ylim(0,5.5);ax.axis('off')
def box(x,y,w,h,label,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',linewidth=1,edgecolor='#8d9ca8',facecolor=color))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
def arrow(a,b,dashed=False):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#465b69','lw':1.5,'linestyle':'--'if dashed else'-'})
ax.text(.05,5.18,'Testing gate dimensions in a fixed GPT-2 checkpoint',fontsize=15,weight='bold')
box(.15,2.9,2.7,1.1,'Current causal Q/K scores\n12 heads at one query\nAllocation logit b; mass a', '#e8f1f8')
box(3.45,2.9,3.05,1.1,'r = b − log(q); center r − α\nRetain / remove fixed directions\nb̂ = α + log(q) + residual map', '#eef0fa')
box(7.1,2.9,2.0,1.1,'Mapped gate\nâ = sigmoid(b̂)\n(or fixed mean mass)', '#eaf4ec')
box(9.7,2.9,2.1,1.1,'Patched context\nc′ = c × (â / a)\nbefore output projection', '#faf0e4')
arrow((2.93,3.45),(3.37,3.45));arrow((6.58,3.45),(7.02,3.45));arrow((9.18,3.45),(9.62,3.45))
box(3.45,1.05,3.05,1.15,'Calibration means and PCA\nFixed-seed QR controls\nNo evaluation fitting', '#f0f1f2')
arrow((4.98,2.28),(4.98,2.82),True)
box(7.1,1.05,4.7,1.15,'Source V₀ = 0, so current context c = a × u\nu: current conditional non-source value mixture\nThe patch changes its scalar gain within each head', '#f8ebee')
arrow((10.72,2.28),(10.72,2.82))
ax.text(.15,.35,'Queries 1+ in layers 4–11; source query preserved relative to the zero-V baseline. Later layers use their evolving states.',fontsize=10)
ax.text(.15,4.6,'Matched controls: fixed means, position-only map, random subspaces, exact copies, and changed future suffixes.',fontsize=10)
fig.savefig(out/'gate_intervention.png',dpi=170);fig.savefig(out/'gate_intervention.svg');plt.close(fig)
