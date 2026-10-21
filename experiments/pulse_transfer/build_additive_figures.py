"""Static research figures for reviewed T17 outputs; all prespecified groups shown."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/superweight-matplotlib')
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CORPORA=('wiki_validation','mill','wollstonecraft','optimizers_new')
LABELS=('Wikipedia','Mill','Wollstonecraft','Optimizer code')
FAMILIES=('remove_pc1','orthogonal_control','position')
FLABELS=('PC1 removal','Orthogonal control','Position map')
COLORS={'J':'#176b99','R':'#d4862f','joint_minus_sum':'#584b82'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 data=json.loads((a.run/'additive_summary.json').read_text());ix={(r['corpus'],r['identity'],r['family'],r['view'],r['metric']):r for r in data}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfcfd','axes.facecolor':'#fbfcfd'})
 def save(fig,n):
  for ext in('png','svg'):fig.savefig(a.out/(n+'.'+ext),dpi=160,bbox_inches='tight')
  plt.close(fig)
 fig,axs=plt.subplots(4,2,figsize=(12,12),sharex=True,constrained_layout=True)
 for j,(c,label)in enumerate(zip(CORPORA,LABELS)):
  for k,i in enumerate(('natural','bos')):
   ax=axs[j,k];x=np.arange(3)
   for n,m in enumerate(('J','R','joint_minus_sum')):
    rows=[ix[c,i,f,'full',m]for f in FAMILIES];v=np.array([r['mean']for r in rows]);ci=np.array([r['ci95']for r in rows]).T
    ax.errorbar(x+(n-1)*.2,v,yerr=np.maximum(0,np.stack([v-ci[0],ci[1]-v])),fmt='o',capsize=3,color=COLORS[m],label={'J':'J: joint − additive loss','R':'R: nonlinear loss term','joint_minus_sum':'N: joint − sum'}[m])
   ax.axhline(0,color='#6b7280',lw=.6);ax.set_title(label+' · '+i);ax.set_ylabel('Mean ΔNLL (nats)');ax.set_xticks(x,FLABELS,fontsize=9);ax.grid(axis='y',alpha=.15)
 axs[0,0].legend(fontsize=8);fig.suptitle('Finite output effects: N = J + R\nFour paired passages per group; conditional 95% bootstrap intervals',fontsize=15)
 save(fig,'additive_decomposition')
 fig,axs=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
 metrics=('agreement_joint_additive','agreement_p_additive','agreement_native_additive')
 for ax,m,title in zip(axs,metrics,('Additive vs joint','Additive vs zero-V P','Additive vs native C')):
  vals=np.array([[ix[c,i,f,'full',m]['mean']*100 for f in FAMILIES]for c in CORPORA for i in('natural','bos')])
  im=ax.imshow(vals,vmin=0,vmax=100,cmap='Blues',aspect='auto');ax.set_title(title);ax.set_xticks(range(3),FLABELS,rotation=25,ha='right');ax.set_yticks(range(8),[l+' '+i for l in LABELS for i in('natural','BOS')])
  for(y,x),v in np.ndenumerate(vals):ax.text(x,y,f'{v:.1f}%',ha='center',va='center',color='white'if v>60 else'#111827')
 fig.colorbar(im,ax=axs,shrink=.8,label='Mean top-1 agreement (%)');fig.suptitle('Prediction fidelity is distinct from loss',fontsize=15);save(fig,'additive_fidelity')
 fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
 for ax,m,title in zip(axs,('kl_joint_additive','centered_logit_rms'),('KL(joint || additive)','Centered logit discrepancy, RMS')):
  for k,f in enumerate(FAMILIES):
   rs=[ix[c,i,f,'full',m]for c in CORPORA for i in('natural','bos')];x=np.arange(8)+(k-1)*.19;v=np.array([r['mean']for r in rs]);ci=np.array([r['ci95']for r in rs]).T
   ax.errorbar(x,v,yerr=np.maximum(0,np.stack([v-ci[0],ci[1]-v])),fmt='o',capsize=2,label=FLABELS[k])
  ax.set_xticks(range(8),[l+' '+i for l in LABELS for i in('N','B')],rotation=35,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_ylabel('Mean per query')
 axs[0].legend(fontsize=8);fig.suptitle('Direct output disagreement, with paired uncertainty',fontsize=15);save(fig,'additive_disagreement')
 fig,ax=plt.subplots(figsize=(12,4.5));ax.set_xlim(0,12);ax.set_ylim(0,4.5);ax.axis('off')
 def box(x,y,w,h,text,color='#e4f0f6'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#5b7282'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
 box(.3,2.7,3.1,1.1,'Eight singleton runs\nδₗ = z(single l) − z(P)')
 box(4.3,2.7,3.1,1.1,'Additive output diagnostic\nz(add) = z(P) + Σ δₗ')
 box(8.3,2.7,3.1,1.1,'Actual joint intervention\nz(joint)',color='#eeeaf5')
 ax.annotate('',xy=(4.15,3.25),xytext=(3.55,3.25),arrowprops={'arrowstyle':'->','lw':1.7})
 box(3.9,.5,3.9,1.3,'J = L(joint) − L(add)\nR = L(add) − L(P) − Σ ΔLₗ\nN = J + R',color='#fff1df')
 ax.annotate('',xy=(5.7,1.95),xytext=(5.7,2.5),arrowprops={'arrowstyle':'->','lw':1.5})
 ax.annotate('',xy=(7.95,1.3),xytext=(9.8,2.55),arrowprops={'arrowstyle':'->','lw':1.5})
 ax.text(1.7,1.2,'R: target terms cancel\nfor cross-entropy',ha='center',fontsize=10)
 ax.text(10,1.0,'Also measure KL,\nagreement and centered\nlogit-vector discrepancy',ha='center',fontsize=10)
 ax.set_title('An output diagnostic; J and R do not identify internal mediators',fontsize=14);save(fig,'additive_method')
if __name__=='__main__':main()
