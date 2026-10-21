"""Query allocation/conditional-mixture report with self-contained figures."""
from pathlib import Path
import base64,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parent;FIG=ROOT/'figures10';FIG.mkdir(exist_ok=True)
RUNS=['gpt2_query_source_run','gpt2_query_mixture_run'];S={r:json.loads((ROOT/r/'summary.json').read_text()) for r in RUNS};M={r:json.loads((ROOT/r/'manifest.json').read_text()) for r in RUNS}
CS=['wiki_validation','dubois','hobbes','optimizers'];NAMES=['Wiki validation','Du Bois','Hobbes','Optimizer code']
def row(run,c,t,i='natural'):return next(r for r in S[run] if (r['corpus'],r['trial'],r['identity'])==(c,t,i))
def table(run,conditions,i='natural',metric='ppl_ratio'):
    text='| Intervention | '+' | '.join(NAMES)+' |\n|---|'+'---:|'*4+'\n'
    for t,label in conditions:text+='| '+label+' | '+' | '.join(f'{row(run,c,t,i)[metric]:.5g}' for c in CS)+' |\n'
    return text
def errorplot(run,conditions,name,title):
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,ident in zip(axes,['natural','bos']):
        for k,(trial,label) in enumerate(conditions):
            rr=[row(run,c,trial,ident) for c in CS];y=np.array([r['ppl_ratio'] for r in rr]);ci=np.array([r['ppl_ratio_ci95'] for r in rr])
            ax.errorbar(np.arange(4)+(k-(len(conditions)-1)/2)*.13,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=2,ms=4,label=label)
        ax.axhline(1,c='#687985',ls='--',lw=.8);ax.set_xticks(range(4),NAMES,rotation=15,ha='right');ax.set_title(title+': '+ident,fontsize=11)
        ax.set_ylabel('Perplexity / same-input native');ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.savefig(FIG/name,dpi=170);plt.close(fig)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
errorplot(RUNS[0],[('C_zero_present_causal','Bias present, zero V'),('C_zero_removed_causal','Bias removed, zero V'),('C_zero_present_blocked','Source blocked, zero V')],'query_loss.png','Query bias and source access')
conditions=[('P_GP_DP','P mass / P mixture'),('P_GP_DR','P mass / R mixture'),('P_GR_DP','R mass / P mixture'),('P_GR_DR','R mass / R mixture')]
errorplot(RUNS[1],conditions,'mixture_loss.png','Attention-boundary factorial')
att=json.loads((ROOT/RUNS[0]/'attention_summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
for ax,identity in zip(axes,['natural','bos']):
    for trial,label in [('C_zero_present_causal','Bias present'),('C_zero_removed_causal','Bias removed'),('C_zero_random_causal','Matched random subtraction')]:
        values=[np.mean([r['mean_source_mass'] for r in att if r['identity']==identity and r['trial']==trial and r['layer']==l]) for l in range(4,12)]
        ax.plot(range(4,12),values,'o-',label=label)
    ax.set_ylim(0,1);ax.set_xlabel('Layer (zero-based)');ax.set_ylabel('Mean probability on source token');ax.set_title('Native upstream, zero source V: '+identity);ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.savefig(FIG/'query_mass.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4.5),constrained_layout=True);ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,4.5)
def box(x,y,t,color):ax.text(x,y,t,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.55',fc=color,ec='#8d9eac'))
def arrow(x,y,a,b):ax.annotate('',xy=(a,b),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#425867',lw=1.6))
ax.text(.1,4.1,'Separating allocation from conditional value mixing',fontsize=15,weight='bold')
box(2,2.85,'P: learned query bias present\nR: learned query bias removed','#e9f1f8')
box(6,3.1,'Non-source mass a\n(sum of probabilities on j ≥ 1)','#eaf4eb');box(6,1.3,'Conditional mixture u\n(non-source values, renormalized)','#f7ecee')
box(10.3,2.2,'Delivered context\na_G × u_D\nG, D ∈ {P, R}','#f5f0e6')
arrow(4.1,3,4.5,3.1);arrow(4.1,2.6,4.7,1.6);arrow(7.7,3,8.9,2.5);arrow(7.7,1.4,8.9,1.9)
ax.text(.1,.3,'All heads, layers 4–11, query positions 1+; V₀ = 0. Source query unchanged. Fixed donor trajectories.',fontsize=10)
fig.savefig(FIG/'boundary.png',dpi=170);plt.close(fig)
resource='| Batch | Forwards | Wall seconds | Peak GPU GiB | Peak process RSS GiB |\n|---|---:|---:|---:|---:|\n'
for r,m in M.items():resource+=f"| {r} | {m['forward_passes']} | {m['wall_seconds']:.2f} | {m['peak_gpu_allocated_bytes']/2**30:.3f} | {m['peak_process_rss_kib']/2**20:.3f} |\n"
query=(ROOT/'report10_query_findings.md').read_text();mixture=(ROOT/'report10_mixture_findings.md').read_text()
prior='\n'.join((ROOT/'report10_prior_findings.md').read_text().splitlines()[2:]).replace('\n## ','\n### ').replace('](../REPORT8.md)','](REPORT8.md)').replace("AppendixB.4's zero-K/zero-V control deserves explicit mention in the next report because it is closer to the proposed estimand than a generic constant-bias citation.","AppendixB.4's zero-K/zero-V control is particularly close to the present estimand.")
text=f'''# What does GPT-2's query bias change after source values are removed?

Research continuation, 21 October 2026. Two fresh, sequential experiments
follow the reviewed coordinate-regrowth and source-value experiments. The
model is `openai-community/gpt2`, revision
`75e09b43581151bd1d9ef6700faa605df408979f`, FP32 with native SDPA. Indices
are zero-based. These results concern specified activation interventions
under teacher forcing; they do not establish faithful generation, speedup,
a universal mechanism, or a novel discovery of attention sinks.

The main result is a separation of two effects: using the bias-removed value
mixture generally raises loss, whereas its non-source allocation can lower
loss on books and code when paired with the bias-present mixture. Their
combination can hide these opposing changes. This is a causal result for the
defined donor-boundary intervention; it is not a natural mediation percentage.

## Query bias and retained source access

{query}

![Loss effects of query and source interventions](figures10/query_loss.png)

![Allocation of attention to source0](figures10/query_mass.png)

## Separating allocation from the conditional non-source mixture

With source V0=0, attention at a later query is mathematically
`c = a × u`, where `a = Σ(j≥1) p_j` is non-source probability mass and `u`
is the value mixture conditional on attending a non-source token. Query-bias
removal can change both. The next experiment collects donors P (bias present)
and R (bias removed), then crosses their mass and mixture states at every
attention output before c_proj in L4–11. Each pair is applied to both P and R
recipients. The duplicate recipient is a closure control, not another sample.
Upstream coordinates are native; the E/B lesions from the preceding experiment
are not used in this second factorial.

![Attention boundary and donor-factor decomposition](figures10/boundary.png)

The implementation uses actual native SDPA donor context `c_D × (a_G/a_D)`.
Exact diagonals copy `c_D`. Non-source mass is summed directly, avoiding
subtraction from one. Independently computed conditional softmax mixtures
check division and amplified error at every treated layer and query. The
predeclared absolute tolerance is1e-4; no mass clipping or silent kernel
changes are allowed. Query0 remains untouched. Fixed donor contexts may
create a trajectory absent from native computation; the factorial identifies
effects at this defined boundary, not additive natural mediation fractions.

The eighth panel uses fractions(.125,.325,.525,.725), disjoint actual token
spans from all28 prior spans/corpus. There are four fresh512-token passages
per corpus, natural/BOS starts, targets2–511 scored, with the same paired
bootstrap protocol. PPL ratios below are relative to native C on each input.

Natural starts:

{table(RUNS[1],conditions+[('P_blocked','P, source blocked'),('R_blocked','R, source blocked')])}

BOS starts:

{table(RUNS[1],conditions+[('P_blocked','P, source blocked'),('R_blocked','R, source blocked')],'bos')}

![Crossing donor mass and conditional mixture](figures10/mixture_loss.png)

{mixture}

## Prior work and novelty limits

{prior}

The earlier reports' super-weight, massive-activation, conditional-reserve,
coordinate-regrowth and source-K/V comparisons remain applicable. These two
experiments are targeted mechanism tests on one pinned GPT-2 checkpoint.
Different estimands alone do not establish priority.

## Reproducibility and remaining limits

Exact protocols are `GPT2_QUERY_SOURCE.md` and `GPT2_QUERY_MIXTURE.md`.
Scripts, model/input hashes, source revisions, seeds, fresh-panel manifests,
per-token metrics, raw donors, frozen raw attention snapshots and all controls
are retained locally. Private input/prediction IDs, raw activation donors,
weights and copyrighted corpora are excluded from the numerical repository
export. Full vocabulary logits are not retained; saved KL can be reaggregated
but not independently reconstructed from logits. The second experiment also
hashes the complete final normalized hidden state to check exact donor closures.

{resource}

These batches total {sum(m['forward_passes'] for m in M.values())} forwards and
{sum(m['wall_seconds'] for m in M.values()):.2f} recorded worker seconds.
One local RTX3060Ti ran one model at a time. Independent reviews used CPU and
no pretrained model. Small paired panels, one random control direction,
conditional bootstrap intervals, reused corpus sources and intervention-induced
trajectories limit generalization. Low loss and reduced attention allocation
do not imply unchanged outputs or one underlying mechanism. The continuous
research ledger distinguishes established results from prospective follow-ups.

The private GitHub creation was denied before execution. No remote repository
or push occurred; local research and reviewed deliverables remain available.
'''
(ROOT/'REPORT10.md').write_text(text)
render=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
for p in FIG.glob('*.png'):render=render.replace('src="figures10/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
html='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-2 query allocation and value mixture experiments</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#172631;background:#f7f8fa;max-width:1080px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.2}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:14px}td,th{border:1px solid #ccd5dc;padding:7px 10px}th{background:#e7eef3}a{color:#155a8c}code{font-size:.88em;overflow-wrap:anywhere}p{max-width:1000px}</style></head><body>'+render+'</body></html>'
(ROOT/'report10.html').write_text(html);print('Built',len(text),'Markdown chars and',len(html),'HTML chars')
