"""Reviewed GPT-2 follow-up report with embedded scientific figures."""
from pathlib import Path
import base64,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parent;FIG=ROOT/'figures9';FIG.mkdir(exist_ok=True)
RUNS=['gpt2_layer_writes_run','gpt2_regrowth_kv_run','gpt2_value_geometry_run']
S={r:json.loads((ROOT/r/'summary.json').read_text()) for r in RUNS}
M={r:json.loads((ROOT/r/'manifest.json').read_text()) for r in RUNS}
CS=['wiki_validation','dubois','hobbes','optimizers'];NAMES=['Wiki validation','Du Bois','Hobbes','Optimizer code']
def row(run,c,t,i='natural'):return next(r for r in S[run] if (r['corpus'],r['trial'],r['identity'])==(c,t,i))
def table(run,conditions,i='natural',metric='ppl_ratio'):
    text='| Intervention | '+' | '.join(NAMES)+' |\n|---|'+'---:|'*4+'\n'
    for t,label in conditions:
        rr=[row(run,c,t,i) for c in CS]
        text+='| '+label+' | '+' | '.join(f'{r[metric]:.5g}' for r in rr)+' |\n'
    return text
def errorplot(run,conditions,name,title):
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,ident in zip(axes,['natural','bos']):
        for k,(trial,label) in enumerate(conditions):
            rr=[row(run,c,trial,ident) for c in CS];y=np.array([r['ppl_ratio'] for r in rr]);ci=np.array([r['ppl_ratio_ci95'] for r in rr])
            ax.errorbar(np.arange(4)+(k-(len(conditions)-1)/2)*.12,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=2,ms=4,label=label)
        ax.axhline(1,c='#687985',ls='--',lw=.8);ax.set_yscale('log');ax.set_xticks(range(4),NAMES,rotation=15,ha='right')
        ax.set_title(title+': '+ident,fontsize=11);ax.set_ylabel('Perplexity / same-input native');ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.savefig(FIG/name,dpi=170);plt.close(fig)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
layers=[('E1B30BL0','E: erasure only'),('E1B30BL1','Only L3 selected writes allowed'),('E1B31BL0','B: block selected L3 writes'),('E1B31BL1','Block selected writes, L3–11')]
pathways=[('E','E: erasure only'),('B','B: block L3 writes'),('E_KE_VB','E keys, B values'),('E_KB_VE','B keys, E values'),('E_zeroV_currentK','Zero values in E'),('E_mask_blocked','Block source access')]
values=[('B_DE_NE','E direction, E norms'),('B_DE_NB','E direction, B norms'),('B_DB_NE','B direction, E norms'),('B_DB_NB','B direction, B norms'),('B_randomE','Permuted E values')]
errorplot(RUNS[0],layers,'layer_local.png','Source-coordinate MLP writes')
errorplot(RUNS[1],pathways,'source_kv.png','Source K/V pathways')
errorplot(RUNS[2],values,'value_geometry.png','Fixed E keys; V factorial')
fig,ax=plt.subplots(figsize=(12,4.4),constrained_layout=True);ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,4.4)
def box(x,y,text,color):ax.text(x,y,text,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.55',fc=color,ec='#8d9eac'))
def arrow(a,b,c,d):ax.annotate('',xy=(c,d),xytext=(a,b),arrowprops=dict(arrowstyle='->',color='#425867',lw=1.6))
ax.text(.1,4.05,'Tested causal boundaries in GPT-2',fontsize=15,weight='bold')
box(1.5,2.9,'Source token, after L2\nerase coordinates 447/138','#e9f1f8')
box(5.2,2.9,'L3 MLP source writes\ncoordinates 447/138 only\nE: allowed / B: blocked','#f7ecee')
box(10,2.9,'Source K and V in L4–11\npatched independently','#eaf4eb')
arrow(3.1,2.9,3.65,2.9);arrow(6.8,2.9,8.1,2.9)
box(10,1.0,'Later-token outputs\nloss, KL, predictions','#f5f0e6');arrow(10,2.3,10,1.5)
ax.text(.2,1.55,'E and B initially have identical later-token states and L4 queries.',fontsize=10)
ax.text(.2,1.05,'The K/V donor experiment isolates the source-to-suffix route.',fontsize=10)
ax.text(.2,.55,'Keys fixed across value cells; downstream query feedback remains active.',fontsize=10)
fig.savefig(FIG/'causal_path.png',dpi=170);plt.close(fig)
resources='| Batch | Forwards | Wall seconds | GPU peak GiB | Worker RSS GiB |\n|---|---:|---:|---:|---:|\n'
for r,m in M.items():resources+=f"| {r} | {m['forward_passes']} | {m['wall_seconds']:.2f} | {m['peak_gpu_allocated_bytes']/2**30:.3f} | {m['peak_process_rss_kib']/2**20:.3f} |\n"
text=f'''# How GPT-2's conditional coordinate writes affect later predictions

Research continuation, 21 October 2026. Three fresh, sequential batches extend
the six-batch OLMo-to-GPT-2/Llama transfer report. This report concerns the
pinned GPT-2 checkpoint and deliberately perturbed source-token computation.
It does not claim that a natural forward pass detects an error and repairs it.

## What is established, and what remains open

After erasing source coordinates 447 and 138 after layer 2, the layer-3 MLP's
writes to those coordinates protect much of the tested suffix performance.
Their conditional importance reaches later tokens through source K/V states.
Under either donor-key state, the values from the world with those writes
blocked are more damaging than the values from the world with writes allowed.
Zeroing these source values often preserves substantially better loss than
blocking access to the source altogether. This supports a role in preventing
a harmful value contribution under this lesion; it does not establish that
the regrown values encode indispensable linguistic information.

Raw coordinate reappearance does not restore the native trajectory. Native
attention, all other residual coordinates and most MLP computations remain
active throughout. Keys also matter, especially with BOS starts. Fixed source
keys leave downstream queries and attention gates responsive to the intervention.

![Diagram of the intervened causal boundaries](figures9/causal_path.png)

## Protocol and scope

Model: `openai-community/gpt2`, revision
`75e09b43581151bd1d9ef6700faa605df408979f`, FP32, native Transformers SDPA,
12 layers, 12 heads, hidden width 768. Indices are zero-based. No parameters
are modified; before/after hashes and parameter version counters match.

C is native. E erases only coordinates 447/138 at source position 0 after L2.
B performs E and also zeros those two coordinates in the source's L3 MLP
output. E and B first differ after that MLP; their later-token L3 states and
later-token L4 queries are identical. All K/V interventions below span source
position 0 in L4–11, all 12 heads. The final-layer nontransmission control in
the first batch concerns MLP writes after its attention, not L11 values.

Each batch uses four new 512-token passages from each of Wiki validation,
Du Bois, Hobbes and optimizer code, with natural and BOS start conditions.
Replacing the first token leaves every later token unchanged. Targets 2–511
are scored, so the source's immediate prediction is excluded. The fourth,
fifth and sixth panel fractions are respectively (.25,.45,.65,.85),
(0,.05,.9,.95), and (.175,.375,.575,.775). Actual token-character spans are
disjoint from earlier panels; larger extraction buffers can overlap. The
fifth panel includes source edges and can include front matter/code headers.
No post-result content filtering, selection or calibration uses these panels.
The original coordinate selection was frozen before them.

Tables report exp(mean paired NLL difference) relative to each input's native
prediction. Figures show 2,000-draw paired bootstrap intervals, seed 20261021,
four passages per corpus/start group. These are small, conditional panels,
not population-wide confidence or independent pretraining-distribution tests.
No multiple-comparison adjustment is applied. Duplicate recipient and diagonal
arms are closure controls, not extra samples. Per-token losses, KL and argmax
IDs are retained privately; the repository exports numerical reductions and
native-prediction equality masks without token IDs or corpus text.

## 1. The first post-erasure MLP layer has a conditional role

The layer-local factorial crosses erasure, blocking L3's two-coordinate MLP
writes, and blocking those writes in L4–11. Further arms allow only one later
MLP layer to write, or block successively longer prefixes. Random-coordinate
and final-layer controls accompany these interventions.

Natural starts:

{table(RUNS[0],layers)}

BOS starts:

{table(RUNS[0],layers,'bos')}

![Layer-local conditional effects](figures9/layer_local.png)

Blocking L3 after E raises loss in all 32 input/start pairs; all eight group
intervals exclude zero. Without E, blocking L3 alone gives ratios
0.999423–1.000147. Allowing only L3's selected writes leaves performance close
to E, whereas later single writers generally fail to rescue the lesion.
The last MLP layer cannot broadcast to earlier or other tokens: allowing only
its writes exactly matches blocking all later MLP writes on scored loss, KL
and predictions in all 32 pairs. These controls localize a conditional
coordinate-write dependency, not whole-network sufficiency. Agreement with
native predictions remains below 100% even in the low-loss conditions.

## 2. Value differences carry much of the conditional damage

The full E/B source K/V factorial uses each donor-key state with each
donor-value state in both E and B recipients. Diagonal donors reproduce their
world exactly; equal mixed donors give identical suffix results in the two
recipients. Under source blocking the recipients also agree exactly. These
closures follow the causal boundary and validate the implementation.

Natural starts:

{table(RUNS[1],pathways)}

BOS starts:

{table(RUNS[1],pathways,'bos')}

![Source key and value donor factorial](figures9/source_kv.png)

B values increase NLL relative to E values under either donor-key state in
every one of the 32 input/start pairs; all 16 grouped value-effect intervals
exclude zero. This is a causal effect of the tested source-value replacement,
conditional on the key state. Key effects and signed K/V interactions vary
with corpus and starting token; they are not additive mediation fractions.

Zero values are less damaging than complete source blocking in every pair,
both with endogenous keys and with E keys clamped. All 64 explicit causal-mask
references have zero scored loss drift and exact prediction agreement, so
the reference-adjusted comparison agrees with the raw comparison here. Zero
values preserve the source's role in attention normalization; blocking it
redistributes attention over other positions. Thus these operations do not
test the same mechanism. Lower loss under zero values also does not imply
exact output fidelity or that native values have no function.

## 3. Separating projected-value norm and direction

The next batch uses a B recipient for all four main factorial cells, freezes
E source keys and crosses E/B value direction with
E/B norm separately for every layer/head. “B norm” denotes a joint profile;
it is larger in many earlier layers but is not uniformly larger. Exact
diagonals copy original donors. Mixed cells rescale one direction to the
other world's head norm. Downstream queries remain endogenous. A fixed
signed permutation, drawn before execution with seed 20261023, preserves
each head's absolute-value multiset at both original E and B profiles.
Those two controls are separately permuted source vectors, not one common
random direction compared at two scales. C/E/B zero-value controls test the
broader consequence of eliminating source values.

Natural starts (all four factorial cells use E keys):

{table(RUNS[2],values+[('B_randomB','Permuted B values'),('C_zeroV','Native C with zero values'),('B_EK_zeroV','E keys, zero values')])}

BOS starts:

{table(RUNS[2],values+[('B_randomB','Permuted B values'),('C_zeroV','Native C with zero values'),('B_EK_zeroV','E keys, zero values')],'bos')}

![Headwise direction and norm factorial](figures9/value_geometry.png)

<!-- VALUE_FINDINGS_REVIEWED -->

## Relationship to prior work

These are targeted replication/extension measurements, not a discovery of
GPT-2 massive coordinates or compensatory computation. The detailed novelty
comparison in Report8 remains applicable. In particular, the pinned
[Weight-Dictionary Decomposition author code](https://github.com/sunmoonron/weight-dictionary-decomposition/tree/33504a4d90840bc01d0ef75609265e48a34403bb)
already scales coordinate 447 after block 2, measures later layer/neuron
compensation and conditional reserves, and separates a neuron's coordinate
contribution from its complement. We inspected that code rather than
independently reproducing its reported experiments; full OpenReview PDF
access remained blocked. Our source-only joint-coordinate write lesions,
paired suffix-loss panels and source-K/V/norm-direction interventions define
the estimand here. A different estimand alone does not establish priority.

[Massive Activations](https://arxiv.org/html/2402.17762v2) studied GPT-2,
starting-token concentration and explicit K/V bias alternatives.
[A Mechanistic Account of Attention Sinks in GPT-2](https://arxiv.org/html/2604.14722v1)
connects positional MLP processing, key projections and query biases, and
tests their effects on sink attention. Our source K/V loss experiment does
not by itself test that query-bias explanation.
[The Communication Map](https://arxiv.org/pdf/2608.22007) already identifies
447/138 and tests coordinate deletion. Architecture-dependent clamping
results in [Transient Reserves](https://arxiv.org/html/2607.09279v1) and
collective-norm/magnitude-control results in
[Life Cycle of Massive Activation](https://arxiv.org/html/2610.00423v2)
further constrain novelty claims. The latter's training-time restoration is
different from our fixed-weight within-forward interventions. The present
results do not uniquely attribute cross-model differences to normalization.

## Reproduction, uncertainty and resources

The exact protocols, scripts, checkpoint/source hashes, seeds, token manifests,
raw donors, source trajectories, closures and failed-check preservation are
retained. Independent reviewers used CPU only and did not load pretrained
models or contend for the GPU. Numerical audit validates saved reductions;
full-vocabulary logits are not retained, so KL is not independently rebuilt
from logits. Runtime assertions check hook locality beyond saved source
vectors. A single RTX 3060 Ti ran each experiment sequentially with four CPU
threads; no paid resources or network model fetches were used.

{resources}

The three new batches total {sum(m['forward_passes'] for m in M.values())}
forwards/records and {sum(m['wall_seconds'] for m in M.values()):.2f} worker
seconds, including loading, hashing and checkpoint writes as measured by each
runner. This is research execution cost; any additional reproduction runs
are recorded separately. These interventions still compute native modules
before replacing outputs and establish no efficiency improvement.

Main limitations are four passages per group, one checkpoint, teacher-forced
suffix scoring, source-only interventions, and conditional dependence on
large artificial lesions. Neither low-dimensional correlations nor low loss
prove one mechanism or exact functional replacement. Query-path interventions,
causal low-rank directions fitted on fresh development data, and generated
continuations remain useful tests. Research is not declared exhausted.

The private GitHub creation remains denied and unexecuted; no alternate route
or credential/security change has been attempted. Reviewed local commits and
private raw bundles are preserved while publication confirmation is pending.
'''
extra=ROOT/'report9_value_findings.md'
if extra.exists():text=text.replace('<!-- VALUE_FINDINGS_REVIEWED -->',extra.read_text())
else:raise RuntimeError('Add independently reviewed value findings before building final report')
(ROOT/'REPORT9.md').write_text(text)
body=MarkdownIt('commonmark').enable('table').render(text)
for p in FIG.glob('*.png'):body=body.replace('src="figures9/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-2 conditional writes — downstream value pathways</title><style>body{max-width:1050px;margin:auto;padding:24px;font:17px/1.65 system-ui;color:#183044}h1,h2{line-height:1.25}h2{margin-top:2em}img{max-width:100%}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px}td,th{border:1px solid #ccd6df;padding:8px}th{background:#edf4fa}code{overflow-wrap:anywhere}a{color:#12628c}</style></head><body>'''+body+'</body></html>'
(ROOT/'report9.html').write_text(html)
print('Built report9',len(html.encode()),'bytes')
