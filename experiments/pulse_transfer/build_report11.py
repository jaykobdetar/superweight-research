"""Self-contained reviewed gate report; all tables read frozen saved results."""
from pathlib import Path
import base64,json
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'gpt2_gate_fixed512_run'
assert json.loads((RUN/'audit.json').read_text())['status']=='passed'
assert json.loads((ROOT/'review/gpt2_gate_fixed512_numerical_audit.json').read_text())['status']=='passed'
S=json.loads((RUN/'unexposed_summary.json').read_text())
CS=['wiki_validation','dubois','hobbes','optimizers']
NAMES=['Wiki (n=3)','Du Bois (n=4)','Hobbes (n=4)','Code (n=4)']
TRIALS=['P','meanmass','meanlogit','position','global1']+[f'{base}{k}'for base in ['pca_keep','qr_keep','pca_remove','qr_remove']for k in [1,3,6]]
LABELS={'P':'Zero-V reference P','meanmass':'Fixed mean mass','meanlogit':'Fixed mean logits','position':'Position-only','global1':'Raw-logit PC1'}
def table(identity):
    s='| Map | '+' | '.join(NAMES)+' |\n|---|'+'---:|'*4+'\n'
    for trial in TRIALS:
        label=LABELS.get(trial,trial.replace('pca_','PCA ').replace('qr_','QR ').replace('keep','retain ').replace('remove','remove '))
        rows=[next(r for r in S if(r['corpus'],r['identity'],r['trial'])==(c,identity,trial))for c in CS]
        s+='| '+label+' | '+' | '.join(f"{r['p_ppl_ratio']:.4f}"for r in rows)+' |\n'
    return s
resource='| Run | New forwards | Wall seconds | Peak allocated GPU GiB | Peak process RSS GiB |\n|---|---:|---:|---:|---:|\n'
for name,label in [('gpt2_gate_subspace_run','V1: failed'),('native_prefix_diagnostic_run','Native numerical diagnostic'),('gpt2_gate_calibration_rebuild','Calibration-only rebuild'),('gpt2_gate_subspace_v2_run','V2: failed'),('gpt2_gate_fixed512_run','Fixed512: completed')]:
    m=json.loads((ROOT/name/'manifest.json').read_text())
    def val(k,div=1):return f'{m[k]/div:.3f}'if k in m else 'not recorded'
    resource+=f"| {label} | {m.get('forward_passes',33)} | {val('wall_seconds')} | {val('peak_gpu_allocated_bytes',2**30)} | {val('peak_process_rss_kib',2**20)} |\n"
methods=(ROOT/'report11_methods.md').read_text();findings=(ROOT/'report11_gate_findings.md').read_text();prior=(ROOT/'report11_prior_findings.md').read_text()
text='''# Do the leading gate directions preserve GPT-2's computation?

Research continuation, 21 October 2026. GPT-2 revision
`75e09b43581151bd1d9ef6700faa605df408979f`, FP32/native SDPA;
all layer/head indices are zero-based.

**The leading covariance directions affect behavior, but do not provide a
faithful simple replacement.** Keeping one calibrated direction per layer
raises perplexity about 15–29% relative to the zero-source-value baseline.
Keeping six reduces the gap to about 1–10%, while changing roughly 5–10%
of argmax predictions. Matched random directions sometimes have lower loss
despite worse prediction fidelity. This is evidence about a defined
intervention in one checkpoint, not one universal underlying mechanism.

The report preserves two failed cross-length studies and their numerical
diagnostic. The completed experiment uses one physical shape throughout.
Independent numerical review passed; no previous failed result is relabeled
as successful.

'''+methods+'\n'+findings+'''

## All predeclared maps

Perplexity ratios below use zero-value reference P on the identical input.
The exact-copy controls `current` and `fullrank` equal P; native C and their
full numerical records are retained in the data. All predeclared active maps
are shown. The perplexity figure shows paired bootstrap 95% intervals;
agreement panels show means. The raw tables
also include comparisons to native C, KL, agreement, and full n=4 views.

Natural starts:

'''+table('natural')+'\nBOS replacement at token0:\n\n'+table('bos')+'''

![All PCA and QR retention/removal ranks, with paired intervals](figures11/gate_rank_loss.png)

![Prediction fidelity can favor a different map from loss](figures11/gate_rank_agreement.png)

![Centered calibration spectrum; descriptive rather than causal](figures11/gate_spectrum.png)

'''+prior+'''

## Reproduction and resource accounting

The local reproducibility package includes frozen experiment sources, exact
checkpoint and input hashes, both failed runs, raw private outputs, numerical
exports, protocols, and independent review notes. Scripts rebuild historical
calibration metadata with a tokenizer only, then run eight new GPT-2 calibration
forwards and 1,352 fixed512 forwards. Earlier model experiments and Llama weights
are unnecessary for that route. Exact input files must be obtained under their
own terms; model weights, corpora and private token/activation arrays are not
redistributed in the repository.

'''+resource+'''

The gate investigation totals 1,501 production/diagnostic forwards through this
milestone; independent same-machine replays are counted separately. V1 did not
record its final wall/peak resources before failure. One local RTX3060Ti ran
one model at a time. Exact replays are hardware/runtime-conditional evidence,
not external replication; other kernels may expose numerical differences.

The standalone reproduction then completed 1,360 new forwards (eight calibration
and 1,352 study) in 258.179 driver seconds. Its numerical comparison matched all
680 records and 1,734,000 exported per-token values exactly. Private comparison
matched 123,828 saved tensors containing 546,996,560 elements, with 9,520 scalar
reductions independently recomputed across the reference and replay. Peak CPU
comparison RSS was 1.282 GiB. Successful full logits/contexts outside the fixed
snapshots remain absent, so this does not compare arrays that were never saved.

The first comparison stopped on a calibration bookkeeping-schema mismatch:
the rebuild records omit original phase/identity and check-count fields.
That failed comparison and its code remain preserved. An independently reviewed
compatibility fix validates the original constants and compares every shared
field, including all64 layer gate checks; it neither invents missing counters
nor reruns the model. Both numerical and private comparisons then passed.
The replay adds1,360 forwards to the1,501 production/diagnostic count;
CPU fixtures and metadata-only preparation use zero model forwards.

The private GitHub repository remains local because automatic approval review
denied its creation before execution. No remote or push was created, and no
publication retry or credential change occurred. The next proposed experiment
isolates PC1 removal at each layer, with a matched directional control and
position-only replacement, before separately crossing calibration means and
bases. Fresh-span eligibility and design review precede its execution. This
will test the removal sign and layer composition; it will not by itself explain
all rank-one retention loss. This report is a reviewed checkpoint, not a claim of complete
understanding.
'''
(ROOT/'REPORT11.md').write_text(text)
render=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
for p in(ROOT/'figures11').glob('*.png'):
    render=render.replace('src="figures11/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
html='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-2 causal gate subspaces: results and numerical limits</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#172631;background:#f7f8fa;max-width:1080px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.2}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:14px}td,th{border:1px solid #ccd5dc;padding:7px 10px}th{background:#e7eef3}a{color:#155a8c}code{font-size:.88em;overflow-wrap:anywhere}</style></head><body>'+render+'</body></html>'
(ROOT/'report11.html').write_text(html);print('Built report',len(text),'characters;',len(html),'HTMLbytes')
