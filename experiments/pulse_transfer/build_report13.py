"""Render the reviewed additive-logit report with complete primary group tables."""
from pathlib import Path
import base64,json
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'gpt2_gate_additive_run'
def read(p):return json.loads(p.read_text())
M=read(RUN/'manifest.json');A=read(RUN/'audit.json');S=read(RUN/'additive_summary.json')
assert M['status']=='completed' and A['status']=='passed'
review=read(ROOT/'review/gate_additive_numerical_audit.json');assert review['status']=='passed'
ix={(r['corpus'],r['identity'],r['family'],r['view'],r['metric']):r for r in S}
cs=('wiki_validation','mill','wollstonecraft','optimizers_new');labels=('Wiki','Mill','Wollstonecraft','Optimizer code')
families=(('remove_pc1','PC1 removal'),('orthogonal_control','Orthogonal control'),('position','Position replacement'))
def entry(c,i,f,m):
 r=ix[c,i,f,'full',m];return f"{r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]"
def table(f):
 s='| Corpus/start | J | R | N = J + R |\n|---|---:|---:|---:|\n'
 for c,label in zip(cs,labels):
  for i in('natural','bos'):s+='| '+label+' / '+i+' | '+' | '.join(entry(c,i,f,m)for m in('J','R','joint_minus_sum'))+' |\n'
 return s
parts=['# Why do combined gate changes fail to add?\n\nReviewed research checkpoint, 21 October 2026. Investigation continues.\n',(ROOT/'report13_findings.md').read_text(),(ROOT/'report13_methods.md').read_text(),
 '\n![Output diagnostic and loss identity](figures13/additive_method.png)\n',
 '\n## Every primary decomposition group\n\nValues are mean nats per target, followed by conditional paired 95% bootstrap intervals. All groups contain four paired passages. These are three terms in an output/loss diagnostic, not mediation shares. Intervals are calculated separately from shared bootstrap draws; their endpoints must not be added.\n']
for f,title in families:parts+=['\n### '+title+'\n\n'+table(f)]
parts+=['\n![All three loss terms and intervals](figures13/additive_decomposition.png)\n\n![Prediction fidelity](figures13/additive_fidelity.png)\n\n![Distribution and centered logit disagreement](figures13/additive_disagreement.png)\n',(ROOT/'report13_review_and_limits.md').read_text(),(ROOT/'report13_prior.md').read_text()]
parts += [f'''
## Execution and local reproduction

The experiment completed **{M['forward_passes']:,} forwards**, comprising
**{M['records']:,} main records** and **{M['suffix_forwards']:,} suffix controls**,
in {M['wall_seconds']:.3f} worker seconds. It reused eight original calibration
forwards and performed zero new calibration forwards. Peak allocated GPU
memory was {M['peak_gpu_allocated_bytes']/2**30:.3f} GiB; peak host RSS was
{M['peak_process_rss_kib']/2**20:.3f} GiB. One RTX 3060 Ti ran one model worker.
The frozen manifest binds source revisions, seeds, precision settings, all
intervention definitions and before/after model, map and input identities.

The readable report, scripts, manifests, numerical results and audit guidance
belong in the existing local private-repository package. Model weights,
corpora, token payloads, private paths and raw donor activations stay outside
its export. Original input/source hashes identify restricted dependencies;
retrieval and reconstruction instructions identify what another researcher
must supply. Any metadata portability adaptations are explicit and separately
reviewed. No complete second pretrained replay of this new batch is claimed.

Private GitHub creation remains denied and unexecuted, awaiting the specific
confirmation handled by the parent task. No publication retry, alternate
repository upload, credential change or paid compute purchase occurred.
''']
text='\n'.join(parts);(ROOT/'REPORT13.md').write_text(text)
body=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
for p in(ROOT/'figures13').glob('*.png'):body=body.replace('src="figures13/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
html='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-2 gate effects — output and loss nonadditivity</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#172631;background:#f7f8fa;max-width:1080px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.2}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:14px}td,th{border:1px solid #ccd5dc;padding:7px 10px}th{background:#e7eef3}a{color:#155a8c}code{font-size:.88em;overflow-wrap:anywhere}</style></head><body>'+body+'</body></html>'
(ROOT/'report13.html').write_text(html);print('Built report13',len(html.encode()),'bytes')
