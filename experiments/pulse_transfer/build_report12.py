"""Readable localization report; tables read complete audited frozen outputs."""
from pathlib import Path
import base64,json
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'gpt2_gate_localization_run'
assert json.loads((RUN/'audit.json').read_text())['status']=='passed'
assert json.loads((ROOT/'review/gpt2_gate_localization_numerical_audit.json').read_text())['status']=='passed'
S=json.loads((RUN/'summary.json').read_text());M=json.loads((RUN/'manifest.json').read_text());A=json.loads((RUN/'audit.json').read_text())
cs=['wiki_validation','dubois','hobbes','optimizers'];labels=['Wiki (n=4)','Du Bois (n=4)','Hobbes (n=4)','Code (n=3)'];families=[('remove_pc1','PC1 removal'),('orthogonal_control','Orthogonal control'),('position','Mean + log(position)')];sets=[f'L{l}'for l in range(4,12)]+['all']
def table(identity,family):
 t='| Layers patched | '+' | '.join(labels)+' |\n|---|'+'---:|'*4+'\n'
 for subset in sets:
  rr=[next(r for r in S if(r['corpus'],r['identity'],r['trial'])==(c,identity,family+'_'+subset))for c in cs]
  t+='| '+subset+' | '+' | '.join(f"{r['p_ppl_ratio']:.4f}"for r in rr)+' |\n'
 return t
parts=['# Which layers depend on GPT-2’s leading gate directions?\n\nResearch continuation, 21 October 2026. This is a reviewed checkpoint in an ongoing investigation.\n', (ROOT/'report12_findings.md').read_text(),(ROOT/'report12_methods.md').read_text()]
parts+=['\n## Every predeclared intervention\n\nThese tables show perplexity divided by zero-value reference P. Both exact-copy controls equal P; native C and all per-passage records remain in the numerical results. Figures show all layers, families and starts. Perplexity and nonadditivity figures include paired bootstrap intervals; agreement plots show means.\n']
for family,title in families:
 parts+=['\n### '+title+'\n\nNatural starts:\n\n'+table('natural',family)+'\nBOS replacement:\n\n'+table('bos',family)]
parts+=['\n![All singleton and joint loss effects](figures12/localization_loss.png)\n\n![Prediction fidelity for every intervention](figures12/localization_agreement.png)\n\n![All joint-minus-singleton residuals](figures12/localization_nonadditivity.png)\n',(ROOT/'report12_prior_and_limits.md').read_text()]
parts += [f'''
## Verification, reproduction and resources

The study completed **{M['forward_passes']:,} forwards**, with **{M['records']} main
records and {M['suffix_forwards']} suffix controls**, in {M['wall_seconds']:.3f}
worker seconds. It reused the original eight calibration forwards and made
zero new calibration forwards. Peak allocated GPU memory was
{M['peak_gpu_allocated_bytes']/2**30:.3f} GiB; peak process RSS was
{M['peak_process_rss_kib']/2**20:.3f} GiB. One local RTX 3060 Ti ran one model
worker. The manifest records the pinned runtime, observed SDPA backend,
checkpoint identity and unchanged parameter/map hashes.

Implementation review included 27 independent physical-512 random-model CPU
forwards, a complete fabricated 938-record/930-control analyzer fixture and
seven corruption guards. Those fixtures are separate from the pretrained
experiment. The actual-result audit is independent and uses saved raw arrays,
not a second model load. It checks loss/KL/agreement reductions, every paired
contrast, actual n=3/n=4 bootstrap, all three nonadditivity families, source and
prefix invariants, and the 52 predeclared FP64 attention snapshot states.
The unchanged 1e-4 reconstruction guard and 1e-6 within-state logit-norm guard
are retained. No failed criterion is silently relaxed.

Full attention contexts are saved for the fixed snapshot subset; successful
full output logits are not retained. Outside that subset, context equality
and all full-logit equality rely on runtime comparisons and shape/dtype/content
hashes. Raw gates, per-token metrics, prefix hidden states, source
states and main-run norm vectors are retained. The complete raw directory and
its SHA256 inventory stay local. Repository exports omit token IDs, raw donor
activations, model weights, copyrighted corpora and private configuration.

The standalone path stages historical metadata, reconstructs inputs with a
local tokenizer, runs eight GPT-2 calibration forwards if no verified reference
is supplied, and then runs this 1,868-forward study. It does not rerun earlier
OLMo, Llama or GPT-2 experiments or load their weights. The tokenizer-only path
has reconstructed all 15 inputs, directions and metadata byte for byte and
rechecked 1,192 historical interval exclusions. The first input rebuild failed
on serialized string-sharing bookkeeping despite identical token values;
that attempt is retained, and the narrow fix preserves the original serialization.
Historical selection evidence is explicitly inherited, not presented as a new
independent eligibility proof.

A complete second pretrained replay of this new localization batch has not yet
been performed. The earlier fixed512 study did reproduce all saved tensors
exactly on this machine; that is separate evidence and does not substitute for
localization replay. Independent researchers should use the supplied entry
points and audit instructions on their own runtime. Bootstrap intervals and
same-machine checks are not external replication.

Private GitHub creation remains denied and unexecuted. No remote, push,
publication retry, credential change or paid compute purchase occurred.
Research remains open; the next experiment is chosen from the reviewed
findings with a new frozen protocol and data separation.
''']
text='\n'.join(parts);(ROOT/'REPORT12.md').write_text(text)
html=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
for p in(ROOT/'figures12').glob('*.png'):html=html.replace('src="figures12/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
page='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-2 gate localization — causal layer screen</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#172631;background:#f7f8fa;max-width:1080px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.2}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:14px}td,th{border:1px solid #ccd5dc;padding:7px 10px}th{background:#e7eef3}a{color:#155a8c}code{font-size:.88em;overflow-wrap:anywhere}</style></head><body>'+html+'</body></html>'
(ROOT/'report12.html').write_text(page);print('Built report12',len(page.encode()),'bytes')
