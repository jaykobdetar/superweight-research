"""Build a self-contained T18 report from audited data and explanatory prose."""
from pathlib import Path
import json,base64
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'gpt2_gate_wiki_calibration_run'
def j(p):return json.loads(p.read_text())
M=j(RUN/'manifest.json');A=j(RUN/'wiki_factorial_analysis.json')
V=j(ROOT/'review/gate_wiki_numerical_audit.json');Q=j(RUN/'saved_state_audit.json')
assert M['status']=='completed'and V['status']==Q['status']=='passed'
IX={(r['corpus'],r['identity'],r['view'],r['mode']):r for r in A['summary']}
cs=('wiki_validation','utilitarianism','ethics','optimizers_new');labels=('Wiki','Utilitarianism','Ethics','Optimizer code')
def table(family,which):
 out='| Corpus / start | OO | OF | FO | FF | M8 (secondary) |\n|---|---:|---:|---:|---:|---:|\n'
 for c,label in zip(cs,labels):
  for start in ('natural','bos'):
   vals=[]
   for cell in ('OO','OF','FO','FF','M8'):
    r=IX[c,start,'full',family+'_'+cell+'_'+which]['ppl_ratio_p']
    vals.append(f"{r['point']:.6f} [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]")
   out+='| '+label+' / '+start+' | '+' | '.join(vals)+' |\n'
 return out
parts=['''# Does the gate result survive fresh calibration?

GPT-2 research checkpoint, 21 October 2026. This study extends the causal
super-weight investigation; useful questions remain and research continues.

The existing private repository is [jaykobdetar/superweight-research](https://github.com/jaykobdetar/superweight-research).
The preceding [output/loss analysis](https://github.com/jaykobdetar/superweight-research/blob/main/docs/GPT2_GATE_ADDITIVE.md)
explains why combined changes cannot be inferred from singleton losses.
''',(ROOT/'report14_findings.md').read_text(),
'\n![All removal conditions and intervals](figures14/calibration_effects.png)\n',
'\n![Matched-control comparisons](figures14/calibration_controls.png)\n',
'\n![Calibration direction geometry](figures14/calibration_geometry.png)\n',
(ROOT/'report14_methods.md').read_text(),
'\n![Calibration factorial and attention gate intervention](figures14/calibration_method.png)\n',
'''\n## Complete primary perplexity tables

Each cell is perplexity relative to P, with its conditional paired 95%
bootstrap interval. Each corpus contains four paired passages. OO/OF/FO/FF
cross old/fresh Wiki4 centers and directions; M8 is a separately labeled
mixed-eight anchor. The same passages underlie all cells and both starts.
''']
for family,label in (('remove','PC1 removal'),('orthogonal','Orthogonal control')):
 for which,scope in (('L10','L10 only'),('all','joint L4–L11')):
  parts+=['\n### '+label+' — '+scope+'\n\n'+table(family,which)]
R=V['readout_audit'];e=R['maxima']
parts+=[f'''
## Independent audit and resource accounting

The study completed **{M['forward_passes']:,} forwards**: four fresh calibration,
{M['evaluation_forwards']:,} main/anchor forwards and {M['suffix_forwards']:,}
suffix controls. The original eight calibration forwards were reused, not
repeated. Execution took **{M['wall_seconds']:.3f} seconds** on one RTX 3060 Ti.
Peak allocated/reserved GPU memory was **{M['peak_gpu_allocated_bytes']/2**30:.3f}/{M['peak_gpu_reserved_bytes']/2**30:.3f} GiB**,
and peak host RSS **{M['peak_process_rss_kib']/2**20:.3f} GiB**. Calibration maps
froze before evaluation, and before/after parameter, map and precision checks
passed. The four fresh forwards, not the evaluation outcomes, determined F.

Independent CPU review recomputed 32 full and 128 leave-one-out fresh spectra,
384 subspace-cosine comparisons, 4,032 summary-metric groups, 912 paired
contrasts and 162,432 norm-summary reductions. Those saved statistics matched
exactly. The audit verified all 776 main records, 768 suffix controls, 1,128
main checks, 11,808 gate checks and 3,136 layer norm states. Eleven fixed L10
attention states were reconstructed from full saved arrays; other successful
closures rely on recorded runtime comparisons and fingerprints.

A separate CPU FP64 check loaded only the tied embedding tensor and checked
44 actual GPU logit rows from those 11 P/singleton states. The inherited
conservative dot-product error bound passed. Maximum absolute logit difference
was **{e['max_absolute_logit_error']:.9f}**, maximum sampled NLL difference
**{e['max_NLL_CPU_vs_actual_logit_error']:.9f}**, and maximum bidirectional KL
**{e['max_KL_either_direction_CPU_actual']:.3g}**. There were **{R['CPU_actual_argmax_mismatches']}**
sampled argmax mismatches. This is a sampled readout audit, not a second
pretrained execution, full-query validation or exact CPU/GPU identity. No
complete second pretrained replay of T18 has yet been performed.

The observed minimum non-source mass was **{V['runtime_extrema']['min_mass']:.6g}**,
and the maximum recorded scaling ratio was **{V['runtime_extrema']['max_gain']:.6g}**.
All prospective 1e−4 context guards passed, with maximum conditional-context
error **{V['runtime_extrema']['max_conditional_error']:.6g}**. These large ratios
make numerical scope material: the fixed 11 raw snapshots do not cover every
extreme state, and the checks do not establish stability on arbitrary inputs,
longer contexts or other devices. No clipping or outcome-based tolerance
increase was applied.

The numerical audit peaked at {V['peak_rss_kib']/2**20:.3f} GiB host RSS and
the single-tensor readout audit at {R['peak_rss_kib']/2**20:.3f} GiB. They used
no GPU and no model forwards. Raw outputs, frozen sources, failed preparations,
failed audit and reviewed adapter remain locally preserved.

## Reproduction and evidence scope

The local reproducibility bundle contains the exact source freeze, input and
checkpoint hashes, seeds, 24 intervention definitions, selection/rejection
history, per-target measurements, calibration fits, sampled attention states,
actual GPU readout rows, and independent audits. The private repository
contains the reviewed research through the preceding report; this study's
safe numerical export and standalone preparation wrapper are being reviewed
for its next update. Do not treat that packaging work as a completed replay.

Model weights, source corpora, token IDs, private configuration and raw
hidden/logit payloads are excluded from repository publication. Lawful local
inputs must match their pinned hashes. An existing validated M8 reference
avoids old model experiments; the separate eight-forward bootstrap is not the
earlier full gate study. Its current preparation requires a Llama tokenizer
but no Llama weights. A GPT-2-only bootstrap is proposed separately and is not
yet delivered. The standalone metadata contract preserves all 561 historical
duplicate hashes and explicitly distinguishes inherited attestations from
checks performed in a fresh reconstruction.

Use the manifest and source hashes as the authoritative experimental record.
The next scientific test is the native-source-value boundary experiment
described above; its design must be frozen and reviewed before execution.
''',(ROOT/'report14_prior.md').read_text()]
text='\n'.join(parts);(ROOT/'REPORT14.md').write_text(text)
body=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
for p in(ROOT/'figures14').glob('*.png'):
 body=body.replace('src="figures14/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
html='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fresh calibration and the GPT-2 gate effect</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#172631;background:#f7f8fa;max-width:1100px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.2}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:13px}td,th{border:1px solid #ccd5dc;padding:7px 10px}th{background:#e7eef3}a{color:#155a8c}code{font-size:.88em;overflow-wrap:anywhere}</style></head><body>'+body+'</body></html>'
(ROOT/'report14.html').write_text(html);print('Built report14',len(html.encode()),'bytes')
