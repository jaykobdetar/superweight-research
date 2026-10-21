"""Build static scientific figures and a self-contained, source-linked report."""
from pathlib import Path
import base64, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
CORPORA = ['wiki_validation', 'dubois', 'hobbes', 'optimizers']
NAMES = ['Wiki validation', 'Du Bois', 'Hobbes', 'Optimizer code']
RUNS = ['gpt2_run', 'llama_run', 'gpt2_groups_run', 'llama_source_kv_run', 'gpt2_writes_run']
if (ROOT / 'llama_replacements_run' / 'summary.json').exists():
    RUNS.append('llama_replacements_run')
S = {r: json.loads((ROOT / r / 'summary.json').read_text()) for r in RUNS}
M = {r: json.loads((ROOT / r / 'manifest.json').read_text()) for r in RUNS}

def row(run, corpus, trial, identity='natural'):
    return next(r for r in S[run] if r['corpus'] == corpus and r['trial'] == trial and
                r['identity'] == identity and r.get('length', 512) == 512)

def errorbars(ax, run, conditions, identity='natural', log=False):
    for offset, (trial, label, color) in enumerate(conditions):
        rr = [row(run, c, trial, identity) for c in CORPORA]
        y = np.array([r['ppl_ratio'] for r in rr])
        ci = np.array([r['ppl_ratio_ci95'] for r in rr])
        x = np.arange(4) + (offset - (len(conditions)-1)/2)*.13
        ax.errorbar(x, y, yerr=np.maximum(0, np.stack([y-ci[:,0], ci[:,1]-y])),
                    fmt='o', ms=5, capsize=3, color=color, label=label)
    ax.axhline(1, color='#687985', lw=.8, ls='--')
    ax.set_xticks(range(4), NAMES, rotation=15, ha='right')
    ax.set_ylabel('Perplexity / same-input native')
    if log:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=.15)
    ax.legend(fontsize=8)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for ax, run, title in zip(axes, ['gpt2_run', 'llama_run'], ['GPT-2: weak selected-candidate loss effects', 'Llama: whole column is more consequential']):
    errorbars(ax, run, [('row', 'Selected weight contribution', '#256a9c'), ('column', 'Whole column', '#c55a37')])
    ax.set_title(title, fontsize=11)
fig.savefig(FIG/'transfer_candidates.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, identity in zip(axes, ['natural', 'bos']):
    errorbars(ax, 'llama_source_kv_run', [('N', 'Column deletion', '#6c7580'), ('N_KC_VN', 'Clean K, altered V', '#267ca4'),
              ('N_KN_VC', 'Altered K, clean V', '#45a079'), ('C_zeroV_cleanK', 'Zero V, clean K', '#c48a36'),
              ('C_mask_blocked', 'Block source access', '#bc4565')], identity, True)
    ax.set_title('Llama source-set interventions: '+identity)
fig.savefig(FIG/'llama_source_sets.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, identity in zip(axes, ['natural', 'bos']):
    errorbars(ax, 'gpt2_writes_run', [('E1A0M0', 'Erase only', '#267ca4'), ('E1A0M1', 'Erase + block MLP writes', '#bc4565'),
              ('E1A1M0', 'Erase + block attention writes', '#45a079'), ('E1A1M1', 'Erase + block both', '#955ba3')], identity, True)
    ax.set_title('GPT-2 conditional coordinate writes: '+identity)
fig.savefig(FIG/'gpt2_writer_factorial.png', dpi=170)
plt.close(fig)

geometry=json.loads((ROOT/'gpt2_writes_run/geometry.json').read_text())
fig, axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
for j,ax in enumerate(axes):
    for trial,label,color in [('E0A0M0','Native','#687985'),('E1A0M0','Erase only','#267ca4'),('E1A0M1','Erase + block MLP','#bc4565')]:
        ys=[]
        for layer in range(2,12):
            vals=[r['after_coords'][j] for r in geometry if r['identity']=='natural' and r['trial']==trial and r['layer']==layer and r['component']=='residual']
            assert len(vals)==16
            ys.append(vals)
        y=np.array(ys)
        ax.plot(range(2,12), y.mean(1), label=label, color=color)
        ax.fill_between(range(2,12), y.min(1), y.max(1), color=color, alpha=.1)
    ax.axhline(0,color='#a0a0a0',lw=.6);ax.set_xlabel('Decoder layer (zero-based)')
    ax.set_title('Raw source coordinate '+str([447,138][j]));ax.set_ylabel('Residual value after layer');ax.legend(fontsize=8)
fig.savefig(FIG/'gpt2_regrowth.png',dpi=170)
plt.close(fig)

fig,ax=plt.subplots(figsize=(11,4.2),constrained_layout=True)
ax.set_xlim(0,11);ax.set_ylim(0,4.2);ax.axis('off')
def box(x,y,label,color,width=2.4):
    ax.text(x,y,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.6',fc=color,ec='#879baa'))
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.5,color='#435867'))
ax.text(.1,3.9,'Two scoped causal findings',fontsize=14,weight='bold')
box(1.6,2.8,'Llama source0\nL1 column perturbation','#eaf1f8')
box(5.1,2.8,'Later source K and V\nL2–27, patched separately','#eaf1f8')
box(9.1,2.8,'Later-token predictions\nK and V both matter','#eaf1f8')
arrow(2.9,2.8,3.6,2.8);arrow(6.8,2.8,7.7,2.8)
ax.text(5.6,2.0,'Fixed calibrated K/V can approximate this boundary; attention gates still vary.',ha='center',fontsize=10)
box(1.6,1.1,'GPT-2 source0\nerase447 and138 afterL2','#f8ecef')
box(5.1,1.1,'MLP writes in L3–11\ncoordinates reappear','#f8ecef')
box(9.1,1.1,'Blocking these writes\nhurts after erasure','#f8ecef')
arrow(3,1.1,3.6,1.1);arrow(6.6,1.1,7.7,1.1)
ax.text(5.6,.25,'Regrowth does not restore the native trajectory; layer/neurons are not yet isolated.',ha='center',fontsize=10)
fig.savefig(FIG/'transfer_mechanisms.png',dpi=170)
plt.close(fig)

def table(run, conditions, identity='natural', intervals=False):
    s='| Condition | '+' | '.join(NAMES)+' |\n|---|'+'---:|'*4+'\n'
    for trial,label in conditions:
        rr=[row(run,c,trial,identity) for c in CORPORA]
        cells=[f"{r['ppl_ratio']:.6g}"+(f" [{r['ppl_ratio_ci95'][0]:.4f}, {r['ppl_ratio_ci95'][1]:.4f}]" if intervals else '') for r in rr]
        s+='| '+label+' | '+' | '.join(cells)+' |\n'
    return s

resources='| Batch | Forwards | Evaluation records | Wall seconds | Peak GPU GiB | Peak host GiB |\n|---|---:|---:|---:|---:|---:|\n'
for r in RUNS:
    m=M[r]; resources+=f"| {r} | {m['forward_passes']} | {m['records']} | {m['wall_seconds']:.2f} | {m['peak_gpu_allocated_bytes']/2**30:.3f} | {m['peak_process_rss_kib']/2**20:.3f} |\n"
total=sum(m['forward_passes'] for m in M.values()); records=sum(m['records'] for m in M.values()); wall=sum(m['wall_seconds'] for m in M.values())

text=f'''# Does the super-weight mechanism transfer?

Report 8 · 21 October 2026 · empirical continuation of the OLMo case study

The strongest transfer result is a difference in causal structure. In Llama-3.2-3B-Instruct, removing the full selected neuron-output column is consistently more damaging than removing its largest individual output contribution. Its downstream source values have a substantial role. In GPT-2, the largest development-selected contribution has little effect on held-out loss when removed. Erasing two large source coordinates reveals a different conditional dependence: later MLP writes regrow those coordinates, and blocking these writes after erasure sharply increases loss. Their raw trajectory is not restored to native values.

These are scoped intervention findings for three pinned checkpoints, including the earlier OLMo study. Similar activation magnitudes do not establish a shared causal mechanism. The original OLMo row-versus-column and newline-history signatures do not simply recur in the two transfer candidates.

![Scoped intervention mechanisms](figures/transfer_mechanisms.png)

## What was tested

All indices are zero-based. “Source” means position0 in a freshly reset, teacher-forced sequence. No hidden cache history, chat template, padding or prefix insertion was used. Each baseline uses exactly the same input IDs as its intervention; only the designated first-token replacement differs between start conditions. Scoring excludes the source's immediate next-token prediction and uses targets2..511 for the main tables.

Eight development passages from Wiki test and Hume selected one candidate per model, without evaluation loss information. The score is mean absolute source intermediate activation multiplied by the largest absolute weight in its output column. This selects a large typical contribution, not necessarily the model's most causally important weight. Same-layer random controls were frozen separately; they are not magnitude-matched.

| Checkpoint | Revision | Arithmetic | Selected layer / output / intermediate | Selected coordinate's column-energy share |
|---|---|---|---|---:|
| GPT-2, openai-community/gpt2 | `75e09b43581151bd1d9ef6700faa605df408979f` | FP32, SDPA | 2 / 447 / 666 | 98.88% |
| meta-llama/Llama-3.2-3B-Instruct | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | BF16, SDPA | 1 / 588 / 1419 | 30.17% |

Initial evaluation used four passages each from Wiki validation, Du Bois, Hobbes and PyTorch optimizer source code, at256 and512 tokens, with natural, single-token `The`, and BOS starts. GPT-2 uses its defined BOS/EOS identity. Follow-ups use new fixed character locations, natural/BOS starts and512 tokens. Actual token-to-character spans were checked for overlap within each model's successive panels. Models tokenize the same source passages independently; cross-model inputs are not token-matched.

Uncertainty bars are paired-window percentile bootstrap intervals:2000 draws, seed20261021, n=4 per corpus/start, conditional on fixed selection and calibration. They do not address selection uncertainty, pretraining overlap or multiple comparisons. Perplexity ratios are exp(mean paired NLL difference), with each input compared to its own native baseline. A ratio near1 does not imply identical outputs.

## 1. The initial signature does not transfer uniformly

![Selected-candidate interventions](figures/transfer_candidates.png)

Natural-start512-token results; brackets are conditional95% bootstrap intervals. The two panels have different vertical scales.

### GPT-2

{table('gpt2_run',[('row','Selected output contribution removed'),('column','Whole selected column removed')],intervals=True)}

The selected weight is large (15.0681), and its source contribution is about946 on development inputs. Yet its loss effects are small. The three strongest development-selected columns feeding output447 were then tested in a complete deletion factorial on fresh passages. Joint removal substantially reduced the source MLP output447, from roughly2331–2339 to290–291, while natural-start PPL ratios stayed0.9980–1.0037 for row-only removal and0.9922–0.9988 for full-column removal. Removing the entire MLP output447 was also weak. This does not support the simple explanation that those three columns hide a major loss effect through within-layer redundancy.

### Llama

{table('llama_run',[('row','Selected output contribution removed'),('column','Whole selected column removed')],intervals=True)}

The full-column lesion has greater NLL than the single-output lesion in all96 individual window/length/start comparisons, with positive lower confidence bounds in all24 groups. Row deletion still changes predictions: natural-start argmax agreement is about90–94%, despite modest loss changes. The largest coordinate carries only30.17% of this column's squared weight norm, so the unremoved complement is substantial. Energy accounting alone is not its causal explanation.

Both models failed the predeclared newline-history screen at the frozen candidate. Earlier identical newline history did not produce the required tenfold later-pulse suppression with a sufficiently large initial pulse. This constrains these candidates and this reset-window protocol; it does not show that either model lacks other newline-dependent circuits.

## 2. Llama needs source values as well as source access

On fresh passages, C is clean; R deletes the selected contribution to output588; N deletes its whole column at position0. At every later layer2..27, source0 K and V projections were independently replaced with donors from C, R or N. All eight shared K/V heads were patched together, preserving native24-query-head grouped attention. Donors are pre-RoPE projection outputs.

![Llama source K/V interventions](figures/llama_source_sets.png)

Both natural and BOS starts are shown. Natural-start numerical values:

{table('llama_source_kv_run',[('R','R: row deletion'),('N','N: column deletion'),('N_KC_VN','N with clean keys, N values'),('N_KN_VC','N with N keys, clean values'),('C_zeroV_cleanK','Clean keys, zero source values'),('C_mask_blocked','Block later queries from source')])}

Restoring clean values helps more than restoring clean keys in these comparisons, but neither mixed state restores native outputs. Zeroing values while clamping clean keys gives about3.0–3.3× PPL; removing access to source0 gives19–54×. Thus source access has a role beyond simply adding its value, and the values themselves also matter. Neither effect can be read as an additive mediation percentage.

This reverses the qualitative zero-value-versus-access-block pattern of the earlier OLMo panel. Size, training, instruction tuning, precision and input distribution differ alongside architecture; no experiment here isolates which difference causes the reversal. The comparison is evidence against assuming one interchangeable source mechanism across these models.

All192 full-donor diagonal closures,64 source-blocking closures and64 first-downstream-layer query-equality checks passed. Explicit causal-mask reference forwards matched scored per-token losses and argmax IDs exactly in96 cases. With the same complete K/V donors, the R and N recipients also agree on scored outputs. These expected architectural closures validate intervention boundaries; they are not discoveries by themselves. Later query equality before first transmission separates the initial source route from later dependence.

## 3. GPT-2 regrows erased coordinates through conditional MLP writes

The follow-up froze coordinates447 and138 from development evidence. E sets both source residual coordinates to zero immediately after layer2. A prevents later attention outputs from writing those coordinates at source0 in layers3..11. M similarly prevents later MLP outputs from writing them. All eight E/A/M combinations were run on a third fresh passage panel, with random-coordinate and norm-rescaling controls.

![Conditional writer factorial](figures/gpt2_writer_factorial.png)

Natural-start results:

{table('gpt2_writes_run',[('E1A0M0','Erase E only'),('E1A0M1','E + block MLP writes'),('E1A1M0','E + block attention writes'),('E1A1M1','E + block both writes'),('E0A0M1','Block MLP writes without E'),('random_E1A1M1','Random-coordinate E + both blocks')])}

Erasure alone has modest loss effects. Blocking MLP writes after erasure raises natural-start PPL5.46–37.69×, whereas blocking those same writes without erasure gives1.0011–1.0037×. The paired E×M interaction is positive in all32 inputs and all eight group confidence intervals have positive lower bounds. This supports conditional dependence on the MLP writes after erasure, rather than their necessity in the clean model.

BOS is much more sensitive: E plus MLP-write blocking gives31.29–549.39× PPL, and E plus both blocks gives20.77–262.94×. Blocking attention in addition to MLP writes improves loss in every input relative to MLP-only blocking after E (−0.805 to−0.086 nats). These lesions are therefore strongly nonadditive.

![Raw-coordinate trajectories](figures/gpt2_regrowth.png)

Lines show means across16 natural-start inputs; shading shows their observed range, not a confidence interval. Both erased coordinates reappear after layer3 in all32 inputs. Natural-start mean values after layer3 are about[535,66] after E, versus native[2653,771]. The measured MLP write contributes about[540,66]; attention contributes about[−5.47,0.23]. However, by layer11 the mean erased-run coordinates are approximately[−1121,−341], versus native[378,6]. **Regrowth is not restoration of the native raw trajectory.**

The loss intervention blocks writes across nine later layers, so its effect is not uniquely attributable to layer3. Full-vector residual reconstructions passed3,240 exact additions;640 target-coordinate zero closures passed. A frozen random pair gives nearly native loss, but this is not a magnitude-matched or exhaustive coordinate control.

Rescaling the remainder immediately after E to the original source norm worsens natural-start PPL3.85–6.89×, despite similar early coordinate regrowth. This is evidence that those two early values alone do not establish fidelity. A specific relative-geometry explanation remains a hypothesis; this rescaling changes every remaining source component and its later trajectory.

## Findings, failed hypotheses and next discriminating tests

**Supported within the tested interventions:** Llama's whole-column lesion is more consequential than the selected scalar contribution; its key and value components have distinct conditional roles. GPT-2 has strong conditional dependence on later MLP coordinate writes after a two-coordinate lesion. Complete source K/V replacement and source blocking close exactly at the expected causal boundary.

**Failed or unsupported:** a universal “row deletion worse than column deletion” rule; transfer of the particular newline-history signature to the frozen candidates; hidden major GPT-2 loss sensitivity explained merely by its three strongest same-output columns; restoration of GPT-2's native coordinate trajectory after erasure; an additive interpretation of attention/MLP lesions.

**Still open:** whether a fixed Llama source replacement remains useful during free generation, how much dynamic source gating must be retained, which individual GPT-2 layers/neurons generate conditional regrowth, and how much the two-coordinate lesion shifts normalized geometry rather than a specific semantic feature. These are distinct questions. Source0 means require first-token identity accounting, because source0 has no preceding context in this protocol.

## Relation to established work and novelty limits

[Super Weight in Large Language Models](https://arxiv.org/html/2411.07191v2) established extreme sensitivity to a small set of weights and their relationship with large activations. This investigation begins from that phenomenon and does not claim its discovery.

[Massive Activations in Large Language Models](https://arxiv.org/html/2402.17762v2) already studied GPT-2, identified a starting-token-only category, analyzed attention bias effects, and trained variants with explicit K/V bias. Large GPT-2 source coordinates, approximate constant-source interpretations, and explicit K/V alternatives are therefore prior art. Our particular candidate screen, paired native column/scalar lesions and fresh source-set K/V factorial are the case-study measurements.

[The Hydra Effect](https://arxiv.org/abs/2307.15771) and [Explorations of Self-Repair in Language Models](https://arxiv.org/html/2402.15390v1) established compensatory behavior after ablation; the latter includes GPT-2, normalization effects and sparse MLP contributions. The present raw-coordinate reappearance is an operationally narrower observation than restoration of logit direct effects. The E/A/M factorial tests a specific conditional source-coordinate dependence. It does not discover self-repair in general or establish one universal repair mechanism.

**Exact-coordinate precedents are closer still.** [Weight-Dictionary Decomposition](https://github.com/sunmoonron/weight-dictionary-decomposition) studies GPT-2 channel447 at the same block2→3 boundary. Its pinned primary code [perturbs that boundary](https://github.com/sunmoonron/weight-dictionary-decomposition/blob/33504a4d90840bc01d0ef75609265e48a34403bb/scripts/exp_s8_loop.py), [traces attention/MLP compensation](https://github.com/sunmoonron/weight-dictionary-decomposition/blob/33504a4d90840bc01d0ef75609265e48a34403bb/scripts/exp_t2_compensation.py), [crosses scaling with regulator ablation](https://github.com/sunmoonron/weight-dictionary-decomposition/blob/33504a4d90840bc01d0ef75609265e48a34403bb/scripts/exp_u1_reserves.py), and [separates a neuron's channel contribution from its complementary outputs](https://github.com/sunmoonron/weight-dictionary-decomposition/blob/33504a4d90840bc01d0ef75609265e48a34403bb/scripts/exp_u2_leash.py). The complete OpenReview PDF was access-blocked; these intervention definitions were verified in the author's code at commit33504a4d90840bc01d0ef75609265e48a34403bb, without executing it. We did not independently reproduce its results.

[The Communication Map of a Transformer](https://arxiv.org/pdf/2608.22007) already uses the exact447/138 pair as a causal outlier baseline, with repeated layer-boundary direction deletion. [A Mechanistic Account of Attention Sinks in GPT-2](https://arxiv.org/html/2604.14722v1) names138/378/447 and tests a positional-MLP/key/query-bias route. Our current raw-coordinate loss measurements do not test that complete route or establish that attention-sink changes mediate the loss.

Accordingly, the GPT-2 work is a **targeted replication/extension in intervention scope**, not discovery of these coordinates, coordinate regulation, the scalar/complement distinction, or self-repair. Our joint source0-only erasure crossed with later output-write suppression, fresh paired loss panels and source-K/V comparisons define the measured case study. A different estimand does not establish priority or a unique native repair circuit.

[Transient Reserves, Sink Dampers](https://arxiv.org/html/2607.09279v1) already reports different sink and loss consequences of repeated massive-coordinate clamping across GPT-2, Pythia and Llama. [The Life Cycle of a Massive Activation](https://arxiv.org/html/2610.00423v2) tests collective magnitude and redundant carriers, including magnitude controls and training-time perturb-and-restore experiments. These precede broad nonuniversality, redundancy and magnitude-versus-identity interpretations. Our fixed-weight within-forward regrowth is a different setting; the random-coordinate control remains unmatched for magnitude.

The earlier repository literature comparison also covers [Spike, Sparse, and Sink](https://arxiv.org/html/2603.05498v1), [Refined is Better](https://arxiv.org/pdf/2503.22329v1), StreamingLLM, MAGC and selective-training studies. The remaining contribution is the scoped comparative dataset and its causal controls, without a priority claim. Both close precedents and failed transfers are part of the account.

## Reproduction, audit and resources

The local bundle preserves all scripts, frozen protocols, checkpoint and source hashes, calibration/selection tensors, private input IDs, per-token losses/KL/predictions, source vectors and oracle checks. Native projection outputs are recomputed using copied matrices; model parameters are never changed. GPT-2 uses its native Conv1D addmm with bias; Llama uses native F.linear. BF16 rounding means the copied-matrix counterfactual need not equal a separately subtracted real-arithmetic scalar product.

Independent CPU reviews checked candidate selection, hooks, token-span separation, raw metric aggregation, full-vector additions, donor closures and interpretation. Retained KL reductions can be reaggregated, but full vocabulary logits are not retained for an independent reconstruction of KL. Runtime locality assertions supplement saved source vectors; later-position vectors are not all saved. All completed runs have equal before/after model hashes and unchanged parameter version counters.

The portable six-batch workflow was also rerun on the original machine. All4,568 saved per-token loss, KL and argmax arrays matched exactly, including their input hashes and scoring spans. This is an executable local reproduction check, not replication by an independent researcher. It cost a separate4,592 forwards and512.10seconds of summed worker time; those validation costs are additional to the scientific batches below.

{resources}

Total so far: **{total:,} forwards, {records:,} evaluation records, {wall:.2f} seconds of recorded batch wall time**. Calibration/development forwards explain the record-count difference. Analysis, independent review, installation, model retrieval and report preparation are additional. One RTX3060Ti8GiB was used sequentially; maximum measured allocated GPU memory was about6.36GiB, and the largest worker used about5.5GiB host RAM. CPU reviews stayed below1.1GiB. A practical reproduction allowance is8GiB VRAM,16GiB host RAM,25GiB free disk including two checkpoints/environment/outputs, and15–30minutes after dependencies and authorized inputs are ready; exact bitwise results may vary with hardware/kernels.

The repository export includes numerical results without input or prediction token IDs, model weights or corpus text. Private full bundles remain local. Restricted model inputs require the researcher's own lawful access. No new model download or paid compute was used. Qwen's available small checkpoint lacked its matching tokenizer, so no substitute tokenizer was used.

**Publication status:** the OLMo repository is preserved locally at commit `6c97a96dd3a318125ceb21e55e7c078f9222af6a`. Creation of `jaykobdetar/superweight-causal-circuits` as a private GitHub repository was denied by automatic approval review and did not execute. There is no verified remote repository or push. The denied action has not been retried, and credentials or security settings have not been changed. Transfer additions are being prepared locally; this report does not claim they are part of that earlier commit.
'''

if 'llama_replacements_run' in S:
    extra='''## 4. Fresh calibration of simpler Llama replacements

Eight new natural-start calibration passages fit signed activation and per-layer source K/V means, in FP32 then cast to BF16. All eight calibration first tokens are distinct. Third-panel evaluation uses new passages and both natural/BOS starts; its source identities are explicitly checked against calibration. C/R/N have the definitions above; D removes the selected column's contributions to every output except588. These are native copied-matrix counterfactuals.

'''+table('llama_replacements_run',[('R','R: selected row removed'),('D','D: complement removed'),('N','N: full column removed'),('C_mean_k','Pooled fixed keys'),('C_mean_v','Pooled fixed values'),('C_mean_kv','Pooled fixed keys and values'),('mean_pulse_row','Mean activation, selected row'),('mean_pulse_column','Mean activation, full column'),('C_random_mean_kv','Signed-permutation K/V control')],intervals=True)
    extra+='''
The complement-only lesion is worse than the full-column lesion in all32 inputs. Its natural-start PPL is1.685–2.050, versus1.408–1.504 for full-column deletion; BOS gives1.748–2.220 versus1.206–1.360. The prespecified interaction NLL(N)−NLL(R)−NLL(D)+NLL(C) is negative in all eight corpus/start confidence intervals. Removing the selected scalar reduces loss when the rest of the column is already missing, unlike its small average effect in the clean recipient. This is not an additive partition of column importance or proof that squared weight energy determines causality.

Pooled fixed K/V gives natural-start PPL0.998479–1.001467, argmax agreement97.941–98.676%, and mean KL0.001013–0.001484nats. BOS gives PPL1.001892–1.008919, agreement96.814–98.137%, KL0.002286–0.002802. Only2/16 natural evaluation starts appeared in calibration (15distinct natural starts); BOS was unseen. Thus the approximate replacement extends beyond memorized calibration identities within this panel. Wiki-only and Hume-only means also remain close in loss, without selecting whichever performs better.

The matched-norm signed-permutation K/V control fails: natural-start PPL5.49–9.49 and BOS5.40–8.88. Preserving the per-vector absolute-value multiset is therefore insufficient in this control. It is one fixed randomized control, not a distribution over all matched directions.

Replacing the chosen intermediate by its freshly calibrated signed mean−426 in BF16, then recomputing its whole source output contribution, also gives near-native loss: natural PPL0.99779–1.00178 and BOS0.99471–0.99859. Argmax agreement remains below100% (natural98.68–98.87%; BOS97.55–99.12%). This is a calibrated activation replacement, not another numerical weight sweep. Generic source-contribution fields in the raw window table retain the native potential activation; the applied mean is separately recorded in pulse_checks and calibration_summary.

All72 oracle/recipient closures,320 projection-patch records and64 pulse-locality checks passed independent audit. Calibration means and all52 randomized vectors were recomputed independently; D preserves output588 exactly and its other source coordinates match N exactly. The intervention leaves later query-dependent attention weights endogenous. It does not replace attention by a constant operation, prove faithful free generation, or establish a speedup. The hooks still compute the native modules before replacing their outputs.

'''
    text=text.replace('## Findings, failed hypotheses and next discriminating tests',extra+'## Findings, failed hypotheses and next discriminating tests')

# Readable prose spacing, without altering hashes, model names, URLs or protocols.
for word in ('all','at','and','about','roughly','to','stayed','gives','gives','for','by','in','from','versus','is','was','were','of','only','with','passed','seed','layer','layers','output','position','source','targets','cases','across','intervals'):
    text=re.sub(r'\b'+word+r'(?=\d)',word+' ',text)
text=re.sub(r'(?<=\d)(GiB|MiB|KiB|nats|seconds|distinct|inputs|tokens|windows|forwards|evaluation|cases|layers|copies|percent)',r' \1',text)
text=text.replace('RTX3060Ti8 GiB','RTX 3060 Ti with 8 GiB').replace('about[','about [').replace('native[','native [').replace('roughly[','roughly [')
(ROOT/'REPORT8.md').write_text(text)
body=MarkdownIt('commonmark').enable('table').render(text)
def embed(m):
    p=ROOT/m.group(1);return 'src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
body=re.sub(r'src="(figures/[^\"]+)"',embed,body)
html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Super-weight transfer — conditional regrowth and source pathways</title><style>body{max-width:1050px;margin:auto;padding:24px;font:17px/1.65 system-ui;color:#183044}h1,h2,h3{line-height:1.2}h2{margin-top:2.4em}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px}td,th{border:1px solid #cbd7df;padding:8px;text-align:left}th{background:#edf4fa}code{background:#eef2f4;padding:2px 4px;overflow-wrap:anywhere}a{color:#12628c}</style></head><body>'''+body+'</body></html>'
(ROOT/'report8.html').write_text(html)
print('Report',len(text.split()),'words;',len(html.encode()),'HTML bytes;',len(RUNS),'batches')
