"""CPU-only frozen head-selection, local attention and compensation audit."""
import csv,hashlib,inspect,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
import audit_common as common
from audit_common import interval,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'newline_heads_run'
m=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in m and m['forward_passes']==880
assert m['model_hash_before']==m['model_hash_after']and m['parameter_versions_unchanged']
for f,p in [('script_sha256',ROOT/'newline_heads.py'),('design_sha256',ROOT/'NEWLINE_HEADS.md')]:assert m[f]==hashlib.sha256(p.read_bytes()).hexdigest()
ns=dict(common.__dict__);code=inspect.getsource(common.load_audit).replace('assert len(items)==expected_windows','assert len(items)==(8 if key.startswith("dev_") else 4)');exec(code,ns);raw,groups,result=ns['load_audit'](OUT);assert result['rows']==880 and result['groups']==184
sets=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');traces=torch.load(OUT/'probe_traces.pt',weights_only=True,map_location='cpu');obs=json.loads((OUT/'pulse_observations.json').read_text());assert len(traces)==len(obs)==880;obs={(z['group'],z['trial'],z['window']):z for z in obs}
sel=json.loads((OUT/'selection.json').read_text());selected=sel['selected'];control=sel['control'];assert control==(0 if selected!=0 else 1)and sel['frozen_before_heldout'];screen={}
def edited(xx,pid=187,qid=187):
 x=xx.clone();a=x[:129];a[(a==187)|(a==535)]=209;x[0]=510;x[64]=pid;x[128]=qid;return x
for c,i,xx in sets['development']:
 s=old['starts'][c]['development'][i];assert torch.equal(xx,old['corpora'][c][s:s+512]);g='dev_'+c
 b=traces[f'{g}__clean__{i}']['potential_pulse'][128];a=traces[f'{g}__edge_all__{i}']['potential_pulse'][128];gap=float(a-b);assert abs(gap)>1
 for h in range(16):screen[c,i,h]=float(traces[f'{g}__edge_H{h}__{i}']['potential_pulse'][128]-b)/gap
for z in sel['screen']:assert z['fraction']==screen[z['corpus'],z['window'],z['head']]
ranking=sorted([{'head':h,'mean_fraction':sum(v for (c,i,hh),v in screen.items()if hh==h)/16}for h in range(16)],key=lambda z:-z['mean_fraction']);assert ranking==sel['ranking']and selected==ranking[0]['head']
prior=json.loads((ROOT/'confirmation_run/data_manifest.json').read_text());later=json.loads((ROOT/'later_uses_run/data_manifest.json').read_text());dm=json.loads((OUT/'data_manifest.json').read_text());seen=set()
for c,ws in sets['evaluation'].items():
 d=dm[c];starts=[4096,5120,6144,7168]if c=='optimizers'else[16384,18432,20480,22528];excluded=prior[c]['calibration_starts']+prior[c]['evaluation_starts']+later[c]['evaluation_starts'];assert d['evaluation_starts']==starts and d['excluded_starts']==excluded;assert all(abs(s-e)>=512 for s in starts for e in excluded)
 for source in d['sources']:assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
 for xx in ws:
  digest=hashlib.sha256(xx.numpy().tobytes()).hexdigest();assert digest not in seen;seen.add(digest)
checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==16 and all(z['same_q_query']and z['same_source_and_q_projections']for z in checks)
oracles={};ratios={};pulses={};mass_stats={};conditional_mass_errors=[];factor_errors=[];context_locality_checks=0
freq=1/(10000.0**(torch.arange(0,128,2).float()/128));ang=torch.tensor([0.,64.,128.])[:,None]*freq[None];ang=torch.cat([ang,ang],-1);co=ang.cos()[:,None,:];si=ang.sin()[:,None,:]
def rot(z):
 a,b=z.chunk(2,-1);return torch.cat([-b,a],-1)
for key,items in groups.items():
 g,t=key.split('__');isdev=g.startswith('dev_');world=None if isdev else g.rsplit('_',1)[1];c=g[4:]if isdev else g.removeprefix('heads_').rsplit('_',1)[0];n=8 if isdev else 4
 for i,row in items.items():
  xx=next(z for cc,ii,z in sets['development']if cc==c and ii==i)if isdev else sets['evaluation'][c][i];x=edited(xx,209 if world=='U'else 187,535 if world=='Q535'else 187);assert row['token_sha256']==hashlib.sha256(x.numpy().tobytes()).hexdigest()and row['score_start']==129 and row['scored_tokens']==382
  z=traces[f'{key}__{i}'];b=traces[f'{g}__clean__{i}'];o=obs[g,t,i];assert torch.equal(z['potential_pulse'],z['activation'].float()*.63232421875)
  applied=z['potential_pulse'].clone()
  if 'delete_'in t:
   pos={'p':[64],'q':[128],'both':[64,128]}[t.rsplit('_',1)[1]];applied[pos]=0
   ref='edge_selected'if t.startswith('edge_')else'clean';r=traces[f'{g}__{ref}__{i}']
   for k in ['activation','gate','up','H12_values','H12_contexts']:assert torch.equal(z[k],r[k])
  assert torch.equal(applied,z['applied_pulse'])
  for k,p in [('p_pulse',64),('q_pulse',128),('source_pulse',0)]:assert o[k]==float(z['potential_pulse'][p])
  assert z['activation'][0]==b['activation'][0]and torch.equal(z['activation'][:128],b['activation'][:128])
  for k in 'qkv':assert torch.equal(z['L0_'+k],b['L0_'+k])
  factor_errors.append(float((F.silu(z['gate'])*z['up']-z['activation'][[0,64,128]]).abs().max()))
  local={k:z['L0_'+k].clone()for k in 'qkv'};patchhead=None
  if t.startswith('selected_')or t.startswith('control_'):
   label,kinds=t.split('_');patchhead=selected if label=='selected'else control;sl=slice(128*patchhead,128*(patchhead+1));assert torch.equal(z['L0_context'],b['L0_context'])
   if kinds!='oracle':
    donorworld='Q535'if kinds=='q'else('R'if world=='U'else'U');donor=traces[f'heads_{c}_{donorworld}__clean__{i}']
    for k in kinds:local[k][2 if k=='q'else 1,sl]=donor['L0_'+k][2 if k=='q'else 1,sl]
   before=z['L0_context'][2];after=z['L0_patched_context_q'];assert torch.equal(before[:sl.start],after[:sl.start])and torch.equal(before[sl.stop:],after[sl.stop:]);context_locality_checks+=1
  qq,kk=[local[k].clamp(-8,8).reshape(3,16,128)for k in 'qk'];qq=(qq*co+rot(qq)*si).to(qq.dtype);kk=(kk*co+rot(kk)*si).to(kk.dtype);score=(qq[2].float()[:,None,:]*kk.float().permute(1,0,2)).sum(-1)/(128**.5)
  masked=[]
  if t=='edge_all':masked=list(range(16))
  elif t.startswith('edge_H'):masked=[int(t[6:])]
  elif t=='edge_selected'or t.startswith('edge_delete_'):masked=[selected]
  elif t=='edge_control':masked=[control]
  if masked:score[masked,1]=-torch.inf
  # Sham masks position63, not one of these3 retained keys; normalizing3 masses cancels other keys.
  mass=z['L0_mass_0_p_q'];assert mass.shape==(16,3)and torch.isfinite(mass).all()and(mass>=0).all()and(mass<=1).all()
  if masked:assert (mass[masked,1]==0).all()
  denom=mass.sum(-1,keepdim=True);assert(denom>0).all();conditional_mass_errors.append(float((score.softmax(-1)-mass/denom).abs().max()))
 if not isdev:
  q=effect(groups,g,t,'clean');ratios[key]={'ratio':math.exp(q['mean']),'delta_nll':q,'agreement':sum(z['agreement']for z in items.values())/4}
 pulses[key]={k:interval([obs[g,t,i][k]for i in range(n)])for k in ['p_pulse','q_pulse','source_pulse']}
 mass_stats[key]={str(h):{str(pos):interval([float(traces[f'{key}__{i}']['L0_mass_0_p_q'][h,j])for i in range(n)])for j,pos in enumerate([0,64,128])}for h in [selected,control]}
for c,ws in sets['evaluation'].items():
 for i in range(4):
  u=traces[f'heads_{c}_U__clean__{i}'];r=traces[f'heads_{c}_R__clean__{i}']
  for k in 'qkv':assert torch.equal(u['L0_'+k][[0,2]],r['L0_'+k][[0,2]])
 for world in ['U','R']:
  g=f'heads_{c}_{world}'
  for label,h in [('selected',selected),('control',control)]:
   t=label+'_oracle';d=compare(raw,groups,g,t,'clean');d['q_pulse_difference']=interval([obs[g,t,i]['q_pulse']-obs[g,'clean',i]['q_pulse']for i in range(4)]);errs=[]
   for i in range(4):
    z=traces[f'{g}__{t}__{i}'];a=z['L0_patched_context_q'][h*128:(h+1)*128].float();b=z['L0_context'][2,h*128:(h+1)*128].float();errs.append({'max_absolute':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-12))})
   d['q_context_errors']=errs;oracles[g+'__'+t]=d
ss=list(csv.DictReader((OUT/'pulse_summary.csv').open()));assert len(ss)==184
for row in ss:
 for k in ['p_pulse','q_pulse','source_pulse']:
  q=pulses[row['group']+'__'+row['trial']][k]
  for a,b in [(k,'mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(row[a])-q[b])<1e-12
cs=list(csv.DictReader((OUT/'pathway_contrasts.csv').open()));assert len(cs)==92;contrasts={}
for row in cs:
 g=row['group'];terms=json.loads(row['terms']);n=int(row['windows']);dd={}
 for k in ['nll','q_pulse']:
  table=groups if k=='nll'else None;vals=[sum(coef*(groups[g+'__'+t][i]['nll']if k=='nll'else obs[g,t,i][k])for t,coef in terms)for i in range(n)];q=interval(vals);dd[k]=q
  for a,b in [(k+'_difference','mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(row[a])-q[b])<1e-12
 contrasts[g+'__'+row['contrast']]=dd
result.update(selected=selected,control=control,ranking=ranking,development_rank_and_screen_replayed=True,heldout_source_hashes_and_disjoint_offsets_verified=True,unique_heldout_windows=len(seen),source_and_query_projection_checks_replayed=16,context_locality_checks=context_locality_checks,pulse_summary_rows_replayed=len(ss),pathway_contrast_rows_replayed=len(cs),conditional_three_key_mass_max_error=max(conditional_mass_errors),cpu_half_factor_max_error=max(factor_errors),oracle_controls=oracles,ratios=ratios,pulses=pulses,mass_statistics=mass_stats,pathway_contrasts=contrasts,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['oracle_controls','ratios','pulses','mass_statistics','pathway_contrasts']},indent=2))
