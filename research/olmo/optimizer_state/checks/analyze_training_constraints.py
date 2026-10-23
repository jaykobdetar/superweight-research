#!/usr/bin/env python3
"""Offline post-hoc constraints from authenticated archived OLMo tensor slices.

No model execution, optimizer application to a model, network access, or training.
Run with --source to collect inputs from the recovered original evidence bundle;
then run without --source for a portable offline reproduction.
"""
import argparse, csv, hashlib, json, math, pathlib, shutil
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name, x): (OUT/name).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def csvwrite(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

p=argparse.ArgumentParser(); p.add_argument('--source', type=pathlib.Path)
args=p.parse_args()
if args.source:
    src=args.source
    manifest=json.loads((src/'BUNDLE_MANIFEST.json').read_text())['files']
    chosen=[]
    for q in (src/'sources/trajectory').glob('*/*'):
        if q.name=='result.json' or q.name.endswith('.f32'):
            chosen.append(q)
    records=[]
    for q in sorted(chosen):
        rel=str(q.relative_to(src)); m=manifest[rel]
        assert q.stat().st_size==m['bytes'] and digest(q)==m['sha256'], rel
        dest=ROOT/'inputs'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(q,dest)
        records.append({'path':str(dest.relative_to(ROOT)),**m})
    (ROOT/'inputs/archived_input_manifest.json').write_text(json.dumps(records,indent=2)+'\n')

records=json.loads((ROOT/'inputs/archived_input_manifest.json').read_text())
for x in records:
    q=ROOT/x['path']; assert q.stat().st_size==x['bytes'] and digest(q)==x['sha256']
data=[]
for q in (ROOT/'inputs/sources/trajectory').glob('*/result.json'):
    x=json.loads(q.read_text()); x['row']=np.fromfile(q.with_name('down_row1764.f32'),dtype='<f4').astype(float)
    assert x['row'].size==8192 and x['row'][1710]==x['primary']
    x['dir']=q.parent; data.append(x)
known=sorted([x for x in data if x['step'] is not None],key=lambda x:x['step'])
by_step={x['step']:x for x in known}; final=next(x for x in data if x['name']=='main')
assert len(known)==39 and len(data)==40

# Scenario, NOT a recovered historical LR sequence. The native-release effective
# configuration describes a later restart; the historical code commit is unknown.
peak=4e-4; decay=.1; tokens_per_step=512*4096; tmax=3e12
last=max(by_step); steps=np.arange(last+1,dtype=float)
scenarios={}
for warm in (2000,5000):
    tok=steps*tokens_per_step; tw=warm*tokens_per_step
    lr=np.where(tok<tw,peak*(.1+.9*tok/tw),peak*(.1+.9*(1+np.cos(np.pi*np.clip((tok-tw)/(tmax-tw),0,1)))/2))
    # Updated weight after step b includes eta_b. One-step offset is audited later.
    scenarios[f'cosine_warmup{warm}']=np.r_[0,np.cumsum(np.log1p(-decay*lr[1:]))]
scenarios['constant_lr_floor']=np.arange(last+1)*math.log1p(-decay*4e-5)
scenarios['constant_lr_peak']=np.arange(last+1)*math.log1p(-decay*4e-4)

intervals=[(a['step'],b['step']) for a,b in zip(known,known[1:])]
extra=[(56000,88000),(120000,256000),(120000,512000),(512000,800000),(800000,1200000),(120000,1431000)]
intervals=list(dict.fromkeys(intervals+extra))
budget=[]
for a,b in intervals:
    for scenario,logs in scenarios.items():
        P=math.exp(float(logs[b]-logs[a])); denom=-math.expm1(float(logs[b]-logs[a]))/decay
        for j in [1710,8041,4820,5755]:
            wa=float(by_step[a]['row'][j]); wb=float(by_step[b]['row'][j]); inherited=P*wa
            push=wb-inherited
            threshold=(-math.log(abs(wb/wa))/(decay*(b-a)) if wa*wb>0 and abs(wb)<abs(wa) else None)
            budget.append(dict(start=a,end=b,neuron=j,scenario=scenario,weight_start=wa,weight_end=wb,
                decay_retention=P,passive_endpoint=inherited,net_update_contribution=push,
                decay_weighted_mean_positive_push=push/denom,
                passive_fraction_of_endpoint=inherited/wb if wb else None,
                equivalent_constant_lr_for_passive_decay=(1-math.exp(math.log(abs(wb/wa))/(b-a)))/decay if threshold is not None else None))
csvwrite('optimizer_budget.csv',budget)

# Initializer audit, measured against nominal Mitchell standard deviations.
init=by_step[0]; init_rows=[]
for fn,sd in [('down_row1764.f32',1/math.sqrt(2*8192*2))]+[(f'{p}_row{j}.f32',1/math.sqrt(2048)) for p in ('gate_proj','up_proj') for j in (1710,8041)]:
    w=np.fromfile(init['dir']/fn,dtype='<f4').astype(float); z=w/sd
    init_rows.append(dict(tensor=fn,count=len(w),nominal_std=sd,max_abs_sigma=float(abs(z).max()),
        nominal_second_moment=float(np.mean(z*z)),raw_kurtosis=float(np.mean(z**4)/np.mean(z*z)**2),
        over_three_sigma=int((abs(z)>3).sum())))
a=3.; phi=math.exp(-a*a/2)/math.sqrt(2*math.pi); Z=math.erf(a/math.sqrt(2))
v=1-2*a*phi/Z; fourth=3-2*a*(a*a+3)*phi/Z
init_summary=dict(tensors=init_rows,total_count=sum(r['count'] for r in init_rows),
    predicted_truncated_variance=v,predicted_truncated_kurtosis=fourth/v**2,
    iid_untruncated_probability_all_within_cutoff=Z**sum(r['count'] for r in init_rows),
    note='Descriptive distribution check, not a formal causal test. Independently truncated Gaussian is covariance-isotropic but not rotationally invariant.')
dump('initialization_audit.json',init_summary)

# Whole-row prediction checks. Final axis was selected after training, so these
# are descriptive, not a pre-registered predictive evaluation.
def rank(a):
    # Standard Spearman mid-ranks, including the two tied initial magnitudes.
    order=np.argsort(a); result=np.empty(len(a),dtype=float); i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        result[order[i:j]]=(i+j-1)/2; i=j
    return result
def corr(a,b): return float(np.corrcoef(a,b)[0,1])
initw=init['row']; finw=final['row']; order=np.argsort(-abs(finw))[:10]
prediction=dict(initial_to_final_signed_pearson=corr(initw,finw),
    initial_to_final_absolute_spearman=corr(rank(abs(initw)),rank(abs(finw))),
    top10=[dict(neuron=int(j),initial_weight=float(initw[j]),initial_abs_rank=int((abs(initw)>abs(initw[j])).sum()+1),
        final_weight=float(finw[j]),initial_sign_matches=bool(initw[j]*finw[j]>0)) for j in order])
dump('initial_weight_prediction.json',prediction)

# Nullspace of each LN-fed row. In exact centered LayerNorm, its all-ones
# component does not affect its preactivation and receives zero raw gradient.
nullrows=[]
for x in sorted(data,key=lambda q:q['step'] if q['step'] is not None else last+1):
    for fn in ('gate_proj_row1710.f32','up_proj_row1710.f32','gate_proj_row8041.f32','up_proj_row8041.f32'):
        q=x['dir']/fn
        if not q.exists(): continue
        w=np.fromfile(q,dtype='<f4').astype(float); m=float(w.mean()); initmean=float(np.fromfile(init['dir']/fn,dtype='<f4').astype(float).mean())
        nullrows.append(dict(checkpoint=x['name'],step=x['step'],tensor=fn,mean=m,initial_mean=initmean,
            absolute_mean_ratio_to_initial=abs(m/initmean),null_energy_fraction=len(w)*m*m/float(w@w),
            mean_sign_changed=bool(m*initmean<0)))
csvwrite('layernorm_nullspace.csv',nullrows)

# Analytic Adam bounds and independent algebraic checks. These are not a
# training experiment: no loss is optimized and no model is updated.
b1,b2=.9,.95; ratio=b1/math.sqrt(b2)
bound=math.sqrt((1-b1)**2/((1-b2)*(1-b1*b1/b2)))
tail=peak*bound*ratio/(1-ratio)
rng=np.random.default_rng(20261023)
errors=[]
for n in (1,2,7,31,137):
    eta=rng.uniform(4e-5,4e-4,n); q=rng.normal(size=n); w0=.7; w=w0
    for l,u in zip(eta,q): w=(1-decay*l)*w+l*u
    retention=float(np.prod(1-decay*eta)); kernel=eta*np.r_[np.cumprod((1-decay*eta[:0:-1]))[::-1],1.]
    errors.append(abs(w-retention*w0-float(kernel@q)))
    assert abs(kernel.sum()-(1-retention)/decay)<1e-12
    assert errors[-1]<1e-12

# Identical stored first/diagonal second moments do not identify rotated moments.
R=np.array([[1,1],[-1,1]],dtype=float)/math.sqrt(2)
ambiguity=[]
for sign in (-1,1):
    g1=np.array([1.,sign]); g2=-b1*g1
    m=b1*(1-b1)*g1+(1-b1)*g2
    v=(1-b2)*(b2*g1*g1+g2*g2)
    vr=(1-b2)*(b2*(R@g1)**2+(R@g2)**2)
    ambiguity.append(dict(sign=sign,first=m.tolist(),diagonal_second=v.tolist(),rotated_second=vr.tolist()))
assert np.allclose(ambiguity[0]['first'],ambiguity[1]['first'])
assert np.allclose(ambiguity[0]['diagonal_second'],ambiguity[1]['diagonal_second'])
assert not np.allclose(ambiguity[0]['rotated_second'],ambiguity[1]['rotated_second'])

# Zero-sum gradients can generate a nonzero row mean under diagonal adaptivity.
g=np.array([1.,1.,-2.]); eps=1e-5
null_adam=float(np.mean(-g/(abs(g)+eps)))
assert abs(g.sum())<1e-15 and abs(null_adam)>.3

# Orthogonal transform fixing all-ones: exact LN covariance, SGD covariance,
# but not generally Adam covariance. Parameters here are arbitrary test arrays.
d=9; x=rng.normal(size=d); w=rng.normal(size=d); v=rng.normal(size=d); v-=v.mean(); v/=np.linalg.norm(v)
T=np.eye(d)-2*np.outer(v,v)
ln=lambda z:(z-z.mean())/math.sqrt(np.mean((z-z.mean())**2)+1e-5)
g=rng.normal(size=d); lr=.001
sgd_error=float(np.max(abs(T@(w-lr*g)-(T@w-lr*T@g))))
adam_difference=float(np.linalg.norm(T@(g/(abs(g)+eps))-(T@g)/(abs(T@g)+eps)))
tests=dict(recurrence_max_error=max(errors),ln_equivariance_error=float(np.max(abs(ln(T@x)-T@ln(x)))),
    sgd_equivariance_error=sgd_error,adam_update_non_equivariance_norm=adam_difference,
    zero_sum_gradient_adam_mean_step=null_adam,ambiguous_second_moment_histories=ambiguity)
assert tests['ln_equivariance_error']<1e-12 and sgd_error<1e-12 and adam_difference>.01
dump('algebra_checks.json',tests)
dump('analytic_adam_bounds.json',dict(beta1=b1,beta2=b2,asymptotic_coordinate_update_bound=bound,
    zero_new_gradient_normalized_momentum_ratio=ratio,normalized_momentum_halflife_steps=math.log(.5)/math.log(ratio),
    maximum_total_future_displacement_from_old_momentum_ignoring_decay=tail,
    assumptions='Standard AdamW, consistent moments from zero initialization with unchanged betas, late step where bias corrections are negligible, zero future gradient, nonnegative epsilon, LR<=4e-4. Bound ignores decay, hence conservatively overestimates retained movement. No unknown transplanted moment states.'))
dump('summary.json',dict(known_step_checkpoints=len(known),exact_final_has_unknown_step=True,input_files_verified=len(records),
    max_input_bytes=max(x['bytes'] for x in records),input_total_bytes=sum(x['bytes'] for x in records),
    scenarios=list(scenarios),historical_schedule_verified=False,
    large_weight_initial_abs_rank=prediction['top10'][0]['initial_abs_rank'],
    max_primary_gate_mean_ratio=max(x['absolute_mean_ratio_to_initial'] for x in nullrows if x['tensor']=='gate_proj_row1710.f32'),
    tests_passed=True))
# Off-by-one scheduler indexing sensitivity on the headline intervals.
logs=scenarios['cosine_warmup5000']; offsets=[]
for a,b in extra:
    P=math.exp(float(logs[b]-logs[a]))
    shifted=math.exp(float(logs[b-1]-logs[a-1]))
    offsets.append(dict(start=a,end=b,retention=P,one_step_earlier_retention=shifted,
        relative_retention_difference=abs(shifted/P-1)))
dump('scheduler_offset_sensitivity.json',offsets)
print(json.dumps(json.loads((OUT/'summary.json').read_text()),indent=2))
