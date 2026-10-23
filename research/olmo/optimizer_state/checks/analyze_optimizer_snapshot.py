#!/usr/bin/env python3
"""Offline FP64 arithmetic on authentic selected Adam-state FP32 bytes.

No model execution, new gradient, optimizer application, unpickling, or training.
The settings byte arrays are decoded by a restricted primitive-only interpreter.
"""
import pathlib, json, hashlib, struct, math, csv
import numpy as np
from primitive_pickle import decode
ROOT=pathlib.Path(__file__).resolve().parents[1]; IN=ROOT/'inputs/optimizer_slices'; OUT=ROOT/'results'
receipt=json.loads((IN/'receipt.json').read_text());assert receipt['status']=='complete'
assert receipt['body_bytes']==74641 and receipt['body_bytes']<512*1024
for rec in receipt['requests']:
    data=(IN/rec['local_file']).read_bytes()
    assert rec['status']=='complete' and rec['http_status']==206
    assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256']
group,ops=decode((IN/'param_group0.bin').read_bytes());state_keys,keyops=decode((IN/'state_keys.bin').read_bytes())
assert state_keys==['exp_avg','exp_avg_sq','step']
assert struct.unpack('<q',(IN/'num_param_groups.bin').read_bytes())[0]==1
assert group['_param_names'][6]=='transformer.blocks.1.ff_out.weight'
assert [x.removeprefix('_fsdp_wrapped_module.') for x in group['param_names']]==group['_param_names']
step=struct.unpack('<f',(IN/'step.f32').read_bytes())[0]
assert step==954001
beta1,beta2=group['betas'];eps=group['eps'];lr=group['lr'];decay=group['weight_decay']
assert (beta1,beta2,eps,decay)==(.9,.95,1e-5,.1)
assert not any(group[x] for x in ['amsgrad','maximize','capturable','differentiable'])
w=np.fromfile(ROOT/'inputs/native_identity/down_row1764.f32',dtype='<f4').astype(float)
m=np.fromfile(IN/'row1764_exp_avg.f32',dtype='<f4').astype(float)
v=np.fromfile(IN/'row1764_exp_avg_sq.f32',dtype='<f4').astype(float)
assert all(x.shape==(8192,) and np.isfinite(x).all() for x in (w,m,v)) and (v>0).all()
mh=m/(1-beta1**step);vh=v/(1-beta2**step);rms=np.sqrt(vh)
q=-mh/(rms+eps);q0=-mh/rms;gain=1/(rms+eps)
def rk(x,j):return int(1+np.count_nonzero(abs(x)>abs(x[j])))
selected=[]
for j in [1710,8041,4820,5755,7579,6947,2637,7845,3950,4786]:
    selected.append(dict(neuron=j,weight=float(w[j]),m=float(m[j]),v=float(v[j]),sqrt_v=float(rms[j]),
        abs_m_rank=rk(m,j),v_rank=rk(v,j),inverse_denominator_gain=float(gain[j]),
        inverse_denominator_gain_rank=rk(gain,j),epsilon_over_sqrt_v=float(eps/rms[j]),
        signed_adaptive_push=float(q[j]),abs_push_rank=rk(q,j),
        epsilon_free_push=float(q0[j]),epsilon_free_abs_push_rank=rk(q0,j),
        retained_fraction_after_epsilon=float(rms[j]/(rms[j]+eps)),
        current_weight_decay_direction=float(-decay*w[j]),
        current_net_direction_no_new_gradient=float(q[j]-decay*w[j]),
        frozen_proposal_lr_times_adaptive=float(lr*q[j]),
        frozen_proposal_lr_times_decay=float(-lr*decay*w[j])))
with (OUT/'optimizer_snapshot_selected.csv').open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(selected[0]));wr.writeheader();wr.writerows(selected)
sweeps=[]
for e in (0.,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,.01):
    qq=-mh/(rms+e)
    sweeps.append(dict(epsilon=e,primary_push=float(qq[1710]),primary_abs_push_rank=rk(qq,1710),
        primary_fraction_squared_update_norm=float(qq[1710]**2/np.dot(qq,qq)),
        row_update_norm=float(np.linalg.norm(qq)),median_abs_push=float(np.median(abs(qq))),
        coordinate_eps_dominant_count=int(np.count_nonzero(e>rms))))
with (OUT/'frozen_epsilon_sweep.csv').open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(sweeps[0]));wr.writeheader();wr.writerows(sweeps)
# The isotropic operator is norm-matched only within this selected output row,
# not at whole-model level. It is a diagnostic, NOT an SGD training arm.
qiso=-mh*(np.linalg.norm(q)/np.linalg.norm(mh))
tw=10485760000.;tm=3e12;tok=954000*512*4096
lr_scenario=.0004*(.1+.9*(1+math.cos(math.pi*(tok-tw)/(tm-tw)))/2)
assert lr_scenario==lr
summary=dict(checkpoint_label_step=954000,stored_optimizer_step=step,
    saved_settings={k:v for k,v in group.items() if k not in ('param_names','_param_names','params')},
    parsed_primitive_opcodes=sorted(set(ops+keyops)),
    data_bytes_retrieved=receipt['body_bytes'],
    primary=selected[0],secondary=selected[1],
    primary_abs_m_over_row_median=float(abs(mh[1710])/np.median(abs(mh))),
    primary_rms_over_row_median=float(rms[1710]/np.median(rms)),
    primary_v_over_row_median=float(v[1710]/np.median(v)),
    primary_fraction_squared_first_moment_norm=float(mh[1710]**2/np.dot(mh,mh)),
    primary_fraction_squared_adaptive_update_norm=float(q[1710]**2/np.dot(q,q)),
    row_norm_matched_isotropic_primary_push=float(qiso[1710]),
    epsilon_dominant_coordinates=int(np.count_nonzero(eps>rms)),
    median_epsilon_over_rms=float(np.median(eps/rms)),
    median_fraction_of_epsilon_free_update=float(np.median(rms/(rms+eps))),
    saved_lr_exactly_matches_released_scenario_at_checkpoint_label=True,
    scenario_lr=lr_scenario,
    counter_note='Do not equate optimizer step with checkpoint label. The old official checkpointer may initialize an optimizer with a zero-gradient step, but the exact historical reason for this one-step offset is not established.',
    findings=[
      'At this late snapshot the primary has the largest |m| and v and smallest inverse-denominator gain in the selected row, contrary to a tiny-v explosion account of its current update.',
      'The saved adaptive direction still reinforces the positive primary scalar more strongly than the current decay direction opposes it.',
      'Epsilon is negligible for the primary relative to most of the row; removing epsilon at fixed moments lowers its relative update rank. This is frozen-state arithmetic, not a causal training intervention.',
      'Coordinatewise adaptation greatly disperses first-moment concentration within this output row. It need not cause current update concentration simply because the eventual weight is concentrated.'
    ],
    limitations=[
      'Single late checkpoint, not the onset brackets or a historical replay.',
      'One output row compares neurons at fixed residual coordinate1764; it does not test residual-axis concentration within the primary neuron\'s full output column.',
      'No new gradient was computed. Stored m/v reconstruct a saved direction, not the next minibatch update.',
      'Current high gradient moments could be a consequence of the mature circuit rather than its origin.',
      'Changing epsilon while freezing moments does not reproduce a different training history. Row-only norm matching is not a whole-model matched-loss control.',
      'A byte-identical model row and metadata-consistent optimizer slices do not establish whole-checkpoint, trainer/RNG or data-order identity.'
    ])
(OUT/'optimizer_snapshot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'native_param_group.json').write_text(json.dumps(group,indent=2)+'\n')
print(json.dumps(summary,indent=2))
