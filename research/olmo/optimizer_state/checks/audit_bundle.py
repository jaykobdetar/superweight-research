#!/usr/bin/env python3
"""Independent scalar / struct checks on selected published numerical claims."""
import pathlib, json, math, hashlib, struct
ROOT=pathlib.Path(__file__).resolve().parents[1]
I=ROOT/'inputs'; O=ROOT/'results'
def vec(p):
    b=p.read_bytes();return struct.unpack('<'+'f'*(len(b)//4),b)
m=vec(I/'optimizer_slices/row1764_exp_avg.f32');v=vec(I/'optimizer_slices/row1764_exp_avg_sq.f32')
w=vec(I/'native_identity/down_row1764.f32');p=1710;eps=1e-5
q=[-a/(math.sqrt(b)+eps) for a,b in zip(m,v)]
q0=[-a/math.sqrt(b) for a,b in zip(m,v)]
assert len(m)==len(v)==len(w)==8192
assert sum(abs(a)>abs(m[p]) for a in m)==0 and sum(b>v[p] for b in v)==0
assert 1+sum(abs(x)>abs(q[p]) for x in q)==101
assert 1+sum(abs(x)>abs(q0[p]) for x in q0)==2149
assert sum(math.sqrt(b)<eps for b in v)==6850
s=json.loads((O/'optimizer_snapshot_summary.json').read_text())
assert q[p]==s['primary']['signed_adaptive_push']
assert hashlib.sha256((I/'native_identity/down_row1764.f32').read_bytes()).hexdigest()=='13ada3b9aebad0385385edeea85184fdf93ad7d6002f23c6a8a6bc345268f4f8'
assert (I/'upcycle_config.yaml').read_bytes()==(I/'archive_recovery/native954k_config.txt').read_bytes()
assert (I/'user_optimizer_metadata/metadata.json').read_bytes()==(I/'archive_recovery/native954k_optim_metadata.txt').read_bytes()
# Independent scalar product rather than vectorized cumulative log retention.
P=1.
for step in range(120001,256001):
    tok=step*512*4096;lr=4e-4*(.1+.9*(1+math.cos(math.pi*(tok-10485760000)/(3e12-10485760000)))/2)
    P*=1-.1*lr
assert abs(P-.005325339964709957)<1e-13
thresholds=[(abs(m[i])*math.sqrt(v[p])-abs(m[p])*math.sqrt(v[i]))/(abs(m[p])-abs(m[i])) for i in range(len(m)) if i!=p]
threshold=max(thresholds)
qth=[-a/(math.sqrt(b)+threshold*(1+1e-12)) for a,b in zip(m,v)]
assert max(range(len(m)),key=lambda j:abs(qth[j]))==p
audit=dict(scalar_math_product_retention_120k_256k=P,matches_vectorized_retention=True,
  native_hf_row_sha_match=True,native_hf_match_scope='One complete 8192-value output row only',
  independent_struct_scalar_optimizer_checks_passed=True,
  native_and_archived_effective_configs_byte_identical=True,
  user_and_archived_optimizer_metadata_byte_identical=True,
  frozen_epsilon_threshold_primary_becomes_largest_absolute_update=threshold,
  no_training_performed=True,no_model_or_optimizer_application_performed=True,
  optimizer_selected_raw_state_bytes_read=True,
  official_hf_identity_bytes_retrieved=54988,official_optimizer_slice_bytes_retrieved=74641,
  note='Frozen epsilon threshold is an arithmetic property of this one row and snapshot, not a causal training threshold.')
(O/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
