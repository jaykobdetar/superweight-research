#!/usr/bin/env python3
"""Read-only selected-scalar analysis; no model, optimizer update or network."""
import pathlib,json,struct,hashlib,math,csv
ROOT=pathlib.Path(__file__).resolve().parent
p=json.loads((ROOT/'protocol.json').read_text());r=json.loads((ROOT/'receipt.json').read_text())
assert r['status']=='complete' and r['body_bytes']==1024
assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==r['protocol_sha256']
rows=p['sample_rows'];assert len(rows)==len(set(rows))==128 and 1764 not in rows
metadata=json.loads((ROOT/'inputs/optimizer_metadata.json').read_text())['tensors']
for t in r['requests']:
 assert t['status']=='complete' and t['http_status']==206
 assert hashlib.sha256(bytes.fromhex(t['data_hex'])).hexdigest()==t['sha256']
 file=t['url'].rsplit('/',1)[-1];rank=int(file.split('_')[1].split('.')[0])
 header_bytes=(ROOT/f'inputs/rank{rank}_header.json').read_bytes();header=json.loads(header_bytes)
 key='state.transformer.blocks.1.ff_out.weight.__'+t['moment'];global_index=t['row']*8192+1710
 a,b=metadata[key]['flattened_offsets_per_file'][file];assert a<=global_index<b
 expected=8+len(header_bytes)+header[key]['data_offsets'][0]+4*(global_index-a)
 assert t['range']==[expected,expected+3]
for m in ('exp_avg','exp_avg_sq'):
 expected=b''.join(bytes.fromhex(t['data_hex']) for t in r['requests'] if t['moment']==m)
 assert (ROOT/f'{m}.f32').read_bytes()==expected
snap=json.loads((ROOT/'inputs/focal_snapshot_summary.json').read_text());f=snap['primary'];eps=snap['saved_settings']['eps']
assert struct.unpack('<f',(ROOT/'inputs/focal_exp_avg.f32').read_bytes())[0]==f['m']
assert struct.unpack('<f',(ROOT/'inputs/focal_exp_avg_sq.f32').read_bytes())[0]==f['v']
m=struct.unpack('<128f',(ROOT/'exp_avg.f32').read_bytes());v=struct.unpack('<128f',(ROOT/'exp_avg_sq.f32').read_bytes())
assert all(math.isfinite(x) for x in m+v) and all(x>0 for x in v)
data=[]
for i,a,b in zip(rows,m,v):
 data.append(dict(residual_coordinate=i,m=a,v=b,rms=math.sqrt(b),gain=1/(math.sqrt(b)+eps),q=-a/(math.sqrt(b)+eps),epsilon_over_rms=eps/math.sqrt(b)))
with (ROOT/'sampled_axes.csv').open('w',newline='') as out:
 w=csv.DictWriter(out,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def med(x):
 x=sorted(x);n=len(x);return (x[(n-1)//2]+x[n//2])/2
def logchoose(n,k):
 if k<0 or k>n:return -math.inf
 return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
N=2047;n=128;den=logchoose(N,n)
def tail(K,k,upper):
 lo=max(0,n-(N-K));hi=min(n,K)
 if upper:lo=max(lo,k)
 else:hi=min(hi,k)
 return sum(math.exp(logchoose(K,x)+logchoose(N-K,n-x)-den) for x in range(lo,hi+1))
def interval(k):
 # Exact two-sided equal-tail hypergeometric interval, discrete conservative.
 lo,hi=0,N
 while lo<hi:
  mid=(lo+hi)//2
  if tail(mid,k,True)>=.025:hi=mid
  else:lo=mid+1
 lower=lo;lo,hi=0,N
 while lo<hi:
  mid=(lo+hi+1)//2
  if tail(mid,k,False)>=.025:lo=mid
  else:hi=mid-1
 return [lower,lo]
metrics={}
for name,vals,focal in [('abs_m',[abs(x) for x in m],abs(f['m'])),('v',v,f['v']),('gain',[x['gain'] for x in data],f['inverse_denominator_gain']),('abs_q',[abs(x['q']) for x in data],abs(f['signed_adaptive_push']))]:
 k=sum(x>focal for x in vals);ci=interval(k)
 metrics[name]=dict(focal_value=focal,sample_median=med(vals),focal_over_sample_median=focal/med(vals),sample_greater=k,sample_less=sum(x<focal for x in vals),sample_equal=sum(x==focal for x in vals),other_population_greater_count_95_percent_interval=ci,other_population_greater_fraction_95_percent_interval=[x/N for x in ci])
out=dict(checkpoint='native OLMo-1B-0724 step954000',column_neuron=1710,focal_residual_coordinate=1764,sampled_other_coordinates=128,other_population_size=2047,sample_selection_frozen_before_retrieval=True,payload_bytes=1024,metrics=metrics,
 epsilon_dominant_sample_count=sum(x['epsilon_over_rms']>1 for x in data),median_epsilon_over_rms=med([x['epsilon_over_rms'] for x in data]),
 uncertainty='Each reported interval separately inverts the exact without-replacement hypergeometric sampling distribution at two-sided95%. No adjustment for looking at four correlated metrics. No actual full-column rank has been measured.',
 limits=['One late snapshot and one hidden neuron; not historical causality.','Selection excludes focal1764 by design.','m is a first moment, not the current gradient.','No output-column model weights were retrieved; no weight-axis concentration correlation estimated.','No new gradient, optimizer update or training was performed.'])
(ROOT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
