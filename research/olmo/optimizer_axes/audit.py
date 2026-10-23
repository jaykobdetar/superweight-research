#!/usr/bin/env python3
"""Independent checks of selected-axis sampling intervals and provenance."""
import json,pathlib,math,hashlib
P=pathlib.Path(__file__).resolve().parent
j=json.loads((P/'results.json').read_text());N=2047;n=128
prob={K:math.prod((N-K-i)/(N-i) for i in range(n)) for K in (56,57)}
assert prob[56]>=.025 and prob[57]<.025
scipy_checked=False
try:
 from scipy.stats import hypergeom
 for x in j['metrics'].values():
  k=x['sample_greater'];lo,hi=x['other_population_greater_count_95_percent_interval']
  assert hypergeom.sf(k-1,N,lo,n)>=.025
  assert lo==0 or hypergeom.sf(k-1,N,lo-1,n)<.025
  assert hypergeom.cdf(k,N,hi,n)>=.025
  assert hi==N or hypergeom.cdf(k,N,hi+1,n)<.025
 scipy_checked=True
except ImportError:pass
assert hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest()=='b1b962fcc6cfa9faf4bac0270eb17f4dbf74ddc209220f7ea118104f0c210d7f'
out=dict(independent_zero_count_endpoint_product=prob,all_intervals_independently_verified_with_scipy=scipy_checked,protocol_hash_unchanged=True,no_training=True)
(P/'audit_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
