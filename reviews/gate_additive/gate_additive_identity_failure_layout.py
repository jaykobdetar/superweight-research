"""Additional reachable identity-failure capture; zero model/GPU."""
from pathlib import Path
import ast
source=Path(__file__).with_name('gate_additive_failure_layout.py')
s=source.read_text().split('started=time.time();results={};source_hashes=')[0]
exec(compile(s,str(source),'exec'))
started=time.time();payload=common();a,b=logits(),logits();b[300,123]=3
payload['identity_failure']={'check':{'trial':'repeat','logits_exact':False},'logits':F.logit_pair(a,b)}
del a,b
with tempfile.TemporaryDirectory(prefix='additive-identity-layout-') as td:
 p=Path(td)/'failure.pt';n=F.save_capsule(p,payload);z=torch.load(p,weights_only=True,map_location='cpu')
 v=z['identity_failure']['logits'];assert 300 in v['query_rows'].tolist()and v['per_query_max_abs'][300]==3
 assert v['right'][v['query_rows'].tolist().index(300),123]==3
result={'status':'passed','runner_sha256':sha(ROOT/'gpt2_gate_additive.py'),'failure_helper_sha256':sha(ROOT/'gate_additive_failure.py'),
 'serialized_bytes':n,'cap_bytes':F.CAPSULE_LIMIT,'margin_bytes':F.CAPSULE_LIMIT-n,'max_error_row_preserved':True,'model_forwards':0,'gpu_used':False,
 'scope':'Conservative identity/anchor failure plus common complete accumulator and pending prefix ledger; no impossible union with another raised failure.',
 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.time()-started}
assert result['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
