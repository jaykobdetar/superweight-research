"""Mandatory execution gate for the already-frozen alternate-writer study.

Use only after explicit CPU-slot release by the coordinating worker.
Kept separate so no frozen scientific condition or input identity is changed.
"""
from pathlib import Path
import hashlib,json,resource
from run_alternate_writers import execute,OUT,ROLE

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__=='__main__':
    model=ROLE/'downstream'
    validation_path=model/'results/validation_fp32.json'
    validation=json.loads(validation_path.read_text())
    assert validation['complete'] and validation['all_pass']
    assert len(validation['rows'])==10
    assert all(x['pass'] and not x['half_round'] for x in validation['rows'])
    assert validation['code_sha256']==sha(model/'streamed_olmo.py')
    acquired=json.loads((model/'results/acquisition_complete.json').read_text())
    assert acquired['all_complete'] and acquired['revision']=='d7cbab742d80589e714b1a2d7f838dcd21cbe143'
    receipt=dict(validation_sha256=sha(validation_path),model_code_sha256=validation['code_sha256'],
                 execution_gate_sha256=sha(__file__),
                 frozen_protocol_sha256=sha(OUT/'frozen_protocol.json'),
                 passed_comparisons=len(validation['rows']),
                 note='Execution wrapper added after scientific protocol freeze, without modifying inputs or treatments.')
    target=OUT/'execution_validation.json'
    if target.exists():assert json.loads(target.read_text())==receipt
    else:target.write_text(json.dumps(receipt,indent=2))
    resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
    execute()
