"""Four preplanned component controls after the completed cancellation rescue."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,resource,sys
import numpy as np
from alternate_writer_hooks import make_hook,group_factorial,NEWLINE_POSITIVE_GROUP

R=Path(__file__).resolve().parent;O=R/'axis_controls';O.mkdir(exist_ok=True)
ROLE=Path('/workspace/shared/olmo_computational_role/downstream')
sys.path.insert(0,str(ROLE));from streamed_olmo import StreamedOlmo
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
    validation=json.loads((ROLE/'results/validation_fp32.json').read_text())
    assert validation['all_pass'] and validation['complete'] and len(validation['rows'])==10
    assert validation['code_sha256']==sha(ROLE/'streamed_olmo.py')
    model=StreamedOlmo(half_round=False)
    ids=np.load(R/'downstream/frozen_inputs.npz');proto=json.loads((R/'downstream/frozen_protocol.json').read_text())
    panels=[p for p in proto['panels'] if p['source_token']==187]
    specs=group_factorial(NEWLINE_POSITIVE_GROUP,prefix='newline_positive')[5:7]
    manifest=dict(model_sha256=validation['code_sha256'],runner_sha256=sha(__file__),
        original_scientific_protocol_sha256=sha(R/'INTERVENTION_MANIFEST.json'),
        panels=[p['name'] for p in panels],interventions=[asdict(s) for s in specs],
        purpose='Exact group axis versus raw complementary-vector removal, with1710 removed. These are vector-component interventions, not neuron ablations or scalar value sweeps.')
    mp=O/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2))
    down=np.memmap(R.parent/'tensors/down_proj.f32',mode='r',dtype='<f4',shape=(2048,8192))
    for panel in panels:
        name=panel['name'];base=R/'downstream'
        checks=json.loads((base/(name+'__numerical_controls.json')).read_text())
        assert all(c['max_abs']<1e-3 and c['mean_abs']<1e-4 for c in checks)
        column=np.load(base/(name+'__drop1710.npz'))['loss'][1:].mean()
        rescue=np.load(base/(name+'__newline_positive_drop_with1710.npz'))['loss'][1:].mean()
        assert column-rescue>.1,'Preplanned conditional-stage rescue gate not met'
        for spec in specs:
            stem=name+'__'+spec.name;path=O/(stem+'.npz');rp=O/(stem+'.json')
            if path.exists():
                old=json.loads(rp.read_text());assert old['result_sha256']==sha(path) and old['manifest_sha256']==sha(mp);continue
            result=model.forward(ids[name],mlp_hook=make_hook(down,spec),capture_layers=(1,),diagnostic_sources=(0,))
            source=result['states']['1']['post_layer'][:1];norm=model.ln(source)
            kv={k:np.clip(model.linear(norm,'model.layers.2.self_attn.'+k+'_proj.weight'),-8,8) for k in ('k','v')}
            data={k:result[k] for k in ('loss','argmax','target_logit','log_normalizer')}
            data.update(source_post_l1=source,source_l2_normalized=norm,source_l2_k=kv['k'],source_l2_v=kv['v'])
            for layer,diag in result['attention'].items():data['attention_to_source_l'+layer]=diag['0']
            np.savez_compressed(path,**data)
            receipt=dict(panel=name,treatment=spec.name,seconds=result['seconds'],peak_rss=result['peak_rss'],
                result_sha256=sha(path),manifest_sha256=sha(mp),mean_nll_excluding_q0=float(result['loss'][1:].mean()))
            rp.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
            del result,data

if __name__=='__main__':main()
