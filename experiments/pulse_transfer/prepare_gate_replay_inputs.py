"""Reconstruct historical calibration metadata using only a local tokenizer.

This does not rerun any Llama model experiment. Output contains private input
IDs and stays in the replication workspace, never in the numerical repository.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import json
import resource
from pathlib import Path
import torch
from transformers import AutoTokenizer
from llama_replacements import calibration_inputs
from run_transfer import sha, dump

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--models-manifest',type=Path,required=True)
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--expected-input-sha256',required=True)
    args=ap.parse_args()
    assert not args.out.exists() or not any(args.out.iterdir()), 'Use a new or empty output directory'
    cfg=json.loads(args.config.read_text())
    expected=json.loads(args.models_manifest.read_text())['llama']
    model=cfg['models']['llama'];path=Path(model['path']).expanduser().resolve()
    assert model['model_id']==expected['model_id'] and model['revision']==expected['revision']
    required=('config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json')
    checked={}
    for name in required:
        identity=expected['checkpoint_files'][name];p=path/name
        assert p.stat().st_size==identity['bytes'] and sha(p)==identity['sha256'],name
        checked[name]=identity['sha256']
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    items=calibration_inputs(cfg,tok,args.parent,args.out)
    actual=sha(args.out/'calibration_inputs.json')
    assert actual==args.expected_input_sha256, 'Historical input metadata did not reconstruct exactly'
    assert len(items)==8
    dump(args.out/'input_reconstruction.json',{'status':'passed','passages':8,
        'calibration_inputs_sha256':actual,'calibration_tokens_sha256':sha(args.out/'calibration_tokens.pt'),
        'tokenizer_files':checked,'model_forwards':0,'model_loaded':False,'gpu_used':False,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'note':'Only the historical exclusion metadata is reconstructed. No Llama weights or old model outputs are needed.'})
    print('RECONSTRUCTED_INPUT_METADATA',actual)
if __name__=='__main__':main()
