"""Fresh gate calibration and evaluation inputs; no model inference."""
import hashlib,json,math
from pathlib import Path
import torch
from datasets import Dataset
from run_transfer import ROOT,sha,th,dump,strip_book
from fresh_inputs_v2 import prepare_fresh

def prepare_gate_inputs(cfg,tok,out):
    oldpath=ROOT/'gpt2_run/data_manifest.json';old=json.loads(oldpath.read_text())
    calpath=ROOT/'llama_replacements_run/calibration_inputs.json';previous_cal=json.loads(calpath.read_text())
    cal=[];records=[];excluded={}
    for corpus in ('wiki_test','hume'):
        path=Path(cfg['sources'][corpus]);assert sha(path)==old['sources'][corpus]['sha256']
        text='\n\n'.join(Dataset.from_file(str(path))['text']) if corpus=='wiki_test' else strip_book(path.read_text())
        assert hashlib.sha256(text.encode()).hexdigest()==old['sources'][corpus]['processed_text_sha256']
        spans=[]
        for r in [r for r in old['passages'] if r['split']=='development']+previous_cal['passages']:
            if r['corpus']!=corpus:continue
            a=r['char_start'];assert hashlib.sha256(text[a:a+8192].encode()).hexdigest()==r['passage_sha256'];spans.append([a,a+8192])
        assert len(spans)==8;excluded[corpus]=spans;fresh=[]
        for w,f in enumerate((.25,.45,.65,.85)):
            a=math.floor(f*(len(text)-8192));ss=text[a:a+8192];end=a+8192
            assert all(end<=lo or a>=hi for lo,hi in spans+fresh),(corpus,a,end,spans)
            enc=tok(ss,add_special_tokens=False,return_offsets_mapping=True);assert len(enc['input_ids'])>=512
            x=torch.tensor(enc['input_ids'][:512]);fresh.append([a,end]);cal.append({'corpus':corpus,'window':w,'tokens':x})
            records.append({'corpus':corpus,'window':w,'char_start':a,'used_char_end':a+enc['offset_mapping'][511][1],
                'extraction_char_end':end,'passage_sha256':hashlib.sha256(ss.encode()).hexdigest(),'window_token_sha256':th(x),'first_token_id':int(x[0])})
    torch.save(cal,out/'calibration_tokens.pt');dump(out/'calibration_inputs.json',{'passages':records,'excluded_full_buffers':excluded,
        'gpt2_development_manifest_sha256':sha(oldpath),'previous_llama_calibration_manifest_sha256':sha(calpath),
        'query_positions':[1,510],'vectors_per_layer':4080,'passages_per_layer':8})
    prior=('gpt2_groups_run','gpt2_writes_run','gpt2_layer_writes_run','gpt2_regrowth_kv_run','gpt2_value_geometry_run','gpt2_query_source_run','gpt2_query_mixture_run')
    items,dm=prepare_fresh(cfg,tok,ROOT/'gpt2_run',out,fractions=(.075,.275,.475,.675),additional_parents=tuple(ROOT/r for r in prior))
    assert all(len(v)==32 for v in dm['old_used_spans'].values())
    return cal,items,dm
