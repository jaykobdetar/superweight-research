#!/usr/bin/env python3
"""Reuse the immutable saved tokenization and original anchor records.

Run with --out /new/directory. Requires the original pinned local model cache;
does not download weights. The imported runner source is unchanged from the run.
"""
import shutil
from pathlib import Path
import torch
import run_tranche as experiment

def saved_prepare(out):
 source=Path(__file__).resolve().parent/'run'
 if out.resolve()==source.resolve():raise ValueError('Use a new output directory')
 x=torch.load(source/'tokens.pt',map_location='cpu',weights_only=True)
 shutil.copy2(source/'tokens.pt',out/'tokens.pt')
 shutil.copy2(source/'data_manifest.json',out/'data_manifest.json')
 return x['corpora'],x['starts'],x['start_tokens']

experiment.prepare=saved_prepare
experiment.OLD=Path(__file__).resolve().parent/'provenance/previous_experiment'
experiment.main()
