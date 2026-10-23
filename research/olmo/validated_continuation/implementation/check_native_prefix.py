import json,resource
from pathlib import Path
import numpy as np
from streamed_olmo import StreamedOlmo
R=Path(__file__).resolve().parent;resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000));m=StreamedOlmo();a=np.load(R/'protocols/frozen_inputs.npz');x=a['dubois_00'][:128];p=16
native=m.forward(x,stop_layer=3,prefix_hash_end=p)
post=m.forward(x,stop_layer=3,prefix_hash_end=p,mask_sources=(0,),mask_query_start=p+1)
global_=m.forward(x,stop_layer=3,prefix_hash_end=p,mask_sources=(0,))
r={'post_mask_prefix_all_layers_exact':native['prefix_hashes']==post['prefix_hashes'],'global_mask_prefix_differs_after_layer1':native['prefix_hashes'][2:]!=global_['prefix_hashes'][2:],'layers_checked':4,'sequence_length':128,'seconds':[z['seconds'] for z in [native,post,global_]],'peak_rss':m.peak_rss,'scope':'Partial model hidden-state invariant only, not behavioral outcome'};r['pass']=r['post_mask_prefix_all_layers_exact'] and r['global_mask_prefix_differs_after_layer1'];assert r['pass'];print(json.dumps(r,indent=2));(R/'results/native_prefix_invariant.json').write_text(json.dumps(r,indent=2))
