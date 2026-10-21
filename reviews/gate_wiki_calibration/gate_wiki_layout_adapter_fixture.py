import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,copy,hashlib,resource
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import audit_gate_wiki_results_layout as a
from gate_wiki_storage import read_npz
from gate_wiki_calibration import fit_geometry
p=R/'gpt2_gate_wiki_calibration_run';torch.set_num_threads(2)
m=read_npz(p/'calibration_maps.npz');s=read_npz(p/'calibration_spectra.npz');d=read_npz(p/'calibration_directions.npz');g=read_npz(p/'calibration_geometry.npz')
fixed=a.restore_qr_layout(copy.deepcopy(m));dd,gg=fit_geometry(fixed,s);a.frozen.check_equal(dd,d);a.frozen.check_equal(gg,g)
checks=[]
for name,mutate in [('label',lambda z:z.pop('M')),('layer',lambda z:z['F'].pop(11)),('dtype',lambda z:z['F'][10].__setitem__('basis_qr',z['F'][10]['basis_qr'].float())),('shape',lambda z:z['F'][10].__setitem__('basis_qr',z['F'][10]['basis_qr'][:11])),('nonfinite',lambda z:z['F'][10]['basis_qr'].__setitem__((0,0),float('nan'))),('unexpected_stride',lambda z:z['F'][10].__setitem__('basis_qr',z['F'][10]['basis_qr'].T.contiguous().T))]:
 z=copy.deepcopy(m);mutate(z)
 try:a.restore_qr_layout(z)
 except (AssertionError,KeyError):checks.append(name)
 else:raise AssertionError(name)
old=a.FROZEN_SHA;a.FROZEN_SHA='0'*64
try:a.audit(p)
except AssertionError:checks.append('frozen_source_hash')
else:raise AssertionError('hash')
finally:a.FROZEN_SHA=old
z=copy.deepcopy(m);z['F'][10]['basis_qr'][0,0]+=1e-5
z=a.restore_qr_layout(z);dz,_=fit_geometry(z,s) if False else ({},None)
# Restoration must not hide changed values: production frozen manifest/digest and inventory guards reject them.
from gpt2_gate_subspace import maps_digest
assert maps_digest(z['F'])!=maps_digest(m['F']);checks.append('tamper_not_normalized_away')
out={'status':'passed','adapter_sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),'exact_direction_and_geometry_replay':True,'negative_checks':checks,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_forwards':0,'gpu_used':False}
(R/'review/gate_wiki_layout_adapter_fixture.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
