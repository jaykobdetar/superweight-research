"""Run the immutable T20 safe-array NumPy audit from the repository layout.

This launcher changes paths only. It does not load models or reconstruct omitted
IDs/activations. A report is optional; existing reports are never overwritten.
"""
import argparse,json,os,resource,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--report',type=Path)
 a=p.parse_args()
 package=ROOT/'results/post_write_norm_matching'
 code=ROOT/'results/post_write_norm_reproduction/experiments/pulse_transfer'
 if a.report:
  assert not a.report.exists(),'Preserve existing reports'
  for protected in (package,ROOT/'results/post_write_norm_reproduction'):assert not a.report.resolve().is_relative_to(protected.resolve()),'Report outside immutable package required'
 sys.path.insert(0,str(code))
 import audit_post_write_norm_public as backend
 assert Path(backend.__file__).resolve()==code/'audit_post_write_norm_public.py'
 start=time.monotonic();r=backend.audit(package,code)
 r.update(wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 if a.report:
  with a.report.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(r,indent=2,allow_nan=False))
if __name__=='__main__':main()
