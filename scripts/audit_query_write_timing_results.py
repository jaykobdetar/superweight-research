"""Audit published T21 safe numeric results using the pinned NumPy-only auditor."""
import argparse,hashlib,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'results/query_write_timing_reproduction/experiments/pulse_transfer'
AUDITOR=CODE/'audit_query_write_timing_public.py'
PIN='359249dc3850891fb16b4f9f8cbdd849a2673fb803106ed79ef23f7459c9a1d5'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--report',type=Path,default=ROOT/'query_write_timing_public_audit.json')
 args=p.parse_args()
 if hashlib.sha256(AUDITOR.read_bytes()).hexdigest()!=PIN:raise RuntimeError('Pinned public auditor source differs')
 sys.dont_write_bytecode=True
 sys.argv=[str(AUDITOR),'--code',str(CODE),'--package',str(ROOT/'results/query_write_timing'),'--report',str(args.report)]
 runpy.run_path(str(AUDITOR),run_name='__main__')
if __name__=='__main__':main()
