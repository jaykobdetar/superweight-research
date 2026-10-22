"""Audit Report 18's safe numerical results with the pinned NumPy-only auditor."""
import argparse,hashlib,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'results/l11_key_value_donor_reproduction/experiments/pulse_transfer'
AUDITOR=CODE/'audit_l11_key_value_donor_public.py'
PIN='019a1935bd1b95e3a19f965de9d4db878159e23164d53d8b7e63ff8e889118f3'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--report',type=Path,default=ROOT/'l11_key_value_donor_public_audit.json')
 args=p.parse_args()
 if hashlib.sha256(AUDITOR.read_bytes()).hexdigest()!=PIN:raise RuntimeError('Pinned public auditor source differs')
 sys.dont_write_bytecode=True
 sys.argv=[str(AUDITOR),'--code',str(CODE),'--package',str(ROOT/'results/l11_key_value_donor'),'--report',str(args.report)]
 runpy.run_path(str(AUDITOR),run_name='__main__')
if __name__=='__main__':main()
