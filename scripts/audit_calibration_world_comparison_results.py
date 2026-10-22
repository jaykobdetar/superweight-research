"""Reanalyze Report19 restricted observations with the pinned NumPy-only auditor."""
import argparse,hashlib,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'results/calibration_world_comparison_reproduction/experiments/pulse_transfer'
AUDITOR=CODE/'audit_calibration_world_public.py'
PIN='8693b778cd8409a14eb894afdac667f1ea90e1fd021a523756b958a87f876eee'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 assert hashlib.sha256(AUDITOR.read_bytes()).hexdigest()==PIN,'Pinned auditor source differs'
 sys.dont_write_bytecode=True
 sys.argv=[str(AUDITOR),'--code',str(CODE),'--package',str(ROOT/'results/calibration_world_comparison'),'--report',str(a.report)]
 runpy.run_path(str(AUDITOR),run_name='__main__')
if __name__=='__main__':main()
