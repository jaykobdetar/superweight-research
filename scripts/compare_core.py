"""Compare all exported per-window scalar observations with a new core run."""
import argparse,json,math
from pathlib import Path
from run_core import JOBS,REPO
def records(p):
 rows=list(map(json.loads,p.read_text().splitlines()));result={(x['group'],x['trial'],x['window']):x for x in rows};assert len(rows)==len(result),f'Duplicate record in {p}';assert rows,f'Empty result file {p}';return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--atol',type=float,default=1e-5);a=ap.parse_args();reports=[]
 for job in JOBS:
  old=records(REPO/'results/pulse_continuation'/(job+'_run/windows.jsonl'));new=records(a.work/'experiments/pulse_continuation'/(job+'_run/windows.jsonl'));assert old.keys()==new.keys(),job
  diffs={k:max(abs(old[i][k]-new[i][k]) for i in old) for k in ('nll','baseline_nll','agreement','kl_clean_to_condition')};same=all(old[i]['token_sha256']==new[i]['token_sha256'] and old[i]['score_start']==new[i]['score_start'] and old[i]['scored_tokens']==new[i]['scored_tokens'] for i in old);ok=same and all(v<=a.atol for v in diffs.values());reports.append({'job':job,'rows':len(old),'inputs_scoring_exact':same,'max_absolute_errors':diffs,'pass':ok})
 out={'tolerance':a.atol,'all_pass':all(r['pass'] for r in reports),'batches':reports};(a.work/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['all_pass'],'Inspect differences; never silently adjust tolerance.'
if __name__=='__main__':main()
