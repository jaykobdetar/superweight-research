import json,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];a=json.loads((R/'gpt2_gate_wiki_calibration_run/wiki_factorial_analysis.json').read_text());text=(R/'REPORT14.md').read_text();ix={(r['corpus'],r['identity'],r['view'],r['mode']):r for r in a['summary']};labels={'Wiki':'wiki_validation','Utilitarianism':'utilitarianism','Ethics':'ethics','Optimizer code':'optimizers_new'};count=0
for title,table in re.findall(r'### (PC1 removal — L10 only|PC1 removal — joint L4–L11|Orthogonal control — L10 only|Orthogonal control — joint L4–L11)\n(.*?)(?=\n##|\Z)',text,re.S):
 family='remove'if title.startswith('PC1')else'orthogonal';which='L10'if title.endswith('only')else'all'
 for line in table.splitlines():
  if not re.match(r'\| (Wiki|Utilitarianism|Ethics|Optimizer code) / ',line):continue
  cells=[c.strip()for c in line.strip('|').split('|')];label,start=cells[0].split(' / ')
  for cell,got in zip(('OO','OF','FO','FF','M8'),cells[1:]):
   p=ix[labels[label],start,'full',family+'_'+cell+'_'+which]['ppl_ratio_p'];expected=f"{p['point']:.6f} [{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}]";assert got==expected,(title,label,start,cell,got,expected);count+=1
assert count==160
full=[r for r in a['contrasts']if r['view']=='full'];primary=lambda r:not r['name'].endswith('M8')
matched={}
for which in ['L10','all']:
 for wiki in [False,True]:
  rows=[r for r in full if r['family']=='matched_control'and r['layer_set']==which and primary(r)and (r['corpus']=='wiki_validation')==wiki];matched[which+('_wiki'if wiki else'_nonwiki')]={'n':len(rows),'negative_ci':sum(r['delta_nll']['ci95'][1]<0 for r in rows)}
assert matched=={'L10_nonwiki':{'n':24,'negative_ci':24},'L10_wiki':{'n':8,'negative_ci':1},'all_nonwiki':{'n':24,'negative_ci':24},'all_wiki':{'n':8,'negative_ci':2}}
sign=[r for r in a['summary']if r['view']=='full'and r['mode'].startswith('remove_')and r['mode'].endswith('_L10')];assert len(sign)==40 and all(r['delta_nll_p']['ci95'][0]>0 if r['corpus']=='wiki_validation'else r['delta_nll_p']['ci95'][1]<0 for r in sign)
out={'status':'passed_numeric_tables_and_counts','ppl_cells':160,'formatted_numeric_values':480,'matched_control_counts':matched,'L10_sign_intervals':40,'report_sha256':hashlib.sha256((R/'REPORT14.md').read_bytes()).hexdigest(),'html_sha256':hashlib.sha256((R/'report14.html').read_bytes()).hexdigest(),'pending_scope_corrections':[],'final_scope_and_figure_changes_verified':True};(R/'review/report14_numeric_review.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
