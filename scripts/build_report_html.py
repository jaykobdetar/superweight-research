"""Render the report and novelty appendix into a self-contained HTML artifact."""
import argparse,base64,html,re
from pathlib import Path
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();text=(ROOT/'docs/REPORT.md').read_text()+'\n\n---\n\n'+(ROOT/'docs/NOVELTY.md').read_text();body=MarkdownIt('commonmark').enable('table').render(text)
 def embed(m):
  rel=m.group(1);p=(ROOT/'docs'/rel).resolve();assert p.is_relative_to(ROOT) and p.suffix=='.png';return 'src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
 body=re.sub(r'src="(\.\./figures/[^\"]+)"',embed,body)
 base='https://github.com/jaykobdetar/superweight-causal-circuits/blob/main/docs/'
 body=re.sub(r'href="([A-Z_]+\.md)"',lambda m:'href="'+base+m.group(1)+'"',body)
 head='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Super-weight causal circuits — reviewed findings and reproducibility</title><style>body{max-width:1000px;margin:auto;padding:24px;font:17px/1.65 system-ui;color:#183044}h1,h2{line-height:1.2}h2{margin-top:2em}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:15px}td,th{border:1px solid #cbd7df;padding:8px;text-align:left}th{background:#edf4fa}code{background:#eef2f4;padding:2px 4px;overflow-wrap:anywhere}a{color:#12628c}</style></head><body>'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(head+body+'</body></html>');print(a.output)
if __name__=='__main__':main()
