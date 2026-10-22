"""Build a self-contained Report16 HTML artifact from the reviewed repo report.
No model/network execution; local images are embedded and source links point to
an explicit repository revision. Existing outputs are never overwritten.
"""
import argparse,base64,re
from pathlib import Path
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--revision',default='main');a=p.parse_args()
 assert re.fullmatch(r'[A-Za-z0-9_.-]+',a.revision),'Simple branch or commit required'
 src=ROOT/'docs/GPT2_POST_WRITE_NORM.md';body=MarkdownIt('commonmark',{'html':False}).enable('table').render(src.read_text())
 def img(m):
  path=(src.parent/m.group(1)).resolve();assert path.is_relative_to((ROOT/'figures/post_write_norm_matching').resolve()) and path.suffix=='.png'
  return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
 body=re.sub(r'src="([^"#]+)"',img,body)
 def link(m):
  target=m.group(1)
  if re.match(r'^[a-z]+:',target) or target.startswith('#'):return m.group(0)
  path,sep,fragment=target.partition('#');resolved=(src.parent/path).resolve();assert resolved.is_relative_to(ROOT.resolve()) and resolved.is_file(),('Missing report link',target)
  return 'href="https://github.com/jaykobdetar/superweight-research/blob/'+a.revision+'/'+resolved.relative_to(ROOT.resolve()).as_posix()+(sep+fragment if sep else '')+'"'
 body=re.sub(r'href="([^"]+)"',link,body)
 html='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Report16: actual GPT-2 attention-write norm controls</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#172631;background:#f7f8fa;max-width:1100px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.2}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:13px}td,th{border:1px solid #ccd5dc;padding:7px 10px}th{background:#e7eef3}a{color:#155a8c}code{font-size:.88em;overflow-wrap:anywhere}pre{overflow-x:auto;white-space:pre}</style></head><body>'+body+'</body></html>'
 with a.output.open('x') as f:f.write(html)
 print('Report16 HTML bytes',a.output.stat().st_size)
if __name__=='__main__':main()
