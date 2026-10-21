"""Render the gate report with locally embedded figures, without inference."""
import argparse,base64,re
from pathlib import Path
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    body=MarkdownIt('commonmark').enable('table').render((ROOT/'docs/GPT2_GATE_SUBSPACES.md').read_text())
    def embed(m):
        p=(ROOT/'docs'/m.group(1)).resolve();assert p.is_relative_to(ROOT/'figures')and p.suffix=='.png'
        return 'src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
    body=re.sub(r'src="(\.\./figures/[^\"]+)"',embed,body)
    assert not re.search(r'<img[^>]+src="(?!data:)',body)
    page='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-2 causal gate subspaces</title><style>body{font:17px/1.6 system-ui;max-width:1080px;margin:auto;padding:24px;color:#172631}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px}td,th{border:1px solid #ccd5dc;padding:7px}code{overflow-wrap:anywhere}</style></head><body>'+body+'</body></html>'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(page);print(a.output)
if __name__=='__main__':main()
