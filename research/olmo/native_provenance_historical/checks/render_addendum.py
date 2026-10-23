from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
R=Path(__file__).resolve().parents[1];D=R/'deliverables';F='/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('DV',F+'DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DV-Bold',F+'DejaVuSans-Bold.ttf'))
b=ParagraphStyle('body',fontName='DV',fontSize=9.3,leading=13.2,spaceAfter=9,allowWidows=0,allowOrphans=0,textColor=HexColor('#243341'))
s={'body':b,'title':ParagraphStyle('title',parent=b,fontName='DV-Bold',fontSize=21,leading=25,spaceAfter=12),'h2':ParagraphStyle('h2',parent=b,fontName='DV-Bold',fontSize=12.5,leading=17,spaceBefore=8,spaceAfter=7,keepWithNext=True),'source':ParagraphStyle('source',parent=b,fontSize=7.2,leading=10,spaceAfter=8)}
story=[];sources=False
for p in (D/'native_provenance_addendum.md').read_text().split('\n\n'):
 p=p.strip()
 if not p:continue
 if p=='[PAGEBREAK]':story.append(PageBreak());continue
 if p.startswith('# '):style='title';p=p[2:]
 elif p.startswith('## '):style='h2';p=p[3:];sources=sources or p=='Sources and evidence'
 else:style='source' if sources else 'body'
 p=html.escape(p.replace('\n',' '));p=re.sub(r'https?://[^\s]+',lambda m:'<link href="'+m[0]+'" color="#1767a3">'+m[0]+'</link>',p)
 story.append(Paragraph(p,s[style]))
def footer(c,d):
 c.setStrokeColor(HexColor('#dce2e8'));c.line(45,35,550,35);c.setFont('DV',7);c.drawString(45,23,'OLMo-1B training-state provenance | Metadata-only addendum | 23 October 2026');c.drawRightString(550,23,str(d.page))
SimpleDocTemplate(str(D/'native_provenance_addendum.pdf'),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=45,title='OLMo-1B native training-state provenance').build(story,onFirstPage=footer,onLaterPages=footer)
print(D/'native_provenance_addendum.pdf')
