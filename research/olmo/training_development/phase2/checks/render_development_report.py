from pathlib import Path
import re,html,json,csv
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
R=Path(__file__).resolve().parents[1];D=R/'deliverables';font='/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('DV',font+'DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DV-Bold',font+'DejaVuSans-Bold.ttf'));pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold')
body=ParagraphStyle('body',fontName='DV',fontSize=9.2,leading=13.2,spaceAfter=7,allowWidows=0,allowOrphans=0,textColor=HexColor('#202a33'))
styles={'body':body,'title':ParagraphStyle('title',parent=body,fontName='DV-Bold',fontSize=22,leading=27,spaceAfter=12,textColor=HexColor('#143e5f')),'h2':ParagraphStyle('h2',parent=body,fontName='DV-Bold',fontSize=13.2,leading=17,spaceBefore=12,spaceAfter=8,keepWithNext=True,textColor=HexColor('#143e5f')),'source':ParagraphStyle('source',parent=body,fontSize=7.5,leading=10.7),'caption':ParagraphStyle('caption',parent=body,fontSize=8,leading=11,textColor=HexColor('#55616e'))}
def fmt(t):
 t=html.escape(t);return re.sub(r'https?://[^\s]+',lambda m:f'<link href="{m[0]}" color="#1767a3">{m[0]}</link>',t)
def table():
 src={r['checkpoint']:r for r in json.loads((D/'checkpoint_summary.json').read_text()) if r['neuron']==1710};tr={r['name']:r for r in json.loads((R.parent/'checks/trajectory_results.json').read_text())};names=sorted(src,key=lambda x:10**10 if x=='main' else int(x.split('-')[0][4:]));heads=['Checkpoint','Down weight','Source The','Source period','Later newline U','H12 fraction'];data=[heads]
 for n in names:
  s=json.loads((R/'results'/n/'float32.json').read_text());r=src[n];h=r['H12_restoration_fraction_median'];data.append(['final*' if n=='main' else str(int(n.split('-')[0][4:])//1000)+'k',f"{tr[n]['primary']:.4f}",f"{r['single_The_contribution']:.4g}",f"{s['single_tokens']['15']['contribution'][0][0]:.4g}",f"{r['newline_U_median']:.4g}",'-' if h is None else f'{h:.4g}'])
 t=Table(data,colWidths=[65,75,80,80,95,92],repeatRows=1);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DV'),('FONTSIZE',(0,0),(-1,-1),7.5),('BACKGROUND',(0,0),(-1,0),HexColor('#143e5f')),('TEXTCOLOR',(0,0),(-1,0),white),('FONTNAME',(0,0),(-1,0),'DV-Bold'),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f0f4f7')]),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('ALIGN',(1,1),(-1,-1),'RIGHT'),('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#bccbd6'))]));return t
story=[];sources=False
for p in (D/'training_development_report.md').read_text().split('\n\n'):
 p=p.strip()
 if not p:continue
 if p=='[CHECKPOINT_TABLE]':story += [table(),Spacer(1,7),Paragraph('All contributions use neuron 1710 and output coordinate 1764. Later newline U is the median of four fixed probes without an earlier newline. H12 fraction compares donor-patched, ordinary and isolated empty-history conditions; tiny denominators are omitted. *Final is the pinned revision, not an assigned training step.',styles['caption'])];continue
 if p.startswith('!['):
  name=re.search(r'\(([^)]+)\)',p)[1];width=497;height=width*(4.8/11.5 if name=='source_cross_time.png' else 8/12);story += [Spacer(1,5),Image(str(D/name),width=width,height=height),Spacer(1,7)];continue
 if p.startswith('# '):style='title';t=p[2:]
 elif p.startswith('## '):style='h2';t=p[3:];sources=sources or t=='Primary sources'
 else:style='source' if sources else 'body';t=p.replace('\n',' ')
 if style=='body' and re.match(r'^\d+\. ',p) and '\n' in p:
  for line in p.splitlines():
   if line.strip():story.append(Paragraph(fmt(line),styles[style]))
 else:
  para=Paragraph(fmt(t),styles[style]);story.append(KeepTogether([para]) if t.startswith('Separately vary weight decay') else para)
def footer(c,d):
 c.setStrokeColor(HexColor('#dce2e8'));c.line(45,38,550,38);c.setFont('DV',7);c.setFillColor(HexColor('#627182'));c.drawString(45,25,'OLMo-1B training development | Initial eight-point study | 22 October 2026');c.drawRightString(550,25,str(d.page))
out=D/'training_development_report.pdf';doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=43,bottomMargin=49,title='OLMo-1B: how the super-weight mechanism develops',author='Research investigation');doc.build(story,onFirstPage=footer,onLaterPages=footer);print(out)
