from pathlib import Path
import re,html,json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white,black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
C=Path(__file__).resolve().parents[1];B=C.parent;D=C/'deliverables';fd='/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('DV',fd+'DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DV-Bold',fd+'DejaVuSans-Bold.ttf'));pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold')
body=ParagraphStyle('body',fontName='DV',fontSize=9.1,leading=12.8,spaceAfter=7,allowWidows=0,allowOrphans=0,textColor=HexColor('#202a33'))
styles={'body':body,'title':ParagraphStyle('title',parent=body,fontName='DV-Bold',fontSize=21,leading=26,spaceAfter=12,textColor=black),'h2':ParagraphStyle('h2',parent=body,fontName='DV-Bold',fontSize=12.8,leading=16,spaceBefore=10,spaceAfter=7,keepWithNext=True,textColor=black),'source':ParagraphStyle('source',parent=body,fontSize=7.1,leading=9.4,spaceAfter=6),'caption':ParagraphStyle('caption',parent=body,fontSize=7.5,leading=10.4,textColor=HexColor('#55616e'))}
def fmt(t):
 t=html.escape(t);return re.sub(r'https?://[^\s]+',lambda m:f'<link href="{m[0]}" color="#1767a3">{m[0]}</link>',t)
summary=[r for r in json.loads((D/'checkpoint_summary.json').read_text()) if r['neuron']==1710]
def label(r):return 'release' if r['checkpoint']=='main' else str(r['step']//1000)+'k'+('*' if r['selection_cohort']=='adaptive' else '')
def styled_table(data,widths,fs=7.3,pad=5):
 t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DV'),('FONTSIZE',(0,0),(-1,-1),fs),('BACKGROUND',(0,0),(-1,0),HexColor('#143e5f')),('TEXTCOLOR',(0,0),(-1,0),white),('FONTNAME',(0,0),(-1,0),'DV-Bold'),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f0f4f7')]),('BOTTOMPADDING',(0,0),(-1,-1),pad),('TOPPADDING',(0,0),(-1,-1),pad),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('GRID',(0,0),(-1,-1),.35,HexColor('#D9D9D9'))]));return t
def measurements():
 data=[['Checkpoint','Down weight','Source The','Source period','Later U','H12 fraction','H14 attn']]
 for r in summary:
  n=r['checkpoint'];phase='phase2' if r['selection_cohort']=='initial' else 'phase3';z=json.loads((B/phase/'results'/n/'float32.json').read_text());weight=json.loads((B/'sources/trajectory'/n/'result.json').read_text())['primary'];h=r['H12_restoration_fraction_median'];data.append([label(r),f'{weight:.4f}',f"{r['single_The_contribution']:.4g}",f"{z['single_tokens']['15']['contribution'][0][0]:.4g}",f"{r['newline_U_median']:.4g}",'-' if h is None else f'{h:.4g}',f"{r['H14_R_attention_p_median']:.3f}"])
 t=styled_table(data,[65,66,72,75,80,73,66],7.2,4.5);t.setStyle(TableStyle([('ALIGN',(1,1),(-1,-1),'RIGHT')]));return t
def revisions():
 data=[['Checkpoint','Selection','Immutable model commit']]+[[label(r),r['selection_cohort'],r['commit']] for r in summary];return styled_table(data,[77,68,352],7.1,2.0)
story=[];sources=False
for p in (D/'training_development_report.md').read_text().split('\n\n'):
 p=p.strip()
 if not p:continue
 if p=='[PAGEBREAK]':story.append(PageBreak());continue
 if p=='[CHECKPOINT_TABLE]':story.extend([measurements(),Spacer(1,7)]);continue
 if p=='[REVISION_TABLE]':story.extend([revisions(),Spacer(1,8)]);continue
 if p.startswith('!['):
  name=re.search(r'\(([^)]+)\)',p)[1];rat={'functional_trajectory.png':8/12,'query_compatibility.png':5/11.5,'primary_source_tokens.png':3.5/11.5}[name];story.extend([Spacer(1,4),Image(str(D/name),width=497,height=497*rat),Spacer(1,5)]);continue
 if p.startswith('# '):sty='title';t=p[2:]
 elif p.startswith('## '):sty='h2';t=p[3:];sources=sources or t=='Primary sources'
 else:sty='source' if sources else 'body';t=p.replace('\n',' ')
 if sty=='body' and re.match(r'^\d+\. ',p):
  for line in p.splitlines():
   if line.strip():story.append(Paragraph(fmt(line),styles[sty]))
 else:story.append(Paragraph(fmt(t),styles[sty]))
def footer(c,d):
 c.setStrokeColor(HexColor('#dce2e8'));c.line(45,35,550,35);c.setFont('DV',7);c.setFillColor(HexColor('#627182'));c.drawString(45,23,'OLMo 1B training development | 8 initial + 4 adaptive checkpoints | 22 October 2026');c.drawRightString(550,23,str(d.page))
out=D/'training_development_report.pdf';SimpleDocTemplate(str(out),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=45,title='OLMo 1B super weight development across training',author='Research investigation').build(story,onFirstPage=footer,onLaterPages=footer);print(out)
