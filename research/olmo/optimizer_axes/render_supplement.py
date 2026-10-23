#!/usr/bin/env python3
import pathlib,json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
ROOT=pathlib.Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DV-Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
ss=getSampleStyleSheet();ss.add(ParagraphStyle(name='T',fontName='DV-Bold',fontSize=21,leading=25,spaceAfter=12,textColor=colors.HexColor('#12324A')))
ss.add(ParagraphStyle(name='B',fontName='DV',fontSize=10,leading=14,spaceAfter=9))
ss.add(ParagraphStyle(name='S',fontName='DV',fontSize=8.3,leading=11,spaceAfter=7))
ss.add(ParagraphStyle(name='H',fontName='DV-Bold',fontSize=12,leading=16,spaceBefore=8,spaceAfter=6,textColor=colors.HexColor('#12324A')))
parts=[]
def p(s,style='B'):parts.append(Paragraph(s,ss[style]))
p('Adam’s effect depends on<br/>which comparison we make','T')
p('OLMo-1B-0724 | Native step954000 | Preselected 128-axis follow-up | 23 October 2026','S')
p('<b>New result:</b> within neuron 1710, its special output coordinate 1764 has a lower Adam second moment than <b>all 128</b> preselected other residual coordinates. Its inverse-denominator gain is 2.08 times the sample median. Adam therefore favors this axis within the neuron, even though it damps this neuron relative to other neurons at the same axis.')
p('The two results answer different questions','H')
p('<b>Across neurons, at fixed residual coordinate 1764:</b> the complete 8,192-entry row shows the primary has the largest v and smallest inverse-denominator gain. Its first-moment magnitude is also the largest. A tiny-v advantage relative to other neurons is absent at this snapshot.')
p('<b>Across residual axes, within neuron 1710:</b> all 128 sampled v values exceed the primary’s. Its first-moment magnitude is only 1.29 times the sample median; 50 sampled axes have a larger |m|. After adaptive scaling, only 10 have a larger |q|. This is direct evidence of a current axis-dependent gain advantage, not historical causality.')
rows=[['Quantity','Primary axis 1764','Sample median','Ratio'],['|First moment m|','0.000172333','0.000133881','1.29'],['Second moment v','4.5913e-7','2.0081e-6','0.229'],['1 / (sqrt(v)+epsilon)','1454.35','700.74','2.08'],['|Adaptive push q|','0.250632','0.0986956','2.54']]
t=Table([[Paragraph(x,ss['S']) for x in row] for row in rows],colWidths=[170,118,118,76],hAlign='LEFT')
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F5')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#B8C8D2')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));parts.extend([t,Spacer(1,10)])
p('Epsilon is not the main source of this within-neuron advantage','H')
p('For the primary, epsilon/sqrt(v)=0.0148; the sampled median is 0.00706. None of the sampled axes is epsilon-dominated. Thus the within-neuron gain difference principally reflects v, whereas epsilon strongly filters the many smaller-gradient neurons in the earlier row comparison. The approximately twofold gain advantage is not an unbounded denominator explosion.')
p('Sampling, provenance and limits','H')
p('The 128 other axes were selected uniformly without replacement from 2,047 before acquisition, with seed 202610230619. All 256 official HTTP 206 range reads succeeded: 1,024 tensor bytes total. Previously verified optimizer headers were reused; no model weights, gradients or training were added. Exact ranges, raw bytes, protocol hash and offline checks are included.','S')
p('With zero sampled axes having larger gain, the exact two-sided 95% finite-population interval allows 0–56 such axes among 2,047: 0–2.74%. This is sample uncertainty, not a measured full-column rank. The interval was independently checked using SciPy’s hypergeometric distribution. These observations are limited to one late checkpoint and one neuron; m is momentum, not the current gradient. Lower v may be a consequence of the mature circuit rather than its cause.','S')
p('This supplement preserves and refines the immutable 06:15 UTC report. It does not overturn that report’s row-local result, and it does not establish that Adam originally created the super-weight.','S')
def footer(c,d):
 c.setFont('DV',8);c.setFillColor(colors.HexColor('#476274'));c.drawString(45,25,'OLMo training-origin research | Separate post-06:15 UTC supplement | No training performed')
SimpleDocTemplate(str(ROOT/'axis_probe_supplement.pdf'),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=38,bottomMargin=42,title='OLMo: Adam row-versus-axis comparison').build(parts,onFirstPage=footer,onLaterPages=footer)
