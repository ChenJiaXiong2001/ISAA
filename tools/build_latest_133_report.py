# -*- coding: utf-8 -*-
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(r"C:\Users\15637\PycharmProjects\ISAA")
OUT = ROOT / "docs" / "基于133节点全身骨架的动作识别研究报告.docx"
MD = ROOT / "docs" / "基于133节点全身骨架的动作识别研究报告.md"
SAMPLE = ROOT / "outputs" / "ntu_rgb_samples"

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)
def set_cell(cell, text, bold=False, color='1F2937', size=10):
    cell.text = ''
    p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.bold = bold; r.font.size = Pt(size); r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.style = 'Table Grid'
    for i,h in enumerate(headers): set_cell(t.rows[0].cells[i], h, True, 'FFFFFF', 10); shade(t.rows[0].cells[i], '17365D')
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,v in enumerate(row): set_cell(cells[i], str(v), i==0, '1F2937', 9); shade(cells[i], 'F4F7FA' if ri%2 else 'FFFFFF')
    return t
def para(doc, text, boldlead=None):
    p=doc.add_paragraph()
    if boldlead and text.startswith(boldlead):
        r=p.add_run(boldlead); r.bold=True; p.add_run(text[len(boldlead):])
    else: p.add_run(text)
    p.paragraph_format.space_after=Pt(6); p.paragraph_format.line_spacing=1.15
    return p
def heading(doc, text, level=1):
    p=doc.add_heading(text, level=level); p.paragraph_format.space_before=Pt(10); p.paragraph_format.space_after=Pt(5); return p

doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.8); sec.right_margin=Inches(.8)
styles=doc.styles; styles['Normal'].font.name='Microsoft YaHei'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); styles['Normal'].font.size=Pt(10.5)
title=doc.add_paragraph(style='Title'); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; title.add_run('基于133节点全身骨架的动作识别研究')
sub=doc.add_paragraph(); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER; sub.add_run('NTU RGB+D 60  ·  RTMW-133  ·  Joint单流方法对照').italic=True
para(doc,'本报告围绕基于RTMW-133全身二维骨架的动作识别项目展开，说明数据流程、133节点结构、手部图拓扑、模型方法和实验结果。结果部分以CTR-GCN原论文单流Joint的88.9%作为主要对照，同时保留项目内部原池化方案和SkeleT-CTRGCN Joint单流的已确认结果。')

heading(doc,'1 研究目标',1)
para(doc,'研究目标是利用包含身体、足部、面部和双手局部信息的133节点全身骨架完成NTU RGB+D 60动作分类，并重点观察高密度全身节点对细粒度动作识别的作用。')

heading(doc,'2 数据与输入表示',1)
table(doc,['项目','设置'],[['数据集','NTU RGB+D 60'],['划分','NTU60 X-Sub'],['输入','RTMW-133全身二维骨架'],['时间窗口','100帧'],['最大人物数','2'],['节点特征','x、y、score'],['类别数','60类']])
para(doc,'项目数据链路为：RGB视频经过人体检测与关键点提取，得到RTMW-133全身骨架，再进行节点清理、置信度处理和100帧序列整理，最后输入时空图模型完成60类动作分类。')

heading(doc,'3 133节点全身骨架结构',1)
para(doc,'RTMW-133表示保留身体主体、足部、双手和面部的全身局部信息。身体和足部结构用于表达整体姿态与移动状态，双手结构用于表达拍手、写字、玩手机等局部动作，面部节点用于补充头部和面部相关姿态。')
table(doc,['区域','节点数量','作用'],[['身体主体','17','躯干、头部和四肢主体姿态'],['足部','6','脚尖、脚跟和足部方向'],['左手','21','腕点与五指局部运动'],['右手','21','腕点与五指局部运动'],['面部','68','面部局部姿态与表情相关位置'],['合计','133','全身二维骨架表示']])
para(doc,'左右手采用相同的21点局部拓扑。局部编号为0腕点，1至4拇指，5至8食指，9至12中指，13至16无名指，17至20小指。全身索引中左手为91至111，右手为112至132。')

heading(doc,'4 模型方法',1)
heading(doc,'4.1 CTR-GCN原论文单流Joint',2)
para(doc,'CTR-GCN通过通道级拓扑细化调整节点关系，使不同特征通道能够学习不同的骨架连接模式。单流Joint以关节表示作为输入，通过空间图卷积建模节点关系，再通过时间模块建模动作随时间的变化。')
heading(doc,'4.2 SkeleT-CTRGCN Joint单流',2)
para(doc,'项目中的SkeleT-CTRGCN Joint单流使用RTMW骨架派生节点表达，并采用progressive graph mapping、unit_ctrgcn和mstcn完成空间与时间建模。该结构属于项目中的Joint单流实现，服务于133节点全身骨架动作识别流程。训练入口强制启用torch.compile，编译范围覆盖模型主体、节点映射、神经编码和分类器。')
heading(doc,'4.3 方法不足',2)
para(doc,'节点映射会压缩部分原始133点信息；二维骨架不包含深度信息；手部细粒度动作仍可能出现相似类别混淆；高密度节点带来更高的计算量和数据质量要求。')

heading(doc,'5 实验结果',1)
table(doc,['方案','实验设置','Top-1准确率'],[['CTR-GCN原论文单流Joint','原论文单流Joint设置','88.9%'],['原池化项目','RTMW-133、100帧、100轮','94.6400%'],['SkeleT-CTRGCN Joint单流','RTMW派生节点、100帧、10-clip评估','95.5722%']])
para(doc,'相对CTR-GCN原论文单流Joint，原池化项目提升5.7400个百分点，SkeleT-CTRGCN Joint单流10-clip结果提升6.6722个百分点。SkeleT-CTRGCN Joint单流相较原池化项目提升0.9322个百分点。上述数值用于描述方案结果差异，具体输入表达和评估协议以表中设置为准。')

heading(doc,'6 手部动作样例',1)
para(doc,'服务器原始RGB样例用于展示项目数据链路和手部动作识别现象，不作为整体准确率统计。A010拍手预测为拍手，A012写字预测为写字，A029玩手机或平板被预测为掉落物品，说明局部手部动作仍存在细粒度混淆。')
for name,cap in [('A010.png','A010 拍手'),('A012.png','A012 写字'),('A029.png','A029 玩手机或平板')]:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    f=SAMPLE/name
    if f.exists(): p.add_run().add_picture(str(f), width=Inches(2.05)); c=doc.add_paragraph(cap); c.alignment=WD_ALIGN_PARAGRAPH.CENTER

heading(doc,'7 结论',1)
para(doc,'本项目构建了基于RTMW-133全身骨架的动作识别流程，并围绕133节点组成、手部局部拓扑和Joint单流图建模进行了整理。实验表明，原池化项目达到94.6400% Top-1，SkeleT-CTRGCN Joint单流10-clip评估达到95.5722%，均高于报告中采用的CTR-GCN原论文单流Joint对照值88.9%。')
doc.core_properties.title='基于133节点全身骨架的动作识别研究'; doc.save(OUT)

MD.write_text('''# 基于133节点全身骨架的动作识别研究\n\n## 实验结果\n\n|方案|实验设置|Top-1|\n|---|---|---:|\n|CTR-GCN原论文单流Joint|原论文单流Joint设置|88.9%|\n|原池化项目|RTMW-133、100帧、100轮|94.6400%|\n|SkeleT-CTRGCN Joint单流|RTMW派生节点、100帧、10-clip评估|95.5722%|\n\n原池化相较CTR-GCN原论文单流提升5.7400个百分点；SkeleT-CTRGCN Joint单流10-clip相较CTR-GCN原论文单流提升6.6722个百分点；相较原池化项目提升0.9322个百分点。\n''', encoding='utf-8')
print(OUT)

