"""Focused architecture revision; preserve all content outside chapter 4."""
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/基于133节点全身骨架的动作识别研究报告_论文原图修订版.docx'
FINAL=ROOT/'docs/基于133节点全身骨架的动作识别研究报告_模型结构解释版.docx'
FIGURE=ROOT/'outputs/133节点全身骨架_模型结构总览.png'
d=Document(SOURCE)
start=next(i for i,p in enumerate(d.paragraphs) if p.text=='4 身体与局部并行融合模型')
end=next(i for i,p in enumerate(d.paragraphs) if p.text=='5 实验设置与原始CTR-GCN分析')
original_outside=[p.text for i,p in enumerate(d.paragraphs) if not start<i<end]
chapter=d.paragraphs[start+1:end]
text_paragraphs=[p for p in chapter if p.text and not p.style.name.startswith('Heading') and p.style.name!='Caption']
assert len(text_paragraphs)==4
new_texts=[
 '模型以133节点全身骨架序列为输入，张量形状为B×3×100×133×M，其中3个通道为x、y和置信度，B为批量大小，M最多为2。同一序列在身体、双手和面部分支中并行编码，再融合为一个60类动作识别结果，整体结构见图4。',
 '身体分支选取22个代表点，经10层CTR编码提取整体姿态与运动特征，通道按48×4、96×3、192×3分组。CTR块结合通道级拓扑细化、多尺度时序卷积与残差连接，第5和第8块进行时间下采样。双手各21点，输入6通道的躯干相对坐标、置信度、躯干尺度归一化的跨手距离及单位方向，共享6层CTR编码器，通道按32×2、64×2、96×2分组，再用1×1卷积投影至48维。面部68点按有效mask压缩为6个token，经4层局部时空图卷积编码，通道为24、24、48、48。',
 '各分支按mask对每帧有效节点求均值，双手特征再取均值得到48维手部序列。手部与面部序列采用线性插值对齐身体分支的时间长度，两者取均值后，用有效mask屏蔽手部与面部均无有效观测的位置。随后，身体和局部分别对人物与时间维度求普通均值，经线性投影和ReLU得到192维身体特征与96维局部特征。拼接后的288维向量送入60类线性分类器，输出识别动作编号及类别。',
 '局部融合采用固定均值和有效性门控，无可学习注意力。训练启用torch.compile，并保留原LocalSTBlock.forward已有的编译隔离边界，编译范围不包含该局部时空块的forward。',
]
for p,text in zip(text_paragraphs,new_texts):
 p.clear();p.add_run(text)
pic_paragraph=next(p for p in chapter if p._p.xpath('.//w:drawing'))
pic_paragraph.clear();pic_paragraph.alignment=WD_ALIGN_PARAGRAPH.CENTER
run=pic_paragraph.add_run();run.add_picture(str(FIGURE),width=Inches(6.9))
for el in pic_paragraph._p.xpath('.//wp:docPr'):
 el.set('descr','133节点输入，身体22点CTR编码、左右手21点共享CTR编码、面部68点压缩为6个token并行编码，经池化和288维拼接完成60类动作识别。')
caption=next(p for p in chapter if p.style.name=='Caption')
caption.clear();caption.add_run('图4 本项目模型结构总览 图中乘数表示对应通道宽度的编码块数量')
for p in chapter:
 if p.text=='4 1 分支编码':
  p.clear();p.add_run('4 1 各分支的结构与作用')
 if p.text=='4 2 原池化与分类':
  p.clear();p.add_run('4 2 有效节点池化与融合分类')
new_start=next(i for i,p in enumerate(d.paragraphs) if p.text=='4 身体与局部并行融合模型')
new_end=next(i for i,p in enumerate(d.paragraphs) if p.text=='5 实验设置与原始CTR-GCN分析')
assert original_outside==[p.text for i,p in enumerate(d.paragraphs) if not new_start<i<new_end]
d.save(FINAL)
print(FINAL)
