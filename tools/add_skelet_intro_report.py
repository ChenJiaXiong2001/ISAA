"""Add a paper-grounded SkeleT introduction, preserving existing experiment data."""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/基于133节点全身骨架的动作识别研究报告_背景与单流结果修订版.docx'
FINAL=ROOT/'docs/基于133节点全身骨架的动作识别研究报告_SkeleT简介补充版.docx'
FIG=ROOT/'.skelet_intro_revision/skelet-figure2.png'
d=Document(SOURCE)
source=Document(SOURCE)
intro=next(p for p in d.paragraphs if p.text.startswith('后续文献梳理发现SkeleT-GCN'))
intro.text='后续文献梳理发现，SkeleT也研究如何利用更丰富的手脚关键点进行动作识别。其全称为Skeleton Transformation，即骨架变换，是一种可嵌入CTR-GCN等GCN骨干的策略 [5]。'
anchor=next(p for p in d.paragraphs if p.text=='7 不足与后续发现的SkeleT路线')
h=intro.insert_paragraph_before('7 1 SkeleT方法简介',style='Heading 2')
def after(p,text='',style=None):
    q=d.add_paragraph(text,style=style)
    p._p.addnext(q._p)
    return q
p=after(intro,'论文先从RGB视频提取COCO-WholeBody 133点，再依据关键点统计分析去掉68个面部点，保留身体17点、脚部6点和双手42点，共65个表达性关键点。这样既保留手指和脚部细节，也减少冗余节点。')
p=after(p)
p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.keep_with_next=True
shape=p.add_run().add_picture(str(FIG),width=Inches(6.8))
shape._inline.docPr.set('descr','SkeleT原论文Figure 2，包含关键点选择、骨架变换策略及可选的多人实例池化模块')
p=after(p,'图6 SkeleT原论文Figure 2 方法总览 [5]',style='Caption')
p=after(p,'骨架变换通过Grouped Mapping（分组映射）实现。模型先按通道对特征分组，各组独立学习节点映射。普通块通过对角权重突出重要节点，下采样块则合并相关节点，将图规模从65点逐步压缩为27点和11点。各组经图卷积后拼接，再由时间卷积建模动作变化。将这套策略结合CTR-GCN骨干，就得到SkeleT-CTRGCN。')
p=after(p,'与本项目保留133点输入、采用身体和局部分支并行编码融合的做法相比，SkeleT主要通过节点选择和网络内分层压缩利用细节并控制计算量。图6(c)的Instance Pooling是论文针对多人场景提出的可选扩展，当前本地NTU单流适配实验未启用。')
p=after(p,'7 2 本地适配结果与差距',style='Heading 2')
p.paragraph_format.page_break_before=True
p=after(p,'下表仅比较服务器已有的Joint单流本地实验，SkeleT适配结果与原始CTR-GCN官方单流结果分别列示，不将其混为同一基准。')
next(p for p in d.paragraphs if p.text=='7 1 类别混淆').text='7 3 类别混淆'
next(p for p in d.paragraphs if p.text=='7 2 后续验证与结论').text='7 4 后续验证与结论'
ref=next(p for p in d.paragraphs if p.text.startswith('[5] SkeleT-GCN.'))
ref.text='[5] Yang Y, Zhang J, Zhang J, Tu Z. Expressive Keypoints for Skeleton-based Action Recognition via Skeleton Transformation. 2024. https://arxiv.org/abs/2406.18011 。图6来源为论文第4页Figure 2。官方代码：https://github.com/YijieYang23/SkeleT-GCN 。文中95.5722%来源为本地适配实验，不是原CTR-GCN论文结果。'
assert len(d.inline_shapes)==len(source.inline_shapes)+1
assert [[c.text for row in t.rows for c in row.cells] for t in d.tables]==[[c.text for row in t.rows for c in row.cells] for t in source.tables]
text='\n'.join(p.text for p in d.paragraphs)
assert '95.4802' not in text and '92.4%' not in text
for token in ['94.6400%','89.9%','4.7400个百分点','0.9322个百分点','Skeleton Transformation','Grouped Mapping']:
    assert token in text,token
d.save(FINAL)
print(FINAL)
