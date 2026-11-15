"""Focused background and official Joint-only quantitative comparison revision."""
from copy import deepcopy
from pathlib import Path
import json
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/基于133节点全身骨架的动作识别研究报告_模型结构解释版.docx'
FINAL=ROOT/'docs/基于133节点全身骨架的动作识别研究报告_背景与单流结果修订版.docx'
V=json.loads((ROOT/'.background_single_stream_revision/verified-comparison.json').read_text(encoding='utf-8'))
assert V['baseline']['top1_percent']==89.9
assert V['project']['display_top1']=='94.6400'
assert V['improvement_percentage_points']==4.74
d=Document(SOURCE)
def replace(old,new):
    matches=[p for p in d.paragraphs if p.text==old]
    assert len(matches)==1, old
    p=matches[0]
    p.clear();p.add_run(new)
    return p

replace('1 研究目标与数据流程','1 研究背景与数据流程')
replace('研究重点是保留整体姿态之外的手指形态和局部相对关系。NTU中的阅读、写字、玩手机等类别具有相近的身体姿态，仅用稀疏人体关节难以完整描述其细节。本项目以同一套133节点关节输入开展识别，不进行Joint、Bone和Motion等多流结果融合。',
        '传统骨架GCN常采用较少的人体关节。以NTU原始25点骨架为例，虽然包含手端和拇指点，但缺少完整的手指关节与精细形态。阅读、写字、玩手机等动作的身体姿态相近，仅凭稀疏关节难以充分描述其细节。为加强精细动作识别，本项目采用RTMW 133节点全身骨架，利用双手各21点及身体、面部局部信息建模，并以同一套关节输入完成60类识别，不进行Joint、Bone和Motion多流融合 [1,4]。')

abstract=next(p for p in d.paragraphs if p.text.startswith('本研究从NTU RGB+D原始RGB视频'))
abstract.text=abstract.text.replace('真实视频展示包含', '原作者公开的原始CTR-GCN Joint单流在NTU60 X-Sub上为89.9%，本项目高4.7400个百分点，比较涉及不同输入表示与处理设置。真实视频展示包含')

# Expand the existing native comparison table, retaining column widths and cell styling.
t=d.tables[4]
assert t.cell(0,1).text=='原始CTR-GCN Joint'
for values,insert_at in [
    (['Top-1','89.9%','94.6400%'],1),
    (['评测划分','NTU60 X-Sub','NTU60 X-Sub'],2),
]:
    row=deepcopy(t.rows[1]._tr)
    t._tbl.insert(t._tbl.index(t.rows[insert_at]._tr),row)
    for j,value in enumerate(values):
        c=t.rows[insert_at].cells[j]
        q=c.paragraphs[0]
        for child in list(c._tc):
            if child.tag not in [qn('w:tcPr'),qn('w:p')]:c._tc.remove(child)
        q.clear();run=q.add_run(value);run.font.size=Pt(10);run.bold=insert_at==1
        run.font.color.rgb=RGBColor(0,0,0)
        if j>0:q.alignment=WD_ALIGN_PARAGRAPH.CENTER
for i,row in enumerate(t.rows):
    if i==0:continue
    for c in row.cells:
        pr=c._tc.get_or_add_tcPr()
        sh=pr.find(qn('w:shd'))
        if sh is None:sh=OxmlElement('w:shd');pr.append(sh)
        sh.set(qn('w:fill'),'FFFFFF' if i%2 else 'EFF3F7')
replace('CTR-GCN通过通道级拓扑细化学习动作相关的关节关系 [4]。本项目在CTR风格身体编码之外引入稠密手部与面部局部表示。两者输入来源及维度不同，应将其理解为方案层面的比较，不能直接视为单一结构因素的消融。原论文NTU60主表采用多流融合，不能将该数值充当Joint单流基准计算提升幅度。',
        '原作者官方预训练目录NTU60_Xsub/CTRGCN_joint_89.9标注Joint单流为89.9% [4]，本项目独立评估94.6400%，高4.7400个百分点。该基准不是论文主表的多流融合结果。输入来源、节点维度、样本清理与时序处理不同，提升仅代表方案结果差异，不能归因于单一结构因素。')

conclusion=next(p for p in d.paragraphs if p.text.startswith('后续首先固定数据划分'))
conclusion.text=conclusion.text.replace('同时完成真实RGB骨架叠加与动作识别视频展示。',
 '较原作者公开的原始CTR-GCN Joint单流89.9%高4.7400个百分点。该比较采用不同输入设置，后续仍需同协议实验确认结构增益。本研究同时完成真实RGB骨架叠加与动作识别视频展示。')

replace('[4] Chen Y 等. Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition. ICCV 2021. https://arxiv.org/abs/2107.12213 。',
        '[4] Chen Y 等. Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition. ICCV 2021. https://arxiv.org/abs/2107.12213 。官方代码：https://github.com/Uason-Chen/CTR-GCN 。单流数据依据为该仓库README链接的官方预训练模型，NTU60_Xsub / CTRGCN_joint_89.9，Top-1为89.9%，核验日期2026年10月19日。目录：https://drive.google.com/drive/folders/1b_r1gGuWCxn-YBPij3R4BULkd-HASp7e 。')

assert len(d.inline_shapes)==len(Document(SOURCE).inline_shapes), 'Do not alter source figures'
text='\n'.join(p.text for p in d.paragraphs)+'\n'+'\n'.join(c.text for table in d.tables for row in table.rows for c in row.cells)
for required in ['89.9%','94.6400%','4.7400个百分点','缺少完整的手指关节']:
    assert required in text
assert '95.4802' not in text
d.save(FINAL)
print(FINAL)
