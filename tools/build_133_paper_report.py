"""Research report using verified project metrics and published original figures."""
from pathlib import Path
import json
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
AS=ROOT/'.paper_reference_revision/assets'
APP=ROOT/'outputs/ntu_rgb_samples/project_pool_application_20261019'
OUT=ROOT/'docs/基于133节点全身骨架的动作识别研究报告_论文原图修订版.docx'
E=json.loads((ROOT/'outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json').read_text(encoding='utf-8'))
S=json.loads((ROOT/'.paper_reference_revision/pool_sample_recognition.json').read_text(encoding='utf-8'))
d=Document()
sec=d.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65)
sec.left_margin=Inches(.75);sec.right_margin=Inches(.75)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 st=d.styles[name];st.font.name='Microsoft YaHei';st.font.color.rgb=RGBColor(0,0,0)
 st.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Microsoft YaHei')
 st.font.italic=False
 st.paragraph_format.space_after=Pt(6)
 st.paragraph_format.space_before=Pt(0)
 pr=st.element.get_or_add_pPr()
 for tag in ['w:pBdr','w:shd']:
  for el in list(pr.findall(qn(tag))):pr.remove(el)
 grid=OxmlElement('w:snapToGrid');grid.set(qn('w:val'),'0');pr.append(grid)
 rpr=st.element.get_or_add_rPr()
 color=rpr.find(qn('w:color'))
 if color is not None:
  for attr in ['w:themeColor','w:themeTint','w:themeShade']:
   color.attrib.pop(qn(attr),None)
d.styles['Normal'].font.size=Pt(11)
d.styles['Normal'].paragraph_format.line_spacing=1.13
d.styles['Title'].font.size=Pt(25);d.styles['Title'].font.bold=True
d.styles['Subtitle'].font.size=Pt(12)
d.styles['Heading 1'].font.size=Pt(17);d.styles['Heading 2'].font.size=Pt(12)
d.styles['Heading 1'].paragraph_format.space_after=Pt(10)
d.styles['Heading 2'].paragraph_format.space_before=Pt(6)
d.styles['Caption'].font.size=Pt(9)
NEXT_PAGE=False

def p(text,style=None):
 q=d.add_paragraph(text,style);q.paragraph_format.widow_control=True;return q
def h(text,level=1):
 global NEXT_PAGE
 q=d.add_heading(text,level)
 if NEXT_PAGE:q.paragraph_format.page_break_before=True;NEXT_PAGE=False
 return q
def page():
 global NEXT_PAGE
 NEXT_PAGE=True
def caption(text):
 q=p(text,'Caption');q.alignment=WD_ALIGN_PARAGRAPH.CENTER;return q
def pic(file,width=6.8):
 q=d.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
 q.add_run().add_picture(str(file),width=Inches(width));q.paragraph_format.space_after=Pt(3)
 return q
def pics(files,widths):
 q=d.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
 for i,(f,w) in enumerate(zip(files,widths)):
  if i:q.add_run('   ')
  q.add_run().add_picture(str(f),width=Inches(w))
 q.paragraph_format.space_after=Pt(3)
def table(rows,widths):
 t=d.add_table(rows=0,cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for col,w in zip(t.columns,widths):col.width=Inches(w)
 for i,row in enumerate(rows):
  cells=t.add_row().cells
  for j,v in enumerate(row):
   c=cells[j];c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'12304A' if i==0 else ('FFFFFF' if i%2 else 'EFF3F7'));pr.append(sh)
   bor=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'5');el.set(qn('w:color'),'D9D9D9');bor.append(el)
   pr.append(bor);mar=OxmlElement('w:tcMar')
   for edge in ['top','left','bottom','right']:
    el=OxmlElement('w:'+edge);el.set(qn('w:w'),'70' if edge in ['top','bottom'] else '90');el.set(qn('w:type'),'dxa');mar.append(el)
   pr.append(mar)
   q=c.paragraphs[0];q.paragraph_format.space_after=Pt(1);q.paragraph_format.space_before=Pt(0);q.paragraph_format.line_spacing=1.05
   r=q.add_run(v);r.font.size=Pt(10);r.font.bold=i==0;r.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '111111')
   if j>0 and len(v)<18:q.alignment=WD_ALIGN_PARAGRAPH.CENTER
  trpr=t.rows[-1]._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
  if i==0:trpr.append(OxmlElement('w:tblHeader'))
 spacer=p('');spacer.paragraph_format.space_after=Pt(3);spacer.paragraph_format.line_spacing=Pt(2)
 spacer.paragraph_format.space_before=Pt(0)
 spacer.add_run().font.size=Pt(2)
 return t

# 1 Introduction and extraction.
p('基于133节点全身骨架的动作识别研究','Title')
p('研究报告', 'Subtitle')
h('摘要')
p('本研究从NTU RGB+D原始RGB视频提取RTMW二维全身骨架，利用身体、双手和面部的局部信息完成60类动作识别。模型采用身体CTR风格编码分支、左右手共享CTR编码分支和面部时空编码分支，在分类前融合身体与局部特征。原池化方案使用100帧输入并训练100轮，在NTU60 X-Sub的16,306个验证样本上，统一独立评估Top-1为94.6400%，Top-5为99.5891%。真实视频展示包含人物框、对齐的133点骨架及动作编号与中文名称。')
h('1 研究目标与数据流程')
p('研究重点是保留整体姿态之外的手指形态和局部相对关系。NTU中的阅读、写字、玩手机等类别具有相近的身体姿态，仅用稀疏人体关节难以完整描述其细节。本项目以同一套133节点关节输入开展识别，不进行Joint、Bone和Motion等多流结果融合。')
pic(AS/'rtmw_figure1.png',6.8)
caption('图1 RTMW网络结构，论文Figure 1原图裁剪 [1]')
p('处理顺序为RGB视频读取、RTMW关键点提取、有效性掩码构建、人物槽位整理、置信度加权时序压缩及模型分类。输入张量为B×3×100×133×M，3个通道分别是x、y、score，M最多为2。图1虚线SimCC Z属于论文中的3D扩展，本项目使用2D输出。')
p('关键点需满足score大于0且数值有限，并与有效帧mask联合使用。缺失节点不直接作为有效观测参与逐帧节点池化。应用绘制则保留原始帧索引，通过NPZ中的frame_indices将骨架对应回RGB帧。')

# 2 All regions and indices.
page();h('2 全身133节点表示')
p('133点采用COCO-WholeBody关键点定义，包含17个身体点、6个足部点、68个面部点及左右手各21点。下图直接采用官方编号原图的区域裁剪，未重新构造节点位置或连线 [2]。')
pics([AS/'official_body_crop.png',AS/'official_face_nodes.png',AS/'official_feet_crop.png'],[1.35,2.9,1.3])
caption('图2 身体、面部与足部官方编号图 [2]')
table([['区域','节点数','官方图编号','代码索引'],['身体','17','1–17','0–16'],['足部','6','18–23','17–22'],['面部','68','24–91','23–90'],['左手','21','92–112','91–111'],['右手','21','113–133','112–132']],[1.3,1.1,2.2,2.2])
p('官方图编号从1开始，代码索引从0开始，两者相差1。身体分支选取22个代表点，既包含身体和足部信息，也包含面部代表点及双手腕点。其代码索引为0、5、6、7、8、9、10、11、12、13、14、15、16、17、18、19、20、21、22、56、91、112。')
p('躯干中心由左肩、右肩、左髋和右髋的有效坐标求均值得到，代码索引为5、6、11、12。肩中心与髋中心的距离作为躯干尺度，供双手距离归一化使用。')

# 3 Hand representation.
page();h('3 双手拓扑与几何特征')
p('左右手采用相同的21点拓扑，并共享同一套手部CTR编码参数。局部0号点为手腕，1至4为拇指，5至8为食指，9至12为中指，13至16为无名指，17至20为小指。全局编号保留左右手各自的索引范围。')
pics([AS/'official_hand_21.png',AS/'official_hands_crop.png'],[2.15,4.3])
caption('图3 MMPose官方局部手部图与COCO-WholeBody双手全局编号图 [2,3]')
table([['特征','通道数','含义'],['躯干相对坐标','2','手部x/y减去躯干中心'],['关键点置信度','1','原始score'],['跨手距离','1','左右手对应点距离除以躯干尺度'],['跨手方向','2','左右手对应点连线的单位方向']],[1.6,1.0,4.2])
p('每个手部节点输入共6个通道。左右手按同一局部编号建立对应关系，例如左手腕对应右手腕。距离在双手两侧一致，方向在两侧相反。只有两侧对应点均有效时，该对距离与方向特征才参与计算。')
p('这一表示同时保留单手形态及双手间的几何关系。它属于模型输入设计，当前结果不能单独证明其中某一通道带来的增益，仍需移除距离、移除方向及不共享编码器等消融实验。')

# 4 Architecture image exported from editable slide6.
page();h('4 身体与局部并行融合模型')
p('模型实现为BodyLocalHandCTRWideRelativeFusion，架构标识为rtmw_ctr22_handctr21_wide_crosshanddistdir_facest48_torso_relative_v3。三个区域分支读取同一输入并行编码，身体特征不按“身体、手、脸”的串行路径传播。')
model=Image.open(ROOT/'.paper_reference_revision/deck/slide-06.png').crop((40,145,1240,615));model.save(AS/'project_parallel_model.png')
pic(AS/'project_parallel_model.png',6.9)
caption('图4 本项目模型结构，对应PPT中的可编辑结构图')
h('4 1 分支编码',2)
p('身体分支对22个代表点进行10层CTR风格编码，通道宽度依次为48、48、48、48、96、96、96、192、192、192。手部分别读取左右21点，通过共享的6层CTR编码器，通道宽度为32、32、64、64、96、96，并将96维输出投影至48维局部表示。面部68点按有效mask压缩为6个token，再进行48维局部时空编码。')
h('4 2 原池化与分类',2)
p('各分支先对每帧有效节点求均值。左右手特征取均值，手部与面部特征采用线性插值对齐身体分支的时间长度。随后对手部和面部表示取均值，并通过局部有效性mask门控。原池化对时间与人物维度求均值，身体投影至192维，局部投影至96维，拼接形成288维向量，送入60类线性分类器。')
p('训练配置启用torch.compile。原模型保留LocalSTBlock.forward已有的编译隔离边界，因此不将该实验表述为全模型fullgraph编译。本次报告的数据复核只运行评估，不启动新训练，也不改变已有训练入口或编译设置。')

# 5 Training and original CTR comparison.
page();h('5 实验设置与原始CTR-GCN分析')
table([['项目','设置'],['数据','NTU60 X-Sub，训练39,627，验证16,306'],['输入与训练','100帧，最多2人，100轮，batch 32，seed 1'],['优化器','SGD，lr 0.1，momentum 0.9，Nesterov，weight decay 0.0004'],['学习率','5轮warmup，起始0.001，cosine最小0.0001'],['权重选择','随机初始化训练，最佳第98轮'],['评估','完整序列压缩为单clip，无数据增强']],[1.5,5.3])
table([['评估口径','Top-1','Top-5'],['统一独立评估','94.6400%','99.5891%'],['训练日志最佳','94.6461%','本表不混合日志与独立评估']],[2.4,1.5,2.9])
p('统一独立评估正确15,432个，错误874个，Top-5命中16,239个。训练日志最佳与独立评估相差0.0061个百分点，约为1个样本。二者对应相同最佳权重，但不属于完全相同的评估执行口径。本文定量分析统一采用独立评估结果。')
h('5 1 原始CTR-GCN Joint单流',2)
table([['维度','原始CTR-GCN Joint','本项目'],['节点表示','NTU原始25关节，三维坐标','RTMW 133点，二维坐标与score'],['主要建模','通道级拓扑细化图卷积','身体CTR与共享手部CTR并行'],['局部处理','原始人体关节图','手部几何特征与面部token']],[1.2,2.6,3.0])
p('CTR-GCN通过通道级拓扑细化学习动作相关的关节关系 [4]。本项目在CTR风格身体编码之外引入稠密手部与面部局部表示。两者输入来源及维度不同，应将其理解为方案层面的比较，不能直接视为单一结构因素的消融。原论文NTU60主表采用多流融合，不能将该数值充当Joint单流基准计算提升幅度。')

# 6 Application.
page();h('6 真实RGB模拟应用展示')
p('展示样例来自服务器NTU RGB+D原始RGB压缩包，均为S001C001P007R001序列，动作分别为A010拍手、A012写字、A029玩手机或平板。程序读取同名RTMW NPZ，依据原始帧索引与RGB对齐，画出人物框、躯干、双手、面部和足部骨架，并在框上标注识别动作编号及中文名称。')
pics([APP/'A010_project_recognition.png',APP/'A012_project_recognition.png',APP/'A029_project_recognition.png'],[2.15,2.15,2.15])
caption('图5 本项目权重的整段动作识别展示，骨架与对应RGB人物对齐')
table([['真实类别','识别结果','类别softmax'],['A010 拍手','A010 拍手','0.996080'],['A012 写字','A012 写字','0.491315'],['A029 玩手机或平板','A029 玩手机或平板','0.945294']],[2.3,2.8,1.7])
p('三个样例使用本项目最佳权重，不使用SkeleT权重。完整视频保留原始69、101和98帧，帧率均为30fps。分类采用整段输入后的离线识别结果，并将该结果标注到整段画面，不能据此声称实时逐帧分类性能。softmax表示模型对类别的输出分数，不等同于经过校准的识别准确率。')
p('写字样例的第二候选为A011阅读，softmax为0.461413，与写字0.491315接近。这一现象与整体验证中阅读、写字和玩手机之间的混淆一致，说明识别正确的样例也可能具有较低的类别区分裕量。三个展示样例已属于既有验证数据，不能作为新增独立测试集。')

# 7 Later similar work and limitations.
page();h('7 不足与后续发现的SkeleT路线')
p('后续文献梳理发现SkeleT-GCN也研究表达性关键点，其节点映射与图规模逐层压缩为本项目分析局部信息利用提供了参考 [5]。这里仅在不足分析中讨论服务器已有的Joint单流本地适配实验，明确区分其与原始CTR-GCN论文结果。')
table([['实验身份','输入与评估','Top-1'],['本项目原池化','RTMW 133点，100帧，单clip，100轮','94.6400%'],['SkeleT-CTRGCN本地适配','RTMW派生65点，100帧，10-clip','95.5722%']],[2.0,3.4,1.4])
p('本项目比SkeleT本地适配低0.9322个百分点。SkeleT采用65至27至11的图规模压缩，训练配置为24个逻辑epoch、每个epoch重复5次数据遍历。其输入节点构成、训练预算和10-clip评估均与本项目不同，因此这一差距只反映当前实验结果，不能单独归因于分层压缩或某个模块。')
h('7 1 类别混淆',2)
table([['动作','正确数与样本数','类别准确率','主要混淆'],['A011 阅读','195 / 276','70.6522%','A029 48次，A012 26次'],['A029 玩手机或平板','197 / 275','71.6364%','A012 27次，A011 24次'],['A012 写字','239 / 274','87.2263%','A011 13次']],[1.65,1.55,1.3,2.3])
p('这些类别均包含手部靠近身体前方、小幅操作等相似骨架现象。当前输入仅含关键点，缺少手机、书本和笔等物体外观，稠密关键点也会受遮挡和低置信度影响。面部压缩与固定均值融合可能损失局部差异，但是否构成主要性能瓶颈，仍需通过实验检验。')
h('7 2 后续验证与结论',2)
p('后续首先固定数据划分、输入长度、训练预算和clip数量，在一致协议下比较单流模型。再分别消融手部距离方向、手部编码宽度、面部token及融合方式，检查局部信息的独立贡献。本研究已实现133节点输入的身体局部融合识别，并在原池化100帧100轮设置下取得94.6400%的统一独立评估Top-1，同时完成真实RGB骨架叠加与动作识别视频展示。')

# 8 References and exact provenance.
page();h('参考文献与实验依据')
refs=[
'[1] Jiang T, Xie X, Li Y. RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation. 2024. https://arxiv.org/abs/2407.08634 。图1来源为论文第4页Figure 1。',
'[2] Jin S 等. Whole-Body Human Pose Estimation in the Wild. ECCV 2020. https://arxiv.org/abs/2007.11858 。全身编号图来源：https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png 。',
'[3] MMPose官方文档，手部21点图。https://user-images.githubusercontent.com/100993824/227771101-03a27bd8-ccc0-4eb9-a111-660f191a7a16.png 。',
'[4] Chen Y 等. Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition. ICCV 2021. https://arxiv.org/abs/2107.12213 。',
'[5] SkeleT-GCN. arXiv:2406.18011. https://arxiv.org/abs/2406.18011 。文中95.5722%来源为本地适配实验，不是原CTR-GCN论文结果。']
for r in refs:p(r)
p('节点编号图使用COCO-WholeBody官方示意图，手部局部拓扑使用MMPose官方图，RTMW模型结构使用其论文Figure 1。不同来源在正文图注中分别标明，不将COCO-WholeBody编号图标记为RTMW论文插图。')
page();h('附录 服务器实验记录')
for title,loc in [
('本项目统一评估','/home/xiong/Desktop/ISAA/outputs/evaluations/best_weighted_100_cosine_20261016/summary.json'),
('本项目最佳权重','/home/xiong/Desktop/ISAA/outputs/best_weighted_100_cosine_ntu60_xsub/20261014-074736-ab97a386/best.pt'),
('应用识别记录','/home/xiong/Desktop/ISAA/outputs/report_paper_reference_20261019/pool_sample_recognition.json'),
('SkeleT单流本地评估','/home/xiong/Desktop/ISAA/outputs/skelet_single_eval_20261019/20261019-025950-cc35608a/config.json'),
('原始RGB目录','/home/xiong/Desktop/nturgbd_skeleton_extraction/data/raw_archives')]:
 q=p(title);q.runs[0].bold=True
 q=p(loc);q.paragraph_format.space_after=Pt(6)
 for run in q.runs:run.font.size=Pt(9)
p('本项目权重SHA256：')
q=p(E['checkpoint_sha256'])
for run in q.runs:run.font.size=Pt(9)
p('代码依据：isaa/models/body_local_fusion.py中的BodyLocalHandCTRWideRelativeFusion及其父类实现。论文原图保持来源归属，应用图片使用对应真实视频与提取骨架，不将展示样例替代整体统计结果。')
for q in list(d.paragraphs)+[q for t in d.tables for row in t.rows for c in row.cells for q in c.paragraphs]:
 pr=q._p.get_or_add_pPr()
 for el in list(pr.findall(qn('w:pBdr'))):pr.remove(el)
 grid=OxmlElement('w:snapToGrid');grid.set(qn('w:val'),'0');pr.append(grid)
OUT.parent.mkdir(parents=True,exist_ok=True);d.save(OUT)
print(OUT)
