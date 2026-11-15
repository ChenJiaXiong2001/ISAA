from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION

OUT = r"C:\Users\15637\PycharmProjects\ISAA\docs\高精度全身骨架细粒度动作识别研究说明.docx"

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(.78); sec.bottom_margin = Inches(.75)
sec.left_margin = Inches(.82); sec.right_margin = Inches(.82)

styles = doc.styles
normal = styles['Normal']
normal.font.name = 'Microsoft YaHei'; normal.font.size = Pt(10.5); normal.font.color.rgb = RGBColor(35, 40, 46)
normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.18
for name, size in [('Title', 24), ('Heading 1', 16), ('Heading 2', 12.5), ('Heading 3', 11)]:
    st = styles[name]; st.font.name = 'Microsoft YaHei'; st.font.size = Pt(size); st.font.bold = True; st.font.color.rgb = RGBColor(0,0,0)
    st._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    st.paragraph_format.space_before = Pt(14 if name != 'Title' else 0); st.paragraph_format.space_after = Pt(7)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)
def margins(cell, top=100, start=110, bottom=100, end=110):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr(); mar = OxmlElement('w:tcMar')
    for k,v in [('top',top),('start',start),('bottom',bottom),('end',end)]:
        el=OxmlElement('w:'+k); el.set(qn('w:w'),str(v)); el.set(qn('w:type'),'dxa'); mar.append(el)
    tcPr.append(mar)
def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr(); el=OxmlElement('w:tblHeader'); el.set(qn('w:val'),'true'); trPr.append(el)
def table(headers, rows, widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    t.style='Table Grid'
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=h; shade(c,'24476B'); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; margins(c)
        for p in c.paragraphs:
            p.paragraph_format.space_after=Pt(0)
            for r in p.runs: r.font.name='Microsoft YaHei'; r.font.size=Pt(9); r.font.bold=True; r.font.color.rgb=RGBColor(255,255,255); r._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    set_repeat_table_header(t.rows[0])
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,val in enumerate(row):
            cells[i].text=str(val); cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; margins(cells[i])
            if ri%2: shade(cells[i],'F1F5F8')
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.05
                for r in p.runs: r.font.name='Microsoft YaHei'; r.font.size=Pt(8.7); r._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
        if widths:
            for i,w in enumerate(widths): cells[i].width=Inches(w)
    if widths:
        for i,w in enumerate(widths): t.columns[i].width=Inches(w)
    doc.add_paragraph().paragraph_format.space_after=Pt(1)
    return t
def para(text='', style=None, boldlead=None):
    p=doc.add_paragraph(style=style)
    if boldlead and text.startswith(boldlead):
        p.add_run(boldlead).bold=True; p.add_run(text[len(boldlead):])
    else: p.add_run(text)
    return p
def bullets(items):
    for x in items: doc.add_paragraph(x, style='List Bullet')
def heading(text, level=1): doc.add_heading(text, level=level)

title=doc.add_paragraph(style='Title'); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; title.add_run('基于高精度全身骨架的细粒度人体动作识别研究')
subtitle=doc.add_paragraph(); subtitle.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=subtitle.add_run('研究背景、数据处理、技术依据与方法设计'); r.font.size=Pt(12); r.font.color.rgb=RGBColor(86,101,116)
doc.add_paragraph()
para('本研究面向整体姿态相似、局部动作不同而容易混淆的人体动作类别，使用 NTU RGB+D 视频提取 RTMW-133 全身骨架，并研究如何控制骨架噪声、降低高密度节点带来的计算负担，以及有效融合身体主体与手部、面部信息。当前已形成从视频到分类的完整流程，并在 NTU60 XSub 上得到可复核的分类基线；后续重点是通过数据准确性验证和严格消融，确认各项设计的独立作用。')

heading('一、研究背景与问题提出')
para('人体动作识别根据连续的人体运动判断其所属类别。许多骨架动作识别方法主要使用 Kinect 提供的少量身体关节点，能够描述整体姿态和大范围运动，但对手腕、手指、面部等局部信息表达有限。当多个动作的身体姿态相似时，区分线索可能出现在手指形态、手腕方向、双手相对距离、面部朝向或局部动作出现的时间阶段。如果模型只使用少量身体节点，就容易忽略这些细节并产生误判。')
para('因此，本研究关注：在整体姿态相似的情况下，包含更多身体节点的高精度全身骨架能否提供有效的局部动作信息，并提升分类能力与结果可解释性。增加节点也会带来检测噪声、关键点缺失、人物槽位交换、计算量上升和训练难度增加等问题，所以研究同时覆盖骨架提取、质量控制、骨架表示和动作分类。')

heading('二、研究总体思路')
para('本研究建立从 RGB 视频到动作类别的连续技术流程。数据质量控制和分类建模相互依赖：前者确保模型输入可信，后者负责从整体运动和局部细节中提取类别信息。')
table(['阶段','主要工作','目标'],[
('数据来源','采用 NTU RGB+D RGB 视频及官方动作标签、划分协议','建立统一的分类数据基础'),
('骨架提取','人物检测后由 RTMW-L 提取 RTMW-133 全身关键点','获得身体、手部、面部和足部信息'),
('质量控制','检查人数、关键点有效性、置信度、跳变、缺失和槽位稳定性','控制检测和姿态估计噪声'),
('数据整理','统一节点顺序、时间窗口、坐标通道和人物维度','保证训练输入一致'),
('动作建模','分开处理身体主体与局部区域，再融合分类','利用全身和局部时空信息'),
('验证分析','进行消融、多种子、遮挡和效率评估','确认方法是否有效及其作用来源')],[1.1,3.35,2.15])

heading('三、数据集与相关技术依据')
table(['数据、论文或技术','原有作用','本研究中的使用方式'],[
('NTU RGB+D，Shahroudy 等，CVPR 2016','提供多视角 RGB 视频、动作类别和标准评估协议，也包含 Kinect 骨架数据。','使用 RGB 视频和官方动作标签；当前完整分类实验采用 NTU60 XSub。'),
('COCO-WholeBody，Jin 等，ECCV 2020','定义包含身体、足部、面部和双手的全身关键点。','作为 RTMW-133 节点语义和区域划分的基础。'),
('RTMW-L / MMPose','根据人物检测框预测全身关键点。','从 NTU RGB 视频重新提取 133 点骨架。'),
('RTMDet-tiny','目标检测方法，为自顶向下姿态估计提供人物框。','用于常规样本人物检测。'),
('YOLO26-X','人物检测器。','用于质量检查失败困难样本的重新检测。'),
('ST-GCN，Yan 等，AAAI 2018','在骨架图上用空间图卷积和时间卷积建模动作。','为手部和面部局部分支提供时空图建模依据。'),
('2s-AGCN，Shi 等，CVPR 2019','学习自适应关节关系，并结合不同骨架信息流。','作为自适应关系和多流表示的技术背景；本研究未完整复现该模型。'),
('CTR-GCN，Chen 等，ICCV 2021','按特征通道细化骨架拓扑关系。','为身体主体分支提供图卷积设计依据。'),
('区域化骨架分析方法','分析身体不同区域与动作类别之间的关系。','启发本研究分别建模身体、手部和面部，并设计局部遮挡分析。')],[1.65,2.5,2.45])
para('本研究结合上述数据和方法建立完整流程，并非对某一篇论文的逐项复现。NTU RGB+D 提供动作视频、标签和评估协议；RTMW-133 骨架由本研究从 RGB 视频重新提取；ST-GCN 和 CTR-GCN 提供模型设计思路。')

heading('四、骨架数据提取')
heading('4.1 从视频到骨架',2)
para('首先使用人物检测器获得视频中每个人物的检测框，再将人物框送入 RTMW-L，提取各帧的全身关键点。每个样本保存随时间变化的关键点坐标、置信度、人物框、人物槽位和样本元数据，形成后续质量检查和分类使用的骨架序列。')
table(['提取步骤','处理内容'],[
('人物检测','在每帧中定位人物，过滤不合理检测框，并按需要扩展裁剪区域。'),
('姿态估计','RTMW-L 根据人物框输出 133 点全身关键点及置信度。'),
('人物匹配','在相邻帧之间匹配人物，尽量保持同一人物的槽位一致。'),
('结果保存','保存坐标、置信度、人物框、槽位和标签等信息。'),
('质量检查','检查缺失点、低置信度、异常跳变、人数变化和槽位稳定性。')],[1.25,5.35])
heading('4.2 为什么重新提取骨架',2)
para('NTU 原始 Kinect 骨架能够表达身体主体，但节点较少，无法充分表示手指、手腕和面部细节。本研究从 RGB 视频重新提取 RTMW-133，是为了保留更丰富的局部信息，支持对手部和面部动作进行分类与分析。与此同时，RGB 姿态估计会受到人物检测、遮挡和图像分辨率影响，因此需要配套质量控制流程。')

heading('五、质量控制与困难样本处理')
heading('5.1 主要数据问题',2)
bullets(['双人遮挡时人物检测器可能漏掉其中一人，背景人物也可能被误检。','检测框变化可能导致相邻帧的人物槽位交换。','手部和面部关键点可能缺失、置信度偏低或出现异常跳变。','人物可能连续多帧缺失，或关键点坐标出现 NaN 和无效值。','删除失败样本可能改变各动作类别的样本分布。','不同版本的数据处理和输入定义可能使模型比较失去一致性。'])
heading('5.2 质量检查内容',2)
para('质量检查统计每个样本的人物数量、身体有效节点数量、关键点坐标有效性、关键点置信度、相邻帧位移、人物框中心变化、连续缺失长度和人物槽位稳定性。项目使用 0 到 100 的启发式质量分帮助筛选待修复样本。该分数用于工程筛查，不等同于人工标注误差或姿态估计准确率。')
heading('5.3 困难样本重新检测',2)
para('常规样本使用 RTMDet-tiny 检测人物；质量检查失败的困难样本使用 YOLO26-X 重新检测，再由同一个 RTMW-L 模型重新提取骨架。这样做是因为困难样本中的重要问题往往发生在人物检测阶段：如果人物框漏掉了目标人物，姿态模型无法恢复其完整骨架。')
table(['检测方法','通过数','通过率','平均质量分'],[
('RTMDet-tiny','5/100','5%','80.521'),('YOLO26-X','85/100','85%','96.149')],[2.2,1.2,1.2,1.6])
para('在同一批 100 个困难样本中，YOLO26-X 的通过率比 RTMDet-tiny 高 80 个百分点，平均启发式质量分提高 15.628 分。该结果表明困难样本的骨架质量筛查明显改善，但不能直接推导动作分类准确率也提高了相同幅度。还需要人工抽样复核、关键点误差或 PCK 等指标验证骨架准确性。')
heading('5.4 保证数据稳定和一致',2)
para('正式实验应固定并记录人物检测器和姿态估计器版本、检测阈值、裁剪参数、质量规则、RTMW-133 节点顺序、人物匹配规则、坐标与置信度通道、时间裁剪方法、数据清理和训练/验证划分。建议通过 manifest 逐样本记录处理来源、是否重新检测、是否被清洗或排除以及质量统计，以便复现和公平比较。')

heading('六、数据表示与预处理')
heading('6.1 输入张量',2)
para('骨架序列整理为 B × C × T × N × M，其中 B 为批次大小，C 为输入通道，T 为时间帧数，N 为骨架节点数，M 为人物数。当前主要 Baseline 使用 RTMW-133 的 x、y 和 score 三个通道，时间窗口为 64 帧，最多处理两个人。')
heading('6.2 时间裁剪和补齐',2)
para('不同视频长度不一，因此训练数据统一到 64 帧。训练阶段使用随机时间裁剪，验证阶段使用中心裁剪，短序列通过补齐达到统一长度。这样既控制计算量，也使模型适应动作起止位置和持续时间变化。')
heading('6.3 坐标归一化和运动信息',2)
para('本研究考虑原始坐标、以躯干中心为基准的相对坐标、按身体尺度归一化的坐标，以及相邻帧运动差分。相对坐标可能减轻人物位置和体型差异，运动差分可能突出动作变化。但不同表示改变了输入语义，必须在相同训练条件下单独比较。')
heading('6.4 有效性 Mask',2)
para('对于缺失、低置信度或不存在的人物节点，数据处理中记录有效性 Mask。池化时只统计有效节点，减少无效坐标对特征平均的影响。Mask 能降低无效数据造成的干扰，但无法证明被标记为有效的关键点一定准确。')

heading('七、当前动作分类模型')
heading('7.1 BodyLocalFusion 结构',2)
para('当前主要分类基线将 RTMW-133 划分为身体主体和局部信息两类输入。身体分支选取 22 个节点，使用 CTR-GCN 风格的拓扑细化；局部分支使用 42 个手部节点，并将面部 68 点压缩为 6 个区域 token，再使用 ST-GCN 风格的时空图结构。各分支经过有效位置池化和特征投影后拼接分类。')
table(['分支','节点表示','建模方式','输出特征'],[
('身体主体','22 个身体节点','CTR-GCN 风格通道拓扑细化','192 维'),
('局部区域','双手 42 点，面部 68 点压缩为 6 个 token','ST-GCN 风格局部时空建模','96 维'),
('分类融合','身体与局部特征拼接','分类器输出动作类别','60 类 logits')],[1.25,2.0,2.5,1.15])
heading('7.2 身体主体分支',2)
para('身体分支用于表达整体姿态和大范围动作，建模躯干、肩部、肘部、髋部和腿部之间的协同运动。采用 CTR-GCN 风格设计，是因为不同通道可以关注不同的关节关系，有助于表达身体主体的多样运动模式。')
heading('7.3 手部和面部分支',2)
para('双手保留 42 个节点，用于描述手腕、手指和手势随时间的变化。面部原有 68 个点压缩为 6 个 token，以降低密集面部节点的计算量和低置信度噪声。面部压缩有信息损失风险，是否优于直接使用 68 点，需要后续消融验证。')
heading('7.4 特征融合',2)
para('当前模型将身体主体特征和局部区域特征分别池化、投影后拼接，再送入分类器。简单拼接结构直接、稳定且便于建立基线。后续将比较加权融合和逐时间步跨分支门控。')

heading('八、对相关方法的调整及原因')
table(['本研究的调整','为什么这样做','需要验证的边界'],[
('从少量 Kinect 节点扩展为 RTMW-133','保留身体、足部、面部和双手的局部细节。','节点增加会增加噪声与计算成本。'),
('按身体主体和局部区域分支建模','不同区域的拓扑、运动尺度和可靠性不同。','分支结构可能增加复杂度并导致过拟合。'),
('身体采用 CTR-GCN 风格，局部采用 ST-GCN 风格','主体关系和局部时空变化具有不同建模需求。','需与固定图、统一图及单分支模型公平比较。'),
('面部 68 点压缩为 6 个区域 token','降低密集面部节点的成本和噪声影响。','可能损失细小表情信息。'),
('池化时应用有效节点 Mask','避免缺失点和无效人物影响特征统计。','Mask 不能替代关键点准确性验证。'),
('困难样本使用 YOLO26-X 重新检测','针对遮挡、远距离等场景的人物漏检问题。','需要确认质量修复是否带来分类增益。'),
('统一时间窗口并进行时间裁剪','控制计算量并提高对动作时长和起始位置的适应性。','窗口长度可能影响不同类别的时序信息。')],[1.8,2.55,2.25])

heading('九、前期探索与待验证方法')
para('本研究前期探索过 Joint、Bone、Joint Motion 和 Bone Motion 等多流输入，以及区域参与、局部残差和多原型分类。相关思路为区域化建模和局部动作分析提供了方法参考，但不同阶段的实验设置并不完全一致，不能直接拼成严格的性能提升曲线。')
para('还探索过“133 点固定局部图 + 32 点动态 CTR”，其目标是以固定图处理全部细节点，仅对 32 个代表节点建模动态关系，从而降低节点关系计算量。该方法尚未在相同数据、输入、训练预算和随机种子下与当前基线进行完整比较，因此目前只能作为待评估方案。')
para('另一个待验证方向是以躯干特征为 Query、手部和面部特征为 Key/Value 的跨分支门控。它希望在动作相关的时间阶段提高局部特征权重。目前该结构尚缺完整同条件结果，应视为后续研究方法，而不是已证实有效的改进。')

heading('十、当前实验设置与结果')
table(['项目','当前设置或结果'],[
('数据集与协议','NTU RGB+D 60，XSub'),
('训练/验证样本','39,627 / 16,306'),
('输入','RTMW-133，raw x/y/score，64 帧，最多 2 人'),
('模型','身体 22 点 CTR-GCN + 双手 42 点、面部 6 token ST-GCN'),
('参数量与训练','907,690；65 轮；随机种子 1'),
('训练 Top-1','99.84%'),
('验证 Top-1 / Top-5','89.60% / 98.82%')],[2.0,4.6])
para('结果表明，从 RGB 视频提取 RTMW-133 骨架并完成分类训练的流程已经跑通，BodyLocalFusion 可以作为当前可复核的分类基线。训练集与验证集约 9.24 个百分点的差距提示模型存在明显过拟合。当前整体结果尚不能单独证明手部、面部、CTR-GCN 风格结构或困难样本重检测各自带来的提升。')

heading('十一、后续研究计划')
table(['工作方向','对照或验证方法','目的'],[
('固定数据版本','生成逐样本 manifest，记录清理、修复、质量和划分信息。','保证复现和公平比较。'),
('验证骨架准确性','人工抽样复核，或使用关键点误差、PCK 等指标。','将工程质量分与实际姿态准确性区分。'),
('区域消融','身体单分支、局部单分支、身体加手部、身体加面部和完整模型。','量化不同身体区域的信息贡献。'),
('结构消融','固定图、CTR-GCN 风格图、统一图与分支图。','判断图结构调整是否有效。'),
('面部表示','68 点直接建模、6 token、去除面部。','评估压缩的计算收益和信息损失。'),
('输入与融合','raw、relative、torso-relative、运动通道及拼接/门控。','比较表示与融合方案。'),
('数据修复效果','修复前后同样本使用同模型进行分类对照。','检验骨架修复能否转化为分类提升。'),
('稳定性和效率','多随机种子、多协议；统计参数量、显存、FLOPs、吞吐和延迟。','评估泛化、波动和计算代价。'),
('局部解释','身体、手部、面部、节点和时间窗口遮挡。','验证模型是否实际依赖局部线索。')],[1.55,3.45,1.6])

heading('十二、研究总结')
para('本研究以 NTU RGB+D 为动作数据来源，参考 RTMW 和 COCO-WholeBody 从 RGB 视频提取 133 点全身骨架，通过人物检测、槽位匹配、质量检查和困难样本重检测控制输入数据质量。在模型方面，参考 ST-GCN 的时空图建模与 CTR-GCN 的通道拓扑细化思想，将身体主体与手部、面部局部信息分支处理，并使用面部 token、有效节点 Mask、时间裁剪和特征融合应对高密度节点带来的噪声和计算成本。')
para('当前 BodyLocalFusion 在 NTU60 XSub 上取得验证 Top-1 89.60%、Top-5 98.82%，形成了可复核的分类基线。后续需要通过骨架准确性验证、同条件消融、多随机种子、区域与时间遮挡，以及修复前后分类对照，确认高精度全身骨架中哪些区域和哪些处理方法真正提升了相似动作的识别能力。')

# Footer with page number field
for section in doc.sections:
    fp=section.footer.paragraphs[0]; fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=fp.add_run('高精度全身骨架细粒度动作识别研究  |  '); run.font.size=Pt(8); run.font.color.rgb=RGBColor(110,120,130)
    fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); fp._p.append(fld)

doc.core_properties.title='基于高精度全身骨架的细粒度人体动作识别研究'
doc.core_properties.subject='研究背景、数据处理、技术依据与方法设计'
doc.core_properties.author=''
doc.save(OUT)
print(OUT)
