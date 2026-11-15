"""Reframe the existing report around research methods and experimental evidence."""
from pathlib import Path
import re, json, shutil, zipfile
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/ISAA完整研究报告_20261011.md'
MD = ROOT / 'docs/全身二维骨架动作识别研究报告_20261011.md'
OUT = MD.with_suffix('.docx')
BUILD = ROOT / 'docs/研究报告图表_20261011'
BUILD.mkdir(exist_ok=True)
text = SOURCE.read_text(encoding='utf-8').split('## 附录 D')[0].rstrip() + '\n'

def paragraph(prefix, new):
    global text
    pattern = r'^' + re.escape(prefix) + r'[^\n]*(?:\n|$)'
    text, count = re.subn(pattern, lambda _: new.rstrip() + '\n', text, flags=re.M)
    assert count == 1, (prefix, count)

def section(start, end, new):
    global text
    a = text.index(start)
    b = text.index(end, a)
    text = text[:a] + new.rstrip() + '\n\n' + text[b:]

paragraph('# ISAA 全身骨架动作识别研究报告', '# 全身二维骨架动作识别研究报告')
paragraph('研究对象：', '研究对象：基于身体 手部和面部关键点的动作识别与可解释性分析  ')
paragraph('项目名称中的 Interpretable', '可解释性是本研究的重要目标。分支拆分、显式几何量和门控提供了分析接口，但并不自动证明解释可靠。身体分支也含有手腕观测，其他分支与残差可能替代手部路径的功能。因此，解释应对应明确的预测对象，并通过分支移除、扰动和稳定性实验验证。')
section('### 1.2 三个代码项目的关系', '### 1.3 核心研究问题', '''### 1.2 研究流程及任务分工

研究流程依次包含姿态提取、时序人物匹配、质量控制、表示构造、时空建模和分类评估。姿态提取把视频转换为关键点；质量控制判断观测是否可用；表示构造定义坐标、掩码及几何关系；时空网络学习动作特征；分类评估检验识别表现；解释实验进一步检验模型实际依赖哪些信息。

上游观测误差会传递到下游，分类网络的能力不能替代缺失人物或关键点的真实观测。不同阶段的样本规模也需分别说明：全量提取审计、困难样本诊断和 NTU60 分类实验具有不同统计对象，不能合并为同一实验。''')
paragraph('CTR-GCN 的 Channel-wise', 'CTR-GCN 的 Channel-wise Topology Refinement 表示按特征通道细化图拓扑。不同通道可以学习不同的节点关系，而非所有通道共享一个固定矩阵。最佳方案使用掩码感知的通道拓扑细化模块；完整三分支方案采用完整的图卷积与时间卷积单元。两者遵循同类方法思想，但节点布局、分支容量、时间降采样和缺失值归一化不同，不能视为完全相同的实验条件。[R6]')
paragraph('图由节点集合和边集合组成。', '图由节点集合和边集合组成。节点对应关节或压缩表示单元，边表示允许汇集信息的关系。邻接矩阵 A 的元素给出节点之间的连接权重；本报告采用 A(u,v) 表示从源节点 v 到目标节点 u 的权重。后续图卷积公式都遵循这一约定。邻接归一化的方向和规则也需明确，因为行归一化与列归一化一般产生不同的信息汇集尺度。')
paragraph('项目的三空间分区顺序', '三个空间分区分别为自连接、布局规定方向的连接及其反向连接，常写为 self、inward、outward。此处的方向由节点布局规定，不是逐帧依据动态几何中心重新判定。图归一化控制不同度数节点的汇集规模。最佳方案从完整图中选取分支节点子矩阵并重新行归一化；完整图网络还采用不同的归一化约定，因此不同方案的图算子需分别定义。')
section('### 5.2 文件命名与 XSub60', '### 5.4', '''### 5.2 跨受试者划分与类别编号

NTU 数据标识包含采集设置、摄像机、受试者、重复次数和动作类别五项信息。采集设置表示场地及采集条件，摄像机编号表示观察视角，受试者编号表示参与者身份，重复编号区分重复录制，动作编号对应任务类别。这些元信息用于划分与审计，受试者身份不得作为分类特征。动作 A001 对应模型标签 0，A060 对应标签 59，共 60 个类别。

XSub60 训练受试者为 1、2、4、5、8、9、13、14、15、16、17、18、19、25、27、28、31、34、35、38，其余协议内受试者进入验证。XSub 按受试者分离，XSet 按采集设置分离，XView 按摄像机视角分离，三者评价不同的泛化能力。本报告完成的主要分类实验采用 XSub60，其他划分不作为已完成结果。

### 5.3 数据容器与在线表示构造

NPZ 是用于保存多个 NumPy 数组的容器，NPY 保存单个数组，ZIP 将多个样本打包压缩。按需读取只在使用样本时解码其数组，内存映射则允许访问预先生成的数组而不一次性加载全部内容。当前压缩数据容量记录为 6,178,363,318 字节。早期基线使用预计算数组，其时间裁剪在生成数组时已固定；主要后续方案按需读取原始三通道，并在训练时构造局部几何。

早期八通道表示依次为原始 x、y、置信度，躯干相对 x、y，跨手距离和跨手方向 x、y。五通道表示只含前三项及相对 x、y，缺少跨手关系。最佳手部输入从原始观测中组合出六通道，详见第 6.4 节。预计算与在线构造可能改变裁剪时机、几何生成条件及随机性，因此早期与后续结果不能直接解释为仅改变网络结构的受控实验。''')
paragraph('最佳模型类为 ', '最佳方案接收完整的 133 点序列，各分支仅处理规定的节点和特征。身体分支处理 22 点；两只手分别处理 21 点并共享权重；68 个面部点压缩为 6 个表示单元。身体、手部和面部输出先沿节点池化，再沿时间对齐和融合。')
paragraph('这是一种固定压缩，并非', '这是一种固定压缩，并非按眼睛、鼻子、嘴等解剖语义划分。六个面部表示单元的空间邻接仅含自连接，因此最佳面部分支的该空间步骤不在不同表示单元之间汇集信息，而是分别进行通道变换，再用时间卷积学习变化。该设计不能被描述为已显式学习完整面部拓扑；解剖分组与表示单元之间的连边应作为后续消融变量。')
section('### 8.1 两个 proximity 模型必须区分', '### 8.2 标准化几何参考', '''### 8.1 交互研究的目标与评估范围

近距离交互可以与完整图骨干结合，也可以作为最佳三分支模型的加性残差。本报告已完成 65 轮训练并统一评估的是后一方案：保留三分支路径的坐标、六通道手部特征和融合方式，新增交互与躯干残差。与完整骨干结合并统一尺度坐标的另一方案尚无同等完整评估，不能把现有数值归给它。

新增设计希望描述手靠近头、躯干或腿脚时的相对运动。它不读取 RGB 物体，不执行物理碰撞检测，也不显式建立人与人之间的边。二维距离近可能只是投影重合，不等于三维接触；距离阈值只定义用于特征汇集的候选关系。''')
paragraph('分类偏置仅在 zBase', '分类偏置仅在 zBase 中出现一次。PI、PT 的权重均零初始化，因此初始新增特征为零；若另行加载预训练三分支权重，这种初始化可保持初始预测一致。但本报告得到 93.0455% 的实验是全部参数从头联合训练，不是对 93.0946% 模型进行微调。零初始化可能影响早期梯度传播，最终交互幅度衰减的原因仍需独立诊断。')
paragraph('但是复用的最佳骨干', '复用的最佳骨干已在整个官方训练部分训练，可能见过内部留出受试者。因此内部留出只对新训练的辅助头形成数据隔离，对整个系统并非完全独立。严格验证应仅用内部拟合受试者从头训练骨干，或使用隔离的交叉拟合。当前仅有可运行方案与小规模流程检查，尚无全量准确率或计算节省结果。')
paragraph('官方训练部分按受试者拆成 fit', '官方训练部分按受试者拆为拟合、搜索和选择三个子集，最终模型、需求表与阈值固定后再做官方评估。教师此前见过完整训练集，严格内部泛化还需隔离教师训练。按通道宽度平方构造的相对成本代理只估算卷积规模，不能替代乘加数、显存或 GPU 延迟。目前仅完成流程检查，尚未完成全量训练与效果验证。')
section('### 10.4 工程组件与职责', '## 11 已完成实验及比较口径', '''### 10.4 计算工具及用途

Python 是科学计算与实验组织的编程语言。PyTorch 提供张量运算、神经网络层、自动微分和优化器；自动微分按计算关系求损失对参数的梯度，供反向传播和优化更新使用。NumPy 用于关键点、置信度和评估数组的数值处理。DataLoader 是 PyTorch 的批次数据读取机制，组织样本、并行预取与批次，不负责决定模型结构或评价指标。

CUDA 是 NVIDIA GPU 的并行计算平台，使张量和卷积等运算在 GPU 上执行；cuDNN 提供神经网络算子的优化实现。TF32 是部分 GPU 矩阵与卷积运算可用的数值计算模式，它改变部分内部精度及执行路径，不能简单理解为所有张量都转换为同一种低精度存储格式。使用 GPU 加速还需要区分数据等待、模型运算、同步和记录开销。

MMPose 提供姿态估计工具与模型支持，RTMW-L 预测全身关键点。RTMDet-tiny 和 YOLO 系列用于人物框检测，其中 tiny 表示较小规模版本，X 表示相应系列的较大规模版本；规模标识本身不能保证特定数据上的召回与速度。它们在本研究中作为预训练推理工具使用，动作分类网络则在骨架序列上训练。

实验记录应覆盖实际参数、逐轮训练与验证指标、选择轮次、逐样本预测及运行环境。统计图展示训练变化与逐类差异，结果表明确分母和评价单位。工具的作用是支持可复核的计算，不替代数据协议、受控实验及结果解释。''')
paragraph('这里是注册参数数量，包括', '这里统计的是模型总参数，包含当前前向计算未使用的部分，因此不能直接等同于活跃参数或部署最小参数量。完整版本耗时约最佳方案的 1.90 倍，新增残差方案约 1.56 倍；时间包含实验记录等主机开销，且后端版本和运行时段不同，不是同环境受控的纯架构性能测试。训练吞吐也不等于单样本推理延迟。')
paragraph('本节在同一个已联合训练的 ', '本节使用同一个已联合训练的三分支加交互及躯干残差模型，在推理时把特定残差置零。它测量终态模型对某条路径的依赖，不是重新训练不含该模块的模型。完整模型保留两条残差；移除交互仅去掉 PI 投影；移除躯干仅去掉 PT 投影；同时移除两者则保留本次联合训练后的三分支路径。')
paragraph('历史 HAPM、ISAA', '早期区域参与度方法将 133 节点映射到 32 个区域，用局部特征、区域注意力和时间片段描述参与程度。区域覆盖头部、肩肘腕、髋膝踝、足部、面部、手部非指尖和各指尖；时间片段是对序列位置的分组。注意力是从特征中学习位置权重并据此汇集信息的机制。区域分数表达网络中的加权或响应，不自动构成经人工验证的动作参与度。')
paragraph('旧 RTMWLocalCTR 路线', '另一种局部细节融合策略通过 133 点固定局部图、低宽度辅助编码、区域内均值与注意力池化，把细节注入 25 或 32 个主节点的通道拓扑细化骨干，并可将面部压缩为表示单元。原型学习中的 prototype 是可学习或统计得到的类别代表向量，用样本与代表向量的相似度辅助分类。这些方法构成研究背景；当前最佳方案不使用原型损失或完整区域参与度机制，不能借用这些概念补上尚未验证的解释结论。')
section('## 16 复现说明', '## 17 结论', '''## 16 实验复核与报告规范

### 16.1 固定数据与评估协议

复核先固定实际样本集合与受试者划分：训练 39,627 个、验证 16,306 个视频，采用 NTU60 跨受试者协议。输入为 RGB 派生全身二维骨架，原始三通道观测，时间窗口 64、最多两个人物槽位。验证使用中心窗口、无随机增强，所有验证视频计入；统一复测批次为 32，使用即时执行模式及 TF32。不同窗口、样本集合、多人策略或模态的结果应单独报告。

报告需同时说明名义规模与实际保留规模，列出逐类样本数、缺失处理、参考坐标与时间填充方式。提取阶段使用动作标签决定期望人数的限制必须披露。准确率复核以逐视频正确数除以实际验证总数，不用逐类召回的简单平均替代整体 Top-1。

### 16.2 记录训练与模型选择条件

每个方案记录初始状态、是否使用预训练、随机种子、训练轮数、优化器、学习率计划、增强、归一化、数据读取条件及硬件软件版本。采用验证集选择最佳轮次时应明确选择指标，并保留完整训练曲线，避免仅报告有利轮次。新增交互与躯干残差的实验必须标注从头联合训练。

多方案比较优先统一训练条件并增加随机种子，报告均值与标准差。种子控制随机生成过程，但不同架构和后端仍可能产生不同随机轨迹及数值误差。现代方案同一权重复测相差一个样本时，必须使用各次配对评估的实际基线，不能把差异归为模型结构收益。

### 16.3 配对验证与成本测量

保存每个验证视频的真实类别、预测类别、真类概率、交叉熵和正确性，并在同一视频上配对比较纠正、退步与共同错误。分支移除与独立训练消融分别标明：前者检验已训练模型的功能依赖，后者检验模块能否带来可重复的学习收益。对于接近的总分，配对统计不能替代多种子结论，也不能自动证明等效。

计算成本同时报告总参数、活跃参数、乘加数、显存和真实延迟。延迟测量固定 GPU、批次、精度和输入形状，预热后同步 GPU，再计时完整预处理、路由、分组、骨干与分类过程，报告中位数及第 95 百分位数。训练吞吐、通道宽度成本代理和稀疏边比例都不能单独证明部署加速。视频全链路评估还应包含解码、检测、姿态估计、跟踪与质量处理。''')
paragraph('ISAA 已建立', '本研究形成了 RGB 全身骨架提取、质量控制、分支时空建模与细粒度评估的研究流程。当前效果最好的已训练方案使用身体 22 点、共享手部 21 点和面部 6 个表示单元，NTU60 跨受试者实际验证集复测 Top-1 为 93.0946%。该结果支持分支式全身骨架在当前数据上的识别能力，但不能单独证明每个设计因素的独立收益，或与其他模态方法的排名关系。')
paragraph('[R7]', '[R7] MMPose 全身姿态估计工具与 RTMW 模型文档。https://github.com/open-mmlab/mmpose 。RTMW-L 在本报告中作为预训练姿态提取工具介绍。')
text = re.sub(r'^\[R8\].*\n?', '', text, flags=re.M)
paragraph('单位为百分数，样本数对应', '单位为百分数，样本数对应实际保留的验证集。B 为最佳三分支模型统一复测，R 为软门控模型，F 为完整三分支图网络，P 为三分支加交互及躯干残差。B 与 R、P 配对；F 的配对基线在 A036 少一个正确，其余类别相同，总正确数为 15,179。此一例差异不作为结构效应。类别名称为 NTU60 动作的中文表述；逐类数字按该类真实视频计分，即该类召回率。')
text = text[:text.index('## 附录 C 配置数值与损失权重')] + '''## 附录 C 参数数值与作用

完整图网络的身体、手部和面部分支均采用 10 层：64 通道 4 层、128 通道 3 层、256 通道 3 层；步长依次为 1、1、1、1、2、1、1、2、1、1。身体与手部采用通道拓扑细化，面部采用固定图时空卷积，面部随机失活比例为 0。随机失活又称 dropout，训练时随机屏蔽部分特征以进行正则化，比例为 0 表示未采用该机制。

近距离交互编码宽度为 64，进入与保留边的距离阈值为 0.40 和 0.48，单位均为躯干尺度比；有效点置信度门槛为 0，即仅使用正置信度观测。共同训练条件见第 10 章。下表是类别自适应宽度方案的默认研究设置，其全量性能尚未验证。

| 自适应宽度参数 | 数值 | 含义与作用 |
| --- | --- | --- |
| 三阶段最大宽度与通道组 | 64、128、256；每组 8 | 限定候选宽度及调整粒度 |
| 路由器隐藏宽度 | 128 | 路由中间特征的维数 |
| 高置信与回退阈值 | 0.80、0.45 | 选择单类宽度或共享配置的概率界限 |
| 候选类别数 | 3 | 中等置信时合并前三类需求 |
| 超网 门控 路由训练轮数 | 40、10、15 | 分别优化共享权重、宽度候选和预测路由 |
| 对应基础学习率 | 0.10、0.01、0.01 | 控制三个阶段的更新步幅 |
| 蒸馏权重与温度 | 0.5、2.0 | 控制教师分布损失及软化程度 |
| 宽度惩罚与搜索成本权重 | 0.01、0.05 | 分别作用于软选宽及候选成本偏好 |
| 最大搜索轮数 | 12 | 控制经验搜索预算 |
| 搜索正样本与负样本数 | 各 64 | 检查目标类别及混淆影响 |
| 每类共享样本数 | 16 | 构成共享评估子集 |
| 搜索准确率容差 | 0.02 | 小数比例，即 2 个百分点 |
| 搜索损失容差 | 0.03 | 平均交叉熵数值的允许变化 |
| 路由准确率容差 | 0.002 | 小数比例，即 0.2 个百分点 |
| 每类最少校准样本 | 5 | 不足时使用保守回退策略 |
| 每个宽度的 BN 校准批数 | 4 | 估计该宽度的运行均值及方差 |
| 内部留出比例 | 0.20 | 留出官方训练部分的 20% 用于内部校准 |

宽度惩罚和搜索成本权重分别控制不同训练或选择阶段，不能合为同一个速度系数。上述数值是默认设置，不是已经证明最优的参数。校准集与教师训练数据的隔离限制见第 9 章。

身体优先硬手部路由中，身体分类与无手分类损失权重均为 1，完整路径蒸馏损失权重为 0.5，蒸馏温度为 2。辅助头训练示例为 10 轮，学习率 0.01、预热 2 轮，在零基第 7 和第 9 轮进行衰减，即通常编号的第 8 与第 10 轮。

手部需求映射的收益间隔为 0、温度为 0.1；手部调用需求阈值为 0.5，身体置信度低于 0.8 时保守调用手。收益间隔表示要求无手与完整路径的平均交叉熵差至少超过多少；需求温度控制该差映射为分数时的陡峭程度。它与蒸馏温度用途不同，不得混用。
'''

# Remove local evidence labels while preserving academic references.
text = re.sub(r'\[E\d+\]\s*至\s*\[E\d+\]', '', text)
text = re.sub(r'\[E\d+\]', '', text)
replacements = [
    ('ISAA 研究如何', '本研究考察如何'),
    ('ISAA 的分类网络', '动作分类网络'),
    ('ISAA 最佳模型的面部 LocalSTBlock', '最佳方案的简化面部时空模块'),
    ('项目已实现身体优先的硬手部路由和按类别选择通道宽度的完整 GCN，但尚无可确认的全量性能结果。', '身体优先的硬手部路由和按类别选择通道宽度已形成可运行方案，但尚无全量性能验证。'),
    ('目前最好的已训练模型为 body-local-hand-ctr-wide-relative。', '当前最佳方案为身体 手部和面部三分支相对坐标模型。'),
    ('body-local-hand-ctr-wide-relative-routed', '软门控三分支模型'),
    ('class-adaptive-gcn', '类别自适应宽度模型'),
    ('class-routed', '身体优先的硬路由模型'),
    ('历史 BodyLocal', '早期身体与局部融合基线'),
    ('BodyLocalFullGCNFusion', '完整三分支图网络'),
    ('BodyLocal', '身体与局部融合基线'),
    ('最佳 wide-relative', '最佳三分支'),
    ('wide-relative', '三分支相对坐标'),
    ('软门控 routed', '软门控'),
    ('新增 proximity', '新增交互与躯干残差'),
    ('新 proximity', '新增残差模型'),
    ('最佳结构加 proximity', '三分支加交互与躯干残差'),
    ('加性 proximity', '加性交互与躯干残差'),
    ('proximity', '近距离交互'),
    ('CTRGCNBlock', '掩码感知通道拓扑细化模块'),
    ('CTRGraphConv', '通道拓扑细化图卷积'),
    ('LocalSTBlock', '简化局部时空模块'),
    ('LocalST', '简化局部时空模块'),
    ('STGCNBackbone', '时空图骨干'),
    ('PointBatchNorm', '掩码感知节点归一化'),
    ('TorsoCoordinates', '躯干参考坐标变换'),
    ('remove_interaction', '移除交互'),
    ('remove_torso', '移除躯干'),
    ('remove_both', '同时移除两者'),
    ('| full |', '| 完整模型 |'),
    ('full 与 ', '完整模型与 '),
    ('full 相对该模式', '完整模型相对该模式'),
    ('原最佳 checkpoint', '独立训练的最佳模型权重'),
    ('checkpoint', '训练终态权重'),
    ('gate.weight', '门控层权重'),
    ('gate.bias', '门控层偏置'),
    ('scatter', '按区域汇集'),
    ('gather 选中的样本', '选择需要手部计算的样本'),
    ('manifest', '逐视频样本清单'),
    ('工程上由 MMPose 支持', '由 MMPose 提供工具支持'),
    ('本项目', '本研究'),
    ('项目的', '研究中的'),
    ('项目已经', '该方法已经'),
    ('项目自定义质量规则', '本研究定义的质量规则'),
    ('项目边界', '适用范围'),
    ('工程实现', '计算工具'),
    ('工程组件', '计算工具'),
    ('第 9 至 10 章介绍路由、自适应宽度、训练及计算工具', '第 9 至 10 章介绍路由、自适应宽度、训练及计算工具'),
    ('参数与证据清单', '参数与作用说明'),
    ('“已有实现”', '“已有可运行方案”'),
    ('术语定义、代码路径、实际训练和可复核结果', '术语定义、模型设计、训练条件和可复核结果'),
    ('支持某种协议的加载接口', '采用某种协议'),
    ('本研究中的含义', '本研究中的含义'),
    ('torch.compile', 'PyTorch 编译机制'),
    ('BatchNorm2d', '二维批次归一化'),
    ('Conv1d', '一维卷积'),
    ('detached 坐标', '不参与梯度传播的坐标'),
    ('detached 身体特征', '不参与该支路梯度传播的身体特征'),
    ('训练实现与损失记录', '训练监督与损失记录'),
    ('将其称作', '将其称为'),
]
for a, b in replacements:
    text = text.replace(a, b)
text = text.replace('研究目标与项目边界', '研究目标与适用范围')
text = text.replace('研究目标与适用范围', '研究目标与适用范围')
text = text.replace('| 项目 |', '| 统计项 |')
text = text.replace('原始 NPZ 中 keypoints 的形状为', '原始关键点数组的形状为')
text = text.replace('scores 为 T×M×133', '置信度数组为 T×M×133')
text = text.replace('raw 路径', '原始坐标表示')
text = text.replace('sigmoid(0)', 'sigmoid(0)=0.5')
text = text.replace('实现约束', '结构约束')
text = text.replace('十层堆叠和显式配置', '十层堆叠和明确的阶段结构')
text = text.replace('该方案已有实现', '该方案已可运行')
text = text.replace('一次实现与训练', '一次建模与训练')
text = text.replace('已有实现', '已有可运行方案')
text = text.replace('当前验证复测结果', '当前验证集复测结果')

# Explain recurring public terms without exposing source structure.
paragraph('主要 GPU 为 ', '主要 GPU 为 NVIDIA RTX A6000。最佳方案的训练记录使用 PyTorch 2.14.0+cu126、CUDA 12.6、cuDNN 91002；完整图网络与新增交互及躯干残差使用 PyTorch 2.14.0+cu130、CUDA 13.0、cuDNN 92400，Linux Python 3.10.12。版本后缀 cu126、cu130 表示对应的 CUDA 构建版本。编译训练并不意味着所有操作都进入编译执行；部分简化时空模块和动态筛边保持即时执行，以适应动态形状及执行图限制。')
paragraph('PyTorch 编译机制 通过', '编译通过捕获张量运算关系并由 Inductor 等后端优化执行，减少语言调度开销或优化算子；eager 即即时执行，按当前操作顺序直接运行。CUDA Graph 是复用固定 GPU 执行图的机制，与编译本身不同。自适应多宽度方案对稳定结构分块编译，随机宽度与搜索使用即时执行，并关闭固定执行图捕获，以支持同批多次反向传播与梯度累积。接口检查成功不等于已经测得真实加速。')
text = text.replace('第 14 至 16 章讨论可解释性、局限、后续研究及复现', '第 14 至 16 章讨论可解释性、局限、后续研究及实验复核')
text = text.replace('该头读取不参与该支路梯度传播的身体特征，因此', '粗分类头以停止梯度的身体特征作为输入，因此')
text = text.replace('sandwich 每批运行', '夹层式宽度训练（sandwich）在每批运行')
text = text.replace('soft target', '软目标概率分布')
text = text.replace('pTeacher(T)‖pStudent(T)', 'pTeacher(T)‖pStudent(T)')
text = text.replace('教师高准确率不等于学生自动继承其准确率。', 'KL 表示 Kullback–Leibler 散度，度量两种概率分布的差异；教师和学生分别指提供参考分布的大模型与被训练的目标模型。教师高准确率不等于学生自动继承其准确率。')
text = text.replace('报告中位数和 P95', '报告中位数和 P95（第 95 百分位数）')
text = text.replace('总 FLOPs', '总浮点运算数 FLOPs')
text = text.replace('profiler', '性能分析工具')
text = text.replace('score', '置信度')
text = text.replace('motion_valid', '运动有效标记')
text = text.replace('radius_on=0.40', '进入阈值 0.40').replace('radius_off=0.48', '保留阈值 0.48')
text = text.replace('width=64', '宽度 64')
text = text.replace('conf 控制', '控制').replace('imgsz 是', '检测输入尺寸参数是').replace('crop margin 在', '裁剪留边在')
text = text.replace('conf=0.15、NMS IoU=0.70、imgsz=960、crop margin=0.10', '检测置信度阈值 0.15、NMS 的 IoU 阈值 0.70、输入尺寸 960、裁剪留边比例 0.10')
text = text.replace('body 分支', '身体分支')
text = text.replace('名称“body”是分支名', '名称“身体”是分支名')
text = text.replace('全量性能结果', '全量性能验证结果')
text = text.replace('独立训练的独立训练的', '独立训练的')
text = text.replace('一个终态训练终态权重', '一份训练终态权重')
text = text.replace('早期身体与局部融合基线 基线', '早期身体与局部融合基线')
text = text.replace('双人携物等动作', '双人靠近和遮挡等情形')
text = text.replace('双手插兜', '手伸进口袋')
text = text.replace('检测阈值 控制', '检测阈值控制')
text = text.replace('检测输入尺寸参数是检测输入分辨率参数', '输入尺寸决定检测器处理图像的分辨率')
text = text.replace('裁剪留边在人物框外留边', '裁剪留边是在人物框外增加边界余量')
text = text.replace('早期少量诊断帧也显示检测器存在速度与召回权衡', '早期少量诊断帧也显示检测器存在速度与召回权衡，ms 表示毫秒')
text = text.replace('PCK/AP', 'PCK（正确关键点比例）或 AP（平均精度）')
text = text.replace('ST-GCN 将关节', 'ST-GCN 是 Spatial Temporal Graph Convolutional Network，即时空图卷积网络。GCN 表示 Graph Convolutional Network，即图卷积网络。ST-GCN 将关节')
text = text.replace('2s-AGCN 在固定', '2s-AGCN 是 Two-Stream Adaptive Graph Convolutional Network，即双流自适应图卷积网络。它在固定')
text = text.replace('原论文的 joint/bone 双流集成', '原论文的关节与骨向量双流集成')
text = text.replace('RGB 视频中提取', 'RGB（红 绿 蓝三通道彩色）视频中提取', 1)
text = text.replace('官方数据还提供 RGB、深度和 Kinect 骨架等模态', '官方数据还提供 RGB、深度和 Kinect 骨架等模态。深度表示相机与场景表面的距离，Kinect 是用于采集深度及人体骨架的传感设备')
text = text.replace('以下统一用 B 表示', '张量是按多个轴组织的数值数组；形状给出每个轴的长度，通道则表示同一位置的不同数值属性。以下统一用 B 表示')
text = text.replace('分类 logits 的形状为 B×K', '分类 logits 是尚未转为概率的实数类别分数，其形状为 B×K')
text = text.replace('NaN 和无穷数需要清洗', 'NaN 表示无法定义的数值；它和正负无穷数都需要清洗')
text = text.replace('再用 SGD 更新', '再用随机梯度下降（SGD）更新')
text = text.replace('不同 kernel 的数值差异', '不同计算内核的数值差异')
text = text.replace('valid_frame_mask', '有效帧掩码')
text = text.replace('置信度归一化 auto', '置信度自动归一化')
text = text.replace('身体 手部和面部关键点', '身体、手部和面部关键点')
text = text.replace('身体 手部和面部三分支相对坐标模型', '身体、手部和面部三分支相对坐标模型')
text = text.replace('最佳方案的简化面部时空模块 是基于固定邻接矩阵和时间卷积的简化实现', '最佳方案的面部时空模块采用固定邻接矩阵和时间卷积构造')
text = text.replace('head_face', '头部与面部').replace('left_arm 与 right_arm', '左臂与右臂').replace('left_leg 与 right_leg', '左腿与右腿').replace('left_foot 与 right_foot', '左足与右足')
text = text.replace('| torso |', '| 躯干 |').replace('肩髋被分到 torso，因此 arm 与 leg 在这里仅含肘腕或膝踝', '肩髋被分到躯干，因此臂部与腿部在这里仅含肘腕或膝踝')
text = text.replace('T²·KL(pTeacher(T)‖pStudent(T))。这里 T 是温度参数，不是时间长度；报告公式中明确写 temperatureDistill 以避免符号混用。', 'τ²·KL(pTeacher(τ)‖pStudent(τ))。这里 τ 是蒸馏温度，控制类别分数转为概率时的软化程度；它与时间长度 T 的含义不同。')
text = text.replace('| 训练 / 验证 batch |', '| 训练与验证批次大小 |')
text = text.replace('| epochs 与 seed |', '| 训练轮数与随机种子 |')
text = text.replace('| weight decay |', '| 权重衰减 |')
text = text.replace('| warmup |', '| 学习率预热 |')
text = text.replace('| 每次 lr 衰减 |', '| 每次学习率衰减 |')
text = text.replace('| grad accumulation |', '| 梯度累积批数 |')
text = text.replace('| drop_last | true |', '| 丢弃训练尾批 | 是 |')
text = text.replace('| workers / prefetch |', '| 读取进程与预取批数 |')
text = text.replace('每 worker 预取批数', '每个读取进程的预取批数')
text = text.replace('| compile | 现代训练开启 | reduce-overhead；评估按记录指定 |', '| 编译执行 | 后续主要训练开启 | 减少运行开销；统一复测使用即时执行 |')
text = text.replace('batch32 且 drop_last', '每批 32 且丢弃训练尾批')
text = text.replace('warmup 前 5 轮', '学习率预热的前 5 轮')
text = text.replace('人类编号', '从 1 开始计数')
text = text.replace('torso context', '躯干上下文特征')
text = text.replace('躯干 context', '躯干上下文特征')
text = text.replace('最小和随机宽度，平均分类与蒸馏损失', '最小和随机宽度，对分类与蒸馏损失取平均')
text = text.replace('all、none、adaptive 三种评估', '完整调用、完全不用手、自适应调用三种评估')
text = text.replace('不能将 none 直接解释为', '不能将完全不用手直接解释为')
text = text.replace('top-k 候选类', '概率最高的 k 个候选类')
text = text.replace('fit 部分', '拟合子集')
text = text.replace('训练与验证轮时', '每轮训练与验证耗时')
text = text.replace('all/none/adaptive 手路由；max/shared/adaptive 选宽', '完整调用、无手、自适应手路由；最大、共享、自适应宽度')
text = '\n'.join(line if line.startswith(('#', '|')) else re.sub(r'(?<=[\u4e00-\u9fff]) +(?=[\u4e00-\u9fff])', '', line) for line in text.split('\n'))
text = re.sub(r' +([。，；])', r'\1', text)
text = re.sub(r'\n{4,}', '\n\n\n', text)

# Give each illustration a research-facing name and preserve its data.
old_builder = (ROOT / 'tools/build_complete_report_20261011.py').read_text(encoding='utf-8')
arch_code = old_builder.split("canvas=Image.new('RGB'")[1].split('figures={')[0]
arch_code = "canvas=Image.new('RGB'" + arch_code
arch_code = arch_code.replace('最佳 wide-relative 模型的信息流', '身体 手部和面部三分支模型的信息流')
arch_code = arch_code.replace('4 层 LocalST · 24 / 48', '4 层简化时空卷积 · 24 / 48')
arch_code = arch_code.replace('best_architecture.png', '三分支信息流.png')
exec(compile(arch_code, '<architecture>', 'exec'))
asset_sources = {
    '完整图网络训练曲线.png': ROOT / 'outputs/evaluations/full_gcn_20261010/training_curves.png',
    '新增残差逐类变化.png': ROOT / 'outputs/evaluations/best_hand_body_proximity_20261011/class_changes.png',
    '新增残差训练曲线.png': ROOT / 'outputs/evaluations/best_hand_body_proximity_20261011/training_curves.png',
}
for name, source in asset_sources.items():
    shutil.copyfile(source, BUILD / name)
figures = {
    '### 6.1 分支选择与网络尺寸': (ARCH, '图 1 三分支模型的输入 分支和融合'),
    '### 11.5 训练曲线与完整 GCN 异常': (BUILD / '完整图网络训练曲线.png', '图 2 完整图网络与最佳模型的训练记录'),
    '### 12.3 新增残差模型 的类间重新分配': (BUILD / '新增残差逐类变化.png', '图 3 新增残差模型的逐类提升与下降'),
    '### 13.1 推理移除的实验定义': (BUILD / '新增残差训练曲线.png', '图 4 新增残差模型与最佳模型的训练记录'),
}
# Replacement above retains the original single separator after model names.
class_heading = next(line for line in text.splitlines() if line.startswith('### 12.3'))
figure = figures.pop('### 12.3 新增残差模型 的类间重新分配')
figures[class_heading] = figure
text = re.sub(r'^!\[图 .*?\]\(.*?\)\s*$', '', text, flags=re.M)
for heading, (path, caption) in figures.items():
    assert heading in text, heading
    text = text.replace(heading, heading + '\n\n![' + caption + '](' + BUILD.name + '/' + path.name + ')', 1)

# Confirm this revision preserves the complete experimental table and formulas.
for forbidden in ['ISAA', 'isaa', 'HumanActionParticipationModeling', 'nturgbd_skeleton_extraction', 'HAPM', 'RTMWLocalCTR', '.py', '.json', '.csv', '```', '[E', 'variant', 'BodyLocal', 'FullGCNFusion', 'LocalST', 'checkpoint', '项目', '代码', '源文件', 'outputs/']:
    assert forbidden not in text, forbidden
assert len(re.findall(r'^> 公式 \d+：', text, re.M)) == 24
assert len(re.findall(r'^\| A\d{3} \|', text, re.M)) == 60
MD.write_text(text, encoding='utf-8')

# Reuse only the established report renderer, bypassing the old evidence builder.
renderer = 'doc=Document();sec=doc.sections[0]' + old_builder.split('doc=Document();sec=doc.sections[0]', 1)[1]
renderer = renderer.replace("doc.core_properties.title='ISAA 全身骨架动作识别研究报告'", "doc.core_properties.title='全身二维骨架动作识别研究报告'")
renderer = renderer.replace("doc.core_properties.author='ISAA'", "doc.core_properties.author='研究报告'")
renderer = renderer.replace("doc.core_properties.subject='数据 方法 定义 实验与复现'", "doc.core_properties.subject='数据 方法 定义 实验与研究分析'")
renderer = renderer.replace('p=c.paragraphs[0];p.paragraph_format.space_after=Pt(0);', 'p=c.paragraphs[0];p.paragraph_format.keep_with_next=(i==0);p.paragraph_format.space_after=Pt(0);')
renderer = renderer[:renderer.index('\nprint(json.dumps(')]
exec(compile(renderer, '<report renderer>', 'exec'))
with zipfile.ZipFile(OUT) as package:
    for path in ['word/document.xml', 'docProps/core.xml']:
        content = package.read(path).decode('utf-8')
        assert not any(s in content for s in ['ISAA', 'HumanActionParticipationModeling', 'nturgbd_skeleton_extraction', 'BodyLocal', '代码']), path
print(json.dumps({'docx': str(OUT), 'markdown': str(MD), 'characters': len(text), 'tables': len(doc.tables), 'images': len(doc.inline_shapes), 'equations': 24, 'class_rows': 60}, ensure_ascii=False))
